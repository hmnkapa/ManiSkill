from enum import IntEnum
from typing import Any, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


class PullCubeSkillPhase(IntEnum):
    ALIGN = 0
    PULL = 1
    DONE = 2


class PullCubeSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PullCubeSkillPhase.ALIGN),
            dtype=torch.long,
            device=device,
        )

    def _normalize_env_idx(self, env_idx=None):
        if env_idx is None:
            return None
        if torch.is_tensor(env_idx):
            return env_idx.to(device=self.phase.device, dtype=torch.long).flatten()
        return torch.as_tensor(
            env_idx, device=self.phase.device, dtype=torch.long
        ).flatten()

    def _select(self, tensor: torch.Tensor, env_idx=None) -> torch.Tensor:
        env_idx = self._normalize_env_idx(env_idx)
        if env_idx is None:
            return tensor
        return tensor[env_idx]

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        obj_pos = env.obj.pose.p
        goal_pos = env.goal_region.pose.p
        tcp_pull_pos = obj_pos + torch.tensor(
            [env.cube_half_size + 0.01, 0, 0], device=env.device
        )
        tcp_to_pull_dist = torch.linalg.norm(
            env.agent.tcp.pose.p - tcp_pull_pos, dim=1
        )
        object_goal_point_world = goal_pos.clone()
        object_goal_point_world[:, 2] = obj_pos[:, 2]
        pull_direction_world = torch.zeros_like(obj_pos)
        pull_direction_world[:, 0] = -1
        signals = {
            "success": info["success"],
            "reached_pull_pose": tcp_to_pull_dist < 0.01,
            "tcp_to_pull_dist": tcp_to_pull_dist,
            "obj_to_goal_dist": torch.linalg.norm(
                obj_pos[:, :2] - goal_pos[:, :2], dim=1
            ),
            "pull_direction_world": pull_direction_world,
            "object_goal_point_world": object_goal_point_world,
        }
        if env_idx is None:
            return signals
        return {
            key: value[env_idx] if torch.is_tensor(value) else value
            for key, value in signals.items()
        }

    def reset(self, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        if env_idx is None:
            self.phase.fill_(int(PullCubeSkillPhase.ALIGN))
        else:
            self.phase[env_idx] = int(PullCubeSkillPhase.ALIGN)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        reached_pull_pose = signals["reached_pull_pose"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        align = phase == int(PullCubeSkillPhase.ALIGN)
        pull = phase == int(PullCubeSkillPhase.PULL)
        phase[align & reached_pull_pose] = int(PullCubeSkillPhase.PULL)
        phase[pull & success] = int(PullCubeSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        align = phase == int(PullCubeSkillPhase.ALIGN)
        pull = phase == int(PullCubeSkillPhase.PULL)
        done = phase == int(PullCubeSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[align] = SKILL_IDS["push"]
        skill_id[pull] = SKILL_IDS["push"]

        obj_pos = self._select(env.obj.pose.p, env_idx).reshape(-1, 3)
        goal_pos = self._select(env.goal_region.pose.p, env_idx).reshape(-1, 3)
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))

        align_target_pos = obj_pos + torch.tensor([0.03, 0, 0], device=env.device)
        pull_target_pos = obj_pos.clone()
        pull_target_pos[:, 0] = goal_pos[:, 0] + 0.05
        pull_target_pos[:, 1] = goal_pos[:, 1]
        align_target_pose = Pose.create_from_pq(p=align_target_pos, q=tcp_pose.q)
        pull_target_pose = Pose.create_from_pq(p=pull_target_pos, q=tcp_pose.q)

        target_pose_world = torch.where(
            pull[:, None, None],
            pull_target_pose.to_transformation_matrix(),
            align_target_pose.to_transformation_matrix(),
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.zeros(
            phase.shape, dtype=torch.float32, device=phase.device
        )
        target_gripper_width[done] = float("nan")

        phase_names_by_id = {
            int(PullCubeSkillPhase.ALIGN): "align",
            int(PullCubeSkillPhase.PULL): "pull",
            int(PullCubeSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PullCubeSkillPhase.ALIGN): "push",
            int(PullCubeSkillPhase.PULL): "push",
            int(PullCubeSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PullCubeSkillPhase.ALIGN): "move_tcp_to_cube_pull_side",
            int(PullCubeSkillPhase.PULL): "pull_cube_to_goal",
            int(PullCubeSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PullCubeSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "PullCube-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "reached_pull_pose": signals["reached_pull_pose"],
            "tcp_to_pull_dist": signals["tcp_to_pull_dist"],
            "obj_to_goal_dist": signals["obj_to_goal_dist"],
            "pull_direction_world": signals["pull_direction_world"],
            "object_goal_point_world": signals["object_goal_point_world"],
        }

        return SkillAnnotationContext(
            skill_id=skill_id,
            skill=[skill_names_by_phase_id[x] for x in phase_ids],
            phase_id=phase,
            phase=[phase_names_by_id[x] for x in phase_ids],
            skill_state=[skill_states_by_phase_id[x] for x in phase_ids],
            target_point_world=target_point_world,
            target_pose_world=target_pose_world,
            target_gripper_width=target_gripper_width,
            active_object=["cube"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PullCube-v1", max_episode_steps=50)
class PullCubeEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to pull a cube onto a target.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1].
    - the target goal region is marked by a red and white target. The position of the target is fixed to be the cube's xy position - [0.1 + goal_radius, 0]

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PullCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[-0.5, 0.0, 0.25], target=[0.2, 0.0, -0.5])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # create cube
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        # create target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])

            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PullCubeSkillFSM):
            fsm = PullCubeSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_region.pose.p,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # grippers should close and pull from behind the cube, not grip it
        # distance to backside of cube (+ 2*0.005) sufficiently encourages this
        tcp_pull_pos = self.obj.pose.p + torch.tensor(
            [self.cube_half_size + 2 * 0.005, 0, 0], device=self.device
        )
        tcp_to_pull_pose = tcp_pull_pos - self.agent.tcp.pose.p
        tcp_to_pull_pose_dist = torch.linalg.norm(tcp_to_pull_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_pull_pose_dist)
        reward = reaching_reward

        reached = tcp_to_pull_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

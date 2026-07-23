from enum import IntEnum
from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


class PokeCubeSkillPhase(IntEnum):
    PICK = 0
    ALIGN = 1
    PUSH = 2
    DONE = 3


class PokeCubeSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PokeCubeSkillPhase.PICK),
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

    def reset(self, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        if env_idx is None:
            self.phase.fill_(int(PokeCubeSkillPhase.PICK))
        else:
            self.phase[env_idx] = int(PokeCubeSkillPhase.PICK)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        is_peg_grasped = info["is_peg_grasped"]
        is_peg_cube_fit = info["is_peg_cube_fit"]
        success = info["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()
            is_peg_grasped = is_peg_grasped[env_idx]
            is_peg_cube_fit = is_peg_cube_fit[env_idx]
            success = success[env_idx]

        pick = phase == int(PokeCubeSkillPhase.PICK)
        align = phase == int(PokeCubeSkillPhase.ALIGN)
        push = phase == int(PokeCubeSkillPhase.PUSH)
        active = align | push
        phase[pick & is_peg_grasped] = int(PokeCubeSkillPhase.ALIGN)
        phase[align & is_peg_grasped & is_peg_cube_fit] = int(
            PokeCubeSkillPhase.PUSH
        )
        phase[push & success] = int(PokeCubeSkillPhase.DONE)
        phase[active & ~is_peg_grasped & ~success] = int(PokeCubeSkillPhase.PICK)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(PokeCubeSkillPhase.PICK)
        align = phase == int(PokeCubeSkillPhase.ALIGN)
        push = phase == int(PokeCubeSkillPhase.PUSH)
        done = phase == int(PokeCubeSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[align] = SKILL_IDS["push"]
        skill_id[push] = SKILL_IDS["push"]

        peg_pose = Pose.create(self._select(env.peg.pose.raw_pose, env_idx))
        cube_pose = Pose.create(self._select(env.cube.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        peg_head_offset = Pose.create(
            self._select(env.peg_head_offsets.raw_pose, env_idx)
        )

        pick_target_pose = build_panda_topdown_grasp_pose(
            center=peg_pose.p,
            tcp_pose=tcp_pose,
            object_pose=peg_pose,
        )

        peg_head_pose = peg_pose * peg_head_offset
        desired_peg_head_pos = peg_head_pose.p.clone()
        desired_peg_head_pos[:, :2] = cube_pose.p[:, :2]
        desired_peg_head_pose = Pose.create_from_pq(
            p=desired_peg_head_pos,
            q=cube_pose.q,
        )
        desired_peg_pose = desired_peg_head_pose * peg_head_offset.inv()
        align_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=peg_pose,
            desired_object_pose=desired_peg_pose,
        )

        desired_cube_pos = cube_pose.p.clone()
        goal_pos = self._select(env.goal_region.pose.p, env_idx).reshape(-1, 3)
        desired_cube_pos[:, :2] = goal_pos[:, :2]
        desired_cube_pose = Pose.create_from_pq(p=desired_cube_pos, q=cube_pose.q)
        push_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=cube_pose,
            desired_object_pose=desired_cube_pose,
        )

        target_pose_world = pick_target_pose.to_transformation_matrix()
        target_pose_world = torch.where(
            align[:, None, None],
            align_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world = torch.where(
            push[:, None, None],
            push_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.full(
            phase.shape,
            float("nan"),
            dtype=torch.float32,
            device=phase.device,
        )
        target_gripper_width[pick | align | push] = 0.045

        phase_names_by_id = {
            int(PokeCubeSkillPhase.PICK): "pick",
            int(PokeCubeSkillPhase.ALIGN): "align",
            int(PokeCubeSkillPhase.PUSH): "push",
            int(PokeCubeSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PokeCubeSkillPhase.PICK): "pick",
            int(PokeCubeSkillPhase.ALIGN): "push",
            int(PokeCubeSkillPhase.PUSH): "push",
            int(PokeCubeSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PokeCubeSkillPhase.PICK): "move_to_peg",
            int(PokeCubeSkillPhase.ALIGN): "align_peg_with_cube",
            int(PokeCubeSkillPhase.PUSH): "push_cube_to_goal",
            int(PokeCubeSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PokeCubeSkillPhase.DONE) else "tcp" for x in phase_ids
        ]

        info = env.evaluate()
        task_meta = {
            "task": "PokeCube-v1",
            "target_frame": "world",
            "success": self._select(info["success"], env_idx),
            "is_cube_placed": self._select(info["is_cube_placed"], env_idx),
            "is_peg_cube_fit": self._select(info["is_peg_cube_fit"], env_idx),
            "is_peg_grasped": self._select(info["is_peg_grasped"], env_idx),
            "angle_diff": self._select(info["angle_diff"], env_idx),
            "head_to_cube_dist": self._select(info["head_to_cube_dist"], env_idx),
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
            active_object=["peg"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PokeCube-v1", max_episode_steps=50)
class PokeCubeEnv(BaseEnv):
    r"""
    **Task Description:**
    A simple task where the objective is to poke a red cube with a peg and push it to a target goal position.

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table
    - the cube's x-coordinate is fixed to peg's x-coordinate + peg half-length (0.12) + 0.1 and y-coordinate is randomized in range [-0.1, 0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized in range [-$\pi$/ 6, $\pi$ / 6]
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.05 + goal_radius, 0]

    **Success Conditions:**
    - the cube's xy position is within goal_radius (default 0.05) of the target's xy position by euclidean distance
    - the robot is static
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PokeCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    cube_half_size = 0.02
    peg_half_width = 0.025
    peg_half_length = 0.12
    goal_radius = 0.05

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.2, 0.2, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[1, 0, self.cube_half_size]),
        )

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.peg_half_width]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

        self.peg_head_offsets = Pose.create_from_pq(
            p=[self.peg_half_length, 0, 0], device=self.device
        )

    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # initialize the peg
            peg_xyz = torch.zeros((b, 3))
            peg_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            peg_xyz[..., 2] = self.peg_half_width
            peg_q = [1, 0, 0, 0]
            peg_pose = Pose.create_from_pq(p=peg_xyz, q=peg_q)
            self.peg.set_pose(peg_pose)
            # initialize the cube
            cube_xyz = torch.zeros((b, 3))
            cube_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            cube_xyz[..., 0] = peg_xyz[..., 0] + self.peg_half_length + 0.1
            cube_xyz[..., 2] = self.cube_half_size
            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
            )
            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)
            # initialize the goal region
            goal_region_xyz = cube_xyz + torch.tensor([0.05 + self.goal_radius, 0, 0])
            goal_region_xyz[..., 2] = 1e-3
            goal_region_q = euler2quat(0, np.pi / 2, 0)
            goal_region_pose = Pose.create_from_pq(p=goal_region_xyz, q=goal_region_q)
            self.goal_region.set_pose(goal_region_pose)
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PokeCubeSkillFSM):
            fsm = PokeCubeSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                peg_pose=self.peg.pose.raw_pose,
                goal_pos=self.peg.pose.p,
                tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp.pose.p,
                peg_to_cube_pos=self.cube.pose.p - self.peg.pose.p,
                cube_to_goal_pos=self.goal_region.pose.p - self.cube.pose.p,
                peghead_to_cube_pos=self.peg_head_pos - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_cube_placed = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        peg_q = self.peg_head_pose.q
        peg_qmat = rotation_conversions.quaternion_to_matrix(peg_q)
        peg_euler = rotation_conversions.matrix_to_euler_angles(peg_qmat, "XYZ")
        cube_q = self.cube.pose.q
        cube_qmat = rotation_conversions.quaternion_to_matrix(cube_q)
        cube_euler = rotation_conversions.matrix_to_euler_angles(cube_qmat, "XYZ")
        angle_diff = torch.abs(peg_euler[:, 2] - cube_euler[:, 2])
        is_peg_cube_aligned = angle_diff < 0.05

        head_to_cube_dist = torch.linalg.norm(
            self.peg_head_pos[..., :2] - self.cube.pose.p[..., :2], axis=1
        )
        is_peg_cube_close = head_to_cube_dist <= self.cube_half_size + 0.005

        is_peg_cube_fit = torch.logical_and(is_peg_cube_aligned, is_peg_cube_close)
        is_peg_grasped = self.agent.is_grasping(self.peg)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_cube_placed & is_robot_static,
            "is_cube_placed": is_cube_placed,
            "is_peg_cube_fit": is_peg_cube_fit,
            "is_peg_grasped": is_peg_grasped,
            "angle_diff": angle_diff,
            "head_to_cube_dist": head_to_cube_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reach peg
        tcp_pos = self.agent.tcp.pose.p
        tgt_tcp_pose = self.peg.pose
        tcp_to_peg_dist = torch.linalg.norm(tcp_pos - tgt_tcp_pose.p, axis=1)
        reached = tcp_to_peg_dist < 0.01
        reaching_reward = 2 * (1 - torch.tanh(5.0 * tcp_to_peg_dist))
        reward = reaching_reward

        # peg to cube
        angle_diff = info["angle_diff"]
        align_reward = 1 - torch.tanh(5.0 * angle_diff)
        head_to_cube_dist = info["head_to_cube_dist"]
        close_reward = 1 - torch.tanh(5.0 * head_to_cube_dist)
        is_peg_grasped = info["is_peg_grasped"] * reached
        reward[is_peg_grasped] = (4 + close_reward + align_reward)[is_peg_grasped]

        # cube to goal
        cube_to_goal_dist = torch.linalg.norm(
            self.goal_region.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        is_peg_cube_fit = info["is_peg_cube_fit"] * is_peg_grasped
        reward[is_peg_cube_fit] = (7 + place_reward)[is_peg_cube_fit]

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward[info["is_cube_placed"]] += static_reward[info["is_cube_placed"]]

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

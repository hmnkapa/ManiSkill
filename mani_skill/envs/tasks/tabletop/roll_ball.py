from enum import IntEnum
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class RollBallSkillPhase(IntEnum):
    ALIGN = 0
    ROLL = 1
    DONE = 2


class RollBallSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(RollBallSkillPhase.ALIGN),
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

    def _compute_roll_direction(self, ball_pos: torch.Tensor, goal_pos: torch.Tensor):
        direction_xy = goal_pos[:, :2] - ball_pos[:, :2]
        direction_xy = direction_xy / torch.linalg.norm(
            direction_xy, dim=1, keepdim=True
        ).clamp_min(1e-6)
        direction = torch.zeros_like(ball_pos)
        direction[:, :2] = direction_xy
        return direction

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        ball_pos = env.ball.pose.p
        goal_pos = env.goal_region.pose.p
        roll_direction = self._compute_roll_direction(ball_pos, goal_pos)
        tcp_hit_pos = ball_pos - roll_direction * (env.ball_radius + 0.05)
        tcp_to_hit_dist = torch.linalg.norm(
            env.agent.tcp.pose.p - tcp_hit_pos, dim=1
        )
        object_goal_point_world = goal_pos.clone()
        object_goal_point_world[:, 2] = ball_pos[:, 2]
        signals = {
            "success": info["success"],
            "reached_hit_pose": tcp_to_hit_dist < 0.04,
            "tcp_to_hit_dist": tcp_to_hit_dist,
            "ball_to_goal_dist": torch.linalg.norm(
                ball_pos[:, :2] - goal_pos[:, :2], dim=1
            ),
            "roll_direction_world": roll_direction,
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
            self.phase.fill_(int(RollBallSkillPhase.ALIGN))
        else:
            self.phase[env_idx] = int(RollBallSkillPhase.ALIGN)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        reached_hit_pose = signals["reached_hit_pose"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        align = phase == int(RollBallSkillPhase.ALIGN)
        roll = phase == int(RollBallSkillPhase.ROLL)
        phase[align & reached_hit_pose] = int(RollBallSkillPhase.ROLL)
        phase[roll & success] = int(RollBallSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        align = phase == int(RollBallSkillPhase.ALIGN)
        roll = phase == int(RollBallSkillPhase.ROLL)
        done = phase == int(RollBallSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[align] = SKILL_IDS["push"]
        skill_id[roll] = SKILL_IDS["push"]

        ball_pos = self._select(env.ball.pose.p, env_idx).reshape(-1, 3)
        goal_pos = self._select(env.goal_region.pose.p, env_idx).reshape(-1, 3)
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        roll_direction = self._compute_roll_direction(ball_pos, goal_pos)
        offset = env.ball_radius + 0.05

        align_target_pos = ball_pos - roll_direction * offset
        roll_target_pos = ball_pos.clone()
        roll_target_pos[:, :2] = goal_pos[:, :2] - roll_direction[:, :2] * offset
        align_target_pose = Pose.create_from_pq(p=align_target_pos, q=tcp_pose.q)
        roll_target_pose = Pose.create_from_pq(p=roll_target_pos, q=tcp_pose.q)

        target_pose_world = torch.where(
            roll[:, None, None],
            roll_target_pose.to_transformation_matrix(),
            align_target_pose.to_transformation_matrix(),
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.zeros(
            phase.shape, dtype=torch.float32, device=phase.device
        )
        target_gripper_width[done] = float("nan")

        phase_names_by_id = {
            int(RollBallSkillPhase.ALIGN): "align",
            int(RollBallSkillPhase.ROLL): "roll",
            int(RollBallSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(RollBallSkillPhase.ALIGN): "push",
            int(RollBallSkillPhase.ROLL): "push",
            int(RollBallSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(RollBallSkillPhase.ALIGN): "move_tcp_behind_ball",
            int(RollBallSkillPhase.ROLL): "roll_ball_to_goal",
            int(RollBallSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(RollBallSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "RollBall-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "reached_hit_pose": signals["reached_hit_pose"],
            "tcp_to_hit_dist": signals["tcp_to_hit_dist"],
            "ball_to_goal_dist": signals["ball_to_goal_dist"],
            "roll_direction_world": signals["roll_direction_world"],
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
            active_object=["ball"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("RollBall-v1", max_episode_steps=80)
class RollBallEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to push and roll a ball to a goal region at the other end of the table

    **Randomizations:**
    - The ball's xy position is randomized on top of a table in the region [0.2, 0.5] x [-0.4, 0.7]. It is placed flat on the table
    - The target goal region is marked by a red/white circular target. The position of the target is randomized on top of a table in the region [-0.4, -0.7] x [0.2, -0.9]

    **Success Conditions:**
    - The ball's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/RollBall-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda"]

    agent: Panda

    goal_radius: float = 0.1  # radius of the goal region
    ball_radius: float = 0.035  # radius of the ball
    reached_status: torch.Tensor

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((b)) * 0.2 + 0.5
            xyz[..., 2] = self.ball_radius
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.ball.set_pose(obj_pose)

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz_goal[..., 1] = torch.rand((b)) * 0.2 - 1.0 + self.goal_radius
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
        self.reached_status[env_idx] = 0.0
        self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, RollBallSkillFSM):
            fsm = RollBallSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):

        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: dict):

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                ball_pose=self.ball.pose.raw_pose,
                ball_vel=self.ball.linear_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.ball_radius + 0.05),
        )
        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

        obj_to_goal_dist = torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )

        reached_reward = 1 - torch.tanh(obj_to_goal_dist)

        reward = (
            20 * reached_reward * self.reached_status
            + reaching_reward * (1 - self.reached_status)
            + self.reached_status
        )

        reward[info["success"]] = 30.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

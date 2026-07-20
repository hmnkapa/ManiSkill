from enum import IntEnum
from typing import Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.structs.pose import Pose


class StackPyramidSkillPhase(IntEnum):
    PUSH_BASE = 0
    PICK_TOP = 1
    PLACE_TOP = 2
    DONE = 3


class StackPyramidSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(StackPyramidSkillPhase.PUSH_BASE),
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

    def _cube_half_size(self, env, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(env.cube_half_size, device=env.device, dtype=dtype)

    def _compute_push_direction(
        self, cubeA_pos: torch.Tensor, cubeB_pos: torch.Tensor
    ) -> torch.Tensor:
        direction_xy = cubeB_pos[:, :2] - cubeA_pos[:, :2]
        return direction_xy / torch.linalg.norm(
            direction_xy, dim=1, keepdim=True
        ).clamp_min(1e-6)

    def _compute_cubeC_goal_point(self, env) -> torch.Tensor:
        half_size = self._cube_half_size(env, dtype=env.cubeA.pose.p.dtype)
        top_offset = torch.zeros(3, device=env.device, dtype=half_size.dtype)
        top_offset[2] = 2 * half_size[2]
        return ((env.cubeA.pose.p + top_offset) + (env.cubeB.pose.p + top_offset)) / 2

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        cubeA_pos = env.cubeA.pose.p
        cubeB_pos = env.cubeB.pose.p
        cubeC_pos = env.cubeC.pose.p
        half_size = self._cube_half_size(env, dtype=cubeA_pos.dtype)
        xy_thresh = torch.linalg.norm(2 * half_size[:2]) + 0.005

        base_pair_ready = (
            torch.linalg.norm((cubeA_pos - cubeB_pos)[:, :2], dim=1) <= xy_thresh
        ) & env.cubeA.is_static(lin_thresh=0.01, ang_thresh=0.5)
        base_pair_ready = base_pair_ready & (~env.agent.is_grasping(env.cubeA))

        is_cubeC_grasped = env.agent.is_grasping(env.cubeC)
        is_C_on_A = (
            torch.linalg.norm((cubeC_pos - cubeA_pos)[:, :2], dim=1) <= xy_thresh
        ) & (torch.abs(cubeC_pos[:, 2] - cubeA_pos[:, 2]) > 0.02)
        is_C_on_B = (
            torch.linalg.norm((cubeC_pos - cubeB_pos)[:, :2], dim=1) <= xy_thresh
        ) & (torch.abs(cubeC_pos[:, 2] - cubeB_pos[:, 2]) > 0.02)
        is_C_static = env.cubeC.is_static(lin_thresh=0.01, ang_thresh=0.5)

        signals = {
            "success": info["success"],
            "base_pair_ready": base_pair_ready,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_C_on_A": is_C_on_A,
            "is_C_on_B": is_C_on_B,
            "is_C_static": is_C_static,
            "cubeC_goal_point_world": self._compute_cubeC_goal_point(env),
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
            self.phase.fill_(int(StackPyramidSkillPhase.PUSH_BASE))
        else:
            self.phase[env_idx] = int(StackPyramidSkillPhase.PUSH_BASE)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        base_pair_ready = signals["base_pair_ready"]
        is_cubeC_grasped = signals["is_cubeC_grasped"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        push_base = phase == int(StackPyramidSkillPhase.PUSH_BASE)
        pick_top = phase == int(StackPyramidSkillPhase.PICK_TOP)
        place_top = phase == int(StackPyramidSkillPhase.PLACE_TOP)
        phase[push_base & base_pair_ready] = int(StackPyramidSkillPhase.PICK_TOP)
        phase[pick_top & is_cubeC_grasped] = int(StackPyramidSkillPhase.PLACE_TOP)
        phase[place_top & success] = int(StackPyramidSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        push_base = phase == int(StackPyramidSkillPhase.PUSH_BASE)
        pick_top = phase == int(StackPyramidSkillPhase.PICK_TOP)
        place_top = phase == int(StackPyramidSkillPhase.PLACE_TOP)
        done = phase == int(StackPyramidSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[push_base] = SKILL_IDS["push"]
        skill_id[pick_top] = SKILL_IDS["pick"]
        skill_id[place_top] = SKILL_IDS["place"]

        cubeA_pose = Pose.create(self._select(env.cubeA.pose.raw_pose, env_idx))
        cubeB_pos = self._select(env.cubeB.pose.p, env_idx).reshape(-1, 3)
        cubeC_pose = Pose.create(self._select(env.cubeC.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))

        half_size = self._cube_half_size(env, dtype=cubeA_pose.p.dtype)
        push_direction = self._compute_push_direction(cubeA_pose.p, cubeB_pos)
        target_cubeA_center = cubeA_pose.p.clone()
        target_cubeA_center[:, :2] = cubeB_pos[:, :2] - 0.04 * push_direction
        push_target_pos = target_cubeA_center.clone()
        push_target_pos[:, :2] = push_target_pos[:, :2] - push_direction * (
            half_size[0] + 0.005
        )
        push_target_pose = Pose.create_from_pq(p=push_target_pos, q=tcp_pose.q)

        pick_target_pose = build_panda_topdown_grasp_pose(
            center=cubeC_pose.p,
            tcp_pose=tcp_pose,
            object_pose=cubeC_pose,
        )

        cubeC_goal_point = self._select(
            self._compute_cubeC_goal_point(env), env_idx
        ).reshape(-1, 3)
        desired_cubeC_pose = Pose.create_from_pq(p=cubeC_goal_point, q=cubeC_pose.q)
        place_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=cubeC_pose,
            desired_object_pose=desired_cubeC_pose,
        )

        target_pose_world = push_target_pose.to_transformation_matrix()
        target_pose_world = torch.where(
            pick_top[:, None, None],
            pick_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world = torch.where(
            place_top[:, None, None],
            place_target_pose.to_transformation_matrix(),
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
        target_gripper_width[push_base] = 0.0
        target_gripper_width[place_top] = 0.08

        phase_names_by_id = {
            int(StackPyramidSkillPhase.PUSH_BASE): "push_base",
            int(StackPyramidSkillPhase.PICK_TOP): "pick_top",
            int(StackPyramidSkillPhase.PLACE_TOP): "place_top",
            int(StackPyramidSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(StackPyramidSkillPhase.PUSH_BASE): "push",
            int(StackPyramidSkillPhase.PICK_TOP): "pick",
            int(StackPyramidSkillPhase.PLACE_TOP): "place",
            int(StackPyramidSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(StackPyramidSkillPhase.PUSH_BASE): "push_cubeA_next_to_cubeB",
            int(StackPyramidSkillPhase.PICK_TOP): "grasp_cubeC",
            int(StackPyramidSkillPhase.PLACE_TOP): "place_cubeC_on_base_pair",
            int(StackPyramidSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(StackPyramidSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]
        active_objects = []
        for phase_id in phase_ids:
            if phase_id == int(StackPyramidSkillPhase.PUSH_BASE):
                active_objects.append("cubeA")
            elif phase_id == int(StackPyramidSkillPhase.DONE):
                active_objects.append(None)
            else:
                active_objects.append("cubeC")

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "StackPyramid-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "base_pair_ready": signals["base_pair_ready"],
            "is_cubeC_grasped": signals["is_cubeC_grasped"],
            "is_C_on_A": signals["is_C_on_A"],
            "is_C_on_B": signals["is_C_on_B"],
            "is_C_static": signals["is_C_static"],
            "cubeC_goal_point_world": signals["cubeC_goal_point_world"],
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
            active_object=active_objects,
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("StackPyramid-v1", max_episode_steps=250)
class StackPyramidEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.05, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.2]),
        )
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="cubeC",
            initial_pose=sapien.Pose(p=[-1, 0, 0.2]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            xy = xyz[:, :2]
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

            # Cube A
            xyz[:, :2] = cubeA_xy

            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube B
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube C
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, StackPyramidSkillFSM):
            fsm = StackPyramidSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (
                torch.linalg.norm(offset[..., :2], axis=1)
                <= torch.linalg.norm(2 * self.cube_half_size[:2]) + 0.005
            )
            z_flag = torch.abs(offset[..., 2]) > 0.02
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(
                    f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}"
                )

            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)
            return success.bool()

        success_A_B = evaluate_cube_distance(
            offset_AB, self.cubeA, self.cubeB, "next_to"
        )
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(
            success_A_B, torch.logical_and(success_C_B, success_C_A)
        )
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs

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
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


class LiftPegUprightSkillPhase(IntEnum):
    PICK = 0
    LIFT = 1
    ROTATE = 2
    LOWER = 3
    DONE = 4


class LiftPegUprightSkillFSM:
    _SAFE_HEIGHT_CLEARANCE = 0.10
    _LIFT_HEIGHT_TOLERANCE = 0.02
    _UPRIGHT_AXIS_TOLERANCE = 0.08

    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(LiftPegUprightSkillPhase.PICK),
            dtype=torch.long,
            device=device,
        )
        self._rotate_target_pose_world = torch.full(
            (num_envs, 4, 4),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        self._rotate_target_valid = torch.zeros(
            (num_envs,), dtype=torch.bool, device=device
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

    def _safe_peg_height(self, env, peg_pose: Pose) -> torch.Tensor:
        return torch.full_like(
            peg_pose.p[:, 2],
            env.peg_half_length + self._SAFE_HEIGHT_CLEARANCE,
        )

    def _nearest_upright_quaternion(self, quaternion: torch.Tensor) -> torch.Tensor:
        candidates = torch.as_tensor(
            [
                [0.5, 0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5, 0.5],
            ],
            dtype=quaternion.dtype,
            device=quaternion.device,
        )
        scores = torch.abs(quaternion @ candidates.T)
        return candidates[torch.argmax(scores, dim=1)]

    def _peg_pose_at_height(
        self,
        peg_pose: Pose,
        height,
        upright: bool,
    ) -> Pose:
        desired_pos = peg_pose.p.clone()
        desired_pos[:, 2] = height
        desired_q = (
            self._nearest_upright_quaternion(peg_pose.q)
            if upright
            else peg_pose.q
        )
        return Pose.create_from_pq(p=desired_pos, q=desired_q)

    def _lifted_peg_pose(self, env, peg_pose: Pose, upright: bool) -> Pose:
        return self._peg_pose_at_height(
            peg_pose=peg_pose,
            height=self._safe_peg_height(env, peg_pose),
            upright=upright,
        )

    def _desired_peg_pose(self, env, peg_pose: Pose) -> Pose:
        return self._peg_pose_at_height(
            peg_pose=peg_pose,
            height=env.peg_half_length,
            upright=True,
        )

    def _rotation_target_pose(self, peg_pose: Pose, tcp_pose: Pose) -> Pose:
        desired_peg_pose = Pose.create_from_pq(
            p=peg_pose.p,
            q=self._nearest_upright_quaternion(peg_pose.q),
        )
        target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=peg_pose,
            desired_object_pose=desired_peg_pose,
        )
        return Pose.create_from_pq(p=tcp_pose.p, q=target_pose.q)

    def _cache_rotation_target(self, env, mask: torch.Tensor, env_idx=None):
        peg_pose = Pose.create(self._select(env.peg.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        target_pose_world = self._rotation_target_pose(
            peg_pose, tcp_pose
        ).to_transformation_matrix()
        if env_idx is None:
            self._rotate_target_pose_world[mask] = target_pose_world[mask]
            self._rotate_target_valid[mask] = True
        else:
            target_env_idx = env_idx[mask]
            self._rotate_target_pose_world[target_env_idx] = target_pose_world[mask]
            self._rotate_target_valid[target_env_idx] = True

    def _invalidate_rotation_target(self, mask: torch.Tensor, env_idx=None):
        target_env_idx = mask if env_idx is None else env_idx[mask]
        self._rotate_target_pose_world[target_env_idx] = float("nan")
        self._rotate_target_valid[target_env_idx] = False

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        peg_pose = env.peg.pose
        peg_rotation = rotation_conversions.quaternion_to_matrix(peg_pose.q)
        peg_euler_xyz = rotation_conversions.matrix_to_euler_angles(
            peg_rotation, "XYZ"
        )
        peg_axis_world = peg_rotation[:, :, 0]
        upright_axis_error = torch.acos(peg_axis_world[:, 2].abs().clamp(-1.0, 1.0))
        orientation_aligned = upright_axis_error < self._UPRIGHT_AXIS_TOLERANCE
        safe_peg_height = self._safe_peg_height(env, peg_pose)
        lift_height_error = (safe_peg_height - peg_pose.p[:, 2]).clamp_min(0.0)
        is_lifted = lift_height_error < self._LIFT_HEIGHT_TOLERANCE
        z_error = torch.abs(peg_pose.p[:, 2] - env.peg_half_length)
        is_peg_upright = torch.abs(torch.abs(peg_euler_xyz[:, 2]) - np.pi / 2) < 0.08
        close_to_table = z_error < 0.005
        desired_peg_pose = self._desired_peg_pose(env, peg_pose)
        signals = {
            "success": info["success"],
            "is_grasped": env.agent.is_grasping(env.peg),
            "is_peg_upright": is_peg_upright,
            "close_to_table": close_to_table,
            "safe_peg_height": safe_peg_height,
            "lift_height_error": lift_height_error,
            "is_lifted": is_lifted,
            "upright_axis_error": upright_axis_error,
            "orientation_aligned": orientation_aligned,
            "z_error": z_error,
            "desired_peg_pose_world": desired_peg_pose.to_transformation_matrix(),
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
            self.phase.fill_(int(LiftPegUprightSkillPhase.PICK))
            self._rotate_target_pose_world.fill_(float("nan"))
            self._rotate_target_valid.fill_(False)
        else:
            self.phase[env_idx] = int(LiftPegUprightSkillPhase.PICK)
            self._rotate_target_pose_world[env_idx] = float("nan")
            self._rotate_target_valid[env_idx] = False

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        is_grasped = signals["is_grasped"]
        is_lifted = signals["is_lifted"]
        orientation_aligned = signals["orientation_aligned"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        pick = phase == int(LiftPegUprightSkillPhase.PICK)
        lift = phase == int(LiftPegUprightSkillPhase.LIFT)
        rotate = phase == int(LiftPegUprightSkillPhase.ROTATE)
        lower = phase == int(LiftPegUprightSkillPhase.LOWER)
        active = lift | rotate | lower
        enter_lift = pick & is_grasped
        enter_rotate = lift & is_grasped & is_lifted
        grasp_lost = active & ~is_grasped & ~success
        self._invalidate_rotation_target(enter_lift | grasp_lost, env_idx)
        self._cache_rotation_target(env, enter_rotate, env_idx)
        phase[enter_lift] = int(LiftPegUprightSkillPhase.LIFT)
        phase[enter_rotate] = int(LiftPegUprightSkillPhase.ROTATE)
        phase[rotate & is_grasped & orientation_aligned] = int(
            LiftPegUprightSkillPhase.LOWER
        )
        phase[lower & success] = int(LiftPegUprightSkillPhase.DONE)
        phase[grasp_lost] = int(LiftPegUprightSkillPhase.PICK)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(LiftPegUprightSkillPhase.PICK)
        lift = phase == int(LiftPegUprightSkillPhase.LIFT)
        rotate = phase == int(LiftPegUprightSkillPhase.ROTATE)
        lower = phase == int(LiftPegUprightSkillPhase.LOWER)
        done = phase == int(LiftPegUprightSkillPhase.DONE)
        active = lift | rotate | lower

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[active] = SKILL_IDS["place"]

        peg_pose = Pose.create(self._select(env.peg.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        pick_target_pose = build_panda_topdown_grasp_pose(
            center=peg_pose.p,
            tcp_pose=tcp_pose,
            object_pose=peg_pose,
        )
        pick_target_pose = pick_target_pose * Pose.create_from_pq(
            p=torch.tensor([0.10, 0.0, 0.0], device=env.device)
        )

        lifted_peg_pose = self._lifted_peg_pose(env, peg_pose, upright=False)
        lift_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=peg_pose,
            desired_object_pose=lifted_peg_pose,
        )
        rotate_target_valid = self._select(self._rotate_target_valid, env_idx)
        self._cache_rotation_target(env, rotate & ~rotate_target_valid, env_idx)
        rotate_target_pose_world = self._select(
            self._rotate_target_pose_world, env_idx
        )
        desired_peg_pose = self._desired_peg_pose(env, peg_pose)
        lower_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=peg_pose,
            desired_object_pose=desired_peg_pose,
        )

        target_pose_world = pick_target_pose.to_transformation_matrix()
        target_pose_world = torch.where(
            lift[:, None, None],
            lift_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world = torch.where(
            rotate[:, None, None],
            rotate_target_pose_world,
            target_pose_world,
        )
        target_pose_world = torch.where(
            lower[:, None, None],
            lower_target_pose.to_transformation_matrix(),
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
        target_gripper_width[pick | lift | rotate] = 0.045
        target_gripper_width[lower] = 0.08

        phase_names_by_id = {
            int(LiftPegUprightSkillPhase.PICK): "pick",
            int(LiftPegUprightSkillPhase.LIFT): "lift",
            int(LiftPegUprightSkillPhase.ROTATE): "rotate",
            int(LiftPegUprightSkillPhase.LOWER): "lower",
            int(LiftPegUprightSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(LiftPegUprightSkillPhase.PICK): "pick",
            int(LiftPegUprightSkillPhase.LIFT): "place",
            int(LiftPegUprightSkillPhase.ROTATE): "place",
            int(LiftPegUprightSkillPhase.LOWER): "place",
            int(LiftPegUprightSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(LiftPegUprightSkillPhase.PICK): "grasp_peg_near_end",
            int(LiftPegUprightSkillPhase.LIFT): "lift_peg_to_safe_height",
            int(LiftPegUprightSkillPhase.ROTATE): "rotate_peg_upright",
            int(LiftPegUprightSkillPhase.LOWER): "lower_peg_to_table",
            int(LiftPegUprightSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(LiftPegUprightSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "LiftPegUpright-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "is_grasped": signals["is_grasped"],
            "is_peg_upright": signals["is_peg_upright"],
            "close_to_table": signals["close_to_table"],
            "safe_peg_height": signals["safe_peg_height"],
            "lift_height_error": signals["lift_height_error"],
            "is_lifted": signals["is_lifted"],
            "upright_axis_error": signals["upright_axis_error"],
            "orientation_aligned": signals["orientation_aligned"],
            "z_error": signals["z_error"],
            "desired_peg_pose_world": signals["desired_peg_pose_world"],
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


@register_env("LiftPegUpright-v1", max_episode_steps=50)
class LiftPegUprightEnv(BaseEnv):
    r"""
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.08 of $\pi$/2 and the z position of the peg is within 0.005 of its half-length (0.12).
    """

    _sample_video_link = "https://github.com/mani-skillll/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, LiftPegUprightSkillFSM):
            fsm = LiftPegUprightSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        return {
            "success": is_peg_upright & close_to_table,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        vec = torch.tensor([1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.peg)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

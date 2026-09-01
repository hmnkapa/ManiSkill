from enum import IntEnum
from typing import Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


class PlugChargerSkillPhase(IntEnum):
    PICK = 0
    PRE_INSERT = 1
    INSERT = 2
    DONE = 3


class PlugChargerSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PlugChargerSkillPhase.PICK),
            dtype=torch.long,
            device=device,
        )
        self._suppress_transition_once = torch.zeros(
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

    def _pre_insert_object_pose(self, goal_pose: Pose) -> Pose:
        return goal_pose * Pose.create_from_pq(
            p=torch.tensor([-0.05, 0.0, 0.0], device=goal_pose.device)
        )

    def _quaternion_angle(self, quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = quaternion / torch.linalg.norm(
            quaternion, dim=1, keepdim=True
        ).clamp_min(1e-6)
        return 2 * torch.acos(quaternion[:, 0].abs().clamp(max=1.0))

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        charger_pose = Pose.create(env.charger.pose.raw_pose)
        goal_pose = Pose.create(env.goal_pose.raw_pose)
        pre_insert_object_pose = self._pre_insert_object_pose(goal_pose)
        pre_rel = pre_insert_object_pose.inv() * charger_pose
        pre_x_error = torch.abs(pre_rel.p[:, 0])
        pre_yz_error = torch.linalg.norm(pre_rel.p[:, 1:], dim=1)
        pre_angle_error = self._quaternion_angle(pre_rel.q)
        pre_inserted = (
            (pre_x_error < 0.01)
            & (pre_yz_error < 0.005)
            & (pre_angle_error < 0.10)
        )
        signals = {
            "success": info["success"],
            "is_grasped": env.agent.is_grasping(env.charger, max_angle=20),
            "obj_to_goal_dist": info["obj_to_goal_dist"],
            "obj_to_goal_angle": info["obj_to_goal_angle"],
            "pre_inserted": pre_inserted,
            "pre_x_error": pre_x_error,
            "pre_yz_error": pre_yz_error,
            "pre_angle_error": pre_angle_error,
            "charger_base_pose_world": env.charger_base_pose.to_transformation_matrix(),
            "object_goal_pose_world": goal_pose.to_transformation_matrix(),
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
            self.phase.fill_(int(PlugChargerSkillPhase.PICK))
            self._suppress_transition_once.fill_(True)
        else:
            self.phase[env_idx] = int(PlugChargerSkillPhase.PICK)
            self._suppress_transition_once[env_idx] = True

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        is_grasped = signals["is_grasped"]
        pre_inserted = signals["pre_inserted"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()
        suppress_transition_once = self._select(
            self._suppress_transition_once, env_idx
        )

        pick = phase == int(PlugChargerSkillPhase.PICK)
        pre_insert = phase == int(PlugChargerSkillPhase.PRE_INSERT)
        insert = phase == int(PlugChargerSkillPhase.INSERT)
        phase[pick & is_grasped & ~suppress_transition_once] = int(
            PlugChargerSkillPhase.PRE_INSERT
        )
        phase[pre_insert & is_grasped & pre_inserted] = int(
            PlugChargerSkillPhase.INSERT
        )
        phase[insert & success] = int(PlugChargerSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
            self._suppress_transition_once.fill_(False)
        else:
            self.phase[env_idx] = phase
            self._suppress_transition_once[env_idx] = False

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(PlugChargerSkillPhase.PICK)
        pre_insert = phase == int(PlugChargerSkillPhase.PRE_INSERT)
        insert = phase == int(PlugChargerSkillPhase.INSERT)
        done = phase == int(PlugChargerSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[pre_insert] = SKILL_IDS["insert"]
        skill_id[insert] = SKILL_IDS["insert"]

        charger_pose = Pose.create(self._select(env.charger.pose.raw_pose, env_idx))
        charger_base_pose = Pose.create(
            self._select(env.charger_base_pose.raw_pose, env_idx)
        )
        goal_pose = Pose.create(self._select(env.goal_pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))

        pick_target_pose = build_panda_topdown_grasp_pose(
            center=charger_base_pose.p,
            tcp_pose=tcp_pose,
            object_pose=charger_base_pose,
        )
        grasp_angle = torch.zeros((len(phase), 3), device=phase.device)
        grasp_angle[:, 1] = np.deg2rad(15)
        grasp_q = rotation_conversions.matrix_to_quaternion(
            rotation_conversions.euler_angles_to_matrix(grasp_angle, "XYZ")
        )
        pick_target_pose = pick_target_pose * Pose.create_from_pq(q=grasp_q)

        pre_insert_object_pose = self._pre_insert_object_pose(goal_pose)
        pre_insert_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=charger_pose,
            desired_object_pose=pre_insert_object_pose,
        )
        insert_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=charger_pose,
            desired_object_pose=goal_pose,
        )

        target_pose_world = pick_target_pose.to_transformation_matrix()
        target_pose_world = torch.where(
            pre_insert[:, None, None],
            pre_insert_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world = torch.where(
            insert[:, None, None],
            insert_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.full(
            phase.shape,
            0.025,
            dtype=torch.float32,
            device=phase.device,
        )
        target_gripper_width[done] = float("nan")

        phase_names_by_id = {
            int(PlugChargerSkillPhase.PICK): "pick",
            int(PlugChargerSkillPhase.PRE_INSERT): "pre_insert",
            int(PlugChargerSkillPhase.INSERT): "insert",
            int(PlugChargerSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PlugChargerSkillPhase.PICK): "pick",
            int(PlugChargerSkillPhase.PRE_INSERT): "insert",
            int(PlugChargerSkillPhase.INSERT): "insert",
            int(PlugChargerSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PlugChargerSkillPhase.PICK): "grasp_charger_base",
            int(PlugChargerSkillPhase.PRE_INSERT): "align_charger_with_receptacle",
            int(PlugChargerSkillPhase.INSERT): "insert_charger",
            int(PlugChargerSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PlugChargerSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "PlugCharger-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "is_grasped": signals["is_grasped"],
            "obj_to_goal_dist": signals["obj_to_goal_dist"],
            "obj_to_goal_angle": signals["obj_to_goal_angle"],
            "pre_inserted": signals["pre_inserted"],
            "pre_x_error": signals["pre_x_error"],
            "pre_yz_error": signals["pre_yz_error"],
            "pre_angle_error": signals["pre_angle_error"],
            "charger_base_pose_world": signals["charger_base_pose_world"],
            "object_goal_pose_world": signals["object_goal_pose_world"],
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
            active_object=["charger"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PlugCharger-v1", max_episode_steps=200)
class PlugChargerEnv(BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
    - The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
    fixed relative to the receptacle.

    **Success Conditions:**
    - The charger is inserted into the receptacle
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PlugCharger-v1_rt.mp4"

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            if self.agent.uid == "panda_wristcam":
                qpos = torch.tensor(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        np.pi / 4,
                        0.04,
                        0.04,
                    ]
                )
                qpos = (
                    torch.normal(
                        0,
                        self.robot_init_qpos_noise,
                        (b, len(qpos)),
                        device=self.device,
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            xy = randomization.uniform(
                [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
            )
            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.1
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PlugChargerSkillFSM):
            fsm = PlugChargerSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs

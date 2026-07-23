from enum import IntEnum
from typing import Any, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class PlaceSphereSkillPhase(IntEnum):
    PICK = 0
    PLACE = 1
    RELEASE = 2
    DONE = 3


class PlaceSphereSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PlaceSphereSkillPhase.PICK),
            dtype=torch.long,
            device=device,
        )
        self._cached_place_target_pose_world = torch.full(
            (num_envs, 4, 4),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        self._cached_place_target_pose_valid = torch.zeros(
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

    def reset(self, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        if env_idx is None:
            self.phase.fill_(int(PlaceSphereSkillPhase.PICK))
            self._cached_place_target_pose_world.fill_(float("nan"))
            self._cached_place_target_pose_valid.fill_(False)
        else:
            self.phase[env_idx] = int(PlaceSphereSkillPhase.PICK)
            self._cached_place_target_pose_world[env_idx] = float("nan")
            self._cached_place_target_pose_valid[env_idx] = False

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        is_grasped = info["is_obj_grasped"]
        is_on_bin = info["is_obj_on_bin"]
        is_static = info["is_obj_static"]
        success = info["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()
            is_grasped = is_grasped[env_idx]
            is_on_bin = is_on_bin[env_idx]
            is_static = is_static[env_idx]
            success = success[env_idx]

        pick = phase == int(PlaceSphereSkillPhase.PICK)
        place = phase == int(PlaceSphereSkillPhase.PLACE)
        release = phase == int(PlaceSphereSkillPhase.RELEASE)
        phase[pick & is_grasped] = int(PlaceSphereSkillPhase.PLACE)
        phase[place & ~is_grasped] = int(PlaceSphereSkillPhase.RELEASE)
        phase[release & success] = int(PlaceSphereSkillPhase.DONE)
        phase[release & is_grasped & ~success] = int(PlaceSphereSkillPhase.PLACE)
        release_failed = (
            release & ~is_grasped & ~is_on_bin & is_static & ~success
        )
        phase[release_failed] = int(PlaceSphereSkillPhase.PICK)

        selected_env_idx = (
            torch.arange(self.phase.shape[0], device=self.phase.device)
            if env_idx is None
            else env_idx
        )
        failed_env_idx = selected_env_idx[release_failed]
        self._cached_place_target_pose_world[failed_env_idx] = float("nan")
        self._cached_place_target_pose_valid[failed_env_idx] = False

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(PlaceSphereSkillPhase.PICK)
        place = phase == int(PlaceSphereSkillPhase.PLACE)
        release = phase == int(PlaceSphereSkillPhase.RELEASE)
        done = phase == int(PlaceSphereSkillPhase.DONE)
        place_active = place | release

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[place_active] = SKILL_IDS["place"]

        obj_pose = Pose.create(self._select(env.obj.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        pick_target_pose = build_panda_topdown_grasp_pose(
            center=obj_pose.p,
            tcp_pose=tcp_pose,
        )

        release_obj_pos = self._select(env.bin.pose.p, env_idx).reshape(-1, 3).clone()
        release_obj_pos[:, 2] = (
            release_obj_pos[:, 2] + 2 * env.block_half_size[2]
        )
        desired_obj_pose = Pose.create_from_pq(p=release_obj_pos, q=obj_pose.q)
        place_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=obj_pose,
            desired_object_pose=desired_obj_pose,
        )
        place_target_pose_world = place_target_pose.to_transformation_matrix()

        # The object-to-TCP transform is only rigid while grasped. Keep refreshing
        # the target in PLACE, then freeze the last value while the sphere falls.
        selected_env_idx = (
            torch.arange(self.phase.shape[0], device=self.phase.device)
            if env_idx is None
            else env_idx
        )
        cached_target_valid = self._cached_place_target_pose_valid[selected_env_idx]
        cache_target = place | (release & ~cached_target_valid)
        cache_env_idx = selected_env_idx[cache_target]
        self._cached_place_target_pose_world[cache_env_idx] = (
            place_target_pose_world[cache_target]
        )
        self._cached_place_target_pose_valid[cache_env_idx] = True

        active_place_target_pose_world = torch.where(
            release[:, None, None],
            self._cached_place_target_pose_world[selected_env_idx],
            place_target_pose_world,
        )

        target_pose_world = torch.where(
            place_active[:, None, None],
            active_place_target_pose_world,
            pick_target_pose.to_transformation_matrix(),
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.full(
            phase.shape,
            float("nan"),
            dtype=torch.float32,
            device=phase.device,
        )
        target_gripper_width[pick | place] = 0.035
        target_gripper_width[release] = 0.08

        phase_names_by_id = {
            int(PlaceSphereSkillPhase.PICK): "pick",
            int(PlaceSphereSkillPhase.PLACE): "place",
            int(PlaceSphereSkillPhase.RELEASE): "release",
            int(PlaceSphereSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PlaceSphereSkillPhase.PICK): "pick",
            int(PlaceSphereSkillPhase.PLACE): "place",
            int(PlaceSphereSkillPhase.RELEASE): "place",
            int(PlaceSphereSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PlaceSphereSkillPhase.PICK): "move_to_sphere",
            int(PlaceSphereSkillPhase.PLACE): "move_to_bin",
            int(PlaceSphereSkillPhase.RELEASE): "wait_for_sphere_to_settle",
            int(PlaceSphereSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PlaceSphereSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        info = env.evaluate()
        task_meta = {
            "task": "PlaceSphere-v1",
            "target_frame": "world",
            "is_obj_grasped": self._select(info["is_obj_grasped"], env_idx),
            "is_obj_on_bin": self._select(info["is_obj_on_bin"], env_idx),
            "is_obj_static": self._select(info["is_obj_static"], env_idx),
            "success": self._select(info["success"], env_idx),
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
            active_object=["sphere"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PlaceSphere-v1", max_episode_steps=50)
class PlaceSphereEnv(BaseEnv):
    """
    **Task Description:**
    Place the sphere into the shallow bin.

    **Randomizations:**
    - The position of the bin and the sphere are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - The sphere is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        # build the kinematic bin
        return builder.build_kinematic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )

        # load the bin
        self.bin = self._build_bin(self.radius)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[
                ..., 0
            ]  # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[
                ..., 0
            ]  # spanning all possible ys
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PlaceSphereSkillFSM):
            fsm = PlaceSphereSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        success = is_obj_on_bin & is_obj_static & (~is_obj_grasped)
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        obj_pos = self.obj.pose.p
        self.bin.pose.p
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_obj_grasped
        ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(
            0.2
        )  # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_bin"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_bin"]]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

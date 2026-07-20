from enum import IntEnum
from typing import Any

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# extending TableSceneBuilder and only making 2 changes:
# 1.Making table smooth and white, 2. adding support for keyframes of new robots - panda stick
class WhiteTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.662,
                    0.212,
                    0.086,
                    -2.685,
                    -0.115,
                    2.898,
                    1.673,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def build(self):
        super().build()
        # cheap way to un-texture table
        for part in self.table._objs:
            render_body = part.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            for render_shape in render_body.render_shapes:
                for triangle in render_shape.parts:
                    triangle.material.set_base_color(
                        np.array([255, 255, 255, 255]) / 255
                    )
                    triangle.material.set_base_color_texture(None)
                    triangle.material.set_normal_texture(None)
                    triangle.material.set_emission_texture(None)
                    triangle.material.set_transmission_texture(None)
                    triangle.material.set_metallic_texture(None)
                    triangle.material.set_roughness_texture(None)


class PushTSkillPhase(IntEnum):
    PUSH_ORIENT = 0
    PUSH_PLACE = 1
    DONE = 2


class PushTSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PushTSkillPhase.PUSH_ORIENT),
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

    def _wrap_to_pi(self, angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi

    def _rotate_xy(self, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        if xy.ndim == 3:
            cos_yaw = cos_yaw[:, None]
            sin_yaw = sin_yaw[:, None]
        return torch.stack(
            [
                xy[..., 0] * cos_yaw - xy[..., 1] * sin_yaw,
                xy[..., 0] * sin_yaw + xy[..., 1] * cos_yaw,
            ],
            dim=-1,
        )

    def _compute_push_direction(self, tee_pos: torch.Tensor, goal_pos: torch.Tensor):
        direction_xy = goal_pos[:, :2] - tee_pos[:, :2]
        return direction_xy / torch.linalg.norm(
            direction_xy, dim=1, keepdim=True
        ).clamp_min(1e-6)

    def _compute_support_radius(
        self, direction_xy: torch.Tensor, goal_yaw: torch.Tensor
    ) -> torch.Tensor:
        footprint_vertices = torch.as_tensor(
            [
                [-0.1000, -0.0125],
                [0.1000, -0.0125],
                [-0.1000, -0.0625],
                [0.1000, -0.0625],
                [-0.0250, 0.1375],
                [0.0250, 0.1375],
                [-0.0250, -0.0125],
                [0.0250, -0.0125],
            ],
            dtype=direction_xy.dtype,
            device=direction_xy.device,
        )
        rotated_vertices = self._rotate_xy(
            footprint_vertices[None].expand(direction_xy.shape[0], -1, -1), goal_yaw
        )
        return torch.max(
            torch.sum(rotated_vertices * (-direction_xy[:, None, :]), dim=-1),
            dim=1,
        ).values.clamp_min(0.0)

    def _compute_signals(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        intersection = env.pseudo_render_intersection()
        tee_pos = env.tee.pose.p
        goal_pos = env.goal_tee.pose.p
        tee_yaw = env.quat_to_z_euler(env.tee.pose.q)
        goal_yaw = env.quat_to_z_euler(env.goal_tee.pose.q)
        yaw_delta = self._wrap_to_pi(tee_yaw - goal_yaw)
        yaw_error = torch.abs(yaw_delta)
        orientation_aligned = yaw_error < 0.15
        rotation_contact_side = [
            "right" if item > 0 else "left"
            for item in yaw_delta.detach().cpu().tolist()
        ]
        signals = {
            "success": info["success"],
            "intersection": intersection,
            "tee_to_goal_dist": torch.linalg.norm(
                tee_pos[:, :2] - goal_pos[:, :2], dim=1
            ),
            "yaw_error": yaw_error,
            "yaw_delta": yaw_delta,
            "orientation_aligned": orientation_aligned,
            "rotation_contact_side": rotation_contact_side,
            "object_goal_pose_world": env.goal_tee.pose.to_transformation_matrix(),
        }
        if env_idx is None:
            return signals
        index_list = [int(i) for i in env_idx.detach().cpu().tolist()]
        return {
            key: value[env_idx]
            if torch.is_tensor(value)
            else [value[i] for i in index_list]
            if isinstance(value, list)
            else value
            for key, value in signals.items()
        }

    def reset(self, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        if env_idx is None:
            self.phase.fill_(int(PushTSkillPhase.PUSH_ORIENT))
        else:
            self.phase[env_idx] = int(PushTSkillPhase.PUSH_ORIENT)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        orientation_aligned = signals["orientation_aligned"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        push_orient = phase == int(PushTSkillPhase.PUSH_ORIENT)
        push_place = phase == int(PushTSkillPhase.PUSH_PLACE)
        phase[push_orient & orientation_aligned] = int(PushTSkillPhase.PUSH_PLACE)
        phase[push_place & success] = int(PushTSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        push_orient = phase == int(PushTSkillPhase.PUSH_ORIENT)
        push_place = phase == int(PushTSkillPhase.PUSH_PLACE)
        done = phase == int(PushTSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[push_orient] = SKILL_IDS["push"]
        skill_id[push_place] = SKILL_IDS["push"]

        tee_pos = self._select(env.tee.pose.p, env_idx).reshape(-1, 3)
        goal_pos = self._select(env.goal_tee.pose.p, env_idx).reshape(-1, 3)
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        goal_yaw = self._select(env.quat_to_z_euler(env.goal_tee.pose.q), env_idx)
        signals = self._compute_signals(env, env_idx)

        orient_contact_xy = torch.zeros((len(phase), 2), device=phase.device)
        orient_contact_xy[:, 0] = torch.where(
            signals["yaw_delta"] > 0,
            torch.full_like(signals["yaw_delta"], 0.10),
            torch.full_like(signals["yaw_delta"], -0.10),
        )
        orient_contact_xy[:, 1] = -0.0375
        orient_target_pos = tee_pos.clone()
        orient_target_pos[:, :2] = tee_pos[:, :2] + self._rotate_xy(
            orient_contact_xy, goal_yaw
        )

        push_direction = self._compute_push_direction(tee_pos, goal_pos)
        support_radius = self._compute_support_radius(push_direction, goal_yaw)
        place_target_pos = tee_pos.clone()
        place_target_pos[:, :2] = goal_pos[:, :2] - push_direction * (
            support_radius[:, None] + 0.005
        )

        orient_target_pose = Pose.create_from_pq(p=orient_target_pos, q=tcp_pose.q)
        place_target_pose = Pose.create_from_pq(p=place_target_pos, q=tcp_pose.q)

        target_pose_world = torch.where(
            push_place[:, None, None],
            place_target_pose.to_transformation_matrix(),
            orient_target_pose.to_transformation_matrix(),
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        phase_names_by_id = {
            int(PushTSkillPhase.PUSH_ORIENT): "push_orient",
            int(PushTSkillPhase.PUSH_PLACE): "push_place",
            int(PushTSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PushTSkillPhase.PUSH_ORIENT): "push",
            int(PushTSkillPhase.PUSH_PLACE): "push",
            int(PushTSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PushTSkillPhase.PUSH_ORIENT): "rotate_tee_toward_goal_yaw",
            int(PushTSkillPhase.PUSH_PLACE): "push_aligned_tee_into_goal",
            int(PushTSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PushTSkillPhase.DONE) else "tcp" for x in phase_ids
        ]

        task_meta = {
            "task": "PushT-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "intersection": signals["intersection"],
            "tee_to_goal_dist": signals["tee_to_goal_dist"],
            "yaw_error": signals["yaw_error"],
            "orientation_aligned": signals["orientation_aligned"],
            "rotation_contact_side": signals["rotation_contact_side"],
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
            target_gripper_width=None,
            active_object=["tee"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PushT-v1", max_episode_steps=100)
class PushTEnv(BaseEnv):
    """
    **Task Description:**
    A simulated version of the real-world push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

    In this task, the robot needs to:
    1. Precisely push the T-shaped block into the target region, and
    2. Move the end-effector to the end-zone which terminates the episode. [2 Not required for PushT-easy-v1]

    **Randomizations:**
    - 3D T block initial position on table  [-1,1] x [-1,2] + T Goal initial position
    - 3D T block initial z rotation         [0,2pi]

    **Success Conditions:**
    - The T block covers 90% of the 2D goal T's area
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PushT-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    # # # # # # # # All Unspecified real-life Parameters Here # # # # # # # #
    # Randomizations
    # 3D T center of mass spawnbox dimensions
    tee_spawnbox_xlength = 0.2
    tee_spawnbox_ylength = 0.3

    # translation of the spawnbox from goal tee as upper left of spawnbox
    tee_spawnbox_xoffset = -0.1
    tee_spawnbox_yoffset = -0.1
    #  end randomizations - rotation around z is simply uniform

    # Hand crafted params to match visual of real life setup
    # T Goal initial position on table
    goal_offset = torch.tensor([-0.156, -0.1])
    goal_z_rot = (5 / 3) * np.pi

    # end effector goal - NOTE that chaning this will not change the actual
    # ee starting position of the robot - need to change joint position resting
    # keyframe in table setup to change ee starting location, then copy that location here
    ee_starting_pos2D = torch.tensor([-0.321, 0.284, 1e-3])
    # this will be used in the state observations
    ee_starting_pos3D = torch.tensor([-0.321, 0.284, 0.024])

    # intersection threshold for success in T position
    intersection_thresh = 0.90

    # T block design choices
    T_mass = 0.8
    T_dynamic_friction = 3
    T_static_friction = 3

    def __init__(
        self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.02, **kwargs
    ):
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # have to put these parmaeters to device - defined before we had access to device
        # load scene is a convienent place for this one time operation
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # returns 3d cad of create_tee - center of mass at (0,0,0)
        # cad Tee is upside down (both 3D tee and target)
        TARGET_RED = (
            np.array([194, 19, 22, 255]) / 255
        )  # same as mani_skill.utils.building.actors.common - goal target

        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            # dimensions of boxes that make tee
            # box2 is same as box1, except (3/4) the lenght, and rotated 90 degrees
            # these dimensions are an exact replica of the 3D tee model given by diffusion policy: https://cad.onshape.com/documents/f1140134e38f6ed6902648d5/w/a78cf81827600e4ff4058d03/e/f35f57fb7589f72e05c76caf
            box1_half_w = 0.2 / 2
            box1_half_h = 0.05 / 2
            half_thickness = 0.04 / 2 if not target else 1e-4

            # we have to center tee at its com so rotations are applied to com
            # vertical block is (3/4) size of horizontal block, so
            # center of mass is (1*com_horiz + (3/4)*com_vert) / (1+(3/4))
            # # center of mass is (1*(0,0)) + (3/4)*(0,(.025+.15)/2)) / (1+(3/4)) = (0,0.0375)
            com_y = 0.0375

            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            if not target:
                builder._mass = self.T_mass
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction,
                    dynamic_friction=self.T_static_friction,
                    restitution=0,
                )
                builder.add_box_collision(
                    pose=first_block_pose,
                    half_size=first_block_size,
                    material=tee_material,
                )
                # builder.add_box_collision(pose=first_block_pose, half_size=first_block_size)
            builder.add_box_visual(
                pose=first_block_pose,
                half_size=first_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )

            # for the second block (vertical part), we translate y by 4*(box1_half_h)-com_y to align flush with horizontal block
            # note that the cad model tee made here is upside down
            second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])
            second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]
            if not target:
                builder.add_box_collision(
                    pose=second_block_pose,
                    half_size=second_block_size,
                    material=tee_material,
                )
                # builder.add_box_collision(pose=second_block_pose, half_size=second_block_size)
            builder.add_box_visual(
                pose=second_block_pose,
                half_size=second_block_size,
                material=sapien.render.RenderMaterial(
                    base_color=base_color,
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            if not target:
                return builder.build(name=name)
            else:
                return builder.build_kinematic(name=name)

        self.tee = create_tee(name="Tee", target=False)
        self.goal_tee = create_tee(
            name="goal_Tee",
            target=True,
            base_color=np.array([128, 128, 128, 255]) / 255,
        )

        # adding end-effector end-episode goal position
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=0.02,
            half_length=1e-4,
            material=sapien.render.RenderMaterial(
                base_color=np.array([128, 128, 128, 255]) / 255
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")

        # Rest of function is setting up for Custom 2D "Pseudo-Rendering" function below
        res = 64
        uv_half_width = 0.15
        self.uv_half_width = uv_half_width
        self.res = res
        oned_grid = torch.arange(res, dtype=torch.float32).view(1, res).repeat(
            res, 1
        ) - (res / 2)
        self.uv_grid = (
            torch.cat([oned_grid.unsqueeze(0), (-1 * oned_grid.T).unsqueeze(0)], dim=0)
            + 0.5
        ) / ((res / 2) / uv_half_width)
        self.uv_grid = self.uv_grid.to(self.device)
        self.homo_uv = torch.cat(
            [self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0
        )

        # tee render
        # tee is made of two different boxes, and then translated by center of mass
        self.center_of_mass = (
            0,
            0.0375,
        )  # in frame of upside tee with center of horizontal box (add cetner of mass to get to real tee frame)
        box1 = torch.tensor(
            [[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]]
        )
        box2 = torch.tensor(
            [[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]]
        )
        box1[:, 1] -= self.center_of_mass[1]
        box2[:, 1] -= self.center_of_mass[1]

        # convert tee boxes to indices
        box1 *= (res / 2) / uv_half_width
        box1 += res / 2

        box2 *= (res / 2) / uv_half_width
        box2 += res / 2

        box1 = box1.long()
        box2 = box2.long()

        self.tee_render = torch.zeros(res, res)
        # image map has flipped x and y, set values in transpose to undo
        self.tee_render.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1
        self.tee_render.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1
        # image map y is flipped of xy plane, flip to unflip
        self.tee_render = self.tee_render.flip(0).to(self.device)

        goal_fake_quat = torch.tensor(
            [(torch.tensor([self.goal_z_rot]) / 2).cos(), 0, 0, 0.0]
        ).unsqueeze(0)
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(
            0
        )  # 3x3 rot matrix for goal to world transform
        goal_trans = torch.eye(3)
        goal_trans[:2, :2] = zrot[:2, :2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(
            self.device
        )  # this is just a 3x3 matrix (2d homogenious transform)

    def quat_to_z_euler(self, quats):
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # z rotation == can be defined by just qw = cos(alpha/2), so alpha = 2*cos^{-1}(qw)
        # for fixing quaternion double covering
        # for some reason, torch.sign() had bugs???
        signs = torch.ones_like(quats[:, -1])
        signs[quats[:, -1] < 0] = -1.0
        qw = quats[:, 0] * signs
        z_euler = 2 * qw.acos()
        return z_euler

    def quat_to_zrot(self, quats):
        # expecting batch of quaternions (b,4)
        assert len(quats.shape) == 2 and quats.shape[-1] == 4
        # output is batch of rotation matrices (b,3,3)
        alphas = self.quat_to_z_euler(quats)
        # constructing rot matrix with rotation around z
        rot_mats = torch.zeros(quats.shape[0], 3, 3).to(quats.device)
        rot_mats[:, 2, 2] = 1
        rot_mats[:, 0, 0] = alphas.cos()
        rot_mats[:, 1, 1] = alphas.cos()
        rot_mats[:, 0, 1] = -alphas.sin()
        rot_mats[:, 1, 0] = alphas.sin()
        return rot_mats

    def pseudo_render_intersection(self):
        """'pseudo render' algo for calculating the intersection
        made custom 'psuedo renderer' to compute intersection area
        all computation in parallel on cuda, zero explicit loops
        views blocks in 2d in the goal tee frame to see overlap"""
        # we are given T_{a->w} where a == actor frame and w == world frame
        # we are given T_{g->w} where g == goal frame and w == world frame
        # applying T_{a->w} and then T_{w->g}, we get the actor's orientation in the goal tee's frame
        # T_{w->g} is T_{g->w}^{-1}, we already have the goal's orientation, and it doesn't change
        tee_to_world_trans = self.quat_to_zrot(
            self.tee.pose.q
        )  # should be (b,3,3) rot matrices
        tee_to_world_trans[:, 0:2, 2] = self.tee.pose.p[
            :, :2
        ]  # should be (b,3,3) rigid trans matrices

        # these matrices convert egocentric 3d tee to 2d goal tee frame
        tee_to_goal_trans = (
            self.world_to_goal_trans @ tee_to_world_trans
        )  # should be (b,3,3) rigid trans matrices

        # making homogenious coords of uv map to apply transformations to view tee in goal tee frame
        b = tee_to_world_trans.shape[0]
        res = self.uv_grid.shape[1]
        homo_uv = self.homo_uv

        # finally, get uv coordinates of tee in goal tee frame
        tees_in_goal_frame = (tee_to_goal_trans @ homo_uv.view(3, -1)).view(
            b, 3, res, res
        )
        # convert from homogenious coords to normal coords
        tees_in_goal_frame = tees_in_goal_frame[:, 0:2, :, :] / tees_in_goal_frame[
            :, -1, :, :
        ].unsqueeze(
            1
        )  #  now (b,2,res,res)

        # we now have a collection of coordinates xy that are the coordinates of the tees in the goal frame
        # we just extract the indices in the uv map where the egocentic T is, to get the transformed T coords
        # this works because while we transformed the coordinates of the uv map -
        # the indices where the egocentric T is is still the indices of the T in the uv map (indices of uv map never chnaged, just values)
        tee_coords = tees_in_goal_frame[:, :, self.tee_render == 1].view(
            b, 2, -1
        )  #  (b,2,num_points_in_tee)

        # convert tee_coords to indices - this is basically a batch of indices - same shape as tee_coords
        # this is the inverse function of creating the uv map from image indices used in load_scene
        tee_indices = (
            (tee_coords * ((res / 2) / self.uv_half_width) + (res / 2))
            .long()
            .view(b, 2, -1)
        )  #  (b,2,num_points_in_tee)

        # setting all of our work in image format to compare with egocentric image of goal T
        final_renders = torch.zeros(b, res, res).to(self.device)
        # for batch indexing
        num_tee_pixels = tee_indices.shape[-1]
        batch_indices = (
            torch.arange(b).view(-1, 1).repeat(1, num_tee_pixels).to(self.device)
        )

        # # ensure no out of bounds indexing - it's fine to not fully 'render' tee, just need to fully see goal tee which is insured
        # # because we are in the goal tee frame, and 'cad' tee render setup of egocentric view includes full tee
        # # also, the reward isn't miou, it's intersection area / goal area - don't need union -> don't need full T 'render'
        # #ugly solution for now to keep parallelism no loop - set out of bound image t indices to [0,0]
        # # anywhere where x or y is out of bounds, make indices (0,0)
        invalid_xs = (tee_indices[:, 0, :] < 0) | (tee_indices[:, 0, :] >= self.res)
        invalid_ys = (tee_indices[:, 1, :] < 0) | (tee_indices[:, 1, :] >= self.res)
        tee_indices[:, 0, :][invalid_xs] = 0
        tee_indices[:, 1, :][invalid_xs] = 0
        tee_indices[:, 0, :][invalid_ys] = 0
        tee_indices[:, 1, :][invalid_ys] = 0

        final_renders[batch_indices, tee_indices[:, 0, :], tee_indices[:, 1, :]] = 1
        # coord to image fix - need to transpose each image in the batch, then reverse y coords to correctly visualize
        final_renders = final_renders.permute(0, 2, 1).flip(1)

        # finally, we can calculate intersection/goal_area for reward
        intersection = (
            (final_renders.bool() & self.tee_render.bool()).sum(dim=[-1, -2]).float()
        )
        goal_area = self.tee_render.bool().sum().float()

        reward = intersection / goal_area

        # del tee_to_world_trans; del tee_to_goal_trans; del tees_in_goal_frame; del tee_coords; del tee_indices
        # del final_renders; del invalid_xs; del invalid_ys; batch_indices; del intersection; del goal_area
        # torch.cuda.empty_cache()
        return reward

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # setting the goal tee position, which is fixed, offset from center, and slightly rotated
            target_region_xyz = torch.zeros((b, 3))
            target_region_xyz[:, 0] += self.goal_offset[0]
            target_region_xyz[:, 1] += self.goal_offset[1]
            # set a little bit above 0 so the target is sitting on the table
            target_region_xyz[..., 2] = 1e-3
            self.goal_tee.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, 0, self.goal_z_rot),
                )
            )

            # randomization code that randomizes the x, y position of the tee we
            # goal tee is alredy at y = -0.1 relative to robot, so we allow the tee to be only -0.2 y relative to robot arm
            target_region_xyz[..., 0] += (
                torch.rand(b) * (self.tee_spawnbox_xlength) + self.tee_spawnbox_xoffset
            )
            target_region_xyz[..., 1] += (
                torch.rand(b) * (self.tee_spawnbox_ylength) + self.tee_spawnbox_yoffset
            )

            target_region_xyz[..., 2] = (
                0.04 / 2 + 1e-3
            )  # this is the half thickness of the tee plus a little
            # rotation for pose is just random rotation around z axis
            # z axis rotation euler to quaternion = [cos(theta/2),0,0,sin(theta/2)]
            q_euler_angle = torch.rand(b) * (2 * torch.pi)
            q = torch.zeros((b, 4))
            q[:, 0] = (q_euler_angle / 2).cos()
            q[:, -1] = (q_euler_angle / 2).sin()

            obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
            self.tee.set_pose(obj_pose)

            # ee starting/ending position marked on table like irl task
            xyz = torch.zeros((b, 3))
            xyz[:] = self.ee_starting_pos2D
            self.ee_goal_pos.set_pose(
                Pose.create_from_pq(
                    p=xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PushTSkillFSM):
            fsm = PushTSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def evaluate(self):
        # success is where the overlap is over intersection thresh and ee dist to start pos is less than it's own thresh
        inter_area = self.pseudo_render_intersection()
        tee_place_success = (inter_area) >= self.intersection_thresh

        success = tee_place_success

        return {"success": success}

    def _get_obs_extra(self, info: dict):
        # ee position is super useful for pandastick robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # state based gets info on goal position and t full pose - necessary to learn task
            obs.update(
                goal_pos=self.goal_tee.pose.p,
                obj_pose=self.tee.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # reward for overlap of the tees

        # legacy reward
        # reward = self.pseudo_render_reward()
        # Pose based reward below is preferred over legacy reward
        # legacy reward gets stuck in local maxs of 50-75% intersection
        # and then fails to promote large explorations to perfectly orient the T, for PPO algorithm

        # new pose based reward: cos(z_rot_euler) + function of translation, between target and goal both in [0,1]
        # z euler cosine similarity reward: -- quat_to_z_euler guarenteed to reutrn value from [0,2pi]
        tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
        # subtract the goal z rotatation to get relative rotation
        rot_rew = (tee_z_eulers - self.goal_z_rot).cos()
        # cos output [-1,1], we want reward of 0.5
        reward = (((rot_rew + 1) / 2) ** 2) / 2

        # x and y distance as reward
        tee_to_goal_pose = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
        tee_to_goal_pose_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tee_to_goal_pose_dist)) ** 2) / 2

        # giving the robot a little help by rewarding it for having its end-effector close to the tee center of mass
        tcp_to_push_pose = self.tee.pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

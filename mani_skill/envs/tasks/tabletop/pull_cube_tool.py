from enum import IntEnum
from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.skill_annotation.schema import SkillAnnotationContext, SKILL_IDS
from mani_skill.utils.skill_annotation.targets import (
    build_panda_topdown_grasp_pose,
    target_tcp_pose_from_object_goal,
)
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


class PullCubeToolSkillPhase(IntEnum):
    PICK = 0
    ALIGN = 1
    PULL = 2
    DONE = 3


class PullCubeToolSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PullCubeToolSkillPhase.PICK),
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
        cube_pos = env.cube.pose.p
        tool_pos = env.l_shape_tool.pose.p
        robot_base_pos = env.agent.robot.get_links()[0].pose.p
        ideal_tool_pos = cube_pos + torch.tensor(
            [-(env.hook_length + env.cube_half_size), -0.067, 0],
            device=env.device,
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_tool_pos, dim=1)
        workspace_target_world = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=env.device
        )
        signals = {
            "success": info["success"],
            "is_tool_grasped": env.agent.is_grasping(
                env.l_shape_tool, max_angle=20
            ),
            "tool_positioned": tool_positioning_dist < 0.05,
            "tool_positioning_dist": tool_positioning_dist,
            "cube_to_base_dist": torch.linalg.norm(
                cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
            ),
            "workspace_target_world": workspace_target_world,
            "ideal_tool_pos": ideal_tool_pos,
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
            self.phase.fill_(int(PullCubeToolSkillPhase.PICK))
        else:
            self.phase[env_idx] = int(PullCubeToolSkillPhase.PICK)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        signals = self._compute_signals(env, env_idx)
        is_tool_grasped = signals["is_tool_grasped"]
        tool_positioned = signals["tool_positioned"]
        success = signals["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()

        pick = phase == int(PullCubeToolSkillPhase.PICK)
        align = phase == int(PullCubeToolSkillPhase.ALIGN)
        pull = phase == int(PullCubeToolSkillPhase.PULL)
        phase[pick & is_tool_grasped] = int(PullCubeToolSkillPhase.ALIGN)
        phase[align & is_tool_grasped & tool_positioned] = int(
            PullCubeToolSkillPhase.PULL
        )
        phase[pull & success] = int(PullCubeToolSkillPhase.DONE)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(PullCubeToolSkillPhase.PICK)
        align = phase == int(PullCubeToolSkillPhase.ALIGN)
        pull = phase == int(PullCubeToolSkillPhase.PULL)
        done = phase == int(PullCubeToolSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[align] = SKILL_IDS["push"]
        skill_id[pull] = SKILL_IDS["push"]

        tool_pose = Pose.create(self._select(env.l_shape_tool.pose.raw_pose, env_idx))
        cube_pose = Pose.create(self._select(env.cube.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        grasp_center = tool_pose.p + torch.tensor([0.02, 0, 0], device=env.device)
        pick_target_pose = build_panda_topdown_grasp_pose(
            center=grasp_center,
            tcp_pose=tcp_pose,
            object_pose=tool_pose,
        )

        ideal_tool_pos = cube_pose.p + torch.tensor(
            [-(env.hook_length + env.cube_half_size), -0.067, 0],
            device=env.device,
        )
        ideal_tool_pose = Pose.create_from_pq(p=ideal_tool_pos, q=tool_pose.q)
        align_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=tool_pose,
            desired_object_pose=ideal_tool_pose,
        )
        pull_target_pos = align_target_pose.p.clone()
        pull_target_pos[:, 0] = pull_target_pos[:, 0] - 0.35
        pull_target_pose = Pose.create_from_pq(p=pull_target_pos, q=align_target_pose.q)

        target_pose_world = pick_target_pose.to_transformation_matrix()
        target_pose_world = torch.where(
            align[:, None, None],
            align_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world = torch.where(
            pull[:, None, None],
            pull_target_pose.to_transformation_matrix(),
            target_pose_world,
        )
        target_pose_world[done] = float("nan")
        target_point_world = target_pose_world[:, :3, 3].clone()

        target_gripper_width = torch.zeros(
            phase.shape, dtype=torch.float32, device=phase.device
        )
        target_gripper_width[done] = float("nan")

        phase_names_by_id = {
            int(PullCubeToolSkillPhase.PICK): "pick",
            int(PullCubeToolSkillPhase.ALIGN): "align",
            int(PullCubeToolSkillPhase.PULL): "pull",
            int(PullCubeToolSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PullCubeToolSkillPhase.PICK): "pick",
            int(PullCubeToolSkillPhase.ALIGN): "push",
            int(PullCubeToolSkillPhase.PULL): "push",
            int(PullCubeToolSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PullCubeToolSkillPhase.PICK): "move_to_tool_grasp",
            int(PullCubeToolSkillPhase.ALIGN): "position_hook_behind_cube",
            int(PullCubeToolSkillPhase.PULL): "pull_cube_into_workspace",
            int(PullCubeToolSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PullCubeToolSkillPhase.DONE) else "tcp"
            for x in phase_ids
        ]

        signals = self._compute_signals(env, env_idx)
        task_meta = {
            "task": "PullCubeTool-v1",
            "target_frame": "world",
            "target_entity": "tcp",
            "success": signals["success"],
            "is_tool_grasped": signals["is_tool_grasped"],
            "tool_positioned": signals["tool_positioned"],
            "tool_positioning_dist": signals["tool_positioning_dist"],
            "cube_to_base_dist": signals["cube_to_base_dist"],
            "workspace_target_world": signals["workspace_target_world"],
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
            active_object=["l_shape_tool"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PullCubeTool-v1", max_episode_steps=100)
class PullCubeToolEnv(BaseEnv):
    """
    **Task Description**
    Given an L-shaped tool that is within the reach of the robot, leverage the
    tool to pull a cube that is out of it's reach

    **Randomizations**
    - The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
    reach, but within reach of tool>". It is placed flat on the table
    - The target goal region is the region on top of the table marked by "<within reach of arm>"

    **Success Conditions**
    - The cube's xy position is within the goal region of the arm's base (marked by reachability)
    """

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.05
    width = 0.05
    height = 0.05
    cube_size = 0.02
    arm_reach = 0.35

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[..., 0] = (
                self.arm_reach
                + torch.rand(b, device=self.device) * (self.handle_length)
                - 0.3
            )
            cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            cube_xyz[..., 2] = self.cube_size / 2 + 0.015

            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
                device=self.device,
            )

            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PullCubeToolSkillFSM):
            fsm = PullCubeToolSkillFSM(num_envs=self.num_envs, device=self.device)
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
                tool_pose=self.l_shape_tool.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p

        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        cube_to_base_dist = torch.linalg.norm(
            cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
        )

        # Success condition - cube is pulled close enough
        cube_pulled_close = cube_to_base_dist < 0.6

        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "cube_progress": progress.mean(),
            "cube_distance": cube_to_workspace_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        if "success" in info:
            reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward

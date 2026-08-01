from enum import IntEnum
from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
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
from mani_skill.utils.structs.pose import Pose

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


class PickCubeSkillPhase(IntEnum):
    PICK = 0
    PLACE = 1
    DONE = 2


class PickCubeSkillFSM:
    def __init__(self, num_envs: int, device):
        self.phase = torch.full(
            (num_envs,),
            int(PickCubeSkillPhase.PICK),
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
            self.phase.fill_(int(PickCubeSkillPhase.PICK))
        else:
            self.phase[env_idx] = int(PickCubeSkillPhase.PICK)

    def update(self, env, env_idx=None):
        env_idx = self._normalize_env_idx(env_idx)
        info = env.evaluate()
        is_grasped = info["is_grasped"]
        success = info["success"]

        if env_idx is None:
            phase = self.phase.clone()
        else:
            phase = self.phase[env_idx].clone()
            is_grasped = is_grasped[env_idx]
            success = success[env_idx]

        pick = phase == int(PickCubeSkillPhase.PICK)
        place = phase == int(PickCubeSkillPhase.PLACE)
        phase[pick & is_grasped] = int(PickCubeSkillPhase.PLACE)
        phase[place & success] = int(PickCubeSkillPhase.DONE)
        phase[place & ~is_grasped & ~success] = int(PickCubeSkillPhase.PICK)

        if env_idx is None:
            self.phase.copy_(phase)
        else:
            self.phase[env_idx] = phase

    def build_context(self, env, env_idx=None) -> SkillAnnotationContext:
        env_idx = self._normalize_env_idx(env_idx)
        phase = self.phase if env_idx is None else self.phase[env_idx]
        phase = phase.reshape(-1)

        pick = phase == int(PickCubeSkillPhase.PICK)
        place = phase == int(PickCubeSkillPhase.PLACE)
        done = phase == int(PickCubeSkillPhase.DONE)

        skill_id = torch.full_like(phase, SKILL_IDS["none"])
        skill_id[pick] = SKILL_IDS["pick"]
        skill_id[place] = SKILL_IDS["place"]

        cube_pose = Pose.create(self._select(env.cube.pose.raw_pose, env_idx))
        tcp_pose = Pose.create(self._select(env.agent.tcp.pose.raw_pose, env_idx))
        pick_target_pose = build_panda_topdown_grasp_pose(
            center=cube_pose.p,
            tcp_pose=tcp_pose,
            object_pose=cube_pose,
        )

        goal_pos = self._select(env.goal_site.pose.p, env_idx).reshape(-1, 3)
        desired_cube_pose = Pose.create_from_pq(p=goal_pos, q=cube_pose.q)
        place_target_pose = target_tcp_pose_from_object_goal(
            current_tcp_pose=tcp_pose,
            current_object_pose=cube_pose,
            desired_object_pose=desired_cube_pose,
        )

        target_pose_world = torch.where(
            place[:, None, None],
            place_target_pose.to_transformation_matrix(),
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

        phase_names_by_id = {
            int(PickCubeSkillPhase.PICK): "pick",
            int(PickCubeSkillPhase.PLACE): "place",
            int(PickCubeSkillPhase.DONE): "done",
        }
        skill_names_by_phase_id = {
            int(PickCubeSkillPhase.PICK): "pick",
            int(PickCubeSkillPhase.PLACE): "place",
            int(PickCubeSkillPhase.DONE): "none",
        }
        skill_states_by_phase_id = {
            int(PickCubeSkillPhase.PICK): "move_to_cube",
            int(PickCubeSkillPhase.PLACE): "move_to_goal",
            int(PickCubeSkillPhase.DONE): "task_done",
        }
        phase_ids = [int(x) for x in phase.detach().cpu().tolist()]
        target_objects = [
            None if x == int(PickCubeSkillPhase.DONE) else "tcp" for x in phase_ids
        ]

        info = env.evaluate()
        task_meta = {
            "task": "PickCube-v1",
            "target_frame": "world",
            "is_grasped": self._select(info["is_grasped"], env_idx),
            "is_obj_placed": self._select(info["is_obj_placed"], env_idx),
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
            active_object=["cube"] * len(phase_ids),
            target_object=target_objects,
            task_meta=task_meta,
        )


@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv(BaseEnv):

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.sensor_cam_width = cfg.get("sensor_cam_width", 128)
        self.sensor_cam_height = cfg.get("sensor_cam_height", 128)
        self.sensor_cam_fov = cfg.get("sensor_cam_fov", np.pi / 2)
        self.sensor_cam_near = cfg.get("sensor_cam_near", 0.01)
        self.sensor_cam_far = cfg.get("sensor_cam_far", 100)
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                self.sensor_cam_width,
                self.sensor_cam_height,
                self.sensor_cam_fov,
                self.sensor_cam_near,
                self.sensor_cam_far,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
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
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
            self._get_or_create_skill_annotation_fsm().reset(env_idx)

    def _get_or_create_skill_annotation_fsm(self):
        fsm = getattr(self, "_skill_annotation_fsm", None)
        if not isinstance(fsm, PickCubeSkillFSM):
            fsm = PickCubeSkillFSM(num_envs=self.num_envs, device=self.device)
            self._skill_annotation_fsm = fsm
        return fsm

    def get_skill_annotation_context(self, env_idx=None):
        fsm = self._get_or_create_skill_annotation_fsm()
        fsm.update(self, env_idx)
        return fsm.build_context(self, env_idx)

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickCubeEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda")


@register_env("PickCubeSO100-v1", max_episode_steps=50)
class PickCubeSO100Env(PickCubeEnv):
    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


PickCubeSO100Env.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="SO100")


@register_env("PickCubeWidowXAI-v1", max_episode_steps=50)
class PickCubeWidowXAIEnv(PickCubeEnv):
    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCubeWidowXAI-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


PickCubeWidowXAIEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="WidowXAI")

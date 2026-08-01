"""Extract RR-compatible robot, camera, part, and guidance observations."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from .schema import (
    CAMERA_TO_IMAGE_KEYS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PICK_CUBE_PART_NAMES,
    CameraInfo,
    GraspAnnotation2D,
    PickleObservation,
    RobotState,
)
from .transforms import (
    as_numpy,
    matrix_to_pose,
    normalize_quaternion_xyzw,
    world_point_to_base,
    world_pose_to_base,
    world_vector_to_base,
)


class PickCubeStateAdapter:
    """Capture a single fixed-base Panda observation in its base frame."""

    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        if int(getattr(self.base_env, "num_envs", 1)) != 1:
            raise NotImplementedError("Pickle state recording supports num_envs=1 only")
        if getattr(self.base_env, "robot_uids", None) not in {
            "panda",
            "panda_wristcam",
        }:
            raise NotImplementedError(
                "Pickle state recording currently supports fixed-base Panda only"
            )
        if not hasattr(self.base_env, "cube") or not hasattr(
            self.base_env, "goal_site"
        ):
            raise NotImplementedError(
                "PickCube state recording requires cube and goal_site actors"
            )
        self.agent = self.base_env.agent

    def capture(
        self,
        raw_observation: Mapping[str, Any],
        annotation_bundle: Optional[Mapping[str, Any]] = None,
    ) -> tuple[PickleObservation, CameraInfo]:
        """Capture one state-aligned observation and front-camera calibration."""

        world_base = self._world_base_matrix()
        robot_state = self._robot_state(world_base)
        color1, depth1 = self._camera_images(raw_observation, "hand_camera")
        color2, depth2 = self._camera_images(raw_observation, "base_camera")
        parts_poses = self._parts_poses(world_base)
        annotations = self._annotations(annotation_bundle, world_base)

        observation: PickleObservation = {
            "robot_state": robot_state,
            "color_image1": color1,
            "color_image2": color2,
            "depth_image1": depth1,
            "depth_image2": depth2,
            "parts_poses": parts_poses,
            "point_cloud": None,
            **annotations,
        }
        camera_info = self._front_camera_info(raw_observation, world_base, color2)
        return observation, camera_info

    def _world_base_matrix(self) -> np.ndarray:
        return self._single_matrix(
            self.agent.robot.pose.to_transformation_matrix(), "Panda base pose"
        )

    def _robot_state(self, world_base: np.ndarray) -> RobotState:
        robot = self.agent.robot
        tcp = self.agent.tcp
        world_tcp = self._single_matrix(
            tcp.pose.to_transformation_matrix(), "TCP pose"
        )
        base_tcp = world_pose_to_base(world_tcp, world_base)
        ee_pos, ee_quat = matrix_to_pose(base_tcp)
        ee_quat = normalize_quaternion_xyzw(ee_quat).reshape(4)

        tcp_linear = self._single_vector(tcp.get_linear_velocity(), "TCP velocity")
        tcp_angular = self._single_vector(
            tcp.get_angular_velocity(), "TCP angular velocity"
        )
        root = robot.root
        base_linear = self._single_vector(
            root.get_linear_velocity(), "base velocity"
        )
        base_angular = self._single_vector(
            root.get_angular_velocity(), "base angular velocity"
        )
        base_position = world_base[:3, 3]
        tcp_position = world_tcp[:3, 3]
        relative_linear_world = (
            tcp_linear
            - base_linear
            - np.cross(base_angular, tcp_position - base_position)
        )
        relative_angular_world = tcp_angular - base_angular
        ee_pos_vel = world_vector_to_base(relative_linear_world, world_base).reshape(3)
        ee_ori_vel = world_vector_to_base(relative_angular_world, world_base).reshape(
            3
        )

        qpos = self._single_row(robot.get_qpos(), "robot qpos")
        qvel = self._single_row(robot.get_qvel(), "robot qvel")
        qf = self._single_row(robot.get_qf(), "robot qf")
        controller = self.agent.controller
        if not hasattr(controller, "controllers") or "arm" not in controller.controllers:
            raise ValueError("Panda controller does not expose an arm component")
        arm_indices = as_numpy(
            controller.controllers["arm"].active_joint_indices
        ).astype(np.int64)
        if arm_indices.shape != (7,):
            raise ValueError(f"Expected seven Panda arm joints, got {arm_indices.shape}")

        if not hasattr(self.agent, "gripper_joint_names"):
            raise ValueError("Panda agent does not expose gripper_joint_names")
        finger_indices = []
        for joint_name in self.agent.gripper_joint_names:
            joint = robot.joints_map[joint_name]
            index = as_numpy(joint.active_index).reshape(-1)
            if index.size != 1:
                raise ValueError(f"Could not resolve active index for {joint_name}")
            finger_indices.append(int(index[0]))
        if len(finger_indices) != 2:
            raise ValueError("Panda state recording requires two finger joints")
        finger1, finger2 = (float(qpos[index]) for index in finger_indices)

        # Keep the full nine-DoF generalized force vector as requested by the
        # raw rollout schema extension.  RR's 14D training state ignores this
        # optional diagnostic field.
        if qf.shape != (9,):
            raise ValueError(f"Expected Panda qf shape (9,), got {qf.shape}")
        return {
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "ee_pos_sim": ee_pos.astype(np.float32).copy(),
            "ee_quat_sim": ee_quat.astype(np.float32).copy(),
            "ee_pos_vel": ee_pos_vel.astype(np.float32),
            "ee_ori_vel": ee_ori_vel.astype(np.float32),
            "gripper_width": float(finger1 + finger2),
            "joint_positions": qpos[arm_indices].astype(np.float32),
            "joint_velocities": qvel[arm_indices].astype(np.float32),
            "joint_torques": qf.astype(np.float32),
            "gripper_finger_1_pos": finger1,
            "gripper_finger_2_pos": finger2,
        }

    def _parts_poses(self, world_base: np.ndarray) -> np.ndarray:
        parts = []
        for name in PICK_CUBE_PART_NAMES:
            actor = getattr(self.base_env, name)
            world_part = self._single_matrix(
                actor.pose.to_transformation_matrix(), f"{name} pose"
            )
            part_position, part_quaternion = matrix_to_pose(
                world_pose_to_base(world_part, world_base)
            )
            part_quaternion = normalize_quaternion_xyzw(part_quaternion)
            parts.append(
                np.concatenate([part_position, part_quaternion]).astype(np.float32)
            )
        return np.concatenate(parts).astype(np.float32)

    def _camera_images(
        self, raw_observation: Mapping[str, Any], camera_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        sensor_data = raw_observation.get("sensor_data")
        if not isinstance(sensor_data, Mapping) or camera_name not in sensor_data:
            raise KeyError(
                f"Observation is missing sensor_data[{camera_name!r}]; "
                "use obs_mode='rgbd' and robot_uids='panda_wristcam'"
            )
        camera_data = sensor_data[camera_name]
        if not isinstance(camera_data, Mapping):
            raise TypeError(f"sensor_data[{camera_name!r}] must be a mapping")
        if "rgb" not in camera_data or "depth" not in camera_data:
            raise KeyError(f"Camera {camera_name!r} must provide RGB and depth")

        color = self._single_image(camera_data["rgb"], f"{camera_name} RGB")
        if color.shape[-1:] == (4,):
            color = color[..., :3]
        if color.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
            raise ValueError(
                f"{camera_name} RGB must be 224x224x3, got {color.shape}"
            )
        if np.issubdtype(color.dtype, np.floating):
            if not np.isfinite(color).all():
                raise ValueError(f"{camera_name} RGB contains a non-finite value")
            if color.size and float(color.max()) <= 1.000001:
                color = np.rint(np.clip(color, 0.0, 1.0) * 255.0)
            else:
                color = np.rint(np.clip(color, 0.0, 255.0))
        color = color.astype(np.uint8, copy=False)

        depth = self._single_image(camera_data["depth"], f"{camera_name} depth")
        if depth.shape[-1:] == (1,):
            depth = depth[..., 0]
        if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"{camera_name} depth must be 224x224, got {depth.shape}")
        # Standard ManiSkill RGB-D cameras encode depth in millimetres.
        depth = depth.astype(np.float32) / np.float32(1000.0)
        if not np.isfinite(depth).all():
            raise ValueError(f"{camera_name} depth contains a non-finite value")
        return color.copy(), depth.copy()

    def _front_camera_info(
        self,
        raw_observation: Mapping[str, Any],
        world_base: np.ndarray,
        front_color: np.ndarray,
    ) -> CameraInfo:
        sensor_params = raw_observation.get("sensor_param")
        if not isinstance(sensor_params, Mapping):
            sensor_params = self.base_env.get_sensor_params()
        if "base_camera" not in sensor_params:
            raise KeyError("Missing base_camera calibration")
        params = sensor_params["base_camera"]
        intrinsic = self._single_matrix_shape(
            params["intrinsic_cv"], (3, 3), "base_camera intrinsics"
        )
        extrinsic_cv = self._single_extrinsic(params["extrinsic_cv"])
        base_to_cv = extrinsic_cv @ world_base
        cv_to_rr = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        base_to_camera_rr = cv_to_rr @ base_to_cv
        camera_to_base = np.linalg.inv(base_to_camera_rr).astype(np.float32)
        height, width = front_color.shape[:2]
        return {
            "image_size": np.asarray([width, height], dtype=np.int32),
            "intrinsics": intrinsic.astype(np.float32),
            "camera_to_sim_local": camera_to_base,
            "sim_local_to_camera": base_to_camera_rr.astype(np.float32),
        }

    def _annotations(
        self,
        bundle: Optional[Mapping[str, Any]],
        world_base: np.ndarray,
    ) -> dict[str, Any]:
        image_keys = tuple(
            CAMERA_TO_IMAGE_KEYS[camera][0] for camera in CAMERA_TO_IMAGE_KEYS
        )
        empty = {
            "skill": None,
            "guidance_point": None,
            "guidance_point_clean": None,
            "guidance_pose": None,
            "guidance_pose_clean": None,
            "guidance_gripper_width": None,
            "guidance_point_2d": {key: None for key in image_keys},
            "grasp_annotation_2d": {key: None for key in image_keys},
        }
        if bundle is None:
            return empty

        skills = bundle.get("skill", [])
        skill = skills[0] if skills else "none"
        empty["skill"] = None if skill == "none" else str(skill)
        target = bundle.get("target", {})
        point_valid = self._first_bool(target.get("point_valid", False))
        pose_valid = self._first_bool(target.get("pose_valid", False))
        width_valid = self._first_bool(
            target.get("gripper_width_valid", False)
        )
        if point_valid:
            point_world = self._single_vector(
                target["point_world"], "guidance point"
            )
            point_base = world_point_to_base(point_world, world_base).astype(
                np.float32
            )
            empty["guidance_point"] = point_base
            empty["guidance_point_clean"] = point_base.copy()
        if pose_valid:
            pose_world = self._single_matrix(
                target["pose_world"], "guidance pose"
            )
            pose_base = world_pose_to_base(pose_world, world_base).astype(np.float32)
            empty["guidance_pose"] = pose_base
            empty["guidance_pose_clean"] = pose_base.copy()
        if width_valid:
            width = as_numpy(target["gripper_width"], dtype=np.float32).reshape(-1)
            if width.size != 1 or not np.isfinite(width[0]):
                raise ValueError("Guidance gripper width must be one finite scalar")
            empty["guidance_gripper_width"] = float(width[0])

        projection = bundle.get("projection", {})
        for camera_name, (color_key, _) in CAMERA_TO_IMAGE_KEYS.items():
            camera_projection = projection.get(camera_name)
            if not isinstance(camera_projection, Mapping):
                continue
            if self._first_bool(camera_projection.get("point_visible", False)):
                point_uv = self._single_vector_size(
                    camera_projection["point_uv"], 2, f"{camera_name} point_uv"
                )
                empty["guidance_point_2d"][color_key] = point_uv.astype(np.float32)
            if self._first_bool(camera_projection.get("grasp_visible", False)):
                corners = as_numpy(
                    camera_projection["grasp_rect_uv"], dtype=np.float32
                )
                if corners.ndim == 3 and corners.shape[0] == 1:
                    corners = corners[0]
                if corners.shape != (4, 2) or not np.isfinite(corners).all():
                    raise ValueError(
                        f"{camera_name} grasp rectangle must have shape (4, 2)"
                    )
                center = self._single_vector_size(
                    camera_projection["point_uv"], 2, f"{camera_name} point_uv"
                ).astype(np.float32)
                annotation: GraspAnnotation2D = {
                    "style": "grasp_rect",
                    "center": center,
                    "corners": corners.astype(np.float32),
                }
                empty["grasp_annotation_2d"][color_key] = annotation
        return empty

    @staticmethod
    def _single_row(value: Any, name: str) -> np.ndarray:
        array = as_numpy(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"{name} must have one batch row, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")
        return array[0]

    @staticmethod
    def _single_vector(value: Any, name: str) -> np.ndarray:
        return PickCubeStateAdapter._single_vector_size(value, 3, name)

    @staticmethod
    def _single_vector_size(value: Any, size: int, name: str) -> np.ndarray:
        array = as_numpy(value, dtype=np.float32)
        if array.shape == (1, size):
            array = array[0]
        if array.shape != (size,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        return array

    @staticmethod
    def _single_matrix(value: Any, name: str) -> np.ndarray:
        return PickCubeStateAdapter._single_matrix_shape(value, (4, 4), name)

    @staticmethod
    def _single_matrix_shape(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
        array = as_numpy(value, dtype=np.float32)
        if array.shape == (1,) + shape:
            array = array[0]
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        return array

    @staticmethod
    def _single_extrinsic(value: Any) -> np.ndarray:
        array = as_numpy(value, dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.shape == (3, 4):
            homogeneous = np.eye(4, dtype=np.float32)
            homogeneous[:3] = array
            array = homogeneous
        if array.shape != (4, 4) or not np.isfinite(array).all():
            raise ValueError(
                f"base_camera extrinsic must be 3x4 or 4x4, got {array.shape}"
            )
        return array

    @staticmethod
    def _single_image(value: Any, name: str) -> np.ndarray:
        array = as_numpy(value)
        if array.ndim >= 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim not in (2, 3):
            raise ValueError(f"{name} has unsupported shape {array.shape}")
        return array

    @staticmethod
    def _first_bool(value: Any) -> bool:
        array = as_numpy(value).reshape(-1)
        return bool(array[0]) if array.size else False


__all__ = ["PickCubeStateAdapter"]

"""RR-compatible pickle trajectory schema.

The schema in this module intentionally mirrors
``robust-rearrangement-custom.src.data_collection.io.save_raw_rollout``.  The
RR repository is not imported at runtime; these definitions are the local
contract used by the adapters, validator, and writer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, TypedDict

import numpy as np


class PickleEnv(str, Enum):
    """Tasks supported by the first pickle recorder implementation."""

    PICK_CUBE = "PickCube-v1"


ACTION_DIM = 8
PART_POSE_DIM = 7
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

ACTION_FIELDS = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_qx",
    "delta_qy",
    "delta_qz",
    "delta_qw",
    "gripper",
)

PICK_CUBE_PART_NAMES = ("cube", "goal_site")

# RR image names are historical.  Stream 1 is the Panda wrist camera and
# stream 2 is the fixed front camera.
CAMERA_TO_IMAGE_KEYS = {
    "hand_camera": ("color_image1", "depth_image1"),
    "base_camera": ("color_image2", "depth_image2"),
}
IMAGE_KEY_TO_CAMERA = {
    image_key: camera_name
    for camera_name, image_keys in CAMERA_TO_IMAGE_KEYS.items()
    for image_key in image_keys
}

ROBOT_STATE_KEYS = (
    "ee_pos",
    "ee_quat",
    "ee_pos_sim",
    "ee_quat_sim",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
    "joint_positions",
    "joint_velocities",
    "joint_torques",
    "gripper_finger_1_pos",
    "gripper_finger_2_pos",
)

OBSERVATION_KEYS = (
    "robot_state",
    "color_image1",
    "color_image2",
    "depth_image1",
    "depth_image2",
    "parts_poses",
    "point_cloud",
    "skill",
    "guidance_point",
    "guidance_point_clean",
    "guidance_pose",
    "guidance_pose_clean",
    "guidance_gripper_width",
    "guidance_point_2d",
    "grasp_annotation_2d",
)

CAMERA_INFO_KEYS = (
    "image_size",
    "intrinsics",
    "camera_to_sim_local",
    "sim_local_to_camera",
)

TRAJECTORY_KEYS = (
    "observations",
    "actions",
    "rewards",
    "camera_info",
    "success",
    "task",
    "action_type",
)


class RobotState(TypedDict):
    """Panda state expressed in the Panda base frame.

    Quaternions use ``xyzw`` ordering.  ``joint_torques`` contains the full
    nine-DoF generalized force vector, while joint positions and velocities
    contain only the seven arm joints.
    """

    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_pos_sim: np.ndarray
    ee_quat_sim: np.ndarray
    ee_pos_vel: np.ndarray
    ee_ori_vel: np.ndarray
    gripper_width: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    gripper_finger_1_pos: float
    gripper_finger_2_pos: float


class CameraInfo(TypedDict):
    """RR front-camera calibration, with ``sim_local`` equal to Panda base."""

    image_size: np.ndarray
    intrinsics: np.ndarray
    camera_to_sim_local: np.ndarray
    sim_local_to_camera: np.ndarray


class GraspAnnotation2D(TypedDict):
    style: str
    center: np.ndarray
    corners: np.ndarray


class PickleObservation(TypedDict):
    robot_state: RobotState
    color_image1: np.ndarray
    color_image2: np.ndarray
    depth_image1: np.ndarray
    depth_image2: np.ndarray
    parts_poses: np.ndarray
    point_cloud: None
    skill: Optional[str]
    guidance_point: Optional[np.ndarray]
    guidance_point_clean: Optional[np.ndarray]
    guidance_pose: Optional[np.ndarray]
    guidance_pose_clean: Optional[np.ndarray]
    guidance_gripper_width: Optional[float]
    guidance_point_2d: dict[str, Optional[np.ndarray]]
    grasp_annotation_2d: dict[str, Optional[GraspAnnotation2D]]


class PickleTrajectory(TypedDict):
    observations: list[PickleObservation]
    actions: list[list[float]]
    rewards: list[float]
    camera_info: dict[str, CameraInfo]
    success: bool
    task: str
    action_type: str


SchemaMapping = dict[str, Any]


__all__ = [
    "ACTION_DIM",
    "ACTION_FIELDS",
    "CAMERA_INFO_KEYS",
    "CAMERA_TO_IMAGE_KEYS",
    "CameraInfo",
    "GraspAnnotation2D",
    "IMAGE_HEIGHT",
    "IMAGE_KEY_TO_CAMERA",
    "IMAGE_WIDTH",
    "OBSERVATION_KEYS",
    "PART_POSE_DIM",
    "PICK_CUBE_PART_NAMES",
    "PickleEnv",
    "PickleObservation",
    "PickleTrajectory",
    "ROBOT_STATE_KEYS",
    "RobotState",
    "TRAJECTORY_KEYS",
]

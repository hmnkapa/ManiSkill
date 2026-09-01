"""RR-compatible online pickle trajectory recording."""

from .camera_contract import rr_aligned_sensor_overrides, rr_front_camera_parameters
from .schema import (
    ACTION_DIM,
    ACTION_FIELDS,
    CAMERA_TO_IMAGE_KEYS,
    PART_POSE_DIM,
    PICK_CUBE_PART_NAMES,
    CameraInfo,
    PickleEnv,
    PickleObservation,
    PickleTrajectory,
    RobotState,
    SOURCE_ENV,
)
from .task_registry import (
    PICKLE_TASK_SPECS,
    PartPoseSource,
    PickleTaskSpec,
    get_pickle_task_spec,
)
from .validator import TrajectoryValidationError, validate_trajectory
from .writer import read_trajectory, write_trajectory

__all__ = [
    "ACTION_DIM",
    "ACTION_FIELDS",
    "CAMERA_TO_IMAGE_KEYS",
    "PART_POSE_DIM",
    "PICKLE_TASK_SPECS",
    "PICK_CUBE_PART_NAMES",
    "CameraInfo",
    "PartPoseSource",
    "PickleEnv",
    "PickleObservation",
    "PickleTaskSpec",
    "PickleTrajectory",
    "RobotState",
    "SOURCE_ENV",
    "TrajectoryValidationError",
    "get_pickle_task_spec",
    "read_trajectory",
    "rr_aligned_sensor_overrides",
    "rr_front_camera_parameters",
    "validate_trajectory",
    "write_trajectory",
]

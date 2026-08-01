"""Strict validation for RR-compatible pickle trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .schema import (
    ACTION_DIM,
    CAMERA_INFO_KEYS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBSERVATION_KEYS,
    PICK_CUBE_PART_NAMES,
    ROBOT_STATE_KEYS,
    TRAJECTORY_KEYS,
    PickleEnv,
)
from .transforms import normalize_quaternion_xyzw


class TrajectoryValidationError(ValueError):
    pass


def _fail(path: str, message: str) -> None:
    raise TrajectoryValidationError(f"{path}: {message}")


def _exact_keys(value: Any, expected: Sequence[str], path: str) -> Mapping:
    if not isinstance(value, Mapping):
        _fail(path, f"expected mapping, got {type(value).__name__}")
    actual = set(value.keys())
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        extra = sorted(actual - expected_set)
        _fail(path, f"key mismatch (missing={missing}, extra={extra})")
    return value


def _array(
    value: Any,
    shape: tuple[int, ...],
    dtype,
    path: str,
    *,
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _fail(path, f"expected numpy.ndarray, got {type(value).__name__}")
    if value.shape != shape:
        _fail(path, f"expected shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(dtype):
        _fail(path, f"expected dtype {np.dtype(dtype)}, got {value.dtype}")
    if finite and not np.isfinite(value).all():
        _fail(path, "contains a non-finite value")
    return value


def _unit_quaternion(value: np.ndarray, path: str, atol: float = 1e-4) -> None:
    norm = np.linalg.norm(value, axis=-1)
    if not np.allclose(norm, 1.0, atol=atol, rtol=0.0):
        _fail(path, f"expected unit quaternion, norm={norm}")
    canonical = normalize_quaternion_xyzw(value)
    if not np.allclose(value, canonical, atol=atol, rtol=0.0):
        _fail(path, "quaternion sign is not canonical")


def _scalar(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        _fail(path, "expected numeric scalar, got bool")
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        _fail(path, "expected one numeric scalar")
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        _fail(path, "expected finite scalar")
    if nonnegative and number < -1e-7:
        _fail(path, f"expected nonnegative scalar, got {number}")
    return number


def _rigid_pose(value: Any, path: str) -> np.ndarray:
    pose = _array(value, (4, 4), np.float32, path)
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5, rtol=0.0):
        _fail(path, "last row is not homogeneous [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=2e-4, rtol=0.0
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4, rtol=0.0):
        _fail(path, "rotation block is not a proper rotation matrix")
    return pose


def _camera_transform(value: Any, path: str) -> np.ndarray:
    """Validate RR's left-handed x-right/y-up/z-forward camera transform."""

    transform = _array(value, (4, 4), np.float32, path)
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5, rtol=0.0):
        _fail(path, "last row is not homogeneous [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=2e-4, rtol=0.0
    ) or not np.isclose(abs(np.linalg.det(rotation)), 1.0, atol=2e-4, rtol=0.0):
        _fail(path, "camera basis is not orthonormal")
    return transform


def _validate_robot_state(robot_state: Any, path: str) -> None:
    state = _exact_keys(robot_state, ROBOT_STATE_KEYS, path)
    ee_pos = _array(state["ee_pos"], (3,), np.float32, f"{path}.ee_pos")
    ee_quat = _array(state["ee_quat"], (4,), np.float32, f"{path}.ee_quat")
    ee_pos_sim = _array(
        state["ee_pos_sim"], (3,), np.float32, f"{path}.ee_pos_sim"
    )
    ee_quat_sim = _array(
        state["ee_quat_sim"], (4,), np.float32, f"{path}.ee_quat_sim"
    )
    _unit_quaternion(ee_quat, f"{path}.ee_quat")
    _unit_quaternion(ee_quat_sim, f"{path}.ee_quat_sim")
    if not np.array_equal(ee_pos, ee_pos_sim):
        _fail(path, "ee_pos_sim must equal base-frame ee_pos")
    if not np.array_equal(ee_quat, ee_quat_sim):
        _fail(path, "ee_quat_sim must equal base-frame ee_quat")
    _array(state["ee_pos_vel"], (3,), np.float32, f"{path}.ee_pos_vel")
    _array(state["ee_ori_vel"], (3,), np.float32, f"{path}.ee_ori_vel")
    _array(
        state["joint_positions"],
        (7,),
        np.float32,
        f"{path}.joint_positions",
    )
    _array(
        state["joint_velocities"],
        (7,),
        np.float32,
        f"{path}.joint_velocities",
    )
    _array(
        state["joint_torques"], (9,), np.float32, f"{path}.joint_torques"
    )
    width = _scalar(state["gripper_width"], f"{path}.gripper_width", nonnegative=True)
    finger1 = _scalar(
        state["gripper_finger_1_pos"], f"{path}.gripper_finger_1_pos"
    )
    finger2 = _scalar(
        state["gripper_finger_2_pos"], f"{path}.gripper_finger_2_pos"
    )
    if not np.isclose(width, finger1 + finger2, atol=1e-6, rtol=0.0):
        _fail(path, "gripper_width must equal the sum of both finger positions")


def _validate_optional_array(
    value: Any, shape: tuple[int, ...], path: str
) -> None:
    if value is not None:
        _array(value, shape, np.float32, path)


def _validate_annotations(observation: Mapping, path: str) -> None:
    skill = observation["skill"]
    if skill is not None and skill not in {"pick", "place", "insert", "screw", "push"}:
        _fail(f"{path}.skill", f"unknown RR skill {skill!r}")
    _validate_optional_array(
        observation["guidance_point"], (3,), f"{path}.guidance_point"
    )
    _validate_optional_array(
        observation["guidance_point_clean"],
        (3,),
        f"{path}.guidance_point_clean",
    )
    for key in ("guidance_pose", "guidance_pose_clean"):
        if observation[key] is not None:
            _rigid_pose(observation[key], f"{path}.{key}")
    if observation["guidance_point"] is None:
        if observation["guidance_point_clean"] is not None:
            _fail(path, "clean guidance point exists without guidance point")
    elif not np.array_equal(
        observation["guidance_point"], observation["guidance_point_clean"]
    ):
        _fail(path, "clean and raw guidance points must match when noise is disabled")
    if observation["guidance_pose"] is None:
        if observation["guidance_pose_clean"] is not None:
            _fail(path, "clean guidance pose exists without guidance pose")
    elif not np.array_equal(
        observation["guidance_pose"], observation["guidance_pose_clean"]
    ):
        _fail(path, "clean and raw guidance poses must match when noise is disabled")
    if observation["guidance_gripper_width"] is not None:
        _scalar(
            observation["guidance_gripper_width"],
            f"{path}.guidance_gripper_width",
            nonnegative=True,
        )

    image_keys = ("color_image1", "color_image2")
    points = _exact_keys(
        observation["guidance_point_2d"], image_keys, f"{path}.guidance_point_2d"
    )
    grasps = _exact_keys(
        observation["grasp_annotation_2d"],
        image_keys,
        f"{path}.grasp_annotation_2d",
    )
    for image_key in image_keys:
        point = points[image_key]
        if point is not None:
            point = _array(
                point,
                (2,),
                np.float32,
                f"{path}.guidance_point_2d.{image_key}",
            )
            if not (
                0 <= point[0] < IMAGE_WIDTH and 0 <= point[1] < IMAGE_HEIGHT
            ):
                _fail(
                    f"{path}.guidance_point_2d.{image_key}",
                    "visible point lies outside the image",
                )
        grasp = grasps[image_key]
        if grasp is None:
            continue
        grasp = _exact_keys(
            grasp,
            ("style", "center", "corners"),
            f"{path}.grasp_annotation_2d.{image_key}",
        )
        if grasp["style"] != "grasp_rect":
            _fail(
                f"{path}.grasp_annotation_2d.{image_key}.style",
                "expected 'grasp_rect'",
            )
        _array(
            grasp["center"],
            (2,),
            np.float32,
            f"{path}.grasp_annotation_2d.{image_key}.center",
        )
        _array(
            grasp["corners"],
            (4, 2),
            np.float32,
            f"{path}.grasp_annotation_2d.{image_key}.corners",
        )


def _validate_observation(observation: Any, index: int) -> None:
    path = f"observations[{index}]"
    observation = _exact_keys(observation, OBSERVATION_KEYS, path)
    _validate_robot_state(observation["robot_state"], f"{path}.robot_state")
    for key in ("color_image1", "color_image2"):
        _array(
            observation[key],
            (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            np.uint8,
            f"{path}.{key}",
            finite=False,
        )
    for key in ("depth_image1", "depth_image2"):
        depth = _array(
            observation[key],
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            np.float32,
            f"{path}.{key}",
        )
        if np.any(depth < 0):
            _fail(f"{path}.{key}", "depth in metres cannot be negative")
    parts = _array(
        observation["parts_poses"],
        (len(PICK_CUBE_PART_NAMES) * 7,),
        np.float32,
        f"{path}.parts_poses",
    ).reshape((-1, 7))
    _unit_quaternion(parts[:, 3:7], f"{path}.parts_poses.quaternions")
    if observation["point_cloud"] is not None:
        _fail(f"{path}.point_cloud", "first schema version requires None")
    _validate_annotations(observation, path)


def _validate_camera_info(camera_info: Any) -> None:
    camera_info = _exact_keys(camera_info, ("front_camera",), "camera_info")
    front = _exact_keys(
        camera_info["front_camera"], CAMERA_INFO_KEYS, "camera_info.front_camera"
    )
    image_size = _array(
        front["image_size"], (2,), np.int32, "camera_info.front_camera.image_size"
    )
    if not np.array_equal(image_size, [IMAGE_WIDTH, IMAGE_HEIGHT]):
        _fail(
            "camera_info.front_camera.image_size",
            f"expected [{IMAGE_WIDTH}, {IMAGE_HEIGHT}], got {image_size.tolist()}",
        )
    intrinsics = _array(
        front["intrinsics"],
        (3, 3),
        np.float32,
        "camera_info.front_camera.intrinsics",
    )
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        _fail("camera_info.front_camera.intrinsics", "focal lengths must be positive")
    camera_to_base = _camera_transform(
        front["camera_to_sim_local"],
        "camera_info.front_camera.camera_to_sim_local",
    )
    base_to_camera = _camera_transform(
        front["sim_local_to_camera"],
        "camera_info.front_camera.sim_local_to_camera",
    )
    if not np.allclose(
        camera_to_base @ base_to_camera,
        np.eye(4),
        atol=3e-4,
        rtol=0.0,
    ) or not np.allclose(
        base_to_camera @ camera_to_base,
        np.eye(4),
        atol=3e-4,
        rtol=0.0,
    ):
        _fail("camera_info.front_camera", "camera transforms are not mutual inverses")


def validate_trajectory(
    trajectory: Any, env: PickleEnv | str = PickleEnv.PICK_CUBE
) -> None:
    """Validate the complete, strict RR raw-rollout contract."""

    try:
        env = PickleEnv(env)
    except ValueError as error:
        raise NotImplementedError(f"Unsupported pickle environment: {env!r}") from error
    if env is not PickleEnv.PICK_CUBE:
        raise NotImplementedError(f"Unsupported pickle environment: {env.value}")

    trajectory = _exact_keys(trajectory, TRAJECTORY_KEYS, "trajectory")
    observations = trajectory["observations"]
    actions = trajectory["actions"]
    rewards = trajectory["rewards"]
    if not isinstance(observations, list):
        _fail("observations", "must be a Python list")
    if not isinstance(actions, list):
        _fail("actions", "must be a Python list")
    if not isinstance(rewards, list):
        _fail("rewards", "must be a Python list")
    if len(actions) == 0:
        _fail("actions", "trajectory must contain at least one transition")
    if len(observations) != len(actions) + 1:
        _fail(
            "observations",
            f"expected T+1={len(actions) + 1}, got {len(observations)}",
        )
    if len(rewards) != len(actions):
        _fail("rewards", f"expected T={len(actions)}, got {len(rewards)}")

    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.shape != (len(actions), ACTION_DIM):
        _fail(
            "actions",
            f"expected shape ({len(actions)}, {ACTION_DIM}), got {action_array.shape}",
        )
    if not np.isfinite(action_array).all():
        _fail("actions", "contains a non-finite value")
    _unit_quaternion(action_array[:, 3:7], "actions[:, 3:7]")
    if np.any(action_array[:, 7] < -1.0) or np.any(action_array[:, 7] > 1.0):
        _fail("actions[:, 7]", "gripper command must be in [-1, 1]")
    reward_array = np.asarray(rewards, dtype=np.float32)
    if reward_array.shape != (len(actions),) or not np.isfinite(reward_array).all():
        _fail("rewards", "must contain T finite scalar values")

    for index, observation in enumerate(observations):
        _validate_observation(observation, index)
    _validate_camera_info(trajectory["camera_info"])
    if type(trajectory["success"]) is not bool:
        _fail("success", "must be a Python bool")
    if trajectory["task"] != env.value:
        _fail("task", f"expected {env.value!r}, got {trajectory['task']!r}")
    if trajectory["action_type"] != "delta":
        _fail("action_type", "expected 'delta'")


__all__ = ["TrajectoryValidationError", "validate_trajectory"]

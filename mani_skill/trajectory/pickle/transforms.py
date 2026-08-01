"""Numpy pose and quaternion transforms used by the RR pickle adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(value: Any, dtype=None) -> np.ndarray:
    """Convert numpy/torch-like values to a detached CPU numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def first_env(value: Any, dtype=np.float32) -> np.ndarray:
    """Return environment zero from a ManiSkill batched value."""

    array = as_numpy(value, dtype=dtype)
    if array.ndim == 0:
        return array
    if array.shape[0] != 1:
        raise ValueError(f"Expected one environment, got shape {array.shape}")
    return array[0]


def _check_quaternion(quaternion: Any) -> np.ndarray:
    quaternion = as_numpy(quaternion, dtype=np.float64)
    if quaternion.shape[-1:] != (4,):
        raise ValueError(
            f"Expected quaternion with final dimension 4, got {quaternion.shape}"
        )
    if not np.isfinite(quaternion).all():
        raise ValueError("Quaternion contains a non-finite value")
    return quaternion


def wxyz_to_xyzw(quaternion: Any) -> np.ndarray:
    quaternion = _check_quaternion(quaternion)
    return np.concatenate([quaternion[..., 1:], quaternion[..., :1]], axis=-1)


def xyzw_to_wxyz(quaternion: Any) -> np.ndarray:
    quaternion = _check_quaternion(quaternion)
    return np.concatenate([quaternion[..., 3:], quaternion[..., :3]], axis=-1)


def normalize_quaternion_xyzw(
    quaternion: Any, *, canonicalize_sign: bool = True
) -> np.ndarray:
    """Normalize an ``xyzw`` quaternion and optionally choose ``w >= 0``."""

    quaternion = _check_quaternion(quaternion)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Cannot normalize a zero quaternion")
    result = quaternion / norm
    if canonicalize_sign:
        # Prefer w > 0.  Exact pi rotations have w == 0, so use x, y, then z
        # as deterministic tie breakers instead of leaving q and -q distinct.
        priority = result[..., [3, 0, 1, 2]]
        significant = np.abs(priority) > 1e-12
        first_index = np.argmax(significant, axis=-1)
        first_value = np.take_along_axis(
            priority, first_index[..., None], axis=-1
        )
        result = np.where(first_value < 0.0, -result, result)
    return result.astype(np.float32)


def quaternion_conjugate_xyzw(quaternion: Any) -> np.ndarray:
    quaternion = normalize_quaternion_xyzw(
        quaternion, canonicalize_sign=False
    ).astype(np.float64)
    result = quaternion.copy()
    result[..., :3] *= -1.0
    return result.astype(np.float32)


def quaternion_inverse_xyzw(quaternion: Any) -> np.ndarray:
    return quaternion_conjugate_xyzw(quaternion)


def quaternion_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    """Hamilton product ``left ⊗ right`` for ``xyzw`` quaternions."""

    left = normalize_quaternion_xyzw(left, canonicalize_sign=False).astype(np.float64)
    right = normalize_quaternion_xyzw(right, canonicalize_sign=False).astype(
        np.float64
    )
    left, right = np.broadcast_arrays(left, right)
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    result = np.stack(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        axis=-1,
    )
    return normalize_quaternion_xyzw(result)


def relative_quaternion_xyzw(current: Any, target: Any) -> np.ndarray:
    """Return RR's right/body delta ``inverse(current) ⊗ target``."""

    return quaternion_multiply_xyzw(quaternion_inverse_xyzw(current), target)


def apply_delta_quaternion_xyzw(current: Any, delta: Any) -> np.ndarray:
    """Reconstruct an RR target as ``current ⊗ delta``."""

    return quaternion_multiply_xyzw(current, delta)


def quaternion_xyzw_to_matrix(quaternion: Any) -> np.ndarray:
    quaternion = normalize_quaternion_xyzw(
        quaternion, canonicalize_sign=False
    ).astype(np.float64)
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        axis=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))
    return matrix.astype(np.float32)


def matrix_to_quaternion_xyzw(matrix: Any) -> np.ndarray:
    """Convert one or more rotation matrices to canonical ``xyzw`` quaternions."""

    matrix = as_numpy(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) rotation matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix contains a non-finite value")

    # Branch-free eigen decomposition is stable at rotations near pi.  The
    # dominant eigenvector of K is the quaternion in xyzw ordering.
    flat = matrix.reshape((-1, 3, 3))
    quaternions = []
    for rotation in flat:
        k = np.array(
            [
                [
                    rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                    rotation[0, 1] + rotation[1, 0],
                    rotation[0, 2] + rotation[2, 0],
                    rotation[2, 1] - rotation[1, 2],
                ],
                [
                    rotation[0, 1] + rotation[1, 0],
                    rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
                    rotation[1, 2] + rotation[2, 1],
                    rotation[0, 2] - rotation[2, 0],
                ],
                [
                    rotation[0, 2] + rotation[2, 0],
                    rotation[1, 2] + rotation[2, 1],
                    rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
                    rotation[1, 0] - rotation[0, 1],
                ],
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                    rotation.trace(),
                ],
            ],
            dtype=np.float64,
        ) / 3.0
        _, eigenvectors = np.linalg.eigh(k)
        quaternions.append(eigenvectors[:, -1])
    result = np.stack(quaternions).reshape(matrix.shape[:-2] + (4,))
    return normalize_quaternion_xyzw(result)


def pose_to_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    position = as_numpy(position, dtype=np.float32)
    quaternion_xyzw = normalize_quaternion_xyzw(quaternion_xyzw)
    if position.shape[-1:] != (3,):
        raise ValueError(f"Expected position with final dimension 3, got {position.shape}")
    batch_shape = np.broadcast_shapes(position.shape[:-1], quaternion_xyzw.shape[:-1])
    position = np.broadcast_to(position, batch_shape + (3,))
    quaternion_xyzw = np.broadcast_to(quaternion_xyzw, batch_shape + (4,))
    matrix = np.broadcast_to(np.eye(4, dtype=np.float32), batch_shape + (4, 4)).copy()
    matrix[..., :3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    matrix[..., :3, 3] = position
    return matrix


def matrix_to_pose(matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = as_numpy(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (..., 4, 4) pose matrix, got {matrix.shape}")
    return matrix[..., :3, 3].copy(), matrix_to_quaternion_xyzw(matrix[..., :3, :3])


def world_pose_to_base(world_pose: Any, world_base_pose: Any) -> np.ndarray:
    """Transform a homogeneous world pose into the Panda base frame."""

    world_pose = as_numpy(world_pose, dtype=np.float32)
    world_base_pose = as_numpy(world_base_pose, dtype=np.float32)
    if world_pose.shape[-2:] != (4, 4) or world_base_pose.shape[-2:] != (4, 4):
        raise ValueError("world_pose and world_base_pose must end in shape (4, 4)")
    return (np.linalg.inv(world_base_pose) @ world_pose).astype(np.float32)


def world_point_to_base(point_world: Any, world_base_pose: Any) -> np.ndarray:
    point_world = as_numpy(point_world, dtype=np.float32)
    world_base_pose = as_numpy(world_base_pose, dtype=np.float32)
    if point_world.shape[-1:] != (3,):
        raise ValueError(f"Expected point with final dimension 3, got {point_world.shape}")
    rotation = world_base_pose[..., :3, :3]
    translation = world_base_pose[..., :3, 3]
    return (
        np.swapaxes(rotation, -1, -2) @ (point_world - translation)[..., None]
    )[..., 0].astype(np.float32)


def world_vector_to_base(vector_world: Any, world_base_pose: Any) -> np.ndarray:
    vector_world = as_numpy(vector_world, dtype=np.float32)
    world_base_pose = as_numpy(world_base_pose, dtype=np.float32)
    if vector_world.shape[-1:] != (3,):
        raise ValueError(
            f"Expected vector with final dimension 3, got {vector_world.shape}"
        )
    rotation = world_base_pose[..., :3, :3]
    return (np.swapaxes(rotation, -1, -2) @ vector_world[..., None])[
        ..., 0
    ].astype(np.float32)


__all__ = [
    "apply_delta_quaternion_xyzw",
    "as_numpy",
    "first_env",
    "matrix_to_pose",
    "matrix_to_quaternion_xyzw",
    "normalize_quaternion_xyzw",
    "pose_to_matrix",
    "quaternion_conjugate_xyzw",
    "quaternion_inverse_xyzw",
    "quaternion_multiply_xyzw",
    "quaternion_xyzw_to_matrix",
    "relative_quaternion_xyzw",
    "world_point_to_base",
    "world_pose_to_base",
    "world_vector_to_base",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
]

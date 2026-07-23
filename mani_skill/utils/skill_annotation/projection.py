from __future__ import annotations

from typing import Any

import torch

from mani_skill.utils.structs.pose import Pose


DEFAULT_GRASP_WIDTH_M = 0.05
DEFAULT_GRASP_HEIGHT_M = 0.02


def project_3d_to_2d(
    points_world: torch.Tensor,
    intrinsic_cv: torch.Tensor,
    extrinsic_cv: torch.Tensor,
    image_size: tuple[int, int] | list[int] | torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    points = _as_points(points_world)
    intrinsic = _as_intrinsic(intrinsic_cv, points.device)
    extrinsic = _as_extrinsic(extrinsic_cv, points.device)
    batch_size = max(points.shape[0], intrinsic.shape[0], extrinsic.shape[0])
    points = _broadcast_first_dim(points, batch_size, "points_world")
    intrinsic = _broadcast_first_dim(intrinsic, batch_size, "intrinsic_cv")
    extrinsic = _broadcast_first_dim(extrinsic, batch_size, "extrinsic_cv")

    point_uv, point_visible = _project_points(points, intrinsic, extrinsic, image_size, eps)
    return {"point_uv": point_uv, "point_visible": point_visible}


def build_grasp_rect_corners_3d(
    target_pose_world: Pose | torch.Tensor,
    gripper_width: torch.Tensor | float | None = None,
    grasp_height: float = DEFAULT_GRASP_HEIGHT_M,
) -> torch.Tensor:
    """Build grasp-rectangle corners using the ManiSkill TCP axis convention.

    A gripper TCP pose uses local X along the fingers, local Y along the
    gripper closing direction, and local Z along the approach direction.
    Therefore ``gripper_width`` controls the corner separation along local Y,
    while ``grasp_height`` controls the fixed finger extent along local X.
    """
    pose_mat = _as_pose_matrix(target_pose_world)
    device = pose_mat.device
    dtype = pose_mat.dtype
    batch_size = pose_mat.shape[0]

    if gripper_width is None:
        width = torch.full((batch_size,), DEFAULT_GRASP_WIDTH_M, device=device, dtype=dtype)
    else:
        width = torch.as_tensor(gripper_width, device=device, dtype=dtype).reshape(-1)
        width = _broadcast_first_dim(width, batch_size, "gripper_width")
    width = torch.clamp(width, min=1e-4)
    height = torch.full((batch_size,), max(float(grasp_height), 1e-4), device=device, dtype=dtype)

    center = pose_mat[:, :3, 3]
    rot = pose_mat[:, :3, :3]
    finger_axis = rot[:, :, 0]
    closing_axis = rot[:, :, 1]
    half_finger_extent = 0.5 * height[:, None]
    half_opening = 0.5 * width[:, None]

    return torch.stack(
        [
            center - half_finger_extent * finger_axis - half_opening * closing_axis,
            center + half_finger_extent * finger_axis - half_opening * closing_axis,
            center + half_finger_extent * finger_axis + half_opening * closing_axis,
            center - half_finger_extent * finger_axis + half_opening * closing_axis,
        ],
        dim=1,
    )


def project_pose_to_grasp_annotation_2d(
    target_pose_world: Pose | torch.Tensor,
    intrinsic_cv: torch.Tensor,
    extrinsic_cv: torch.Tensor,
    image_size: tuple[int, int] | list[int] | torch.Tensor,
    gripper_width: torch.Tensor | float | None = None,
    grasp_height: float = DEFAULT_GRASP_HEIGHT_M,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    corners_world = build_grasp_rect_corners_3d(
        target_pose_world,
        gripper_width=gripper_width,
        grasp_height=grasp_height,
    )
    intrinsic = _as_intrinsic(intrinsic_cv, corners_world.device)
    extrinsic = _as_extrinsic(extrinsic_cv, corners_world.device)
    batch_size = max(corners_world.shape[0], intrinsic.shape[0], extrinsic.shape[0])
    corners_world = _broadcast_first_dim(corners_world, batch_size, "corners_world")
    intrinsic = _broadcast_first_dim(intrinsic, batch_size, "intrinsic_cv")
    extrinsic = _broadcast_first_dim(extrinsic, batch_size, "extrinsic_cv")

    width, height = _image_width_height(image_size)
    ones = torch.ones(
        (*corners_world.shape[:-1], 1),
        device=corners_world.device,
        dtype=corners_world.dtype,
    )
    corners_h = torch.cat([corners_world, ones], dim=-1)
    camera_points = torch.matmul(extrinsic[:, None, :, :], corners_h[..., None]).squeeze(-1)
    z = camera_points[..., 2]
    image_points = torch.matmul(intrinsic[:, None, :, :], camera_points[..., None]).squeeze(-1)
    uv = image_points[..., :2] / torch.clamp(image_points[..., 2:3], min=eps)

    corner_visible = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(camera_points).all(dim=-1)
        & (z > eps)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < height)
    )
    grasp_visible = corner_visible.all(dim=-1)
    grasp_rect_uv = torch.where(
        grasp_visible[:, None, None],
        uv,
        torch.full_like(uv, -1.0),
    )
    return {"grasp_rect_uv": grasp_rect_uv, "grasp_visible": grasp_visible}


def _project_points(
    points_world: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    image_size: tuple[int, int] | list[int] | torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = _image_width_height(image_size)
    ones = torch.ones(
        (points_world.shape[0], 1),
        device=points_world.device,
        dtype=points_world.dtype,
    )
    points_h = torch.cat([points_world, ones], dim=-1)
    camera_points = torch.matmul(extrinsic, points_h[..., None]).squeeze(-1)
    z = camera_points[:, 2]
    image_points = torch.matmul(intrinsic, camera_points[..., None]).squeeze(-1)
    uv = image_points[:, :2] / torch.clamp(image_points[:, 2:3], min=eps)
    visible = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(camera_points).all(dim=-1)
        & (z > eps)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    uv = torch.where(visible[:, None], uv, torch.full_like(uv, -1.0))
    return uv, visible


def _as_points(points_world: Any) -> torch.Tensor:
    points = torch.as_tensor(points_world, dtype=torch.float32)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points_world must have shape [N, 3], got {tuple(points.shape)}")
    return points


def _as_intrinsic(intrinsic_cv: Any, device: torch.device) -> torch.Tensor:
    intrinsic = torch.as_tensor(intrinsic_cv, device=device, dtype=torch.float32)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, ...]
    if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsic_cv must have shape [N, 3, 3], got {tuple(intrinsic.shape)}")
    return intrinsic


def _as_extrinsic(extrinsic_cv: Any, device: torch.device) -> torch.Tensor:
    extrinsic = torch.as_tensor(extrinsic_cv, device=device, dtype=torch.float32)
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...]
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"extrinsic_cv must have shape [N, 3, 4] or [N, 4, 4], got {tuple(extrinsic.shape)}"
        )
    if extrinsic.shape[-2:] == (4, 4):
        extrinsic = extrinsic[:, :3, :4]
    return extrinsic


def _as_pose_matrix(target_pose_world: Pose | torch.Tensor) -> torch.Tensor:
    if isinstance(target_pose_world, Pose):
        return target_pose_world.to_transformation_matrix().to(dtype=torch.float32)
    pose = torch.as_tensor(target_pose_world, dtype=torch.float32)
    if pose.ndim == 2 and pose.shape == (4, 4):
        return pose[None, ...]
    if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
        return pose
    if pose.ndim in (1, 2) and pose.shape[-1] == 7:
        return Pose.create(pose).to_transformation_matrix().to(dtype=torch.float32)
    raise ValueError("target_pose_world must be a Pose, [7], [N, 7], [4, 4], or [N, 4, 4]")


def _image_width_height(
    image_size: tuple[int, int] | list[int] | torch.Tensor,
) -> tuple[float, float]:
    if torch.is_tensor(image_size):
        size = image_size.detach().cpu().flatten().tolist()
    else:
        size = list(image_size)
    if len(size) != 2:
        raise ValueError(f"image_size must be (width, height), got {image_size}")
    return float(size[0]), float(size[1])


def _broadcast_first_dim(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((batch_size,) + (1,) * (tensor.ndim - 1))
    raise ValueError(f"{name} has batch size {tensor.shape[0]}, expected {batch_size}")

import torch

from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.pose import Pose


def build_panda_topdown_grasp_pose(
    center: torch.Tensor,
    tcp_pose: Pose,
    object_pose: Pose | None = None,
) -> Pose:
    """Build a batched Panda TCP grasp pose with a world-frame downward approach."""
    tcp_closing = tcp_pose.to_transformation_matrix()[..., :3, 1]
    default_closing = torch.zeros_like(tcp_closing)
    default_closing[..., 1] = 1
    tcp_closing = _normalize_horizontal_axis(tcp_closing, default_closing)

    if object_pose is None:
        closing = tcp_closing
    else:
        object_closing = object_pose.to_transformation_matrix()[..., :3, 1]
        closing = _normalize_horizontal_axis(object_closing, tcp_closing)
        direction = torch.where(
            torch.sum(closing * tcp_closing, dim=-1, keepdim=True) < 0,
            -torch.ones_like(closing[..., :1]),
            torch.ones_like(closing[..., :1]),
        )
        closing = closing * direction

    approaching = torch.zeros_like(closing)
    approaching[..., 2] = -1
    orthogonal = torch.cross(closing, approaching, dim=-1)
    rotation = torch.stack([orthogonal, closing, approaching], dim=-1)
    quaternion = rotation_conversions.matrix_to_quaternion(rotation)
    return Pose.create_from_pq(p=center, q=quaternion)


def target_tcp_pose_from_object_goal(
    current_tcp_pose: Pose,
    current_object_pose: Pose,
    desired_object_pose: Pose,
) -> Pose:
    return desired_object_pose * current_object_pose.inv() * current_tcp_pose


def _normalize_horizontal_axis(
    axis: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    axis = axis.clone()
    axis[..., 2] = 0
    norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
    normalized = axis / norm.clamp_min(1e-6)
    return torch.where(norm > 1e-6, normalized, fallback)

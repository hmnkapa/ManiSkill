from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TypedDict

import torch

from mani_skill.utils.structs.pose import Pose


SKILL_IDS = {
    "none": 0,
    "pick": 1,
    "place": 2,
    "insert": 3,
    "screw": 4,
    "push": 5,
}
SKILL_NAMES = {skill_id: skill for skill, skill_id in SKILL_IDS.items()}
SKILL_VOCAB = tuple(SKILL_IDS.keys())
SKILL_NAME_TO_ID = SKILL_IDS
SKILL_ID_TO_NAME = SKILL_NAMES


class TargetBundle(TypedDict):
    point_world: torch.Tensor
    point_valid: torch.Tensor
    pose_world: torch.Tensor
    pose_valid: torch.Tensor
    gripper_width: torch.Tensor
    gripper_width_valid: torch.Tensor


class CameraProjectionBundle(TypedDict):
    point_uv: torch.Tensor
    point_visible: torch.Tensor
    grasp_rect_uv: torch.Tensor
    grasp_visible: torch.Tensor


class SkillAnnotationBundle(TypedDict, total=False):
    skill_id: torch.Tensor
    skill: list[str]
    skill_state: list[Optional[str]]
    phase_id: torch.Tensor
    phase: list[Optional[str]]
    active_object: list[Optional[str]]
    target_object: list[Optional[str]]
    target: TargetBundle
    projection: dict[str, CameraProjectionBundle]
    task_meta: dict[str, Any]
    debug: dict[str, Any]


@dataclass
class SkillAnnotationContext:
    skill: str | list[str] | None = None
    skill_id: torch.Tensor | int | list[int] | None = None
    skill_state: str | list[str] | None = None
    phase_id: torch.Tensor | int | list[int] | None = None
    phase: str | list[str] | None = None
    target_point_world: torch.Tensor | None = None
    target_pose_world: Pose | torch.Tensor | None = None
    target_gripper_width: torch.Tensor | None = None
    active_object: str | list[str] | None = None
    target_object: str | list[str] | None = None
    allow_no_target: torch.Tensor | bool | list[bool] = False
    task_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedSkillAnnotationContext:
    skill: list[str]
    skill_id: torch.Tensor
    skill_state: list[Optional[str]]
    phase_id: torch.Tensor
    phase: list[Optional[str]]
    target_point_world: torch.Tensor
    target_point_valid: torch.Tensor
    target_pose_world: torch.Tensor
    target_pose_valid: torch.Tensor
    target_gripper_width: torch.Tensor
    target_gripper_width_valid: torch.Tensor
    active_object: list[Optional[str]]
    target_object: list[Optional[str]]
    allow_no_target: torch.Tensor
    task_meta: dict[str, Any]

    @property
    def num_envs(self) -> int:
        return int(self.skill_id.shape[0])

    @property
    def has_target(self) -> torch.Tensor:
        return self.target_point_valid | self.target_pose_valid


def skill_to_id(skill: str | None) -> int:
    if skill is None:
        skill = "none"
    if skill not in SKILL_NAME_TO_ID:
        raise ValueError(f"Unknown skill '{skill}'. Valid skills: {SKILL_VOCAB}")
    return SKILL_NAME_TO_ID[skill]


def id_to_skill(skill_id: int | torch.Tensor) -> str:
    if torch.is_tensor(skill_id):
        if skill_id.numel() != 1:
            raise ValueError("id_to_skill expects a scalar skill id")
        skill_id = int(skill_id.item())
    skill_id = int(skill_id)
    if skill_id not in SKILL_ID_TO_NAME:
        raise ValueError(f"Unknown skill id {skill_id}. Valid ids: {tuple(SKILL_ID_TO_NAME)}")
    return SKILL_ID_TO_NAME[skill_id]


def normalize_skill_context(
    context: SkillAnnotationContext | Mapping[str, Any] | None,
    num_envs: int | None = None,
    device: torch.device | str | None = None,
) -> NormalizedSkillAnnotationContext:
    if context is None:
        context = SkillAnnotationContext()
    elif isinstance(context, Mapping):
        context = SkillAnnotationContext(**context)

    inferred_num_envs = _infer_num_envs(context)
    n = int(num_envs if num_envs is not None else inferred_num_envs)
    if n <= 0:
        raise ValueError(f"num_envs must be positive, got {n}")

    device = torch.device(device) if device is not None else _infer_device(context)
    skill, skill_id = _normalize_skill(context.skill, context.skill_id, n, device)
    skill_state = _normalize_optional_string_list(context.skill_state, n, "skill_state")
    phase_id = _normalize_phase_id(context.phase_id, n, device)
    phase = _normalize_optional_string_list(context.phase, n, "phase")
    active_object = _normalize_optional_string_list(context.active_object, n, "active_object")
    target_object = _normalize_optional_string_list(context.target_object, n, "target_object")
    allow_no_target = _normalize_bool_mask(
        context.allow_no_target, n, device, "allow_no_target"
    )
    point_world, point_valid = _normalize_point(context.target_point_world, n, device)
    pose_world, pose_valid = _normalize_pose(context.target_pose_world, n, device)
    gripper_width, gripper_width_valid = _normalize_gripper_width(
        context.target_gripper_width, n, device
    )

    return NormalizedSkillAnnotationContext(
        skill=skill,
        skill_id=skill_id,
        skill_state=skill_state,
        phase_id=phase_id,
        phase=phase,
        target_point_world=point_world,
        target_point_valid=point_valid,
        target_pose_world=pose_world,
        target_pose_valid=pose_valid,
        target_gripper_width=gripper_width,
        target_gripper_width_valid=gripper_width_valid,
        active_object=active_object,
        target_object=target_object,
        allow_no_target=allow_no_target,
        task_meta=dict(context.task_meta or {}),
    )


def make_empty_normalized_context(
    num_envs: int,
    device: torch.device | str | None = None,
) -> NormalizedSkillAnnotationContext:
    return normalize_skill_context(SkillAnnotationContext(), num_envs=num_envs, device=device)


def select_normalized_context(
    context: NormalizedSkillAnnotationContext, indices: Sequence[int] | torch.Tensor
) -> NormalizedSkillAnnotationContext:
    if torch.is_tensor(indices):
        index_tensor = indices.to(device=context.skill_id.device, dtype=torch.long).flatten()
        index_list = [int(i) for i in index_tensor.detach().cpu().tolist()]
    else:
        index_list = [int(i) for i in indices]
        index_tensor = torch.as_tensor(index_list, device=context.skill_id.device, dtype=torch.long)

    return NormalizedSkillAnnotationContext(
        skill=[context.skill[i] for i in index_list],
        skill_id=context.skill_id[index_tensor].clone(),
        skill_state=[context.skill_state[i] for i in index_list],
        phase_id=context.phase_id[index_tensor].clone(),
        phase=[context.phase[i] for i in index_list],
        target_point_world=context.target_point_world[index_tensor].clone(),
        target_point_valid=context.target_point_valid[index_tensor].clone(),
        target_pose_world=context.target_pose_world[index_tensor].clone(),
        target_pose_valid=context.target_pose_valid[index_tensor].clone(),
        target_gripper_width=context.target_gripper_width[index_tensor].clone(),
        target_gripper_width_valid=context.target_gripper_width_valid[index_tensor].clone(),
        active_object=[context.active_object[i] for i in index_list],
        target_object=[context.target_object[i] for i in index_list],
        allow_no_target=context.allow_no_target[index_tensor].clone(),
        task_meta=dict(context.task_meta),
    )


def make_target_bundle(context: NormalizedSkillAnnotationContext) -> TargetBundle:
    return {
        "point_world": context.target_point_world,
        "point_valid": context.target_point_valid,
        "pose_world": context.target_pose_world,
        "pose_valid": context.target_pose_valid,
        "gripper_width": context.target_gripper_width,
        "gripper_width_valid": context.target_gripper_width_valid,
    }


def _infer_device(context: SkillAnnotationContext) -> torch.device:
    for value in (
        context.skill_id,
        context.phase_id,
        context.target_point_world,
        context.target_pose_world,
        context.target_gripper_width,
        context.allow_no_target,
    ):
        if isinstance(value, Pose):
            return value.device
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _infer_num_envs(context: SkillAnnotationContext) -> int:
    candidates: list[int] = []
    for value in (
        context.skill,
        context.skill_state,
        context.phase,
        context.active_object,
        context.target_object,
    ):
        if isinstance(value, (list, tuple)):
            candidates.append(len(value))
    if context.skill_id is not None:
        ids = torch.as_tensor(context.skill_id)
        candidates.append(1 if ids.ndim == 0 else int(ids.reshape(-1).shape[0]))
    if context.phase_id is not None:
        phase_ids = torch.as_tensor(context.phase_id)
        candidates.append(
            1 if phase_ids.ndim == 0 else int(phase_ids.reshape(-1).shape[0])
        )
    if context.target_point_world is not None:
        point = torch.as_tensor(context.target_point_world)
        candidates.append(1 if point.ndim == 1 else int(point.shape[0]))
    if context.target_pose_world is not None:
        if isinstance(context.target_pose_world, Pose):
            candidates.append(len(context.target_pose_world))
        else:
            pose = torch.as_tensor(context.target_pose_world)
            if pose.ndim == 1:
                candidates.append(1)
            elif pose.ndim == 2 and pose.shape == (4, 4):
                candidates.append(1)
            else:
                candidates.append(int(pose.shape[0]))
    if context.target_gripper_width is not None:
        width = torch.as_tensor(context.target_gripper_width)
        candidates.append(1 if width.ndim == 0 else int(width.reshape(-1).shape[0]))
    allow_no_target = torch.as_tensor(context.allow_no_target)
    candidates.append(
        1
        if allow_no_target.ndim == 0
        else int(allow_no_target.reshape(-1).shape[0])
    )
    return max(candidates) if candidates else 1


def _normalize_skill(
    skill: str | Sequence[str | None] | None,
    skill_id: torch.Tensor | int | Sequence[int] | None,
    num_envs: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    id_from_input = None
    if skill_id is not None:
        id_from_input = torch.as_tensor(skill_id, device=device, dtype=torch.long).flatten()
        if id_from_input.numel() == 0:
            raise ValueError("skill_id cannot be empty")
        id_from_input = _broadcast_first_dim(id_from_input, num_envs, "skill_id")
        for item in id_from_input.detach().cpu().tolist():
            id_to_skill(int(item))
        return [id_to_skill(i) for i in id_from_input.detach().cpu().tolist()], id_from_input

    if skill is None:
        skill_names = ["none"] * num_envs
    elif isinstance(skill, str):
        skill_names = [skill] * num_envs
    else:
        skill_names = ["none" if item is None else str(item) for item in skill]
        skill_names = _broadcast_list(skill_names, num_envs, "skill")

    ids = torch.as_tensor(
        [skill_to_id(item) for item in skill_names],
        device=device,
        dtype=torch.long,
    )
    if id_from_input is not None and not torch.equal(ids, id_from_input):
        raise ValueError("skill and skill_id disagree")
    return skill_names, ids


def _normalize_phase_id(
    phase_id: torch.Tensor | int | Sequence[int] | None,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    if phase_id is None:
        return torch.full((num_envs,), -1, device=device, dtype=torch.long)

    phase_id_tensor = torch.as_tensor(
        phase_id, device=device, dtype=torch.long
    ).flatten()
    if phase_id_tensor.numel() == 0:
        raise ValueError("phase_id cannot be empty")
    return _broadcast_first_dim(phase_id_tensor, num_envs, "phase_id")


def _normalize_optional_string_list(
    value: str | Sequence[str | None] | None, num_envs: int, name: str
) -> list[Optional[str]]:
    if value is None:
        return [None] * num_envs
    if isinstance(value, str):
        return [value] * num_envs
    return _broadcast_list(
        [None if item is None else str(item) for item in value], num_envs, name
    )


def _normalize_bool_mask(
    value: torch.Tensor | bool | Sequence[bool],
    num_envs: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    mask = torch.as_tensor(value, device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    return _broadcast_first_dim(mask, num_envs, name)


def _broadcast_list(values: list[Any], num_envs: int, name: str) -> list[Any]:
    if len(values) == num_envs:
        return values
    if len(values) == 1:
        return values * num_envs
    raise ValueError(f"{name} has length {len(values)}, expected {num_envs}")


def _to_float_tensor(
    value: Any, device: torch.device, name: str, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    try:
        return torch.as_tensor(value, device=device, dtype=dtype)
    except Exception as exc:
        raise ValueError(f"Could not convert {name} to tensor") from exc


def _normalize_point(
    value: torch.Tensor | None, num_envs: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if value is None:
        return (
            torch.zeros((num_envs, 3), device=device, dtype=torch.float32),
            torch.zeros((num_envs,), device=device, dtype=torch.bool),
        )
    point = _to_float_tensor(value, device, "target_point_world")
    if point.ndim == 1:
        point = point[None, :]
    if point.ndim != 2 or point.shape[-1] != 3:
        raise ValueError(
            f"target_point_world must have shape [N, 3], got {tuple(point.shape)}"
        )
    point = _broadcast_first_dim(point, num_envs, "target_point_world")
    valid = torch.isfinite(point).all(dim=-1)
    point = torch.where(valid[:, None], point, torch.zeros_like(point))
    return point, valid


def _normalize_pose(
    value: Pose | torch.Tensor | None, num_envs: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(4, device=device, dtype=torch.float32)[None, ...].repeat(
        num_envs, 1, 1
    )
    if value is None:
        return identity, torch.zeros((num_envs,), device=device, dtype=torch.bool)

    if isinstance(value, Pose):
        pose = value.to(device)
        pose_mat = pose.to_transformation_matrix().to(dtype=torch.float32)
    else:
        pose_tensor = _to_float_tensor(value, device, "target_pose_world")
        if pose_tensor.ndim == 2 and pose_tensor.shape == (4, 4):
            pose_mat = pose_tensor[None, ...]
        elif pose_tensor.ndim == 3 and pose_tensor.shape[-2:] == (4, 4):
            pose_mat = pose_tensor
        elif pose_tensor.ndim in (1, 2) and pose_tensor.shape[-1] == 7:
            pose_mat = (
                Pose.create(pose_tensor, device=device)
                .to_transformation_matrix()
                .to(dtype=torch.float32)
            )
        else:
            raise ValueError(
                "target_pose_world must be a Pose, [N, 7], [7], [N, 4, 4], or [4, 4]"
            )

    pose_mat = _broadcast_first_dim(pose_mat, num_envs, "target_pose_world")
    valid = torch.isfinite(pose_mat).flatten(start_dim=1).all(dim=-1)
    pose_mat = torch.where(valid[:, None, None], pose_mat, identity)
    return pose_mat, valid


def _normalize_gripper_width(
    value: torch.Tensor | None, num_envs: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if value is None:
        return (
            torch.zeros((num_envs,), device=device, dtype=torch.float32),
            torch.zeros((num_envs,), device=device, dtype=torch.bool),
        )
    width = _to_float_tensor(value, device, "target_gripper_width").reshape(-1)
    width = _broadcast_first_dim(width, num_envs, "target_gripper_width")
    valid = torch.isfinite(width)
    width = torch.where(valid, width, torch.zeros_like(width))
    return width, valid


def _broadcast_first_dim(tensor: torch.Tensor, num_envs: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == num_envs:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.repeat((num_envs,) + (1,) * (tensor.ndim - 1))
    raise ValueError(f"{name} has batch size {tensor.shape[0]}, expected {num_envs}")

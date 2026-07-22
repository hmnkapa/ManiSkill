from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch

from mani_skill.utils.skill_annotation.projection import (
    DEFAULT_GRASP_WIDTH_M,
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
)
from mani_skill.utils.skill_annotation.schema import (
    NormalizedSkillAnnotationContext,
    SkillAnnotationBundle,
    SkillAnnotationContext,
    make_empty_normalized_context,
    make_target_bundle,
    normalize_skill_context,
    select_normalized_context,
)

EnvIndex: TypeAlias = int | list[int] | tuple[int, ...] | np.ndarray | torch.Tensor


@dataclass
class _PreviousAnnotation:
    skill: str
    skill_id: int
    skill_state: str | None
    phase_id: int
    phase: str | None
    target_point_world: torch.Tensor
    target_point_valid: bool
    target_pose_world: torch.Tensor
    target_pose_valid: bool
    target_gripper_width: torch.Tensor
    target_gripper_width_valid: bool
    active_object: str | None
    target_object: str | None
    task_meta: dict[str, Any]

    @classmethod
    def from_context(
        cls, context: NormalizedSkillAnnotationContext, row: int
    ) -> "_PreviousAnnotation":
        return cls(
            skill=context.skill[row],
            skill_id=int(context.skill_id[row].item()),
            skill_state=context.skill_state[row],
            phase_id=int(context.phase_id[row].item()),
            phase=context.phase[row],
            target_point_world=context.target_point_world[row].detach().clone(),
            target_point_valid=bool(context.target_point_valid[row].item()),
            target_pose_world=context.target_pose_world[row].detach().clone(),
            target_pose_valid=bool(context.target_pose_valid[row].item()),
            target_gripper_width=context.target_gripper_width[row].detach().clone(),
            target_gripper_width_valid=bool(context.target_gripper_width_valid[row].item()),
            active_object=context.active_object[row],
            target_object=context.target_object[row],
            task_meta=dict(context.task_meta),
        )

    def write_to(self, context: NormalizedSkillAnnotationContext, row: int) -> None:
        device = context.skill_id.device
        context.skill[row] = self.skill
        context.skill_id[row] = self.skill_id
        context.skill_state[row] = self.skill_state
        context.phase_id[row] = self.phase_id
        context.phase[row] = self.phase
        context.target_point_world[row] = self.target_point_world.to(device=device)
        context.target_point_valid[row] = self.target_point_valid
        context.target_pose_world[row] = self.target_pose_world.to(device=device)
        context.target_pose_valid[row] = self.target_pose_valid
        context.target_gripper_width[row] = self.target_gripper_width.to(device=device)
        context.target_gripper_width_valid[row] = self.target_gripper_width_valid
        context.active_object[row] = self.active_object
        context.target_object[row] = self.target_object
        context.task_meta.update(self.task_meta)


class SkillAnnotationManager:
    def __init__(self, num_envs: int = 1):
        self._previous: list[_PreviousAnnotation | None] = [None] * max(int(num_envs), 1)

    @property
    def num_envs(self) -> int:
        return len(self._previous)

    def ensure_num_envs(self, num_envs: int) -> None:
        num_envs = max(int(num_envs), 1)
        if num_envs == self.num_envs:
            return
        if num_envs > self.num_envs:
            self._previous.extend([None] * (num_envs - self.num_envs))
        else:
            self._previous = self._previous[:num_envs]

    def reset(self, env_idx: EnvIndex | None = None) -> None:
        if env_idx is None:
            self._previous = [None] * self.num_envs
            return
        for idx in _normalize_env_indices(env_idx, self.num_envs):
            self._previous[idx] = None

    def get_annotation_bundle(
        self,
        env,
        cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None = None,
        use_previous: bool = True,
        env_idx: EnvIndex | None = None,
    ) -> SkillAnnotationBundle:
        total_num_envs = _get_env_num_envs(env)
        device = _get_env_device(env)
        self.ensure_num_envs(total_num_envs)
        indices = _normalize_env_indices(env_idx, total_num_envs)
        raw_context = _get_raw_context(
            env, _normalize_env_idx_for_context(env_idx, device)
        )
        normalized = _normalize_context_for_indices(
            raw_context,
            indices,
            env_idx,
            total_num_envs,
            device,
        )
        effective, used_previous, current_valid = self._apply_previous_cache(
            normalized, indices, use_previous=use_previous
        )
        projection = _build_projection_bundle(
            env,
            effective,
            cameras,
            indices,
            total_num_envs,
        )
        return {
            "skill_id": effective.skill_id,
            "skill": effective.skill,
            "skill_state": effective.skill_state,
            "phase_id": effective.phase_id,
            "phase": effective.phase,
            "active_object": effective.active_object,
            "target_object": effective.target_object,
            "target": make_target_bundle(effective),
            "projection": projection,
            "task_meta": effective.task_meta,
            "debug": {
                "env_indices": torch.as_tensor(
                    indices, device=effective.skill_id.device, dtype=torch.long
                ),
                "current_valid": current_valid,
                "used_previous": used_previous,
            },
        }

    def _apply_previous_cache(
        self,
        context: NormalizedSkillAnnotationContext,
        env_indices: Sequence[int],
        use_previous: bool,
    ) -> tuple[NormalizedSkillAnnotationContext, torch.Tensor, torch.Tensor]:
        device = context.skill_id.device
        effective = make_empty_normalized_context(context.num_envs, device=device)
        used_previous = torch.zeros((context.num_envs,), device=device, dtype=torch.bool)
        current_valid = (context.skill_id != 0) & (
            context.has_target | context.allow_no_target
        )

        for row, env_idx in enumerate(env_indices):
            if bool(current_valid[row].item()):
                _write_context_row(effective, row, context, row)
                self._previous[env_idx] = _PreviousAnnotation.from_context(context, row)
            elif use_previous and self._previous[env_idx] is not None:
                self._previous[env_idx].write_to(effective, row)
                used_previous[row] = True

        return effective, used_previous, current_valid


def get_annotation_bundle(
    env,
    cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None = None,
    use_previous: bool = True,
) -> SkillAnnotationBundle:
    manager = _get_or_create_manager(env)
    return manager.get_annotation_bundle(env, cameras=cameras, use_previous=use_previous)


def get_annotation_bundle_for_env(
    env,
    env_idx: EnvIndex,
    cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None = None,
    use_previous: bool = True,
) -> SkillAnnotationBundle:
    manager = _get_or_create_manager(env)
    return manager.get_annotation_bundle(
        env,
        cameras=cameras,
        use_previous=use_previous,
        env_idx=env_idx,
    )


def reset_skill_annotator(env, env_idx: EnvIndex | None = None) -> None:
    manager = _get_or_create_manager(env)
    manager.reset(env_idx=env_idx)


def get_skill_label(env, env_idx: EnvIndex = 0) -> str:
    bundle = get_annotation_bundle_for_env(env, env_idx=env_idx)
    return bundle["skill"][0] if bundle["skill"] else "none"


def _get_or_create_manager(env) -> SkillAnnotationManager:
    manager = getattr(env, "_skill_annotation_manager", None)
    num_envs = _get_env_num_envs(env)
    if manager is None or not isinstance(manager, SkillAnnotationManager):
        manager = SkillAnnotationManager(num_envs=num_envs)
        env._skill_annotation_manager = manager
    else:
        manager.ensure_num_envs(num_envs)
    return manager


def _get_raw_context(env, env_idx: EnvIndex | None):
    if not hasattr(env, "get_skill_annotation_context"):
        raise AttributeError(
            "Skill annotation requires env.get_skill_annotation_context(env_idx=None)"
        )
    return env.get_skill_annotation_context(env_idx=env_idx)


def _normalize_context_for_indices(
    raw_context: (
        SkillAnnotationContext | Mapping[str, Any] | NormalizedSkillAnnotationContext | None
    ),
    indices: Sequence[int],
    requested_env_idx: EnvIndex | None,
    total_num_envs: int,
    device: torch.device,
) -> NormalizedSkillAnnotationContext:
    if isinstance(raw_context, NormalizedSkillAnnotationContext):
        normalized = raw_context
    else:
        requested_num_envs = len(indices) if requested_env_idx is None else None
        normalized = normalize_skill_context(
            raw_context, num_envs=requested_num_envs, device=device
        )

    if requested_env_idx is not None and normalized.num_envs == total_num_envs:
        return select_normalized_context(normalized, indices)
    if normalized.num_envs == len(indices):
        return normalized
    if normalized.num_envs == total_num_envs:
        return select_normalized_context(normalized, indices)
    if (
        normalized.num_envs == 1
        and len(indices) > 1
        and not isinstance(raw_context, NormalizedSkillAnnotationContext)
    ):
        return normalize_skill_context(raw_context, num_envs=len(indices), device=device)

    raise ValueError(
        "get_skill_annotation_context returned batch size "
        f"{normalized.num_envs}, but requested {len(indices)} env(s)"
    )


def _write_context_row(
    dst: NormalizedSkillAnnotationContext,
    dst_row: int,
    src: NormalizedSkillAnnotationContext,
    src_row: int,
) -> None:
    dst.skill[dst_row] = src.skill[src_row]
    dst.skill_id[dst_row] = src.skill_id[src_row]
    dst.skill_state[dst_row] = src.skill_state[src_row]
    dst.phase_id[dst_row] = src.phase_id[src_row]
    dst.phase[dst_row] = src.phase[src_row]
    dst.target_point_world[dst_row] = src.target_point_world[src_row]
    dst.target_point_valid[dst_row] = src.target_point_valid[src_row]
    dst.target_pose_world[dst_row] = src.target_pose_world[src_row]
    dst.target_pose_valid[dst_row] = src.target_pose_valid[src_row]
    dst.target_gripper_width[dst_row] = src.target_gripper_width[src_row]
    dst.target_gripper_width_valid[dst_row] = src.target_gripper_width_valid[src_row]
    dst.active_object[dst_row] = src.active_object[src_row]
    dst.target_object[dst_row] = src.target_object[src_row]
    dst.allow_no_target[dst_row] = src.allow_no_target[src_row]
    dst.task_meta.update(src.task_meta)


def _build_projection_bundle(
    env,
    context: NormalizedSkillAnnotationContext,
    cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None,
    env_indices: Sequence[int],
    total_num_envs: int,
) -> dict[str, dict[str, torch.Tensor]]:
    camera_params = _resolve_camera_params(env, cameras)
    projection = {}
    for camera_name, params in camera_params.items():
        params = _select_camera_params(params, env_indices, total_num_envs)
        image_size = _get_camera_image_size(env, camera_name, params)
        point_projection = project_3d_to_2d(
            context.target_point_world,
            params["intrinsic_cv"],
            params["extrinsic_cv"],
            image_size,
        )
        point_visible = point_projection["point_visible"] & context.target_point_valid
        point_uv = torch.where(
            point_visible[:, None],
            point_projection["point_uv"],
            torch.full_like(point_projection["point_uv"], -1.0),
        )

        gripper_width = torch.where(
            context.target_gripper_width_valid,
            context.target_gripper_width,
            torch.full_like(context.target_gripper_width, DEFAULT_GRASP_WIDTH_M),
        )
        grasp_projection = project_pose_to_grasp_annotation_2d(
            context.target_pose_world,
            params["intrinsic_cv"],
            params["extrinsic_cv"],
            image_size,
            gripper_width=gripper_width,
        )
        grasp_visible = grasp_projection["grasp_visible"] & context.target_pose_valid
        grasp_rect_uv = torch.where(
            grasp_visible[:, None, None],
            grasp_projection["grasp_rect_uv"],
            torch.full_like(grasp_projection["grasp_rect_uv"], -1.0),
        )
        projection[camera_name] = {
            "point_uv": point_uv,
            "point_visible": point_visible,
            "grasp_rect_uv": grasp_rect_uv,
            "grasp_visible": grasp_visible,
        }
    return projection


def _select_camera_params(
    params: Mapping[str, Any], env_indices: Sequence[int], total_num_envs: int
) -> dict[str, Any]:
    selected = dict(params)
    index_tensor = None
    for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
        value = selected.get(key)
        if not torch.is_tensor(value) or value.ndim < 3 or value.shape[0] != total_num_envs:
            continue
        if index_tensor is None:
            index_tensor = torch.as_tensor(env_indices, device=value.device, dtype=torch.long)
        selected[key] = value[index_tensor]
    return selected


def _resolve_camera_params(
    env,
    cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(cameras, Mapping):
        return dict(cameras)

    if not hasattr(env, "get_sensor_params"):
        return {}
    all_params = env.get_sensor_params()
    if cameras is None:
        camera_names = list(all_params.keys())
    elif isinstance(cameras, str):
        camera_names = [cameras]
    else:
        camera_names = list(cameras)

    missing = [name for name in camera_names if name not in all_params]
    if missing:
        raise KeyError(f"Unknown camera(s): {missing}")
    return {name: all_params[name] for name in camera_names}


def _get_camera_image_size(env, camera_name: str, params: Mapping[str, Any]) -> tuple[int, int]:
    if "image_size" in params:
        size = params["image_size"]
        if torch.is_tensor(size):
            size = size.detach().cpu().flatten().tolist()
        return int(size[0]), int(size[1])
    if "width" in params and "height" in params:
        return int(params["width"]), int(params["height"])

    for attr in ("_sensors", "_sensor_configs"):
        container = getattr(env, attr, None)
        if container is None or camera_name not in container:
            continue
        item = container[camera_name]
        config = getattr(item, "config", item)
        if hasattr(config, "width") and hasattr(config, "height"):
            return int(config.width), int(config.height)

    raise ValueError(
        f"Could not infer image size for camera '{camera_name}'. "
        "Provide image_size/width/height in camera params or a ManiSkill sensor config."
    )


def _normalize_env_indices(env_idx: EnvIndex | None, num_envs: int) -> list[int]:
    if env_idx is None:
        indices = list(range(num_envs))
    elif torch.is_tensor(env_idx):
        indices = [int(i) for i in env_idx.detach().cpu().flatten().tolist()]
    else:
        indices = [int(i) for i in torch.as_tensor(env_idx).flatten().tolist()]

    for idx in indices:
        if idx < 0 or idx >= num_envs:
            raise IndexError(f"env_idx {idx} is out of range for num_envs={num_envs}")
    return indices


def _normalize_env_idx_for_context(
    env_idx: EnvIndex | None, device: torch.device
) -> int | torch.Tensor | None:
    if env_idx is None:
        return None
    if torch.is_tensor(env_idx):
        return env_idx.to(device=device, dtype=torch.long).flatten()
    if isinstance(env_idx, (int, np.integer)):
        return int(env_idx)
    return torch.as_tensor(env_idx, device=device, dtype=torch.long).flatten()


def _get_env_num_envs(env) -> int:
    return max(int(getattr(env, "num_envs", 1)), 1)


def _get_env_device(env) -> torch.device:
    device = getattr(env, "device", None)
    return torch.device(device) if device is not None else torch.device("cpu")

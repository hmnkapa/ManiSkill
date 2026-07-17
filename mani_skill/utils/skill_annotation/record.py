from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mani_skill.utils import common
from mani_skill.utils.skill_annotation.manager import (
    get_annotation_bundle,
    reset_skill_annotator,
)
from mani_skill.utils.skill_annotation.schema import SKILL_VOCAB, SkillAnnotationBundle


class SkillAnnotationEpisodeRecorder:
    """Record state-aligned skill annotation bundles for HDF5 trajectories."""

    def __init__(
        self,
        cameras: str | Sequence[str] | Mapping[str, Mapping[str, Any]] | None = None,
        use_previous: bool = True,
    ):
        self.cameras = cameras
        self.use_previous = use_previous
        self.buffer: dict[str, Any] | None = None

    def __len__(self) -> int:
        if self.buffer is None:
            return 0
        return _tree_length(self.buffer)

    def reset(self, env, env_idx=None) -> None:
        env_indices = _normalize_env_indices(env_idx)
        if env_indices is None:
            reset_skill_annotator(env)
        else:
            for idx in env_indices:
                reset_skill_annotator(env, env_idx=int(idx))
        frame = _bundle_to_frame(
            get_annotation_bundle(
                env,
                cameras=self.cameras,
                use_previous=self.use_previous,
            )
        )
        if self.buffer is None:
            self.buffer = frame
            return

        _replace_last_frame(self.buffer, frame, env_indices)

    def step(self, env) -> None:
        frame = _bundle_to_frame(
            get_annotation_bundle(
                env,
                cameras=self.cameras,
                use_previous=self.use_previous,
            )
        )
        if self.buffer is None:
            self.buffer = frame
        else:
            self.buffer = common.append_dict_array(self.buffer, frame)

    def flush_to_h5(
        self,
        group: Any,
        start_ptr: int,
        end_ptr: int,
        env_idx: int,
    ) -> None:
        if self.buffer is None:
            return
        if end_ptr > len(self):
            raise ValueError(
                "Skill annotation buffer is shorter than the trajectory slice: "
                f"buffer={len(self)}, requested end_ptr={end_ptr}"
            )

        annotation_group = group.create_group("skill_annotations", track_order=True)
        annotation_group.attrs["skill_vocab"] = json.dumps(SKILL_VOCAB)
        _write_tree_to_h5(annotation_group, self.buffer, start_ptr, end_ptr, env_idx)

    def truncate(self, slice_) -> None:
        if self.buffer is not None:
            self.buffer = common.index_dict_array(self.buffer, slice_)

    def clear(self) -> None:
        self.buffer = None


def _bundle_to_frame(bundle: SkillAnnotationBundle) -> dict[str, Any]:
    frame: dict[str, Any] = {}
    for key in ("skill_id", "phase_id", "valid"):
        if key in bundle:
            frame[key] = _time_batch(bundle[key])

    target = bundle.get("target")
    if target:
        frame["target"] = {
            key: _time_batch(target[key])
            for key in (
                "point_world",
                "point_valid",
                "pose_world",
                "pose_valid",
                "gripper_width",
                "gripper_width_valid",
            )
            if key in target
        }

    projection = bundle.get("projection")
    if projection:
        frame["projection"] = {}
        for camera_name, camera_bundle in projection.items():
            frame["projection"][camera_name] = {
                key: _time_batch(camera_bundle[key])
                for key in (
                    "point_uv",
                    "point_visible",
                    "grasp_rect_uv",
                    "grasp_visible",
                )
                if key in camera_bundle
            }

    return frame


def _time_batch(value) -> np.ndarray:
    array = common.to_numpy(value)
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array[None, ...]


def _normalize_env_indices(env_idx) -> np.ndarray | None:
    if env_idx is None:
        return None
    env_idx = common.to_numpy(env_idx)
    return np.asarray(env_idx, dtype=np.int64).reshape(-1)


def _replace_last_frame(
    buffer: dict[str, Any] | np.ndarray,
    frame: dict[str, Any] | np.ndarray,
    env_indices: np.ndarray | None,
) -> None:
    if isinstance(buffer, dict):
        for key in buffer:
            _replace_last_frame(buffer[key], frame[key], env_indices)
        return

    if env_indices is None:
        buffer[-1] = frame[-1]
    else:
        buffer[-1, env_indices] = frame[-1, env_indices]


def _write_tree_to_h5(
    group: Any,
    tree: dict[str, Any] | np.ndarray,
    start_ptr: int,
    end_ptr: int,
    env_idx: int,
    key: str | None = None,
) -> None:
    if isinstance(tree, dict):
        subgroup = group if key is None else group.create_group(key, track_order=True)
        for child_key, child_value in tree.items():
            _write_tree_to_h5(
                subgroup,
                child_value,
                start_ptr,
                end_ptr,
                env_idx,
                child_key,
            )
        return

    data = tree[start_ptr:end_ptr, env_idx]
    group.create_dataset(key, data=data, dtype=data.dtype)


def _tree_length(tree: dict[str, Any] | np.ndarray) -> int:
    if isinstance(tree, dict):
        if not tree:
            return 0
        return _tree_length(next(iter(tree.values())))
    return int(tree.shape[0])

"""Strict T+1 observation / T transition buffering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import numpy as np

from .schema import (
    ACTION_DIM,
    ANNOTATION_SOURCE,
    CameraInfo,
    IMAGE_ANNOTATION_MODE,
    PickleEnv,
    PickleObservation,
    PickleTrajectory,
    SOURCE_ENV,
)


def _to_numpy(value: Any, *, dtype: np.dtype) -> np.ndarray:
    """Copy an array-like value to host NumPy, including device tensors."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class TrajectoryBuffer:
    """Build one trajectory without silently crossing episode boundaries."""

    def __init__(self):
        self._observations: list[PickleObservation] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._front_camera_info: Optional[CameraInfo] = None

    @property
    def active(self) -> bool:
        return bool(self._observations)

    @property
    def transition_count(self) -> int:
        return len(self._actions)

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def start(
        self, initial_observation: PickleObservation, front_camera_info: CameraInfo
    ) -> None:
        if self.active:
            raise RuntimeError(
                "Cannot reset over an unflushed pickle trajectory; call "
                "flush_episode() or discard_episode() first"
            )
        observation_copy = deepcopy(initial_observation)
        camera_info_copy = deepcopy(front_camera_info)
        self._observations = [observation_copy]
        self._front_camera_info = camera_info_copy

    def append(
        self,
        action: Any,
        reward: Any,
        next_observation: PickleObservation,
    ) -> None:
        """Atomically append one transition and its resulting observation."""

        if not self.active:
            raise RuntimeError("Trajectory buffer has not been started by reset()")
        action_array = _to_numpy(action, dtype=np.float32).reshape(-1)
        if action_array.shape != (ACTION_DIM,):
            raise ValueError(
                f"Canonical action must have shape ({ACTION_DIM},), got "
                f"{action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ValueError("Canonical action contains a non-finite value")
        reward_array = _to_numpy(reward, dtype=np.float32).reshape(-1)
        if reward_array.size != 1 or not np.isfinite(reward_array[0]):
            raise ValueError("Reward must be one finite scalar")

        # All validation and copying happens before mutation so an exception
        # cannot leave the three time series out of sync.
        action_copy = action_array.copy()
        reward_value = float(reward_array[0])
        observation_copy = deepcopy(next_observation)
        previous_lengths = (
            len(self._actions),
            len(self._rewards),
            len(self._observations),
        )
        try:
            self._actions.append(action_copy)
            self._rewards.append(reward_value)
            self._observations.append(observation_copy)
        except BaseException:
            del self._actions[previous_lengths[0] :]
            del self._rewards[previous_lengths[1] :]
            del self._observations[previous_lengths[2] :]
            raise

    def finalize(
        self, *, success: bool, task: PickleEnv | str = PickleEnv.PICK_CUBE
    ) -> PickleTrajectory:
        if not self.active:
            raise RuntimeError("No active trajectory to finalize")
        if not self._actions:
            raise ValueError("Cannot finalize an empty trajectory")
        if len(self._observations) != len(self._actions) + 1:
            raise RuntimeError(
                "Internal trajectory length invariant is broken: expected T+1 "
                "observations"
            )
        if len(self._rewards) != len(self._actions):
            raise RuntimeError(
                "Internal trajectory length invariant is broken: expected T rewards"
            )
        if self._front_camera_info is None:
            raise RuntimeError("Front-camera calibration was not recorded")
        return {
            "observations": deepcopy(self._observations),
            "actions": np.stack(self._actions).astype(np.float32).tolist(),
            "rewards": np.asarray(self._rewards, dtype=np.float32).tolist(),
            "camera_info": {
                "front_camera": deepcopy(self._front_camera_info),
            },
            "success": bool(success),
            "task": task.value if isinstance(task, PickleEnv) else str(task),
            "action_type": "delta",
            "env": SOURCE_ENV,
            "annotation_source": ANNOTATION_SOURCE,
            "image_annotation_mode": IMAGE_ANNOTATION_MODE,
        }

    def clear(self) -> None:
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._front_camera_info = None


__all__ = ["TrajectoryBuffer"]

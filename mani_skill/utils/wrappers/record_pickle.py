"""Gym wrapper for online RR-compatible pickle recording."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from mani_skill.trajectory.pickle.action_adapter import CanonicalActionAdapter
from mani_skill.trajectory.pickle.buffer import TrajectoryBuffer
from mani_skill.trajectory.pickle.state_adapter import PickleStateAdapter
from mani_skill.trajectory.pickle.task_registry import get_pickle_task_spec
from mani_skill.trajectory.pickle.writer import write_trajectory
from mani_skill.utils.skill_annotation.manager import (
    get_annotation_bundle,
    reset_skill_annotator,
)


class RecordPickle(gym.Wrapper):
    """Record one supported task rollout without changing Gym I/O.

    The wrapper never splits or writes a trajectory on ``terminated`` or
    ``truncated``.  Callers must explicitly call :meth:`flush_episode` or
    :meth:`discard_episode` before the next reset.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str | Path,
        *,
        compress: bool = False,
        overwrite: bool = False,
        use_previous_annotations: bool = False,
    ):
        super().__init__(env)
        base_env = env.unwrapped
        if int(getattr(base_env, "num_envs", 1)) != 1:
            raise NotImplementedError("RecordPickle supports num_envs=1 only")
        if getattr(base_env, "robot_uids", None) != "panda_wristcam":
            raise NotImplementedError(
                "RecordPickle currently requires robot_uids='panda_wristcam'"
            )
        task_id = getattr(getattr(base_env, "spec", None), "id", None)
        if task_id is None:
            task_id = getattr(getattr(env, "spec", None), "id", None)
        self.task_spec = get_pickle_task_spec(task_id)
        self.task_id = self.task_spec.env.value
        if base_env.control_mode != "pd_joint_pos":
            raise NotImplementedError(
                "Online pickle recording requires control_mode='pd_joint_pos'"
            )
        if base_env.obs_mode != "rgbd":
            raise NotImplementedError(
                "RecordPickle requires obs_mode='rgbd' for both RGB-D streams"
            )

        self.output_dir = Path(output_dir)
        self.compress = bool(compress)
        self.overwrite = bool(overwrite)
        self.use_previous_annotations = bool(use_previous_annotations)
        self.buffer = TrajectoryBuffer()
        self.action_adapter = CanonicalActionAdapter(env)
        self.state_adapter = PickleStateAdapter(env)
        self._last_info: Optional[dict[str, Any]] = None

    def reset(self, **kwargs):
        if self.buffer.active:
            raise RuntimeError(
                "Cannot reset over an unflushed pickle trajectory; call "
                "flush_episode() or discard_episode() first"
            )
        observation, info = self.env.reset(**kwargs)
        reset_skill_annotator(self.env.unwrapped)
        annotation = self._annotation_bundle()
        pickle_observation, camera_info = self.state_adapter.capture(
            observation, annotation
        )
        self.buffer.start(pickle_observation, camera_info)
        self._last_info = info
        return observation, info

    def step(self, action):
        if not self.buffer.active:
            raise RuntimeError("Call reset() before recording pickle transitions")
        canonical = self.action_adapter.native_to_canonical(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        annotation = self._annotation_bundle()
        pickle_observation, _ = self.state_adapter.capture(observation, annotation)
        self.buffer.append(canonical, reward, pickle_observation)
        self._last_info = info
        return observation, reward, terminated, truncated, info

    def flush_episode(self, success: Optional[bool] = None) -> Path:
        """Validate and write the active episode, then clear it on success."""

        if success is None:
            success = self._infer_success()
        trajectory = self.buffer.finalize(
            success=bool(success), task=self.task_id
        )
        status = "success" if success else "failure"
        suffix = ".pkl.xz" if self.compress else ".pkl"
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        output_path = self.output_dir / status / f"{timestamp}{suffix}"
        written_path = write_trajectory(
            trajectory,
            output_path,
            overwrite=self.overwrite,
            validate=True,
        )
        self.buffer.clear()
        self._last_info = None
        return written_path

    def discard_episode(self) -> None:
        self.buffer.clear()
        self._last_info = None

    def close(self):
        if self.buffer.active:
            warnings.warn(
                "Discarding an unflushed RecordPickle trajectory during close()",
                RuntimeWarning,
                stacklevel=2,
            )
            self.discard_episode()
        return self.env.close()

    def _annotation_bundle(self):
        return get_annotation_bundle(
            self.env.unwrapped,
            cameras=("hand_camera", "base_camera"),
            use_previous=self.use_previous_annotations,
        )

    def _infer_success(self) -> bool:
        if not isinstance(self._last_info, dict) or "success" not in self._last_info:
            raise ValueError(
                "flush_episode(success=None) could not infer success from the last info"
            )
        value = self._last_info["success"]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).reshape(-1)
        if array.size != 1:
            raise ValueError(f"Expected one success value, got shape {array.shape}")
        return bool(array[0])


__all__ = ["RecordPickle"]

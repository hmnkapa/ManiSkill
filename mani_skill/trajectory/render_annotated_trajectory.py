"""Render ManiSkill trajectories as videos with skill annotations."""

from __future__ import annotations

import copy
import json
import os
import re
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import tyro

import mani_skill.envs  # noqa: F401 - registers ManiSkill environments with Gymnasium
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.skill_annotation import (
    SKILL_VOCAB,
    draw_grasp_annotation_on_image,
    draw_guidance_point_on_image,
    draw_skill_on_image,
    get_annotation_bundle,
    reset_skill_annotator,
)
from mani_skill.utils.skill_annotation.projection import (
    project_3d_to_2d,
    project_pose_to_grasp_annotation_2d,
)


AnnotationSource = Literal["auto", "stored", "runtime"]
ReplayMode = Literal["auto", "actions", "env-states"]


@dataclass
class Args:
    traj_path: str
    """Path to a ManiSkill trajectory .h5 file."""

    episode_id: int | None = None
    """Render one episode. By default, render every annotatable episode."""

    output_path: str | None = None
    """Output .mp4 path. This can only be used with --episode-id."""

    output_dir: str | None = None
    """Directory for generated videos. Defaults to <trajectory>_annotated/."""

    annotation_source: AnnotationSource = "auto"
    """Use stored annotations, runtime annotations, or stored with runtime fallback."""

    replay_mode: ReplayMode = "auto"
    """Replay actions, replay actions plus recorded states, or choose automatically."""

    camera_name: str | None = None
    """Human render camera or RGB sensor to render."""

    fps: int = 30
    """Output video frames per second."""

    sim_backend: str | None = None
    """Simulation backend. Defaults to trajectory metadata, with auto resolved to physx_cpu."""

    shader: str | None = None
    """Optional shader pack override, such as default, rt-fast, or rt."""

    overwrite: bool = False
    """Allow replacing existing output videos."""


@dataclass
class FrameAnnotation:
    skill: str = "none"
    phase_id: int = -1
    valid: bool = True
    point_world: np.ndarray | None = None
    point_valid: bool = False
    pose_world: np.ndarray | None = None
    pose_valid: bool = False
    gripper_width: float | None = None
    point_uv: np.ndarray | None = None
    point_visible: bool = False
    grasp_rect_uv: np.ndarray | None = None
    grasp_visible: bool = False


@dataclass
class RenderedVideo:
    episode_id: int
    output_path: Path
    frame_count: int
    camera_name: str
    annotation_source: Literal["stored", "runtime"]


@dataclass
class RenderSummary:
    rendered: list[RenderedVideo]
    skipped_episode_ids: list[int]


@dataclass
class _EpisodeJob:
    episode: dict[str, Any]
    trajectory: h5py.Group
    source: Literal["stored", "runtime"]
    stored_annotations: "StoredAnnotationSequence | None"
    use_env_states: bool
    output_path: Path


class AnnotationUnavailableError(RuntimeError):
    pass


class StoredAnnotationSequence:
    """A state-aligned view over one HDF5 skill_annotations group."""

    def __init__(self, group: h5py.Group, expected_length: int | None = None):
        if "skill_id" not in group:
            raise ValueError("skill_annotations is missing required dataset 'skill_id'")
        self.group = group
        self.length = len(group["skill_id"])
        self.skill_names = _parse_skill_vocab(group.attrs.get("skill_vocab"))
        if expected_length is not None and self.length != expected_length:
            raise ValueError(
                "skill annotation length must equal the trajectory state count: "
                f"annotations={self.length}, expected={expected_length}"
            )
        for path, dataset in _walk_datasets(group):
            if dataset.ndim == 0:
                raise ValueError(f"skill annotation dataset '{path}' must be state-aligned")
            if len(dataset) != self.length:
                raise ValueError(
                    f"skill annotation dataset '{path}' has length {len(dataset)}, "
                    f"expected {self.length}"
                )

    def frame(self, index: int, camera_name: str) -> FrameAnnotation:
        if index < 0 or index >= self.length:
            raise IndexError(index)

        skill_id = int(np.asarray(self.group["skill_id"][index]).item())
        if skill_id not in self.skill_names:
            raise ValueError(
                f"skill id {skill_id} is not present in skill_vocab "
                f"{tuple(self.skill_names)}"
            )

        target = self.group.get("target")
        projection_root = self.group.get("projection")
        projection = (
            projection_root.get(camera_name)
            if isinstance(projection_root, h5py.Group)
            else None
        )

        return FrameAnnotation(
            skill=self.skill_names[skill_id],
            phase_id=_read_scalar(self.group, "phase_id", index, default=-1, cast=int),
            valid=_read_scalar(self.group, "valid", index, default=True, cast=bool),
            point_world=_read_array(target, "point_world", index),
            point_valid=_read_scalar(
                target, "point_valid", index, default=False, cast=bool
            ),
            pose_world=_read_array(target, "pose_world", index),
            pose_valid=_read_scalar(
                target, "pose_valid", index, default=False, cast=bool
            ),
            gripper_width=_read_valid_gripper_width(target, index),
            point_uv=_read_array(projection, "point_uv", index),
            point_visible=_read_scalar(
                projection, "point_visible", index, default=False, cast=bool
            ),
            grasp_rect_uv=_read_array(projection, "grasp_rect_uv", index),
            grasp_visible=_read_scalar(
                projection, "grasp_visible", index, default=False, cast=bool
            ),
        )


class CameraView:
    def __init__(self, base_env, name: str, kind: Literal["human", "sensor"]):
        self.base_env = base_env
        self.name = name
        self.kind = kind
        if kind == "human":
            self.camera = base_env._human_render_cameras[name]
        else:
            self.camera = base_env._sensors[name]

    def render(self) -> np.ndarray:
        if self.kind == "human":
            image = self.base_env.render_rgb_array(camera_name=self.name)
        else:
            for obj in getattr(self.base_env, "_hidden_objects", []):
                obj.hide_visual()
            self.base_env.scene.update_render(
                update_sensors=True, update_human_render_cameras=False
            )
            self.camera.capture()
            camera_images = self.camera.get_obs(
                rgb=True,
                depth=False,
                position=False,
                segmentation=False,
            )
            image = camera_images.get("rgb")
            if image is None:
                raise RuntimeError(f"Sensor camera '{self.name}' does not provide RGB images")
        if image is None:
            raise RuntimeError(f"Camera '{self.name}' did not produce an image")
        return _to_rgb_uint8(image)

    def params(self, image: np.ndarray) -> dict[str, Any]:
        if not hasattr(self.camera, "get_params"):
            raise RuntimeError(f"Camera '{self.name}' does not expose camera parameters")
        params = {
            key: common.to_numpy(value)
            for key, value in self.camera.get_params().items()
        }
        params["image_size"] = (image.shape[1], image.shape[0])
        return params


def parse_args(args=None) -> Args:
    return tyro.cli(Args, args=args)


def main(args: Args) -> RenderSummary:
    _validate_args(args)
    traj_path = Path(args.traj_path).expanduser().resolve()
    json_path = traj_path.with_suffix(".json")
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file does not exist: {traj_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Trajectory metadata does not exist: {json_path}")

    json_data = io_utils.load_json(json_path)
    episodes = _select_episodes(json_data.get("episodes", []), args.episode_id)
    control_modes = {
        episode.get("control_mode")
        for episode in episodes
        if episode.get("control_mode") is not None
    }
    if len(control_modes) > 1:
        raise ValueError(
            "Selected episodes use multiple control modes; render them in separate calls"
        )

    env, env_id = _make_env(json_data, args, next(iter(control_modes), None))
    rendered: list[RenderedVideo] = []
    skipped: list[int] = []
    try:
        base_env = env.unwrapped
        camera = _resolve_camera(base_env, args.camera_name)
        runtime_supported = callable(
            getattr(base_env, "get_skill_annotation_context", None)
        )

        with h5py.File(traj_path, "r") as h5_file:
            jobs: list[_EpisodeJob] = []
            for episode in episodes:
                episode_id = int(episode["episode_id"])
                traj_id = f"traj_{episode_id}"
                if traj_id not in h5_file:
                    message = f"{traj_id} does not exist in {traj_path}"
                    if args.episode_id is not None:
                        raise KeyError(message)
                    logger.warning(message)
                    skipped.append(episode_id)
                    continue

                trajectory = h5_file[traj_id]
                try:
                    action_count = _tree_length(trajectory.get("actions"), "actions")
                    source, stored = _resolve_annotation_source(
                        trajectory,
                        requested=args.annotation_source,
                        runtime_supported=runtime_supported,
                        expected_length=action_count + 1,
                    )
                    use_env_states = _resolve_replay_mode(
                        trajectory, args.replay_mode, action_count + 1
                    )
                    output_path = _resolve_output_path(
                        args, traj_path, episode_id, camera.name
                    )
                    _check_output_path(output_path, args.overwrite)
                except (AnnotationUnavailableError, ValueError, KeyError) as exc:
                    if args.episode_id is not None:
                        raise
                    logger.warning(f"Skipping {traj_id}: {exc}")
                    skipped.append(episode_id)
                    continue

                jobs.append(
                    _EpisodeJob(
                        episode=episode,
                        trajectory=trajectory,
                        source=source,
                        stored_annotations=stored,
                        use_env_states=use_env_states,
                        output_path=output_path,
                    )
                )

            if not jobs:
                raise AnnotationUnavailableError(
                    "No selected episode has usable skill annotations"
                )

            for job in jobs:
                try:
                    result = _render_episode(
                        env=env,
                        base_env=base_env,
                        env_id=env_id,
                        camera=camera,
                        job=job,
                        fps=args.fps,
                        overwrite=args.overwrite,
                    )
                except AnnotationUnavailableError as exc:
                    episode_id = int(job.episode["episode_id"])
                    if args.episode_id is not None:
                        raise
                    logger.warning(f"Skipping traj_{episode_id}: {exc}")
                    skipped.append(episode_id)
                    continue
                rendered.append(result)
                print(f"Annotated video created: {result.output_path}")
            if not rendered:
                raise AnnotationUnavailableError(
                    "No selected episode could be rendered with skill annotations"
                )
    finally:
        env.close()

    if skipped:
        logger.warning(f"Skipped episodes without usable annotations: {skipped}")
    return RenderSummary(rendered=rendered, skipped_episode_ids=skipped)


def _validate_args(args: Args) -> None:
    if args.fps <= 0:
        raise ValueError(f"fps must be positive, got {args.fps}")
    if args.output_path is not None and args.episode_id is None:
        raise ValueError("--output-path requires --episode-id")
    if args.output_path is not None and args.output_dir is not None:
        raise ValueError("--output-path and --output-dir cannot be used together")


def _select_episodes(
    episodes: list[dict[str, Any]], episode_id: int | None
) -> list[dict[str, Any]]:
    if episode_id is None:
        if not episodes:
            raise ValueError("Trajectory metadata contains no episodes")
        return episodes
    for episode in episodes:
        if int(episode["episode_id"]) == episode_id:
            return [episode]
    raise KeyError(f"Episode {episode_id} does not exist in trajectory metadata")


def _make_env(
    json_data: Mapping[str, Any], args: Args, control_mode: str | None
) -> tuple[gym.Env, str]:
    env_info = json_data.get("env_info")
    if not isinstance(env_info, Mapping) or "env_id" not in env_info:
        raise ValueError("Trajectory metadata is missing env_info.env_id")

    env_id = str(env_info["env_id"])
    env_kwargs = copy.deepcopy(env_info.get("env_kwargs", {}))
    if not isinstance(env_kwargs, dict):
        raise ValueError("Trajectory metadata env_info.env_kwargs must be a mapping")
    env_kwargs["obs_mode"] = "none"
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["num_envs"] = 1
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    sim_backend = args.sim_backend or env_kwargs.get("sim_backend")
    if sim_backend in (None, "auto"):
        sim_backend = "physx_cpu"
    env_kwargs["sim_backend"] = sim_backend
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader

    return gym.make(env_id, **env_kwargs), env_id


def _resolve_camera(base_env, requested_name: str | None) -> CameraView:
    human_cameras = getattr(base_env, "_human_render_cameras", {})
    sensors = getattr(base_env, "_sensors", {})

    if requested_name is not None:
        if requested_name in human_cameras:
            return CameraView(base_env, requested_name, "human")
        if requested_name in sensors:
            return CameraView(base_env, requested_name, "sensor")
        available = [*human_cameras.keys(), *sensors.keys()]
        raise KeyError(
            f"Unknown camera '{requested_name}'. Available cameras: {available}"
        )

    if "render_camera" in human_cameras:
        return CameraView(base_env, "render_camera", "human")
    if human_cameras:
        return CameraView(base_env, next(iter(human_cameras)), "human")
    for name, sensor in sensors.items():
        if hasattr(sensor, "get_params"):
            return CameraView(base_env, name, "sensor")
    raise RuntimeError("The reconstructed environment has no renderable camera")


def _resolve_annotation_source(
    trajectory: h5py.Group,
    requested: AnnotationSource,
    runtime_supported: bool,
    expected_length: int,
) -> tuple[Literal["stored", "runtime"], StoredAnnotationSequence | None]:
    stored_error: Exception | None = None
    if requested in ("auto", "stored") and "skill_annotations" in trajectory:
        try:
            sequence = StoredAnnotationSequence(
                trajectory["skill_annotations"], expected_length=expected_length
            )
            return "stored", sequence
        except ValueError as exc:
            stored_error = exc
            if requested == "stored":
                raise AnnotationUnavailableError(str(exc)) from exc

    if requested == "stored":
        raise AnnotationUnavailableError("trajectory has no skill_annotations group")
    if runtime_supported:
        if stored_error is not None:
            logger.warning(
                f"Stored skill annotations are invalid; using runtime annotations: {stored_error}"
            )
        return "runtime", None
    if requested == "runtime":
        raise AnnotationUnavailableError(
            "environment does not implement get_skill_annotation_context"
        )
    if stored_error is not None:
        raise AnnotationUnavailableError(
            "stored annotations are invalid and runtime annotations are unavailable: "
            f"{stored_error}"
        )
    raise AnnotationUnavailableError(
        "trajectory has no stored annotations and the environment has no runtime annotator"
    )


def _resolve_replay_mode(
    trajectory: h5py.Group, replay_mode: ReplayMode, expected_state_count: int
) -> bool:
    has_states = "env_states" in trajectory
    if replay_mode == "actions":
        return False
    if not has_states:
        if replay_mode == "env-states":
            raise ValueError("--replay-mode env-states requires an env_states group")
        return False

    state_count = _tree_length(trajectory["env_states"], "env_states")
    if state_count != expected_state_count:
        message = (
            f"env_states has length {state_count}, expected {expected_state_count}"
        )
        if replay_mode == "env-states":
            raise ValueError(message)
        logger.warning(f"{message}; falling back to action replay")
        return False
    return True


def _resolve_output_path(
    args: Args, traj_path: Path, episode_id: int, camera_name: str
) -> Path:
    if args.output_path is not None:
        output_path = Path(args.output_path).expanduser().resolve()
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("--output-path must end with .mp4")
        return output_path

    if args.output_dir is None:
        output_dir = traj_path.parent / f"{traj_path.stem}_annotated"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    safe_camera_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", camera_name)
    return output_dir / f"traj_{episode_id}_{safe_camera_name}_annotated.mp4"


def _check_output_path(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )


def _render_episode(
    env: gym.Env,
    base_env,
    env_id: str,
    camera: CameraView,
    job: _EpisodeJob,
    fps: int,
    overwrite: bool,
) -> RenderedVideo:
    episode_id = int(job.episode["episode_id"])
    actions = job.trajectory["actions"]
    action_count = _tree_length(actions, "actions")
    env_states = job.trajectory.get("env_states") if job.use_env_states else None
    reset_kwargs = _normalized_reset_kwargs(job.episode)

    _check_output_path(job.output_path, overwrite)
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = job.output_path.with_name(
        f".{job.output_path.stem}.{uuid.uuid4().hex}.tmp.mp4"
    )
    writer = None
    try:
        env.reset(**reset_kwargs)
        if env_states is not None:
            base_env.set_state_dict(trajectory_utils.index_dict(env_states, 0))
        if job.source == "runtime":
            reset_skill_annotator(base_env)

        writer = imageio.get_writer(temporary_path, fps=fps, quality=5)
        for frame_index in range(action_count + 1):
            if frame_index > 0:
                action = trajectory_utils.index_dict(actions, frame_index - 1)
                env.step(action)
                if env_states is not None:
                    base_env.set_state_dict(
                        trajectory_utils.index_dict(env_states, frame_index)
                    )

            image = camera.render()
            camera_params = camera.params(image)
            if job.source == "stored":
                assert job.stored_annotations is not None
                annotation = job.stored_annotations.frame(frame_index, camera.name)
            else:
                annotation = _get_runtime_annotation(
                    base_env, camera.name, camera_params
                )
            annotation = _project_missing_annotations(
                annotation, camera_params, (image.shape[1], image.shape[0])
            )
            image = _draw_annotation(
                image,
                annotation,
                env_id=env_id,
                episode_id=episode_id,
                frame_index=frame_index,
                frame_count=action_count + 1,
            )
            writer.append_data(image)

        writer.close()
        writer = None
        os.replace(temporary_path, job.output_path)
    except Exception:
        if writer is not None:
            try:
                writer.close()
            finally:
                temporary_path.unlink(missing_ok=True)
        else:
            temporary_path.unlink(missing_ok=True)
        raise

    return RenderedVideo(
        episode_id=episode_id,
        output_path=job.output_path,
        frame_count=action_count + 1,
        camera_name=camera.name,
        annotation_source=job.source,
    )


def _get_runtime_annotation(
    base_env, camera_name: str, camera_params: Mapping[str, Any]
) -> FrameAnnotation:
    try:
        bundle = get_annotation_bundle(
            base_env, cameras={camera_name: dict(camera_params)}
        )
    except (AttributeError, NotImplementedError, ValueError) as exc:
        raise AnnotationUnavailableError(
            f"runtime skill annotations are unavailable: {exc}"
        ) from exc
    target = bundle.get("target", {})
    projection = bundle.get("projection", {}).get(camera_name, {})
    skills = bundle.get("skill", [])
    return FrameAnnotation(
        skill=skills[0] if skills else "none",
        phase_id=_first_scalar(bundle.get("phase_id"), default=-1, cast=int),
        point_world=_first_array(target.get("point_world")),
        point_valid=_first_scalar(
            target.get("point_valid"), default=False, cast=bool
        ),
        pose_world=_first_array(target.get("pose_world")),
        pose_valid=_first_scalar(target.get("pose_valid"), default=False, cast=bool),
        gripper_width=_runtime_gripper_width(target),
        point_uv=_first_array(projection.get("point_uv")),
        point_visible=_first_scalar(
            projection.get("point_visible"), default=False, cast=bool
        ),
        grasp_rect_uv=_first_array(projection.get("grasp_rect_uv")),
        grasp_visible=_first_scalar(
            projection.get("grasp_visible"), default=False, cast=bool
        ),
    )


def _project_missing_annotations(
    annotation: FrameAnnotation,
    camera_params: Mapping[str, Any],
    image_size: tuple[int, int],
) -> FrameAnnotation:
    if not annotation.valid:
        return replace(
            annotation,
            point_visible=False,
            grasp_visible=False,
        )

    result = annotation
    if annotation.point_uv is None and annotation.point_valid:
        if annotation.point_world is not None:
            projected = project_3d_to_2d(
                annotation.point_world,
                camera_params["intrinsic_cv"],
                camera_params["extrinsic_cv"],
                image_size,
            )
            result = replace(
                result,
                point_uv=common.to_numpy(projected["point_uv"])[0],
                point_visible=bool(projected["point_visible"][0].item()),
            )

    if annotation.grasp_rect_uv is None and annotation.pose_valid:
        if annotation.pose_world is not None:
            projected = project_pose_to_grasp_annotation_2d(
                annotation.pose_world,
                camera_params["intrinsic_cv"],
                camera_params["extrinsic_cv"],
                image_size,
                gripper_width=annotation.gripper_width,
            )
            result = replace(
                result,
                grasp_rect_uv=common.to_numpy(projected["grasp_rect_uv"])[0],
                grasp_visible=bool(projected["grasp_visible"][0].item()),
            )
    return result


def _draw_annotation(
    image: np.ndarray,
    annotation: FrameAnnotation,
    env_id: str,
    episode_id: int,
    frame_index: int,
    frame_count: int,
) -> np.ndarray:
    if annotation.grasp_rect_uv is not None:
        image = draw_grasp_annotation_on_image(
            image,
            {
                "grasp_rect_uv": annotation.grasp_rect_uv,
                "grasp_visible": annotation.grasp_visible,
            },
            skill=annotation.skill,
            use_skill_color=True,
        )
    if annotation.point_uv is not None:
        image = draw_guidance_point_on_image(
            image,
            {
                "point_uv": annotation.point_uv,
                "point_visible": annotation.point_visible,
            },
            skill=annotation.skill,
            use_skill_color=True,
        )
    compact = min(image.shape[:2]) < 256
    if not compact:
        image = draw_skill_on_image(image, annotation.skill)
        status_lines = [
            f"env: {env_id}",
            f"episode: {episode_id}  skill: {annotation.skill}",
            f"phase_id: {annotation.phase_id}",
            f"frame: {frame_index + 1}/{frame_count}",
        ]
    else:
        status_lines = [
            f"{env_id}  ep:{episode_id}",
            (
                f"{annotation.skill}  phase:{annotation.phase_id}  "
                f"frame:{frame_index + 1}/{frame_count}"
            ),
        ]
    return _draw_status_panel(
        image,
        status_lines,
    )


def _draw_status_panel(image: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2

    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    compact = min(image.shape[:2]) < 256
    scale = 0.34 if compact else 0.48
    thickness = 1
    padding = 4 if compact else 8
    line_height = 13 if compact else 19
    max_width = max(
        cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines
    )
    available_width = max(image.shape[1] - 2 * padding, 1)
    if max_width > available_width:
        scale *= available_width / max_width
        minimum_line_height = 9 if compact else 12
        line_height = max(
            int(line_height * available_width / max_width), minimum_line_height
        )
        max_width = available_width

    panel_height = padding * 2 + line_height * len(lines)
    top = max(image.shape[0] - panel_height - padding, 0)
    bottom = image.shape[0] - padding
    right = min(max_width + padding * 2, image.shape[1])
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, top), (right, bottom), (0, 0, 0), thickness=-1)
    annotated = cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0.0)
    for line_index, line in enumerate(lines):
        y = top + padding + line_height * (line_index + 1) - 4
        cv2.putText(
            annotated,
            line,
            (padding, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated


def _parse_skill_vocab(raw_vocab: Any) -> dict[int, str]:
    if raw_vocab is None:
        return {index: name for index, name in enumerate(SKILL_VOCAB)}
    if isinstance(raw_vocab, np.ndarray) and raw_vocab.ndim == 0:
        raw_vocab = raw_vocab.item()
    if isinstance(raw_vocab, bytes):
        raw_vocab = raw_vocab.decode("utf-8")
    if isinstance(raw_vocab, str):
        raw_vocab = json.loads(raw_vocab)
    if isinstance(raw_vocab, (list, tuple)):
        return {index: str(name) for index, name in enumerate(raw_vocab)}
    if isinstance(raw_vocab, Mapping):
        if all(isinstance(value, (int, np.integer)) for value in raw_vocab.values()):
            return {int(value): str(name) for name, value in raw_vocab.items()}
        return {int(index): str(name) for index, name in raw_vocab.items()}
    raise ValueError(f"Unsupported skill_vocab format: {type(raw_vocab).__name__}")


def _walk_datasets(
    group: h5py.Group, prefix: str = ""
) -> list[tuple[str, h5py.Dataset]]:
    datasets = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            datasets.append((path, value))
        else:
            datasets.extend(_walk_datasets(value, path))
    return datasets


def _tree_length(tree, name: str) -> int:
    if tree is None:
        raise KeyError(f"trajectory is missing required '{name}' data")
    if isinstance(tree, (dict, h5py.Group)):
        if len(tree) == 0:
            raise ValueError(f"'{name}' data is empty")
        child_lengths = {
            key: _tree_length(tree[key], f"{name}/{key}") for key in tree.keys()
        }
        unique_lengths = set(child_lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"'{name}' leaves have inconsistent lengths: {child_lengths}"
            )
        return next(iter(unique_lengths))
    return len(tree)


def _read_array(group, key: str, index: int) -> np.ndarray | None:
    if group is None or key not in group:
        return None
    return np.asarray(group[key][index])


def _read_scalar(group, key: str, index: int, default, cast):
    if group is None or key not in group:
        return default
    return cast(np.asarray(group[key][index]).item())


def _read_valid_gripper_width(group, index: int) -> float | None:
    if group is None or "gripper_width" not in group:
        return None
    if not _read_scalar(
        group, "gripper_width_valid", index, default=False, cast=bool
    ):
        return None
    return float(np.asarray(group["gripper_width"][index]).item())


def _first_array(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(common.to_numpy(value))
    return array[0] if array.ndim > 0 else array


def _first_scalar(value, default, cast):
    if value is None:
        return default
    array = np.asarray(common.to_numpy(value)).reshape(-1)
    return default if array.size == 0 else cast(array[0])


def _runtime_gripper_width(target: Mapping[str, Any]) -> float | None:
    if not _first_scalar(
        target.get("gripper_width_valid"), default=False, cast=bool
    ):
        return None
    return _first_scalar(target.get("gripper_width"), default=None, cast=float)


def _normalized_reset_kwargs(episode: Mapping[str, Any]) -> dict[str, Any]:
    reset_kwargs = copy.deepcopy(episode.get("reset_kwargs", {}))
    seed = reset_kwargs.get("seed", episode.get("episode_seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(
                f"Episode {episode.get('episode_id')} has an ambiguous seed list: {seed}"
            )
        seed = seed[0]
    if seed is not None:
        reset_kwargs["seed"] = seed
    return reset_kwargs


def _to_rgb_uint8(image) -> np.ndarray:
    array = np.asarray(common.to_numpy(image))
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected one HxWx3 RGB image, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        finite_max = np.nanmax(array) if array.size else 0
        if finite_max <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255)
    return np.ascontiguousarray(array.astype(np.uint8, copy=False))


if __name__ == "__main__":
    main(parse_args())

import json

import h5py
import numpy as np
import pytest

from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.render_annotated_trajectory import (
    AnnotationUnavailableError,
    Args,
    CameraView,
    FrameAnnotation,
    StoredAnnotationSequence,
    _check_output_path,
    _draw_annotation,
    _parse_skill_vocab,
    _project_missing_annotations,
    _resolve_annotation_source,
    _resolve_output_path,
    _resolve_replay_mode,
    _select_episodes,
    _tree_length,
    _validate_args,
)


def _add_annotations(trajectory, length=3, camera_name="render_camera"):
    annotations = trajectory.create_group("skill_annotations")
    annotations.attrs["skill_vocab"] = json.dumps(
        ["none", "pick", "place", "insert", "screw", "push"]
    )
    annotations.create_dataset("skill_id", data=np.array([1, 2, 0])[:length])
    annotations.create_dataset("phase_id", data=np.arange(length))

    target = annotations.create_group("target")
    target.create_dataset(
        "point_world",
        data=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (length, 1)),
    )
    target.create_dataset("point_valid", data=np.ones(length, dtype=bool))
    poses = np.tile(np.eye(4, dtype=np.float32), (length, 1, 1))
    poses[:, 2, 3] = 1.0
    target.create_dataset("pose_world", data=poses)
    target.create_dataset("pose_valid", data=np.ones(length, dtype=bool))
    target.create_dataset(
        "gripper_width", data=np.full(length, 0.05, dtype=np.float32)
    )
    target.create_dataset("gripper_width_valid", data=np.ones(length, dtype=bool))

    projection = annotations.create_group("projection").create_group(camera_name)
    projection.create_dataset(
        "point_uv",
        data=np.tile(np.array([[20.0, 30.0]], dtype=np.float32), (length, 1)),
    )
    projection.create_dataset("point_visible", data=np.ones(length, dtype=bool))
    projection.create_dataset(
        "grasp_rect_uv",
        data=np.tile(
            np.array([[[10, 10], [30, 10], [30, 20], [10, 20]]], dtype=np.float32),
            (length, 1, 1),
        ),
    )
    projection.create_dataset("grasp_visible", data=np.ones(length, dtype=bool))
    return annotations


def test_stored_annotations_decode_vocab_and_projection(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        annotations = _add_annotations(trajectory)

        sequence = StoredAnnotationSequence(annotations, expected_length=3)
        frame = sequence.frame(1, "render_camera")

        assert frame.skill == "place"
        assert frame.phase_id == 1
        assert frame.point_visible
        assert frame.point_uv.tolist() == [20.0, 30.0]
        assert frame.grasp_visible
        assert frame.grasp_rect_uv.shape == (4, 2)
        assert frame.gripper_width == pytest.approx(0.05)


def test_stored_annotations_require_t_plus_one_alignment(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        annotations = _add_annotations(trajectory)

        with pytest.raises(ValueError, match="annotations=3, expected=4"):
            StoredAnnotationSequence(annotations, expected_length=4)


def test_stored_annotations_validate_every_nested_dataset(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        annotations = _add_annotations(trajectory)
        del annotations["target/point_valid"]
        annotations["target"].create_dataset(
            "point_valid", data=np.ones(2, dtype=bool)
        )

        with pytest.raises(ValueError, match="target/point_valid.*length 2"):
            StoredAnnotationSequence(annotations)


def test_annotation_source_prefers_stored_and_falls_back_to_runtime(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        stored_trajectory = h5_file.create_group("traj_0")
        _add_annotations(stored_trajectory)
        source, sequence = _resolve_annotation_source(
            stored_trajectory,
            requested="auto",
            runtime_supported=True,
            expected_length=3,
        )
        assert source == "stored"
        assert sequence is not None

        runtime_trajectory = h5_file.create_group("traj_1")
        source, sequence = _resolve_annotation_source(
            runtime_trajectory,
            requested="auto",
            runtime_supported=True,
            expected_length=3,
        )
        assert source == "runtime"
        assert sequence is None

        with pytest.raises(AnnotationUnavailableError, match="no skill_annotations"):
            _resolve_annotation_source(
                runtime_trajectory,
                requested="stored",
                runtime_supported=True,
                expected_length=3,
            )

        del stored_trajectory["skill_annotations/phase_id"]
        stored_trajectory["skill_annotations"].create_dataset(
            "phase_id", data=np.arange(2)
        )
        source, sequence = _resolve_annotation_source(
            stored_trajectory,
            requested="auto",
            runtime_supported=True,
            expected_length=3,
        )
        assert source == "runtime"
        assert sequence is None


def test_world_targets_are_projected_only_when_stored_projection_is_missing():
    camera_params = {
        "intrinsic_cv": np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "extrinsic_cv": np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
    }
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 1.0
    missing_projection = FrameAnnotation(
        skill="pick",
        point_world=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        point_valid=True,
        pose_world=pose,
        pose_valid=True,
        gripper_width=0.1,
    )

    projected = _project_missing_annotations(
        missing_projection, camera_params, image_size=(100, 100)
    )
    assert projected.point_visible
    assert projected.point_uv.tolist() == pytest.approx([50.0, 50.0])
    assert projected.grasp_visible
    assert projected.grasp_rect_uv.shape == (4, 2)

    stored_projection = FrameAnnotation(
        point_world=missing_projection.point_world,
        point_valid=True,
        point_uv=np.array([7.0, 8.0], dtype=np.float32),
        point_visible=False,
    )
    unchanged = _project_missing_annotations(
        stored_projection, camera_params, image_size=(100, 100)
    )
    assert unchanged.point_uv.tolist() == [7.0, 8.0]
    assert not unchanged.point_visible


def test_dictionary_actions_have_a_shared_length_and_can_be_indexed(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        actions = h5_file.create_group("actions")
        actions.create_dataset("arm", data=np.zeros((4, 7), dtype=np.float32))
        actions.create_dataset("gripper", data=np.arange(4, dtype=np.float32))

        assert _tree_length(actions, "actions") == 4
        action = trajectory_utils.index_dict(actions, 2)
        assert action["arm"].shape == (7,)
        assert action["gripper"] == 2

        del actions["gripper"]
        actions.create_dataset("gripper", data=np.arange(3, dtype=np.float32))
        with pytest.raises(ValueError, match="inconsistent lengths"):
            _tree_length(actions, "actions")


def test_replay_mode_validates_t_plus_one_environment_states(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as h5_file:
        trajectory = h5_file.create_group("traj_0")
        states = trajectory.create_group("env_states")
        states.create_dataset("actor", data=np.zeros((4, 13), dtype=np.float32))

        assert _resolve_replay_mode(trajectory, "auto", 4)
        assert not _resolve_replay_mode(trajectory, "actions", 4)
        with pytest.raises(ValueError, match="length 4, expected 5"):
            _resolve_replay_mode(trajectory, "env-states", 5)


def test_episode_selection_supports_single_and_batch_modes():
    episodes = [{"episode_id": 2}, {"episode_id": 7}]
    assert _select_episodes(episodes, None) == episodes
    assert _select_episodes(episodes, 7) == [{"episode_id": 7}]
    with pytest.raises(KeyError, match="Episode 3"):
        _select_episodes(episodes, 3)


def test_output_arguments_and_overwrite_protection(tmp_path):
    with pytest.raises(ValueError, match="requires --episode-id"):
        _validate_args(Args(traj_path="trajectory.h5", output_path="video.mp4"))

    args = Args(
        traj_path="trajectory.h5",
        episode_id=3,
        output_dir=str(tmp_path),
    )
    output = _resolve_output_path(
        args, tmp_path / "trajectory.h5", episode_id=3, camera_name="robot/camera"
    )
    assert output.name == "traj_3_robot_camera_annotated.mp4"

    output.touch()
    with pytest.raises(FileExistsError, match="--overwrite"):
        _check_output_path(output, overwrite=False)
    _check_output_path(output, overwrite=True)


def test_skill_vocab_supports_standard_and_explicit_mappings():
    assert _parse_skill_vocab(None)[1] == "pick"
    assert _parse_skill_vocab(json.dumps(["none", "custom"]))[1] == "custom"
    assert _parse_skill_vocab({"none": 0, "custom": 7})[7] == "custom"


def test_sensor_camera_render_does_not_depend_on_observation_mode():
    class _HiddenObject:
        hidden = False

        def hide_visual(self):
            self.hidden = True

    class _Scene:
        updated = False

        def update_render(self, update_sensors, update_human_render_cameras):
            assert update_sensors
            assert not update_human_render_cameras
            self.updated = True

    class _Sensor:
        captured = False

        def capture(self):
            self.captured = True

        def get_obs(self, **kwargs):
            assert kwargs == {
                "rgb": True,
                "depth": False,
                "position": False,
                "segmentation": False,
            }
            return {"rgb": np.ones((1, 8, 9, 3), dtype=np.float32)}

    hidden = _HiddenObject()
    scene = _Scene()
    sensor = _Sensor()
    base_env = type(
        "BaseEnvStub",
        (),
        {
            "_hidden_objects": [hidden],
            "_sensors": {"camera": sensor},
            "scene": scene,
        },
    )()

    image = CameraView(base_env, "camera", "sensor").render()

    assert hidden.hidden
    assert scene.updated
    assert sensor.captured
    assert image.shape == (8, 9, 3)
    assert image.dtype == np.uint8
    assert image.max() == 255


def test_small_camera_frames_use_a_compact_non_overlapping_status_panel():
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    annotated = _draw_annotation(
        image,
        FrameAnnotation(skill="pick", phase_id=2),
        env_id="Task-v1",
        episode_id=0,
        frame_index=0,
        frame_count=10,
    )

    assert np.all(annotated[:64] == 255)
    assert np.any(annotated[80:] != 255)

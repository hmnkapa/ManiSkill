from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import sapien
import torch

from mani_skill.trajectory.pickle.action_adapter import (
    CanonicalActionAdapter,
    canonical_action,
    canonical_gripper_to_native,
    native_gripper_to_canonical,
)
from mani_skill.trajectory.pickle.buffer import TrajectoryBuffer
from mani_skill.trajectory.pickle.schema import PickleEnv
from mani_skill.trajectory.pickle.state_adapter import (
    PickCubeStateAdapter,
    PickleStateAdapter,
)
from mani_skill.trajectory.pickle.task_registry import (
    PICKLE_TASK_SPECS,
    get_pickle_task_spec,
)
from mani_skill.trajectory.pickle.transforms import (
    apply_delta_quaternion_xyzw,
    normalize_quaternion_xyzw,
    pose_to_matrix,
    relative_quaternion_xyzw,
    world_point_to_base,
    world_pose_to_base,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from mani_skill.trajectory.pickle.validator import (
    TrajectoryValidationError,
    validate_trajectory,
)
from mani_skill.trajectory.pickle.writer import read_trajectory, write_trajectory


def _robot_state():
    return {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "ee_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "ee_pos_sim": np.zeros(3, dtype=np.float32),
        "ee_quat_sim": np.array([0, 0, 0, 1], dtype=np.float32),
        "ee_pos_vel": np.zeros(3, dtype=np.float32),
        "ee_ori_vel": np.zeros(3, dtype=np.float32),
        "gripper_width": 0.08,
        "joint_positions": np.zeros(7, dtype=np.float32),
        "joint_velocities": np.zeros(7, dtype=np.float32),
        "joint_torques": np.zeros(9, dtype=np.float32),
        "gripper_finger_1_pos": 0.04,
        "gripper_finger_2_pos": 0.04,
    }


def _observation(part_count=2):
    identity_pose = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    return {
        "robot_state": _robot_state(),
        "color_image1": np.zeros((224, 224, 3), dtype=np.uint8),
        "color_image2": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth_image1": np.ones((224, 224), dtype=np.float32),
        "depth_image2": np.ones((224, 224), dtype=np.float32),
        "parts_poses": np.tile(identity_pose, int(part_count)),
        "point_cloud": None,
        "skill": None,
        "guidance_point": None,
        "guidance_point_clean": None,
        "guidance_pose": None,
        "guidance_pose_clean": None,
        "guidance_gripper_width": None,
        "guidance_point_2d": {
            "color_image1": None,
            "color_image2": None,
        },
        "grasp_annotation_2d": {
            "color_image1": None,
            "color_image2": None,
        },
    }


def _camera_info():
    # RR camera coordinates are left handed: x right, y up, z forward.
    reflection = np.diag([1, -1, 1, 1]).astype(np.float32)
    return {
        "image_size": np.array([224, 224], dtype=np.int32),
        "intrinsics": np.array(
            [[300, 0, 112], [0, 300, 112], [0, 0, 1]], dtype=np.float32
        ),
        "camera_to_sim_local": reflection.copy(),
        "sim_local_to_camera": reflection.copy(),
    }


def _trajectory(task=PickleEnv.PICK_CUBE):
    task_spec = get_pickle_task_spec(task)
    observation = _observation(len(task_spec.parts))
    buffer = TrajectoryBuffer()
    buffer.start(observation, _camera_info())
    buffer.append(
        np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32),
        1.0,
        observation,
    )
    return buffer.finalize(success=True, task=task_spec.env)


EXPECTED_PART_NAMES = {
    PickleEnv.LIFT_PEG_UPRIGHT: ("peg",),
    PickleEnv.PEG_INSERTION_SIDE: ("peg", "box_with_hole"),
    PickleEnv.PICK_CUBE: ("cube", "goal_site"),
    PickleEnv.PLACE_SPHERE: ("sphere", "bin"),
    PickleEnv.PLUG_CHARGER: ("charger", "receptacle"),
    PickleEnv.POKE_CUBE: ("peg", "cube", "goal_region"),
    PickleEnv.PULL_CUBE: ("cube", "goal_region"),
    PickleEnv.PULL_CUBE_TOOL: ("l_shape_tool", "cube"),
    PickleEnv.PUSH_CUBE: ("cube", "goal_region"),
    PickleEnv.STACK_CUBE: ("cubeA", "cubeB"),
    PickleEnv.STACK_PYRAMID: ("cubeA", "cubeB", "cubeC"),
}


def test_pickle_task_registry_is_complete_ordered_and_read_only():
    assert PickCubeStateAdapter is PickleStateAdapter
    assert set(PICKLE_TASK_SPECS) == set(PickleEnv)
    for env_id, expected_names in EXPECTED_PART_NAMES.items():
        task_spec = get_pickle_task_spec(env_id.value)
        assert task_spec.env is env_id
        assert task_spec.part_names == expected_names
        assert task_spec.parts_pose_dim == len(expected_names) * 7

    with pytest.raises(TypeError):
        PICKLE_TASK_SPECS[PickleEnv.PICK_CUBE] = get_pickle_task_spec(
            PickleEnv.PICK_CUBE
        )
    with pytest.raises(NotImplementedError, match="Unsupported pickle environment"):
        get_pickle_task_spec("UnknownTask-v1")


@pytest.mark.parametrize("env_id", list(PickleEnv))
def test_task_part_pose_order_and_base_frame_transform(env_id):
    task_spec = get_pickle_task_spec(env_id)
    world_base = pose_to_matrix(
        np.array([0.4, -0.2, 0.3], dtype=np.float32),
        normalize_quaternion_xyzw(
            [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
        ),
    )
    actors = {}
    expected = []
    for index, part in enumerate(task_spec.parts):
        local_position = np.array(
            [0.1 * index, -0.03 * index, 0.02 * (index + 1)],
            dtype=np.float32,
        )
        local_pose = pose_to_matrix(
            local_position, np.array([0, 0, 0, 1], dtype=np.float32)
        )
        world_pose = world_base @ local_pose
        actors[part.env_attribute] = SimpleNamespace(
            pose=SimpleNamespace(
                to_transformation_matrix=lambda pose=world_pose: pose[None]
            )
        )
        expected.append(
            np.r_[local_position, [0, 0, 0, 1]].astype(np.float32)
        )

    adapter = object.__new__(PickleStateAdapter)
    adapter.base_env = SimpleNamespace(**actors)
    adapter.task_spec = task_spec
    actual = adapter._parts_poses(world_base).reshape((-1, 7))
    assert np.allclose(actual, np.stack(expected), atol=1e-6)


def test_state_adapter_reports_missing_task_entity():
    class MissingPartEnv:
        num_envs = 1
        robot_uids = "panda_wristcam"
        spec = SimpleNamespace(id=PickleEnv.LIFT_PEG_UPRIGHT.value)

        @property
        def unwrapped(self):
            return self

    with pytest.raises(
        NotImplementedError, match="LiftPegUpright-v1.*peg"
    ):
        PickleStateAdapter(MissingPartEnv())


def test_state_adapter_maps_negative_raw_depth_to_invalid_zero():
    raw_depth = np.full((1, 224, 224, 1), 1000, dtype=np.int16)
    raw_depth[0, 3, 4, 0] = -32768
    raw_observation = {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
                "depth": raw_depth,
            }
        }
    }
    adapter = object.__new__(PickleStateAdapter)
    _, depth = adapter._camera_images(raw_observation, "base_camera")
    assert depth.dtype == np.float32
    assert depth[3, 4] == 0
    assert depth[0, 0] == 1


@pytest.mark.parametrize("env_id", list(PickleEnv))
def test_validator_accepts_every_registered_task(env_id):
    trajectory = _trajectory(env_id)
    validate_trajectory(trajectory)
    validate_trajectory(trajectory, env=env_id.value)
    expected_dim = get_pickle_task_spec(env_id).parts_pose_dim
    assert trajectory["observations"][0]["parts_poses"].shape == (expected_dim,)


def test_validator_rejects_task_shape_mismatch_and_unknown_task():
    trajectory = _trajectory(PickleEnv.POKE_CUBE)
    trajectory["observations"][0]["parts_poses"] = trajectory["observations"][0][
        "parts_poses"
    ][:-7]
    with pytest.raises(TrajectoryValidationError, match=r"expected shape \(21,\)"):
        validate_trajectory(trajectory)

    trajectory = _trajectory(PickleEnv.PICK_CUBE)
    with pytest.raises(TrajectoryValidationError, match="expected 'StackCube-v1'"):
        validate_trajectory(trajectory, env=PickleEnv.STACK_CUBE)

    trajectory["task"] = "UnknownTask-v1"
    with pytest.raises(NotImplementedError, match="Unsupported pickle environment"):
        validate_trajectory(trajectory)


def test_quaternion_order_delta_and_sign_equivalence():
    wxyz = np.array([0.7, 0.1, -0.2, 0.3], dtype=np.float32)
    assert np.allclose(xyzw_to_wxyz(wxyz_to_xyzw(wxyz)), wxyz)

    current = normalize_quaternion_xyzw([0.2, -0.1, 0.3, 0.9])
    target = normalize_quaternion_xyzw([-0.4, 0.2, 0.1, 0.8])
    delta = relative_quaternion_xyzw(current, -target)
    reconstructed = apply_delta_quaternion_xyzw(current, delta)
    assert np.allclose(reconstructed, target, atol=1e-6)
    assert delta[3] >= 0
    assert np.array_equal(
        normalize_quaternion_xyzw([-1, 0, 0, 0]),
        normalize_quaternion_xyzw([1, 0, 0, 0]),
    )


def test_world_to_base_pose_and_point():
    base_world = pose_to_matrix(
        np.array([1, 2, 3], dtype=np.float32),
        normalize_quaternion_xyzw([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
    )
    object_world = base_world @ pose_to_matrix(
        np.array([0.2, -0.3, 0.4], dtype=np.float32),
        np.array([0, 0, 0, 1], dtype=np.float32),
    )
    object_base = world_pose_to_base(object_world, base_world)
    assert np.allclose(object_base[:3, 3], [0.2, -0.3, 0.4], atol=1e-6)
    assert np.allclose(
        world_point_to_base(object_world[:3, 3], base_world),
        [0.2, -0.3, 0.4],
        atol=1e-6,
    )


def test_gripper_semantics_and_canonical_target_reconstruction():
    assert native_gripper_to_canonical(+1) == -1
    assert native_gripper_to_canonical(-1) == +1
    assert canonical_gripper_to_native(-1) == +1
    assert canonical_gripper_to_native(+1) == -1

    current_position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target_position = np.array([0.4, -0.2, 0.8], dtype=np.float32)
    current_quaternion = normalize_quaternion_xyzw([0.1, 0.2, 0.3, 0.9])
    target_quaternion = normalize_quaternion_xyzw([-0.2, 0.1, 0.4, 0.8])
    action = canonical_action(
        current_position,
        current_quaternion,
        target_position,
        target_quaternion,
        native_gripper_command=+1,
    )
    assert action.dtype == np.float32
    assert np.allclose(current_position + action[:3], target_position)
    assert np.allclose(
        apply_delta_quaternion_xyzw(current_quaternion, action[3:7]),
        target_quaternion,
        atol=1e-6,
    )
    assert action[-1] == -1


def test_fk_action_labels_joint_command_target(monkeypatch):
    import mani_skill.trajectory.pickle.action_adapter as action_module

    class FakeArm:
        active_joint_indices = torch.arange(7)
        config = SimpleNamespace(use_delta=False, use_target=False)

        def _preprocess_action(self, action):
            return action

    class FakeRobot:
        def get_qpos(self):
            return torch.zeros((1, 9), dtype=torch.float32)

    class FakePinModel:
        def compute_forward_kinematics(self, qpos):
            self.qpos = qpos.copy()

        def get_link_pose(self, link_index):
            assert link_index == 8
            return sapien.Pose(
                p=self.qpos[:3],
                q=np.array([1, 0, 0, 0], dtype=np.float32),
            )

    class FakeController:
        controllers = {"arm": FakeArm()}
        single_action_space = SimpleNamespace(shape=(8,))

        @staticmethod
        def to_action_dict(action):
            return {"arm": action[:7], "gripper": action[7:]}

    monkeypatch.setattr(action_module, "PDJointPosController", FakeArm)
    adapter = object.__new__(CanonicalActionAdapter)
    adapter.base_env = SimpleNamespace(control_mode="pd_joint_pos", device="cpu")
    adapter.controller = FakeController()
    adapter.agent = SimpleNamespace(
        robot=FakeRobot(), tcp=SimpleNamespace(index=torch.tensor([8]))
    )
    adapter._get_pin_model = lambda: FakePinModel()
    action = adapter.native_to_canonical(
        np.array([0.4, -0.2, 0.7, 0, 0, 0, 0, 1], dtype=np.float32),
        current_position=np.zeros(3, dtype=np.float32),
        current_quaternion_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
    )
    assert np.allclose(action[:3], [0.4, -0.2, 0.7])
    assert np.allclose(action[3:7], [0, 0, 0, 1])
    assert action[7] == -1


def test_buffer_t_plus_one_and_reset_guard():
    buffer = TrajectoryBuffer()
    buffer.start(_observation(), _camera_info())
    with pytest.raises(RuntimeError, match="unflushed"):
        buffer.start(_observation(), _camera_info())
    buffer.append([0, 0, 0, 0, 0, 0, 1, 1], 0.5, _observation())
    result = buffer.finalize(success=False)
    assert len(result["observations"]) == 2
    assert len(result["actions"]) == len(result["rewards"]) == 1
    assert result["task"] == PickleEnv.PICK_CUBE.value
    assert isinstance(result["actions"], list)
    assert isinstance(result["rewards"], list)
    validate_trajectory(result)


def test_validator_rejects_schema_and_quaternion_errors():
    trajectory = _trajectory()
    del trajectory["observations"][0]["depth_image1"]
    with pytest.raises(TrajectoryValidationError, match="key mismatch"):
        validate_trajectory(trajectory)

    trajectory = _trajectory()
    trajectory["actions"][0][3:7] = [0, 0, 0, 2]
    with pytest.raises(TrajectoryValidationError, match="unit quaternion"):
        validate_trajectory(trajectory)


@pytest.mark.parametrize("suffix", [".pkl", ".pkl.xz"])
@pytest.mark.parametrize(
    "task", [PickleEnv.PICK_CUBE, PickleEnv.STACK_PYRAMID]
)
def test_writer_roundtrip_and_refuses_overwrite(tmp_path, suffix, task):
    trajectory = _trajectory(task)
    path = tmp_path / f"episode{suffix}"
    assert write_trajectory(trajectory, path) == path
    loaded = read_trajectory(path)
    validate_trajectory(loaded)
    assert loaded["task"] == task.value
    assert loaded["actions"] == trajectory["actions"]
    with pytest.raises(FileExistsError):
        write_trajectory(trajectory, path)


def test_writer_failure_preserves_target_and_removes_temp(tmp_path, monkeypatch):
    import mani_skill.trajectory.pickle.writer as writer_module

    path = tmp_path / "episode.pkl"
    original = b"existing-data"
    path.write_bytes(original)

    def fail_dump(*args, **kwargs):
        raise RuntimeError("synthetic serialization failure")

    monkeypatch.setattr(writer_module.pickle, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="synthetic"):
        writer_module.write_trajectory(_trajectory(), path, overwrite=True)
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".episode.pkl.tmp-*")) == []


@pytest.mark.parametrize("env_id", list(PickleEnv))
def test_record_pickle_wrapper_keeps_gym_io_and_explicit_boundaries(
    tmp_path, monkeypatch, env_id
):
    wrapper_module = importlib.import_module(
        "mani_skill.utils.wrappers.record_pickle"
    )

    class FakeEnv(gym.Env):
        def __init__(self):
            self.num_envs = 1
            self.robot_uids = "panda_wristcam"
            self.control_mode = "pd_joint_pos"
            self.obs_mode = "rgbd"
            self.spec = SimpleNamespace(id=env_id.value)
            self.reset_count = 0
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            self.reset_count += 1
            self.step_count = 0
            return {"step": 0}, {"success": np.array([False])}

        def step(self, action):
            self.step_count += 1
            # A true termination must not split or flush the motion plan.
            return (
                {"step": self.step_count},
                np.array([1], dtype=np.float32),
                np.array([True]),
                np.array([False]),
                {"success": np.array([True])},
            )

    class FakeActionAdapter:
        def __init__(self, env):
            pass

        def native_to_canonical(self, action):
            return np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32)

    class FakeStateAdapter:
        def __init__(self, env):
            self.part_count = len(get_pickle_task_spec(env_id).parts)

        def capture(self, observation, annotation):
            return _observation(self.part_count), _camera_info()

    monkeypatch.setattr(wrapper_module, "CanonicalActionAdapter", FakeActionAdapter)
    monkeypatch.setattr(wrapper_module, "PickleStateAdapter", FakeStateAdapter)
    monkeypatch.setattr(wrapper_module, "reset_skill_annotator", lambda env: None)
    monkeypatch.setattr(
        wrapper_module, "get_annotation_bundle", lambda *args, **kwargs: None
    )

    recorder = wrapper_module.RecordPickle(FakeEnv(), tmp_path)
    raw_observation, _ = recorder.reset(seed=3)
    assert raw_observation == {"step": 0}
    returned = recorder.step(np.zeros(8, dtype=np.float32))
    assert returned[0] == {"step": 1}
    assert bool(returned[2][0]) is True
    assert recorder.buffer.transition_count == 1
    with pytest.raises(RuntimeError, match="unflushed"):
        recorder.reset()
    path = recorder.flush_episode(success=True)
    assert path.parent.name == "success"
    data = read_trajectory(path)
    validate_trajectory(data)
    assert data["task"] == env_id.value
    assert len(data["observations"]) == 2
    assert len(data["actions"]) == len(data["rewards"]) == 1

    recorder.reset()
    recorder.discard_episode()
    recorder.reset()
    recorder.discard_episode()
    recorder.close()


def test_record_pickle_rejects_unsupported_environment(tmp_path):
    from mani_skill.utils.wrappers import RecordPickle

    class UnsupportedEnv(gym.Env):
        num_envs = 1
        robot_uids = "panda_wristcam"
        control_mode = "pd_joint_pos"
        obs_mode = "rgbd"
        spec = SimpleNamespace(id="UnknownTask-v1")

    with pytest.raises(NotImplementedError, match="UnknownTask-v1"):
        RecordPickle(UnsupportedEnv(), tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("env_id", list(PickleEnv))
def test_real_short_pickle_for_every_task(tmp_path, env_id):
    if os.environ.get("MANISKILL_RUN_RENDER_TESTS") != "1":
        pytest.skip("set MANISKILL_RUN_RENDER_TESTS=1 to exercise SAPIEN rendering")
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import RecordPickle

    try:
        env = gym.make(
            env_id.value,
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_joint_pos",
            robot_uids="panda_wristcam",
            reward_mode="sparse",
            render_mode="sensors",
            sensor_configs={
                "width": 224,
                "height": 224,
                "shader_pack": "minimal",
            },
        )
    except RuntimeError as error:
        if "supported physical device" in str(error):
            pytest.skip(f"SAPIEN rendering device unavailable: {error}")
        raise
    recorder = RecordPickle(env, tmp_path)
    try:
        recorder.reset(seed=0)
        qpos = recorder.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
        target_position, target_quaternion = (
            recorder.action_adapter.fk_target_pose_base(qpos)
        )
        recorder.step(np.r_[qpos, 1].astype(np.float32))
        path = recorder.flush_episode(success=False)
        trajectory = read_trajectory(path)
        validate_trajectory(trajectory)
        assert trajectory["task"] == env_id.value
        assert trajectory["observations"][0]["parts_poses"].shape == (
            get_pickle_task_spec(env_id).parts_pose_dim,
        )
        state = trajectory["observations"][0]["robot_state"]
        action = np.asarray(trajectory["actions"][0], dtype=np.float32)
        assert np.allclose(state["ee_pos"] + action[:3], target_position, atol=1e-5)
        assert np.allclose(
            apply_delta_quaternion_xyzw(state["ee_quat"], action[3:7]),
            target_quaternion,
            atol=1e-5,
        )
    finally:
        recorder.close()


@pytest.mark.slow
@pytest.mark.parametrize(
    "env_id", [PickleEnv.PICK_CUBE, PickleEnv.POKE_CUBE]
)
def test_rr_process_pickle_smoke(tmp_path, env_id):
    if os.environ.get("MANISKILL_RUN_RR_SMOKE") != "1":
        pytest.skip("set MANISKILL_RUN_RR_SMOKE=1 to run the sibling RR smoke test")
    rr_root = Path(__file__).resolve().parents[3] / "robust-rearrangement-custom"
    rr_python = Path.home() / "miniconda3/envs/rr/bin/python"
    if not rr_root.is_dir() or not rr_python.is_file():
        pytest.skip("sibling robust-rearrangement-custom or its Python env is unavailable")
    raw_dir = tmp_path / "raw" / env_id.value / "success"
    path = raw_dir / "episode.pkl"
    write_trajectory(_trajectory(env_id), path)
    parts_pose_dim = get_pickle_task_spec(env_id).parts_pose_dim
    code = (
        "from pathlib import Path; "
        "from src.data_processing.process_pickles import process_pickle_file; "
        f"d=process_pickle_file(Path({str(path)!r}), 0.0, True, False); "
        "assert d['robot_state'].shape == (1, 16); "
        "assert d['action/delta'].shape == (1, 10); "
        f"assert d['parts_poses'].shape == (1, {parts_pose_dim})"
    )
    subprocess.run(
        [str(rr_python), "-c", code],
        cwd=rr_root,
        env={**os.environ, "PYTHONPATH": str(rr_root)},
        check=True,
    )

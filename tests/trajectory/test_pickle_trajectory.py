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


def _observation():
    return {
        "robot_state": _robot_state(),
        "color_image1": np.zeros((224, 224, 3), dtype=np.uint8),
        "color_image2": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth_image1": np.ones((224, 224), dtype=np.float32),
        "depth_image2": np.ones((224, 224), dtype=np.float32),
        "parts_poses": np.array(
            [0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 1],
            dtype=np.float32,
        ),
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


def _trajectory():
    buffer = TrajectoryBuffer()
    buffer.start(_observation(), _camera_info())
    buffer.append(
        np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32),
        1.0,
        _observation(),
    )
    return buffer.finalize(success=True)


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
def test_writer_roundtrip_and_refuses_overwrite(tmp_path, suffix):
    trajectory = _trajectory()
    path = tmp_path / f"episode{suffix}"
    assert write_trajectory(trajectory, path) == path
    loaded = read_trajectory(path)
    validate_trajectory(loaded)
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


def test_record_pickle_wrapper_keeps_gym_io_and_explicit_boundaries(
    tmp_path, monkeypatch
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
            self.spec = SimpleNamespace(id="PickCube-v1")
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
            pass

        def capture(self, observation, annotation):
            return _observation(), _camera_info()

    monkeypatch.setattr(wrapper_module, "CanonicalActionAdapter", FakeActionAdapter)
    monkeypatch.setattr(wrapper_module, "PickCubeStateAdapter", FakeStateAdapter)
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
    assert len(data["observations"]) == 2
    assert len(data["actions"]) == len(data["rewards"]) == 1

    recorder.reset()
    recorder.discard_episode()
    recorder.reset()
    recorder.discard_episode()
    recorder.close()


@pytest.mark.slow
def test_pick_cube_real_short_pickle(tmp_path):
    if os.environ.get("MANISKILL_RUN_RENDER_TESTS") != "1":
        pytest.skip("set MANISKILL_RUN_RENDER_TESTS=1 to exercise SAPIEN rendering")
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import RecordPickle

    try:
        env = gym.make(
            "PickCube-v1",
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
def test_rr_process_pickle_smoke(tmp_path):
    if os.environ.get("MANISKILL_RUN_RR_SMOKE") != "1":
        pytest.skip("set MANISKILL_RUN_RR_SMOKE=1 to run the sibling RR smoke test")
    rr_root = Path(__file__).resolve().parents[3] / "robust-rearrangement-custom"
    rr_python = Path.home() / "miniconda3/envs/rr/bin/python"
    if not rr_root.is_dir() or not rr_python.is_file():
        pytest.skip("sibling robust-rearrangement-custom or its Python env is unavailable")
    raw_dir = tmp_path / "raw" / "PickCube-v1" / "success"
    path = raw_dir / "episode.pkl"
    write_trajectory(_trajectory(), path)
    code = (
        "from pathlib import Path; "
        "from src.data_processing.process_pickles import process_pickle_file; "
        f"d=process_pickle_file(Path({str(path)!r}), 0.0, True, False); "
        "assert d['robot_state'].shape == (1, 16); "
        "assert d['action/delta'].shape == (1, 10)"
    )
    subprocess.run(
        [str(rr_python), "-c", code],
        cwd=rr_root,
        env={**os.environ, "PYTHONPATH": str(rr_root)},
        check=True,
    )

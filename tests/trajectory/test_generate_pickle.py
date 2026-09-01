from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pytest

from mani_skill.trajectory.pickle import PickleEnv


generate_module = importlib.import_module(
    "scripts.data_generation.generate_pickle"
)


def _args(tmp_path: Path, env_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        env_id=env_id,
        num_traj=1,
        record_dir=tmp_path,
        start_seed=7,
        sim_backend="cpu",
        shader="minimal",
        vis=False,
        compress=True,
        max_attempts=1,
        annotation_source="scripted",
    )


def test_generate_pickle_cli_defaults_and_poke_error(capsys):
    parsed = generate_module.parse_args(["--annotation-source", "scripted"])
    assert parsed.env_id == PickleEnv.PICK_CUBE.value

    parsed = generate_module.parse_args(
        [
            "--annotation-source",
            "scripted",
            "--env-id",
            PickleEnv.STACK_CUBE.value,
            "--num-traj",
            "2",
        ]
    )
    assert parsed.env_id == PickleEnv.STACK_CUBE.value
    assert parsed.num_traj == 2

    with pytest.raises(SystemExit):
        generate_module.parse_args(
            [
                "--annotation-source",
                "scripted",
                "--env-id",
                PickleEnv.POKE_CUBE.value,
            ]
        )
    assert "supports RecordPickle" in capsys.readouterr().err


def test_motion_planning_solver_registry_covers_all_but_poke_cube():
    expected = {env.value for env in PickleEnv} - {PickleEnv.POKE_CUBE.value}
    assert set(generate_module.MOTION_PLANNING_SOLVERS) == expected


def test_generate_pickle_diagnostic_cli_contract(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    parsed = generate_module.parse_args(
        [
            "--annotation-source",
            "scripted",
            "--env-id",
            PickleEnv.PULL_CUBE_TOOL.value,
            "--diagnostic-task-successes",
            "20",
            "--attempt-diagnostics-dir",
            str(diagnostics),
        ]
    )
    assert parsed.diagnostic_task_successes == 20
    assert parsed.attempt_diagnostics_dir == diagnostics

    with pytest.raises(SystemExit):
        generate_module.parse_args(
            [
                "--annotation-source",
                "scripted",
                "--env-id",
                PickleEnv.PULL_CUBE_TOOL.value,
                "--diagnostic-task-successes",
                "20",
            ]
        )


def test_motion_planning_front_point_audit_distinguishes_visibility():
    camera = {
        "image_size": np.array([224, 224], dtype=np.int32),
        "intrinsics": np.array(
            [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "camera_to_sim_local": np.eye(4, dtype=np.float32),
        "sim_local_to_camera": np.eye(4, dtype=np.float32),
    }
    trajectory = {
        "camera_info": {"front_camera": camera},
        "observations": [
            {
                "skill": "pick",
                "guidance_point_clean": np.array(
                    [0.0, 0.0, 1.0], dtype=np.float32
                ),
                "guidance_point_2d": {
                    "color_image2": np.array([112.0, 112.0], dtype=np.float32)
                },
            },
            {
                "skill": "push",
                "guidance_point_clean": np.array(
                    [0.0, 2.0, 1.0], dtype=np.float32
                ),
                "guidance_point_2d": {"color_image2": None},
            },
        ],
    }
    audit = generate_module._front_point_audit(trajectory)
    assert audit["active_observations"] == 2
    assert audit["active_recorded_visible"] == 1
    assert audit["active_geometrically_visible"] == 1
    assert audit["observations"][0]["reprojected_uv"] == pytest.approx(
        [112.0, 112.0]
    )
    assert audit["observations"][1]["reprojected_uv"] == pytest.approx(
        [112.0, -88.0]
    )


def test_generate_dispatches_solver_and_uses_task_output_path(
    tmp_path, monkeypatch
):
    env_id = PickleEnv.STACK_PYRAMID.value
    calls = {}

    class FakeEnv:
        pass

    class FakeRecorder:
        def __init__(self, env, output_dir, compress):
            calls["recorder_env"] = env
            calls["output_dir"] = Path(output_dir)
            calls["compress"] = compress
            self.output_dir = Path(output_dir)

        def flush_episode(self, success):
            calls["flush_success"] = success
            return self.output_dir / "success" / "episode.pkl.xz"

        def discard_episode(self):
            calls["discarded"] = True

        def close(self):
            calls["closed"] = True

    def fake_make(requested_env_id, **kwargs):
        calls["gym_env_id"] = requested_env_id
        calls["gym_kwargs"] = kwargs
        return FakeEnv()

    def fake_solve(recorder, seed, debug, vis):
        calls["solver_args"] = (recorder, seed, debug, vis)
        return (None, {"success": np.array([True])})

    monkeypatch.setattr(generate_module.gym, "make", fake_make)
    monkeypatch.setattr(generate_module, "RecordPickle", FakeRecorder)
    monkeypatch.setitem(
        generate_module.MOTION_PLANNING_SOLVERS, env_id, fake_solve
    )

    saved = generate_module.generate(_args(tmp_path, env_id))
    assert saved == [
        tmp_path
        / env_id
        / "motionplanning_pickle"
        / "success"
        / "episode.pkl.xz"
    ]
    assert calls["gym_env_id"] == env_id
    assert calls["output_dir"] == tmp_path / env_id / "motionplanning_pickle"
    assert calls["compress"] is True
    assert calls["solver_args"][1:] == (7, False, False)
    assert calls["flush_success"] is True
    assert calls["closed"] is True
    assert calls["gym_kwargs"]["robot_uids"] == "panda_wristcam"
    assert calls["gym_kwargs"]["sensor_configs"]["width"] == 224
    front = calls["gym_kwargs"]["sensor_configs"]["base_camera"]
    assert front["width"] == 224
    assert front["height"] == 224
    assert front["fov"] == pytest.approx(np.deg2rad(40.0))
    assert len(front["pose"]) == 7


def test_generate_rejects_poke_before_environment_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        generate_module.gym,
        "make",
        lambda *args, **kwargs: pytest.fail("gym.make must not be called"),
    )
    with pytest.raises(NotImplementedError, match="no Panda motion-planning solver"):
        generate_module.generate(_args(tmp_path, PickleEnv.POKE_CUBE.value))

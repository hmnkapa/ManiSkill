from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


generate_module = importlib.import_module(
    "scripts.data_generation.generate_ppo_pickle"
)


def test_ppo_task_set_and_cli_contract(tmp_path):
    assert set(generate_module.PPO_PICKLE_ENVS) == {
        "LiftPegUpright-v1",
        "PickCube-v1",
        "PokeCube-v1",
        "PullCube-v1",
        "PushCube-v1",
        "StackCube-v1",
    }
    checkpoint = tmp_path / "policy.pt"
    parsed = generate_module.parse_args(
        [
            "--env-id",
            "PickCube-v1",
            "--checkpoint",
            str(checkpoint),
            "--annotation-source",
            "scripted",
            "--num-traj",
            "3",
        ]
    )
    assert parsed.checkpoint == checkpoint
    assert parsed.annotation_source == "scripted"
    assert parsed.num_traj == 3
    assert parsed.num_eval_steps == 400
    assert parsed.diagnostic_task_successes is None
    assert parsed.attempt_diagnostics_dir is None

    diagnostics = tmp_path / "diagnostics"
    parsed = generate_module.parse_args(
        [
            "--env-id",
            "LiftPegUpright-v1",
            "--checkpoint",
            str(checkpoint),
            "--annotation-source",
            "scripted",
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
                "--env-id",
                "LiftPegUpright-v1",
                "--checkpoint",
                str(checkpoint),
                "--diagnostic-task-successes",
                "20",
            ]
        )

    with pytest.raises(SystemExit):
        generate_module.parse_args(
            [
                "--env-id",
                "PlaceSphere-v1",
                "--checkpoint",
                str(checkpoint),
                "--annotation-source",
                "scripted",
            ]
        )


def test_policy_state_and_checkpoint_topology():
    state = generate_module._policy_state(
        {"state": np.arange(9, dtype=np.float32)}, torch.device("cpu")
    )
    assert state.shape == (1, 9)
    agent = generate_module.PPOAgent(9, 7, torch.device("cpu"))
    assert agent.actor_mean(state).shape == (1, 7)
    assert agent.actor_logstd.shape == (1, 7)

    with pytest.raises(KeyError, match="state vector"):
        generate_module._policy_state({}, torch.device("cpu"))
    with pytest.raises(ValueError, match="one PPO state row"):
        generate_module._policy_state(
            {"state": np.zeros((2, 9), dtype=np.float32)}, torch.device("cpu")
        )


def test_front_point_audit_distinguishes_recorded_and_geometric_visibility():
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
                "guidance_point": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "guidance_point_clean": np.array(
                    [0.0, 0.0, 1.0], dtype=np.float32
                ),
                "guidance_point_2d": {
                    "color_image2": np.array([112.0, 112.0], dtype=np.float32)
                },
            },
            {
                "skill": "pick",
                "guidance_point": np.array([0.0, 2.0, 1.0], dtype=np.float32),
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

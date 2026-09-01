import math

import numpy as np

from mani_skill.trajectory.pickle.camera_contract import (
    RR_FRONT_EYE_WORLD,
    RR_FRONT_FOV_RADIANS,
    RR_FRONT_TARGET_WORLD,
    rr_front_camera_parameters,
)


def test_unlisted_tasks_keep_default_front_camera():
    assert rr_front_camera_parameters("PickCube-v1") == (
        RR_FRONT_EYE_WORLD,
        RR_FRONT_TARGET_WORLD,
        RR_FRONT_FOV_RADIANS,
    )


def test_poke_and_pull_tool_are_retreat_only_overrides():
    default_eye = np.asarray(RR_FRONT_EYE_WORLD, dtype=np.float64)
    default_target = np.asarray(RR_FRONT_TARGET_WORLD, dtype=np.float64)
    default_forward = (default_target - default_eye) / np.linalg.norm(
        default_target - default_eye
    )
    expected_retreats = {
        "PokeCube-v1": 0.42,
        "PullCubeTool-v1": 0.42,
    }
    for env_id, expected_retreat in expected_retreats.items():
        eye, target, fov = rr_front_camera_parameters(env_id)
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        forward = (target - eye) / np.linalg.norm(target - eye)
        displacement = eye - default_eye
        np.testing.assert_allclose(forward, default_forward, atol=1e-7)
        np.testing.assert_allclose(
            displacement, -expected_retreat * default_forward, atol=1e-7
        )
        assert math.isclose(fov, RR_FRONT_FOV_RADIANS, abs_tol=1e-12)

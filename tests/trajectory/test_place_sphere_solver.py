from types import SimpleNamespace

import numpy as np

from mani_skill.examples.motionplanning.panda.solutions.place_sphere import (
    SPHERE_RELEASE_SETTLE_STEPS,
    _release_and_settle,
    _sphere_closing_axis,
    _sphere_grasp_center,
)


def test_sphere_grasp_center_uses_live_actor_pose_and_returns_a_copy():
    live_center = np.array([-0.08, 0.09, 0.02], dtype=np.float32)
    env = SimpleNamespace(
        obj=SimpleNamespace(pose=SimpleNamespace(sp=SimpleNamespace(p=live_center)))
    )

    center = _sphere_grasp_center(env)

    np.testing.assert_array_equal(center, live_center)
    assert center is not live_center


def test_release_waits_for_sphere_to_settle():
    class FakePlanner:
        def __init__(self):
            self.open_steps = None

        def open_gripper(self, *, t):
            self.open_steps = t
            return "result"

    planner = FakePlanner()

    assert _release_and_settle(planner) == "result"
    assert planner.open_steps == SPHERE_RELEASE_SETTLE_STEPS == 20


def test_sphere_closing_axis_is_horizontal_and_tcp_aligned():
    approaching = np.array([0.0, 0.0, -1.0])
    target = np.array([0.0, 0.8, 0.6])

    closing = _sphere_closing_axis(target, approaching)

    np.testing.assert_allclose(closing, [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(closing), 1.0, atol=1e-7)
    np.testing.assert_allclose(approaching @ closing, 0.0, atol=1e-7)

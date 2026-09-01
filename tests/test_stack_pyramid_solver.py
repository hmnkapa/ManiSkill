from __future__ import annotations

import numpy as np
import pytest

from mani_skill.examples.motionplanning.panda.solutions.stack_pyramid import (
    BASE_PAIR_TARGET_DISTANCE,
    _base_push_target,
    _move_to_pose,
    _perpendicular_xy,
    _yaw_candidates,
)


class _Planner:
    def __init__(self, screw_result, rrt_result):
        self.screw_result = screw_result
        self.rrt_result = rrt_result
        self.calls = []

    def move_to_pose_with_screw(self, pose, *, dry_run, refine_steps):
        self.calls.append(("screw", pose, dry_run, refine_steps))
        return self.screw_result

    def move_to_pose_with_RRTConnect(self, pose, *, dry_run, refine_steps):
        self.calls.append(("rrt", pose, dry_run, refine_steps))
        return self.rrt_result


def test_base_push_target_uses_fixed_spacing_and_preserves_height():
    moving = np.array([-0.08, 0.14, 0.02])
    target = np.array([0.06, -0.18, 0.021])

    result = _base_push_target(moving, target)

    assert np.linalg.norm(result[:2] - target[:2]) == pytest.approx(
        BASE_PAIR_TARGET_DISTANCE
    )
    assert result[2] == moving[2]
    original_side = moving[:2] - target[:2]
    final_side = result[:2] - target[:2]
    assert np.dot(original_side, final_side) > 0


def test_yaw_candidates_cover_full_turn_without_duplicate_zero():
    candidates = _yaw_candidates()

    assert len(candidates) == 8
    assert candidates[0] == 0
    assert len(set(candidates)) == len(candidates)
    assert set(np.round(np.asarray(candidates) / (np.pi / 4)).astype(int)) == {
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    }


def test_perpendicular_xy_is_horizontal_unit_vector():
    direction = np.array([3.0, -4.0, 7.0])

    result = _perpendicular_xy(direction)

    assert result[2] == 0
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert np.dot(result[:2], direction[:2]) == pytest.approx(0.0)


def test_move_to_pose_uses_screw_without_unnecessary_fallback():
    planner = _Planner(screw_result={"status": "Success"}, rrt_result={})
    pose = object()

    result = _move_to_pose(planner, pose, dry_run=True, refine_steps=3)

    assert result == {"status": "Success"}
    assert planner.calls == [("screw", pose, True, 3)]


def test_move_to_pose_falls_back_to_rrtconnect():
    planner = _Planner(screw_result=-1, rrt_result={"status": "Success"})
    pose = object()

    result = _move_to_pose(planner, pose, refine_steps=2)

    assert result == {"status": "Success"}
    assert planner.calls == [
        ("screw", pose, False, 2),
        ("rrt", pose, False, 2),
    ]

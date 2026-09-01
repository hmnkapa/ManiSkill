from __future__ import annotations

import importlib

import numpy as np
import sapien


solver = importlib.import_module(
    "mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side"
)


class _PoseHolder:
    def __init__(self, pose: sapien.Pose):
        self.sp = pose


class _Actor:
    def __init__(self, pose: sapien.Pose):
        self.pose = _PoseHolder(pose)


class _Tcp:
    def __init__(self, pose: sapien.Pose):
        self.pose = _PoseHolder(pose)


class _Agent:
    def __init__(self, tcp_pose: sapien.Pose):
        self.tcp = _Tcp(tcp_pose)


class _Env:
    def __init__(self, peg_pose: sapien.Pose, tcp_pose: sapien.Pose):
        self.peg = _Actor(peg_pose)
        self.agent = _Agent(tcp_pose)


def test_live_peg_grasp_geometry_tracks_reset_pose():
    peg_pose = sapien.Pose([0.2, -0.1, 0.03])
    env = _Env(peg_pose, sapien.Pose())

    center, closing = solver._live_peg_grasp_geometry(env)

    np.testing.assert_allclose(center, [0.14, -0.1, 0.03], atol=1e-7)
    np.testing.assert_allclose(np.abs(closing), [0.0, 1.0, 0.0], atol=1e-7)


def test_target_tcp_pose_preserves_current_object_to_tcp_transform():
    current_peg = sapien.Pose([0.1, -0.2, 0.03])
    current_tcp = sapien.Pose([0.12, -0.2, 0.10])
    desired_peg = sapien.Pose([0.4, 0.1, 0.2])
    env = _Env(current_peg, current_tcp)

    target = solver._target_tcp_pose(env, desired_peg)

    np.testing.assert_allclose(target.p, [0.42, 0.1, 0.27], atol=1e-7)


def test_desired_peg_pose_applies_hole_frame_insertion_bias():
    assert solver.INSERTION_Z_BIAS == 3.0e-3
    goal = sapien.Pose([0.2, -0.1, 0.3])

    target = solver._desired_peg_pose(goal, -0.12)

    np.testing.assert_allclose(target.p, [0.08, -0.1, 0.303], atol=1e-8)

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from mani_skill.examples.motionplanning.base_motionplanner import utils


class _Pose:
    def __init__(self, transform: np.ndarray):
        self._transform = transform

    def to_transformation_matrix(self):
        return self._transform


class _Object:
    def find_component_by_type(self, _component_type):
        return object()


class _Actor:
    def __init__(self, transform: np.ndarray):
        self.pose = _Pose(transform)
        self._objs = [_Object()]


def test_get_actor_obb_applies_live_actor_transform(monkeypatch):
    calls = []

    def local_mesh(_component, *, to_world_frame):
        calls.append(to_world_frame)
        return trimesh.creation.box(extents=[0.2, 0.1, 0.05])

    monkeypatch.setattr(utils, "get_component_mesh", local_mesh)
    transform = np.eye(4, dtype=np.float64)[None]
    transform[0, :3, 3] = [0.21, -0.13, 0.07]

    obb = utils.get_actor_obb(_Actor(transform))

    np.testing.assert_allclose(
        obb.primitive.transform[:3, 3], transform[0, :3, 3], atol=1e-8
    )
    assert calls == [False]


def test_actor_world_transform_rejects_multiple_scene_instances():
    transform = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)

    with pytest.raises(ValueError, match="expects one actor instance"):
        utils._actor_world_transform(_Actor(transform))

"""Task-specific part-pose definitions for pickle trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .schema import PART_POSE_DIM, PICK_CUBE_PART_NAMES, PickleEnv


@dataclass(frozen=True)
class PartPoseSource:
    """One serialized task entity and the attribute exposing its actor."""

    name: str
    env_attribute: str


@dataclass(frozen=True)
class PickleTaskSpec:
    """Immutable per-task contract for the order of ``parts_poses``."""

    env: PickleEnv
    parts: tuple[PartPoseSource, ...]

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(part.name for part in self.parts)

    @property
    def part_attributes(self) -> tuple[str, ...]:
        return tuple(part.env_attribute for part in self.parts)

    @property
    def parts_pose_dim(self) -> int:
        return len(self.parts) * PART_POSE_DIM


def _part(name: str, env_attribute: str | None = None) -> PartPoseSource:
    return PartPoseSource(name=name, env_attribute=env_attribute or name)


_TASK_SPECS = {
    PickleEnv.LIFT_PEG_UPRIGHT: PickleTaskSpec(
        env=PickleEnv.LIFT_PEG_UPRIGHT,
        parts=(_part("peg"),),
    ),
    PickleEnv.PEG_INSERTION_SIDE: PickleTaskSpec(
        env=PickleEnv.PEG_INSERTION_SIDE,
        parts=(_part("peg"), _part("box_with_hole", "box")),
    ),
    PickleEnv.PICK_CUBE: PickleTaskSpec(
        env=PickleEnv.PICK_CUBE,
        parts=tuple(_part(name) for name in PICK_CUBE_PART_NAMES),
    ),
    PickleEnv.PLACE_SPHERE: PickleTaskSpec(
        env=PickleEnv.PLACE_SPHERE,
        parts=(_part("sphere", "obj"), _part("bin")),
    ),
    PickleEnv.PLUG_CHARGER: PickleTaskSpec(
        env=PickleEnv.PLUG_CHARGER,
        parts=(_part("charger"), _part("receptacle")),
    ),
    PickleEnv.POKE_CUBE: PickleTaskSpec(
        env=PickleEnv.POKE_CUBE,
        parts=(_part("peg"), _part("cube"), _part("goal_region")),
    ),
    PickleEnv.PULL_CUBE: PickleTaskSpec(
        env=PickleEnv.PULL_CUBE,
        parts=(_part("cube", "obj"), _part("goal_region")),
    ),
    PickleEnv.PULL_CUBE_TOOL: PickleTaskSpec(
        env=PickleEnv.PULL_CUBE_TOOL,
        parts=(_part("l_shape_tool"), _part("cube")),
    ),
    PickleEnv.PUSH_CUBE: PickleTaskSpec(
        env=PickleEnv.PUSH_CUBE,
        parts=(_part("cube", "obj"), _part("goal_region")),
    ),
    PickleEnv.STACK_CUBE: PickleTaskSpec(
        env=PickleEnv.STACK_CUBE,
        parts=(_part("cubeA"), _part("cubeB")),
    ),
    PickleEnv.STACK_PYRAMID: PickleTaskSpec(
        env=PickleEnv.STACK_PYRAMID,
        parts=(_part("cubeA"), _part("cubeB"), _part("cubeC")),
    ),
}

PICKLE_TASK_SPECS: Mapping[PickleEnv, PickleTaskSpec] = MappingProxyType(
    _TASK_SPECS
)
"""Read-only mapping from environment ID to its part-pose contract."""


def get_pickle_task_spec(env: PickleEnv | str) -> PickleTaskSpec:
    """Resolve one supported task or raise a consistent compatibility error."""

    try:
        env_id = PickleEnv(env)
    except (TypeError, ValueError) as error:
        raise NotImplementedError(
            f"Unsupported pickle environment: {env!r}"
        ) from error
    return PICKLE_TASK_SPECS[env_id]


__all__ = [
    "PICKLE_TASK_SPECS",
    "PartPoseSource",
    "PickleTaskSpec",
    "get_pickle_task_spec",
]

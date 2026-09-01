"""FurnitureBench-aligned camera settings for RR pickle collection."""

from __future__ import annotations

import math

from mani_skill.utils.sapien_utils import look_at


RR_FRONT_EYE_WORLD = (0.585, 0.0, 0.235)
RR_FRONT_TARGET_WORLD = (-1.315, 0.0, -0.115)
RR_FRONT_FOV_RADIANS = math.radians(40.0)

# User-confirmed 2026-08-31 retreat-only views.  Each proposal preserves the
# default optical axis and 40-degree FOV; tasks not listed here retain the
# FurnitureBench-aligned default above.
RR_FRONT_TASK_OVERRIDES = {
    "PegInsertionSide-v1": {
        "eye_world": (1.2144098281860352, -7.708048789106396e-17, 0.350943922996521),
        "target_world": (
            -0.6855901917679821,
            -7.708048789106396e-17,
            0.0009440313183419202,
        ),
        "fov_radians": math.radians(40.0),
    },
    "PokeCube-v1": {
        "eye_world": (0.9980502128601074, -5.058408299981022e-17, 0.3110882043838501),
        "target_world": (-0.9019498070939098, -5.058408299981022e-17, -0.03891168729432898),
        "fov_radians": math.radians(40.0),
    },
    "PullCubeTool-v1": {
        "eye_world": (0.9980502128601074, -5.058408299981022e-17, 0.3110882043838501),
        "target_world": (-0.9019498070939098, -5.058408299981022e-17, -0.03891168729432898),
        "fov_radians": math.radians(40.0),
    },
}


def rr_front_camera_parameters(
    env_id: str | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Resolve one task's front-camera eye, target, and vertical FOV."""

    override = RR_FRONT_TASK_OVERRIDES.get(env_id)
    if override is None:
        return RR_FRONT_EYE_WORLD, RR_FRONT_TARGET_WORLD, RR_FRONT_FOV_RADIANS
    return (
        override["eye_world"],
        override["target_world"],
        override["fov_radians"],
    )


def rr_aligned_sensor_overrides(
    shader_pack: str = "minimal", env_id: str | None = None
) -> dict:
    """Return overrides for 224px wrist and FB-relative front cameras.

    All supported Panda tabletop collectors place the robot base at
    ``(-0.615, 0, 0)``. The fixed camera eye and target preserve FB's relative
    transforms from that base. The camera-specific override is required:
    changing global width/height alone leaves each task's pose and FOV intact.
    """

    eye_world, target_world, fov_radians = rr_front_camera_parameters(env_id)
    front_pose = look_at(eye_world, target_world)
    raw_pose = front_pose.raw_pose[0].detach().cpu().tolist()
    return {
        "width": 224,
        "height": 224,
        "shader_pack": shader_pack,
        "base_camera": {
            "pose": raw_pose,
            "width": 224,
            "height": 224,
            "fov": fov_radians,
            "near": 0.001,
            "far": 2.0,
        },
    }


__all__ = [
    "RR_FRONT_EYE_WORLD",
    "RR_FRONT_FOV_RADIANS",
    "RR_FRONT_TASK_OVERRIDES",
    "RR_FRONT_TARGET_WORLD",
    "rr_aligned_sensor_overrides",
    "rr_front_camera_parameters",
]

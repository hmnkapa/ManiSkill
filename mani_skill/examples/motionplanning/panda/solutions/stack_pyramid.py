"""Motion-planning solution for :class:`StackPyramidEnv`."""

from __future__ import annotations

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackPyramidEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


APPROACH_CLEARANCE = 0.05
# A closed Panda gripper pushes from above and the cube trails its commanded
# TCP center by roughly 1.5--3 cm. Aim 1 cm inside the nominal 4 cm contact
# distance; collision response then settles the two cubes next to each other.
# The task accepts up to about 6.16 cm, and 6 cm leaves a small numeric margin.
BASE_PAIR_TARGET_DISTANCE = 0.010
BASE_PAIR_READY_DISTANCE = 0.060


def _position(actor) -> np.ndarray:
    """Return an actor's unbatched xyz position as a copied float array."""

    return np.asarray(actor.pose.sp.p, dtype=np.float64).copy()


def _base_push_target(
    moving_position: np.ndarray,
    target_position: np.ndarray,
    *,
    target_distance: float = BASE_PAIR_TARGET_DISTANCE,
) -> np.ndarray:
    """Place the moving cube on its current side of the fixed base cube."""

    moving_position = np.asarray(moving_position, dtype=np.float64)
    target_position = np.asarray(target_position, dtype=np.float64)
    direction = moving_position[:2] - target_position[:2]
    distance = float(np.linalg.norm(direction))
    if distance <= 1.0e-8:
        direction = np.array([1.0, 0.0], dtype=np.float64)
    else:
        direction /= distance
    result = target_position.copy()
    result[:2] += target_distance * direction
    result[2] = moving_position[2]
    return result


def _yaw_candidates() -> tuple[float, ...]:
    """Return deterministic, alternating yaw offsets covering a full turn."""

    quarter_turn = np.pi / 4
    return tuple(
        quarter_turn * index
        for index in (0, 1, -1, 2, -2, 3, -3, 4)
    )


def _perpendicular_xy(direction: np.ndarray) -> np.ndarray:
    """Return a horizontal unit vector perpendicular to ``direction``."""

    direction = np.asarray(direction, dtype=np.float64)
    perpendicular = np.array([-direction[1], direction[0], 0.0])
    norm = float(np.linalg.norm(perpendicular))
    if norm <= 1.0e-8:
        return np.array([0.0, 1.0, 0.0])
    return perpendicular / norm


def _move_to_pose(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    *,
    dry_run: bool = False,
    refine_steps: int = 0,
):
    """Prefer a straight screw motion, then fall back to RRTConnect."""

    result = planner.move_to_pose_with_screw(
        pose, dry_run=dry_run, refine_steps=refine_steps
    )
    if result == -1:
        result = planner.move_to_pose_with_RRTConnect(
            pose, dry_run=dry_run, refine_steps=refine_steps
        )
    return result


def _topdown_grasp_pose(
    planner: PandaArmMotionPlanningSolver,
    env: StackPyramidEnv,
    actor,
) -> sapien.Pose | None:
    """Find a reachable top-down grasp, testing its collision-free approach."""

    approaching = np.array([0.0, 0.0, -1.0])
    # Keep the fingers tangent to the nearby A/B support pair. If a finger
    # closes toward either support cube, it can shove cube C sideways before a
    # stable two-sided contact is established.
    base_midpoint = (_position(env.cubeA) + _position(env.cubeB)) / 2
    target_closing = _perpendicular_xy(
        base_midpoint - _position(actor)
    )
    base_pose = env.agent.build_grasp_pose(
        approaching, target_closing, _position(actor)
    )
    for angle in _yaw_candidates():
        candidate = base_pose * sapien.Pose(q=euler2quat(0, 0, angle))
        approach_pose = candidate * sapien.Pose([0, 0, -APPROACH_CLEARANCE])
        if _move_to_pose(planner, approach_pose, dry_run=True) != -1:
            return candidate
    return None


def _base_pair_distance(env: StackPyramidEnv) -> float:
    return float(
        np.linalg.norm((_position(env.cubeA) - _position(env.cubeB))[:2])
    )


def solve(env: StackPyramidEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    env = env.unwrapped

    # The first stage deliberately uses the closed gripper as a pusher. The
    # previous implementation called this a grasp, but never actually grasped
    # cube A; its absolute ``0.8 * cubeB`` target made the final A/B spacing
    # depend on where cube B happened to spawn.
    if _base_pair_distance(env) > BASE_PAIR_READY_DISTANCE:
        planner.close_gripper()
        approaching = np.array([0.0, 0.0, -1.0])
        push_direction = _position(env.cubeB) - _position(env.cubeA)
        # A closed gripper works as a straight pusher only when the line between
        # its fingers is perpendicular to the desired push direction.
        target_closing = _perpendicular_xy(push_direction)
        push_pose = env.agent.build_grasp_pose(
            approaching, target_closing, _position(env.cubeA)
        )
        approach_pose = push_pose * sapien.Pose([0, 0, -APPROACH_CLEARANCE])
        if _move_to_pose(planner, approach_pose) == -1:
            planner.close()
            return -1
        if _move_to_pose(planner, push_pose) == -1:
            planner.close()
            return -1

        moving_position = _position(env.cubeA)
        desired_position = _base_push_target(
            moving_position, _position(env.cubeB)
        )
        push_delta = desired_position - moving_position
        target_push_pose = sapien.Pose(push_pose.p + push_delta, push_pose.q)
        if _move_to_pose(planner, target_push_pose, refine_steps=2) == -1:
            planner.close()
            return -1
        # Opening here lets the cubes settle and advances the scripted
        # annotation FSM from push_base to pick_top before approaching cube C.
        planner.open_gripper()
        if _base_pair_distance(env) > BASE_PAIR_READY_DISTANCE:
            print(
                "Base-pair push missed target: "
                f"distance={_base_pair_distance(env):.6f}"
            )
            planner.close()
            return -1

    # Pick cube C using a reachable top-down approach. Test the approach pose,
    # not the table-level grasp pose, and fail instead of executing the final
    # invalid candidate.
    grasp_pose = _topdown_grasp_pose(planner, env, env.cubeC)
    if grasp_pose is None:
        print("Fail to find a valid grasp pose")
        planner.close()
        return -1

    reach_pose = grasp_pose * sapien.Pose([0, 0, -APPROACH_CLEARANCE])
    if _move_to_pose(planner, reach_pose) == -1:
        planner.close()
        return -1
    if _move_to_pose(planner, grasp_pose, refine_steps=2) == -1:
        planner.close()
        return -1
    # Contact while descending can nudge a small cube a few millimetres. Recenter
    # before closing so both fingers contact it symmetrically.
    recenter_delta = _position(env.cubeC) - np.asarray(grasp_pose.p)
    if np.linalg.norm(recenter_delta[:2]) > 1.0e-4:
        grasp_pose = sapien.Pose(grasp_pose.p + recenter_delta, grasp_pose.q)
        if _move_to_pose(planner, grasp_pose, refine_steps=2) == -1:
            planner.close()
            return -1
    planner.close_gripper(t=10)
    if not bool(env.agent.is_grasping(env.cubeC).item()):
        print("Failed to grasp cube C")
        planner.close()
        return -1

    # Verify the grasp after a short lift before committing to the full
    # transfer. A contact-only ``is_grasping`` result can disappear as soon as
    # the object leaves the table.
    check_lift_pose = sapien.Pose([0, 0, 0.04]) * grasp_pose
    if _move_to_pose(planner, check_lift_pose) == -1:
        planner.close()
        return -1
    if not bool(env.agent.is_grasping(env.cubeC).item()):
        print("Lost cube C during lift")
        planner.close()
        return -1

    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    if _move_to_pose(planner, lift_pose) == -1:
        planner.close()
        return -1

    cube_height = float((env.cube_half_size[2] * 2).item())
    goal_pose_A = env.cubeA.pose * sapien.Pose([0, 0, cube_height])
    goal_pose_B = env.cubeB.pose * sapien.Pose([0, 0, cube_height])
    goal_position = ((goal_pose_A.p + goal_pose_B.p) / 2).cpu().numpy()[0]
    offset = goal_position - _position(env.cubeC)
    align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    if _move_to_pose(planner, align_pose, refine_steps=2) == -1:
        planner.close()
        return -1

    # A few extra open-gripper steps let cube C settle before evaluate() is
    # sampled; this stays comfortably inside the task's 250-step horizon.
    res = planner.open_gripper(t=10)
    planner.close()
    return res

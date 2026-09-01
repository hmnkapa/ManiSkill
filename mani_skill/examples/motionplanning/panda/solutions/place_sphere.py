import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlaceSphereEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils import common


SPHERE_RELEASE_SETTLE_STEPS = 20


def _sphere_grasp_center(env: PlaceSphereEnv) -> np.ndarray:
    """Return the reset-time sphere center from the live actor pose.

    The GPU backend does not refresh the world transform used by
    ``get_actor_obb`` after the episode reset, so its OBB center can remain at
    the actor-builder origin.  A sphere is rotationally symmetric; its live
    actor position is the exact grasp center needed by this solver.
    """

    return env.obj.pose.sp.p.copy()


def _release_and_settle(planner: PandaArmMotionPlanningSolver):
    """Open the gripper while allowing the sphere to reach static success."""

    return planner.open_gripper(t=SPHERE_RELEASE_SETTLE_STEPS)


def _sphere_closing_axis(
    target_closing: np.ndarray, approaching: np.ndarray
) -> np.ndarray:
    """Use a stable TCP-aligned horizontal axis for a symmetric sphere."""

    closing = target_closing - (approaching @ target_closing) * approaching
    return closing / np.linalg.norm(closing)


def solve(env: PlaceSphereEnv, seed=None, debug=False, vis=False):
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

    approaching = np.array([0, 0, -1])
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    # A sphere has no meaningful OBB closing axis.  Reusing the current TCP
    # closing direction avoids arbitrary axis flips from an equivalent OBB.
    closing = _sphere_closing_axis(target_closing, approaching)
    grasp_pose = env.agent.build_grasp_pose(
        approaching, closing, _sphere_grasp_center(env)
    )

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2) + np.pi / 4
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        planner.close()
        return res

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        planner.close()
        return res
    planner.close_gripper()
    if not bool(env.agent.is_grasping(env.obj).item()):
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        planner.close()
        return res
    if not bool(env.agent.is_grasping(env.obj).item()):
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Stack
    # -------------------------------------------------------------------------- #
    block_half_size_torch = common.to_tensor(env.block_half_size)
    goal_pose = env.bin.pose * sapien.Pose(
        [0, 0, (block_half_size_torch[2] * 2).item()]
    )
    # ManiSkill actor poses are batched torch tensors, even with num_envs=1.
    offset = (goal_pose.p - env.obj.pose.p).cpu().numpy()[0]
    # Keep the sphere high while translating over the bin, then lower it.  A
    # single diagonal screw motion can fail IK for otherwise reachable resets.
    transfer_pose = sapien.Pose(
        lift_pose.p + np.array([offset[0], offset[1], 0.0]), lift_pose.q
    )
    res = planner.move_to_pose_with_screw(transfer_pose)
    if res == -1:
        planner.close()
        return res
    align_pose = sapien.Pose(
        transfer_pose.p + np.array([0.0, 0.0, offset[2]]), lift_pose.q
    )
    res = planner.move_to_pose_with_screw(align_pose)
    if res == -1:
        planner.close()
        return res

    # Six controller steps open the fingers but can return before the sphere is
    # static, which makes the final sparse-success observation false.  Keep the
    # same open command active long enough for the released sphere to settle.
    res = _release_and_settle(planner)
    planner.close()
    return res

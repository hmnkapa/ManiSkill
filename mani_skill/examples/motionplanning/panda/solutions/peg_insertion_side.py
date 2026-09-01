import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


GRASP_OFFSET_ALONG_PEG = -0.06
INSERTION_Z_BIAS = 3.0e-3


def _live_peg_grasp_geometry(env: PegInsertionSideEnv):
    """Return a top-down grasp center/closing axis from the reset actor pose.

    The GPU backend can leave the world transform consumed by ``get_actor_obb``
    at the actor-builder pose.  PegInsertionSide repositions the peg on every
    reset, so the bundled OBB-based solver can otherwise move the TCP to an
    unrelated point and never grasp the peg.
    """

    peg_pose = env.peg.pose.sp
    grasp_center = (peg_pose * sapien.Pose([GRASP_OFFSET_ALONG_PEG, 0, 0])).p
    closing = peg_pose.to_transformation_matrix()[:3, 1]
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    if target_closing @ closing < 0:
        closing = -closing
    return grasp_center, closing


def _target_tcp_pose(env: PegInsertionSideEnv, desired_peg_pose: sapien.Pose):
    """Map a desired live peg pose to the TCP target for the current grasp."""

    return desired_peg_pose * env.peg.pose.sp.inv() * env.agent.tcp.pose.sp


def _desired_peg_pose(
    goal_pose: sapien.Pose, x_offset: float
) -> sapien.Pose:
    """Return a hole-frame target with measured vertical contact compensation."""

    return goal_pose * sapien.Pose([x_offset, 0.0, INSERTION_Z_BIAS])


def main():
    env: PegInsertionSideEnv = gym.make(
        "PegInsertionSide-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(env: PegInsertionSideEnv, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )
    env = env.unwrapped
    approaching = np.array([0, 0, -1])
    center, closing = _live_peg_grasp_geometry(env)
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
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
    res = planner.close_gripper()
    if not bool(env.agent.is_grasping(env.peg, max_angle=20).item()):
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Align Peg
    # -------------------------------------------------------------------------- #

    # align the peg with the hole
    offset = sapien.Pose([-0.01 - env.peg_half_sizes[0, 0].item(), 0, 0])
    desired_pre_insert_pose = _desired_peg_pose(env.goal_pose.sp, offset.p[0])
    pre_insert_pose = _target_tcp_pose(env, desired_pre_insert_pose)
    res = planner.move_to_pose_with_screw(pre_insert_pose)
    if res == -1:
        planner.close()
        return res
    # refine the insertion pose
    for i in range(3):
        pre_insert_pose = _target_tcp_pose(env, desired_pre_insert_pose)
        res = planner.move_to_pose_with_screw(pre_insert_pose)
        if res == -1:
            planner.close()
            return res

    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    desired_insert_pose = _desired_peg_pose(env.goal_pose.sp, 0.05)
    insert_pose = _target_tcp_pose(env, desired_insert_pose)
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1:
        planner.close()
        return res
    planner.close()
    return res


if __name__ == "__main__":
    main()

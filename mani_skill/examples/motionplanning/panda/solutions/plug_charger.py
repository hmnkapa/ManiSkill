import gymnasium as gym
import numpy as np
import sapien.core as sapien
import trimesh
from tqdm import tqdm
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import PlugChargerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


INSERTION_Z_BIAS = 5.0e-4


def _desired_charger_pose(goal_pose: sapien.Pose, x_offset: float) -> sapien.Pose:
    """Return the socket-frame object target with calibrated vertical bias."""

    return goal_pose * sapien.Pose([x_offset, 0.0, INSERTION_Z_BIAS])


def main():
    env: PlugChargerEnv = gym.make(
        "PlugCharger-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
    )
    for seed in tqdm(range(100)):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(env: PlugChargerEnv, seed=None, debug=False, vis=False):
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
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    charger_base_pose = env.charger_base_pose
    charger_base_size = np.array(env.unwrapped._base_size) * 2

    obb = trimesh.primitives.Box(
        extents=charger_base_size,
        transform=charger_base_pose.sp.to_transformation_matrix(),
    )

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # add a angle to grasp
    grasp_angle = np.deg2rad(15)
    grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_angle, 0))

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
    if not bool(env.agent.is_grasping(env.charger, max_angle=20).item()):
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Align
    # -------------------------------------------------------------------------- #
    pre_insert_pose = (
        _desired_charger_pose(env.goal_pose.sp, -0.05)
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    # The collision aperture has only about 0.5 mm one-sided clearance.  A
    # paired 60-seed GPU test preserved all 7 baseline successes and improved
    # the total from 7/60 to 11/60 with this socket-frame z compensation.
    insert_pose = (
        _desired_charger_pose(env.goal_pose.sp, 0.0)
        * env.charger.pose.sp.inv()
        * env.agent.tcp.pose.sp
    )
    res = planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
    if res == -1:
        planner.close()
        return res
    res = planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)
    if res == -1:
        planner.close()
        return res
    if not bool(env.agent.is_grasping(env.charger, max_angle=20).item()):
        planner.close()
        return -1
    # -------------------------------------------------------------------------- #
    # Insert
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1:
        planner.close()
        return res

    planner.close()
    return res


if __name__ == "__main__":
    main()

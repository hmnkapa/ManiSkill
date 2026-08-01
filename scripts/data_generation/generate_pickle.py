#!/usr/bin/env python3
"""Generate RR-compatible task pickles with Panda motion planners."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401 - register ManiSkill Gym environments
from mani_skill.examples.motionplanning.panda.solutions import (
    solveLiftPegUpright,
    solvePegInsertionSide,
    solvePickCube,
    solvePlaceSphere,
    solvePlugCharger,
    solvePullCube,
    solvePullCubeTool,
    solvePushCube,
    solveStackCube,
    solveStackPyramid,
)
from mani_skill.trajectory.pickle import PickleEnv
from mani_skill.utils.wrappers import RecordPickle


MOTION_PLANNING_SOLVERS = {
    PickleEnv.LIFT_PEG_UPRIGHT.value: solveLiftPegUpright,
    PickleEnv.PEG_INSERTION_SIDE.value: solvePegInsertionSide,
    PickleEnv.PICK_CUBE.value: solvePickCube,
    PickleEnv.PLACE_SPHERE.value: solvePlaceSphere,
    PickleEnv.PLUG_CHARGER.value: solvePlugCharger,
    PickleEnv.PULL_CUBE.value: solvePullCube,
    PickleEnv.PULL_CUBE_TOOL.value: solvePullCubeTool,
    PickleEnv.PUSH_CUBE.value: solvePushCube,
    PickleEnv.STACK_CUBE.value: solveStackCube,
    PickleEnv.STACK_PYRAMID.value: solveStackPyramid,
}


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Panda motion-planning trajectories in RR pickle format."
    )
    parser.add_argument(
        "-e",
        "--env-id",
        choices=tuple(env.value for env in PickleEnv),
        default=PickleEnv.PICK_CUBE.value,
        help="Task to record. PokeCube-v1 is recordable but has no bundled solver.",
    )
    parser.add_argument(
        "--num-traj",
        type=int,
        default=10,
        help="Number of successful pickle trajectories to save.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("demos"),
        help="Root output directory.",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="auto",
        help="ManiSkill simulation backend, e.g. auto, cpu, or gpu.",
    )
    parser.add_argument(
        "--shader",
        type=str,
        default="minimal",
        help="Sensor shader pack (minimal, default, rt-fast, rt-med, or rt).",
    )
    parser.add_argument("--vis", action="store_true", help="Open the live viewer.")
    parser.add_argument(
        "--compress", action="store_true", help="Write .pkl.xz with LZMA."
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Optional cap on successful plus failed planning attempts.",
    )
    parsed = parser.parse_args(args)
    if parsed.num_traj <= 0:
        parser.error("--num-traj must be positive")
    if parsed.max_attempts is not None and parsed.max_attempts <= 0:
        parser.error("--max-attempts must be positive when provided")
    if parsed.env_id not in MOTION_PLANNING_SOLVERS:
        parser.error(
            f"{parsed.env_id} supports RecordPickle, but no Panda motion-planning "
            "solver is available; use RecordPickle with another rollout source"
        )
    return parsed


def _one_bool(value: Any) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one boolean value, got shape {array.shape}")
    return bool(array[0])


def _planning_success(result: Any) -> bool:
    if result is None or isinstance(result, (int, np.integer)):
        return False
    if not isinstance(result, (tuple, list)) or not result:
        return False
    info = result[-1]
    if not isinstance(info, dict) or "success" not in info:
        return False
    return _one_bool(info["success"])


def generate(args: argparse.Namespace) -> list[Path]:
    env_id = str(args.env_id)
    if env_id not in MOTION_PLANNING_SOLVERS:
        raise NotImplementedError(
            f"{env_id} supports RecordPickle, but no Panda motion-planning "
            "solver is available; use RecordPickle with another rollout source"
        )
    solve = MOTION_PLANNING_SOLVERS[env_id]
    output_dir = args.record_dir / env_id / "motionplanning_pickle"
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_pos",
        robot_uids="panda_wristcam",
        reward_mode="sparse",
        render_mode="human" if args.vis else "sensors",
        sim_backend=args.sim_backend,
        sensor_configs={
            "width": 224,
            "height": 224,
            "shader_pack": args.shader,
        },
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={"shader_pack": args.shader},
    )
    recorder = RecordPickle(env, output_dir=output_dir, compress=args.compress)
    saved: list[Path] = []
    seed = int(args.start_seed)
    attempts = 0
    try:
        while len(saved) < args.num_traj:
            if args.max_attempts is not None and attempts >= args.max_attempts:
                raise RuntimeError(
                    f"Reached --max-attempts={args.max_attempts} after saving "
                    f"{len(saved)}/{args.num_traj} successful trajectories"
                )
            attempts += 1
            try:
                result = solve(recorder, seed=seed, debug=False, vis=bool(args.vis))
                success = _planning_success(result)
            except Exception as error:
                success = False
                print(
                    f"[attempt {attempts}, seed {seed}] planning failed: "
                    f"{type(error).__name__}: {error}"
                )

            if success:
                path = recorder.flush_episode(success=True)
                saved.append(path)
                print(
                    f"[{len(saved)}/{args.num_traj}] saved seed {seed}: {path}"
                )
            else:
                recorder.discard_episode()
                print(f"[attempt {attempts}, seed {seed}] discarded unsuccessful rollout")
            seed += 1
    finally:
        recorder.close()
    return saved


def main(args=None) -> int:
    parsed = parse_args(args)
    saved = generate(parsed)
    print(
        f"Generated {len(saved)} successful {parsed.env_id} pickle trajectories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

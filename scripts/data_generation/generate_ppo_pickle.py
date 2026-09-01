#!/usr/bin/env python3
"""Roll out official state PPO checkpoints into RR-compatible RGB-D pickles."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import mani_skill.envs  # noqa: F401 - register ManiSkill Gym environments
from mani_skill.trajectory.pickle import (
    TrajectoryValidationError,
    rr_aligned_sensor_overrides,
)
from mani_skill.utils.wrappers import RecordPickle


PPO_PICKLE_ENVS = (
    "LiftPegUpright-v1",
    "PickCube-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PushCube-v1",
    "StackCube-v1",
)


class PPOAgent(nn.Module):
    """Exact network topology used by ``examples/baselines/ppo/ppo_fast.py``."""

    def __init__(self, n_obs: int, n_act: int, device: torch.device):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 1, device=device),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(n_obs, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, 256, device=device),
            nn.Tanh(),
            nn.Linear(256, n_act, device=device),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RR pickles from official ManiSkill PPO checkpoints."
    )
    parser.add_argument("--env-id", choices=PPO_PICKLE_ENVS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--annotation-source",
        choices=("scripted",),
        required=True,
        help="Required provenance gate; task annotations are geometry GT.",
    )
    parser.add_argument("--num-traj", type=int, default=100)
    parser.add_argument("--record-dir", type=Path, default=Path("demos"))
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-eval-steps", type=int, default=400)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--shader", default="minimal")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument(
        "--diagnostic-task-successes",
        type=int,
        help=(
            "stop after this many task-success rollouts, regardless of strict "
            "pickle acceptance; intended for visibility diagnostics"
        ),
    )
    parser.add_argument(
        "--attempt-diagnostics-dir",
        type=Path,
        help="write one pre-validation geometry audit JSON per task-success rollout",
    )
    parsed = parser.parse_args(args)
    if parsed.num_traj <= 0:
        parser.error("--num-traj must be positive")
    if parsed.num_eval_steps <= 0:
        parser.error("--num-eval-steps must be positive")
    if parsed.max_attempts is not None and parsed.max_attempts <= 0:
        parser.error("--max-attempts must be positive when provided")
    if (
        parsed.diagnostic_task_successes is not None
        and parsed.diagnostic_task_successes <= 0
    ):
        parser.error("--diagnostic-task-successes must be positive when provided")
    if (parsed.diagnostic_task_successes is None) != (
        parsed.attempt_diagnostics_dir is None
    ):
        parser.error(
            "--diagnostic-task-successes and --attempt-diagnostics-dir "
            "must be supplied together"
        )
    return parsed


def _one_bool(value: Any, name: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected one {name} value, got shape {array.shape}")
    return bool(array[0])


def _policy_state(observation: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(observation, Mapping) or "state" not in observation:
        raise KeyError("RGB-D plus state observation is missing the PPO state vector")
    state = torch.as_tensor(observation["state"], device=device, dtype=torch.float32)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2 or state.shape[0] != 1:
        raise ValueError(f"Expected one PPO state row, got shape {tuple(state.shape)}")
    if not torch.isfinite(state).all():
        raise ValueError("PPO state contains a non-finite value")
    return state


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _front_point_audit(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    camera = trajectory["camera_info"]["front_camera"]
    image_size = np.asarray(camera["image_size"], dtype=np.int64)
    intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
    base_to_camera_rr = np.asarray(camera["sim_local_to_camera"], dtype=np.float64)
    observations = []
    for index, observation in enumerate(trajectory["observations"]):
        skill = observation.get("skill")
        point = observation.get("guidance_point_clean")
        if point is None:
            point = observation.get("guidance_point")
        recorded_uv = observation.get("guidance_point_2d", {}).get("color_image2")
        record: dict[str, Any] = {
            "index": index,
            "skill": skill,
            "point_base": None,
            "recorded_uv": None if recorded_uv is None else _json_value(recorded_uv),
            "reprojected_uv": None,
            "camera_depth": None,
            "geometrically_visible": False,
        }
        if point is not None:
            point_base = np.asarray(point, dtype=np.float64)
            point_camera = base_to_camera_rr @ np.r_[point_base, 1.0]
            depth = float(point_camera[2])
            record["point_base"] = point_base.tolist()
            record["camera_depth"] = depth
            if np.isfinite(point_camera).all() and depth > 1e-8:
                # RR camera coordinates use +Y up; image pixels use +V down.
                u = intrinsic[0, 0] * point_camera[0] / depth + intrinsic[0, 2]
                v = intrinsic[1, 2] - intrinsic[1, 1] * point_camera[1] / depth
                uv = np.asarray([u, v], dtype=np.float64)
                record["reprojected_uv"] = uv.tolist()
                record["geometrically_visible"] = bool(
                    np.isfinite(uv).all()
                    and 0 <= u < image_size[0]
                    and 0 <= v < image_size[1]
                )
        observations.append(record)
    active = [record for record in observations if record["skill"] is not None]
    return {
        "camera": {
            "image_size": image_size.tolist(),
            "intrinsics": intrinsic.tolist(),
            "camera_to_sim_local": _json_value(camera["camera_to_sim_local"]),
            "sim_local_to_camera": base_to_camera_rr.tolist(),
        },
        "active_observations": len(active),
        "active_geometrically_visible": sum(
            bool(record["geometrically_visible"]) for record in active
        ),
        "active_recorded_visible": sum(
            record["recorded_uv"] is not None for record in active
        ),
        "observations": observations,
    }


def _write_attempt_diagnostic(
    directory: Path,
    *,
    task: str,
    seed: int,
    attempt: int,
    task_success_index: int,
    trajectory: Mapping[str, Any],
    validation_error: str | None,
) -> Path:
    audit = _front_point_audit(trajectory)
    audit.update(
        {
            "schema": "rr-maniskill-prevalidation-visibility-v1",
            "task": task,
            "seed": seed,
            "attempt": attempt,
            "task_success_index": task_success_index,
            "task_success": True,
            "strict_pass": validation_error is None,
            "validation_error": validation_error,
            "annotation_source": trajectory.get("annotation_source"),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"task-success-{task_success_index:03d}-seed-{seed}.json"
    if output.exists():
        raise FileExistsError(f"refusing existing diagnostic: {output}")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def generate(args: argparse.Namespace) -> list[Path]:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("PPO pickle collection requires a CUDA GPU")
    device = torch.device("cuda")
    random.seed(args.start_seed)
    np.random.seed(args.start_seed)
    torch.manual_seed(args.start_seed)
    torch.cuda.manual_seed_all(args.start_seed)
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.record_dir) / args.env_id / "ppo_pickle"
    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="rgb+depth+state",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda_wristcam",
        reward_mode="sparse",
        render_mode="sensors",
        sim_backend="physx_cuda",
        sensor_configs=rr_aligned_sensor_overrides(args.shader, args.env_id),
    )
    recorder = RecordPickle(env, output_dir=output_dir, compress=args.compress)
    saved: list[Path] = []
    task_successes = 0
    attempts = 0
    seed = int(args.start_seed)
    agent = None
    try:
        while (
            task_successes < args.diagnostic_task_successes
            if args.diagnostic_task_successes is not None
            else len(saved) < args.num_traj
        ):
            if args.max_attempts is not None and attempts >= args.max_attempts:
                raise RuntimeError(
                    f"Reached --max-attempts={args.max_attempts} after saving "
                    f"{len(saved)}/{args.num_traj} successful trajectories"
                )
            attempts += 1
            observation, _ = recorder.reset(seed=seed)
            state = _policy_state(observation, device)
            if agent is None:
                single_action_space = recorder.unwrapped.single_action_space
                n_act = math.prod(single_action_space.shape)
                agent = PPOAgent(state.shape[1], n_act, device)
                state_dict = torch.load(
                    checkpoint, map_location=device, weights_only=True
                )
                agent.load_state_dict(state_dict, strict=True)
                agent.eval()

            success = False
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    native_action = agent.actor_mean(state)
                observation, _, terminated, truncated, info = recorder.step(
                    native_action
                )
                success = _one_bool(info["success"], "success")
                if success:
                    task_successes += 1
                    diagnostic_trajectory = None
                    if args.attempt_diagnostics_dir is not None:
                        diagnostic_trajectory = recorder.buffer.finalize(
                            success=True, task=args.env_id
                        )
                    validation_error = None
                    try:
                        path = recorder.flush_episode(success=True)
                    except TrajectoryValidationError as error:
                        validation_error = str(error)
                        success = False
                        print(
                            f"[attempt {attempts}, seed {seed}] rejected by strict "
                            f"training-point gate: {error}",
                            flush=True,
                        )
                    else:
                        saved.append(path)
                        print(
                            f"[{len(saved)}/{args.num_traj}] saved seed {seed}: {path}",
                            flush=True,
                        )
                    if args.attempt_diagnostics_dir is not None:
                        diagnostic_path = _write_attempt_diagnostic(
                            args.attempt_diagnostics_dir,
                            task=args.env_id,
                            seed=seed,
                            attempt=attempts,
                            task_success_index=task_successes,
                            trajectory=diagnostic_trajectory,
                            validation_error=validation_error,
                        )
                        print(f"diagnostic={diagnostic_path}", flush=True)
                    break
                if _one_bool(terminated, "termination") or _one_bool(
                    truncated, "truncation"
                ):
                    break
                state = _policy_state(observation, device)
            if not success:
                recorder.discard_episode()
                print(
                    f"[attempt {attempts}, seed {seed}] discarded unsuccessful rollout",
                    flush=True,
                )
            seed += 1
    finally:
        recorder.close()
    if args.diagnostic_task_successes is not None:
        print(
            f"Diagnostic task successes={task_successes}, strict passes={len(saved)}",
            flush=True,
        )
    return saved


def main(args=None) -> int:
    parsed = parse_args(args)
    saved = generate(parsed)
    print(f"Generated {len(saved)} successful {parsed.env_id} PPO pickles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

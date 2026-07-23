from __future__ import annotations

from dataclasses import dataclass

import torch

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.utils.structs.pose import Pose


@dataclass
class PandaState:
    tcp_pose: Pose
    tcp_pos: torch.Tensor
    gripper_width: torch.Tensor
    left_finger_pos: torch.Tensor
    right_finger_pos: torch.Tensor


def extract_panda_state(agent: Panda, env_idx: int | torch.Tensor | None = None) -> PandaState:
    if not isinstance(agent, Panda):
        raise TypeError(
            "skill_annotation.panda only supports Franka Panda agents "
            f"(got {type(agent).__name__})"
        )

    tcp_pose = agent.tcp_pose
    tcp_pos = _as_batched_tensor(agent.tcp_pos)
    qpos = _as_batched_tensor(agent.robot.get_qpos())
    left_finger_pos = _as_batched_tensor(agent.finger1_link.pose.p)
    right_finger_pos = _as_batched_tensor(agent.finger2_link.pose.p)
    indices = _normalize_env_idx(env_idx, qpos.device)

    if indices is not None:
        tcp_pose = Pose.create(tcp_pose.raw_pose[indices])
        tcp_pos = tcp_pos[indices]
        qpos = qpos[indices]
        left_finger_pos = left_finger_pos[indices]
        right_finger_pos = right_finger_pos[indices]

    return PandaState(
        tcp_pose=tcp_pose,
        tcp_pos=tcp_pos,
        gripper_width=torch.sum(qpos[..., -2:], dim=-1),
        left_finger_pos=left_finger_pos,
        right_finger_pos=right_finger_pos,
    )


def _as_batched_tensor(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 1:
        value = value[None, :]
    return value


def _normalize_env_idx(
    env_idx: int | torch.Tensor | None, device: torch.device
) -> torch.Tensor | None:
    if env_idx is None:
        return None
    if torch.is_tensor(env_idx):
        return env_idx.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor([int(env_idx)], device=device, dtype=torch.long)

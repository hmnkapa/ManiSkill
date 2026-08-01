"""Adapters between ManiSkill controller actions and RR's canonical action."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController
from mani_skill.utils import gym_utils

from .schema import ACTION_DIM
from .transforms import (
    apply_delta_quaternion_xyzw,
    as_numpy,
    matrix_to_pose,
    normalize_quaternion_xyzw,
    quaternion_inverse_xyzw,
    quaternion_multiply_xyzw,
    relative_quaternion_xyzw,
    world_pose_to_base,
    wxyz_to_xyzw,
)


def native_gripper_to_canonical(command: Any) -> np.float32:
    """Map Panda ``+1=open/-1=close`` to RR ``-1=open/+1=close``."""

    value = float(np.asarray(command, dtype=np.float32).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("Gripper command must be finite")
    if value < -1.000001 or value > 1.000001:
        raise ValueError(f"Native gripper command must be in [-1, 1], got {value}")
    return np.float32(-np.clip(value, -1.0, 1.0))


def canonical_gripper_to_native(command: Any) -> np.float32:
    """Map RR ``-1=open/+1=close`` back to the Panda convention."""

    value = float(np.asarray(command, dtype=np.float32).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("Gripper command must be finite")
    if value < -1.000001 or value > 1.000001:
        raise ValueError(f"Canonical gripper command must be in [-1, 1], got {value}")
    return np.float32(-np.clip(value, -1.0, 1.0))


def canonical_action(
    current_position: Any,
    current_quaternion_xyzw: Any,
    target_position: Any,
    target_quaternion_xyzw: Any,
    native_gripper_command: Any,
) -> np.ndarray:
    """Build the exact canonical 8D command from current and target TCP poses."""

    current_position = np.asarray(current_position, dtype=np.float32).reshape(3)
    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    delta_position = target_position - current_position
    delta_quaternion = relative_quaternion_xyzw(
        current_quaternion_xyzw, target_quaternion_xyzw
    ).reshape(4)
    result = np.concatenate(
        [
            delta_position,
            delta_quaternion,
            np.asarray(
                [native_gripper_to_canonical(native_gripper_command)],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    if result.shape != (ACTION_DIM,):
        raise AssertionError(f"Canonical action has unexpected shape {result.shape}")
    return result


class CanonicalActionAdapter:
    """Convert actions for a fixed-base, single-environment Panda.

    ``pd_joint_pos`` conversion labels the command target: the seven arm joint
    targets are evaluated with forward kinematics before the environment step.
    Consequently the recorded action is independent of tracking error during
    the simulator step.
    """

    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        if int(getattr(self.base_env, "num_envs", 1)) != 1:
            raise NotImplementedError("Pickle action recording supports num_envs=1 only")
        if getattr(self.base_env, "robot_uids", None) not in {
            "panda",
            "panda_wristcam",
        }:
            raise NotImplementedError(
                "Pickle action recording currently supports fixed-base Panda only"
            )

        self.agent = self.base_env.agent
        fixed_root = as_numpy(self.agent.robot.fixed_root_link).reshape(-1)
        if fixed_root.size != 1 or not bool(fixed_root[0]):
            raise NotImplementedError("A fixed-base Panda articulation is required")
        controller = self.agent.controller
        if not isinstance(controller, CombinedController):
            raise NotImplementedError("A flat Panda CombinedController is required")
        if "arm" not in controller.controllers or "gripper" not in controller.controllers:
            raise ValueError("Panda controller must expose arm and gripper components")
        self.controller = controller
        self._pin_model = None

    @property
    def control_mode(self) -> str:
        return str(self.base_env.control_mode)

    @property
    def arm_controller(self):
        return self.controller.controllers["arm"]

    def current_tcp_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        world_base = as_numpy(
            self.agent.robot.pose.to_transformation_matrix(), dtype=np.float32
        )[0]
        world_tcp = as_numpy(
            self.agent.tcp.pose.to_transformation_matrix(), dtype=np.float32
        )[0]
        return matrix_to_pose(world_pose_to_base(world_tcp, world_base))

    def native_to_canonical(
        self,
        action: Any,
        *,
        current_position: Optional[Any] = None,
        current_quaternion_xyzw: Optional[Any] = None,
    ) -> np.ndarray:
        """Convert a native ``pd_joint_pos`` command before calling ``step``."""

        if self.control_mode != "pd_joint_pos":
            raise NotImplementedError(
                "Native-to-canonical conversion currently requires pd_joint_pos; "
                f"got {self.control_mode!r}"
            )
        arm_controller = self.arm_controller
        if not isinstance(arm_controller, PDJointPosController):
            raise TypeError("The arm component is not a PDJointPosController")
        if arm_controller.config.use_delta or arm_controller.config.use_target:
            raise NotImplementedError(
                "Only absolute, non-target-accumulating pd_joint_pos is supported"
            )

        flat_action = self._flat_action(action)
        action_dict = self.controller.to_action_dict(flat_action)
        target_position, target_quaternion = self.fk_target_pose_base(
            action_dict["arm"]
        )
        if current_position is None or current_quaternion_xyzw is None:
            current_position, current_quaternion_xyzw = self.current_tcp_pose_base()
        return canonical_action(
            current_position,
            current_quaternion_xyzw,
            target_position,
            target_quaternion,
            action_dict["gripper"],
        )

    def fk_target_pose_base(self, arm_action: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a Panda arm joint command and return its base-frame TCP pose."""

        arm_controller = self.arm_controller
        if not isinstance(arm_controller, PDJointPosController):
            raise TypeError("FK conversion requires a PDJointPosController")
        action_tensor = torch.as_tensor(
            np.asarray(arm_action, dtype=np.float32).reshape(1, -1),
            device=self.base_env.device,
        )
        with torch.no_grad():
            target_arm_qpos = arm_controller._preprocess_action(action_tensor.clone())
        full_qpos = as_numpy(self.agent.robot.get_qpos(), dtype=np.float64)[0].copy()
        active_indices = as_numpy(arm_controller.active_joint_indices).astype(
            np.int64
        )
        full_qpos[active_indices] = as_numpy(target_arm_qpos, dtype=np.float64)[0]

        pin_model = self._get_pin_model()
        pin_model.compute_forward_kinematics(full_qpos)
        tcp_index = int(as_numpy(self.agent.tcp.index).reshape(-1)[0])
        target_pose = pin_model.get_link_pose(tcp_index)
        target_position = np.asarray(target_pose.p, dtype=np.float32)
        target_quaternion = normalize_quaternion_xyzw(
            wxyz_to_xyzw(target_pose.q)
        )
        return target_position, target_quaternion

    def canonical_to_native(
        self,
        action: Any,
        *,
        current_position: Optional[Any] = None,
        current_quaternion_xyzw: Optional[Any] = None,
    ) -> np.ndarray:
        """Convert canonical 8D to native ``pd_ee_delta_pose``.

        Other control modes are deliberately rejected.  The default Panda EE
        controller uses root-frame translation and root-aligned body rotation;
        its Euler action is recovered without clipping.
        """

        if self.control_mode != "pd_ee_delta_pose":
            raise NotImplementedError(
                "Canonical-to-native conversion supports pd_ee_delta_pose only; "
                f"got {self.control_mode!r}"
            )
        arm_controller = self.arm_controller
        if not isinstance(arm_controller, PDEEPoseController):
            raise TypeError("The arm component is not a PDEEPoseController")
        if arm_controller.config.use_target or not arm_controller.config.use_delta:
            raise NotImplementedError(
                "pd_ee_delta_pose must use current-pose deltas (use_target=False)"
            )
        if (
            arm_controller.config.frame
            != "root_translation:root_aligned_body_rotation"
        ):
            raise NotImplementedError(
                "Only root_translation:root_aligned_body_rotation is supported"
            )

        canonical = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if canonical.shape != (ACTION_DIM,):
            raise ValueError(f"Expected canonical action shape (8,), got {canonical.shape}")
        if not np.isfinite(canonical).all():
            raise ValueError("Canonical action contains a non-finite value")
        canonical[-1] = canonical_gripper_to_native(canonical[-1])
        delta_body = normalize_quaternion_xyzw(canonical[3:7])
        if current_position is None or current_quaternion_xyzw is None:
            current_position, current_quaternion_xyzw = self.current_tcp_pose_base()
        current_quaternion_xyzw = normalize_quaternion_xyzw(
            current_quaternion_xyzw
        )
        target_quaternion = apply_delta_quaternion_xyzw(
            current_quaternion_xyzw, delta_body
        )
        delta_root = quaternion_multiply_xyzw(
            target_quaternion,
            quaternion_inverse_xyzw(current_quaternion_xyzw),
        )
        delta_euler = Rotation.from_quat(delta_root).as_euler("XYZ").astype(
            np.float32
        )
        arm_physical = np.concatenate([canonical[:3], delta_euler]).astype(np.float32)
        arm_native = self._inverse_preprocess_ee_action(arm_physical)

        native_dict = {
            "arm": torch.as_tensor(arm_native, device=self.base_env.device),
            "gripper": torch.as_tensor(
                [canonical[-1]], dtype=torch.float32, device=self.base_env.device
            ),
        }
        return as_numpy(self.controller.from_action_dict(native_dict), dtype=np.float32)

    def _inverse_preprocess_ee_action(self, physical: np.ndarray) -> np.ndarray:
        controller = self.arm_controller
        if not controller._normalize_action:
            return physical
        pos = gym_utils.inv_scale_action(
            physical[:3],
            as_numpy(controller.action_space_low[:3]),
            as_numpy(controller.action_space_high[:3]),
        )
        rot_lower = np.broadcast_to(
            np.asarray(controller.config.rot_lower, dtype=np.float32), (3,)
        )
        if np.any(np.abs(rot_lower) < 1e-12):
            raise ValueError("EE rotation scale contains zero")
        rot = physical[3:] / rot_lower
        result = np.concatenate([pos, rot]).astype(np.float32)
        if np.any(np.abs(pos) > 1.000001) or np.linalg.norm(rot) > 1.000001:
            raise ValueError(
                "Canonical delta is outside the pd_ee_delta_pose action range; "
                "conversion would require clipping"
            )
        return result

    def _flat_action(self, action: Any) -> np.ndarray:
        flat = as_numpy(action, dtype=np.float32)
        if flat.ndim == 2:
            if flat.shape[0] != 1:
                raise ValueError(f"Expected one action, got shape {flat.shape}")
            flat = flat[0]
        flat = flat.reshape(-1)
        expected = int(self.controller.single_action_space.shape[0])
        if flat.shape != (expected,):
            raise ValueError(f"Expected native action shape ({expected},), got {flat.shape}")
        if not np.isfinite(flat).all():
            raise ValueError("Native action contains a non-finite value")
        return flat

    def _get_pin_model(self):
        if self._pin_model is not None:
            return self._pin_model
        articulation = self.agent.robot
        try:
            self._pin_model = articulation.create_pinocchio_model()
            return self._pin_model
        except NotImplementedError:
            pass

        # ManiSkill disables the merged-articulation convenience method on GPU,
        # but the underlying single physical articulation still exposes the
        # same state-independent Pinocchio model used by the CPU path.
        try:
            model = articulation._objs[0].create_pinocchio_model()
        except Exception as error:
            raise RuntimeError(
                "Could not construct the Panda FK model for action recording"
            ) from error
        self._pin_model = model
        return self._pin_model


__all__ = [
    "CanonicalActionAdapter",
    "canonical_action",
    "canonical_gripper_to_native",
    "native_gripper_to_canonical",
]

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


from isaaclab.sensors import ContactSensor


def both_feet_airborne(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return 1 when neither foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    in_contact = contact_time > 0.0

    return (~torch.any(in_contact, dim=1)).float()


def base_acceleration_l2(
    env: ManagerBasedRLEnv,
    axis: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize squared base acceleration along one selected direction.

    The lateral y acceleration is measured in the gravity-aligned yaw frame,
    so turning the robot does not change which direction is considered lateral.

    The vertical z acceleration is measured in the world frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Acceleration of base_link's center of mass in the world frame.
    base_acc_w = asset.data.body_lin_acc_w[
        :, asset_cfg.body_ids[0], :
    ]

    if axis == "y":
        # Rotate into the gravity-aligned robot heading frame.
        base_acc_yaw = quat_apply_inverse(
            yaw_quat(asset.data.root_quat_w),
            base_acc_w,
        )
        return torch.square(base_acc_yaw[:, 1])

    if axis == "z":
        return torch.square(base_acc_w[:, 2])

    raise ValueError(
        f"Unsupported acceleration axis: {axis!r}. Use 'y' or 'z'."
    )

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

def feet_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward swing-foot clearance, capped at ``target_height``.

    The grounded foot is used as the ground-height reference. This avoids
    needing to know the vertical offset between the ankle link frame and the
    physical bottom of the foot.

    Reward per swing foot:
        0.00 m clearance -> 0
        0.015 m clearance -> 0.5
        >= 0.03 m clearance -> 1.0

    The reward is disabled when:
    - neither foot is supporting the robot; or
    - the commanded forward/yaw velocity is approximately zero.
    """
    if target_height <= 0.0:
        raise ValueError("target_height must be greater than zero.")

    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Z positions of the two ankle-roll link frames in the world frame.
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]

    # Detect which feet are currently in contact.
    contact_time = contact_sensor.data.current_contact_time[
        :, sensor_cfg.body_ids
    ]
    in_contact = contact_time > 0.0
    has_support_foot = torch.any(in_contact, dim=1)

    # Use the lowest contacting foot as the approximate ground reference.
    large_height = torch.full_like(foot_z, float("inf"))
    contacting_foot_z = torch.where(in_contact, foot_z, large_height)
    support_z = torch.min(contacting_foot_z, dim=1, keepdim=True).values

    # Avoid infinity when both feet are airborne. The final support mask will
    # disable the reward in those environments.
    lowest_foot_z = torch.min(foot_z, dim=1, keepdim=True).values
    support_z = torch.where(
        has_support_foot.unsqueeze(1),
        support_z,
        lowest_foot_z,
    )

    # Clearance of each foot relative to the supporting foot.
    clearance = torch.clamp(
        foot_z - support_z,
        min=0.0,
        max=target_height,
    )

    # Normalize so each swing foot contributes at most 1.
    clearance_reward = clearance / target_height

    # Only the airborne/swing foot receives clearance reward.
    swing_foot = ~in_contact
    clearance_reward = clearance_reward * swing_foot.float()

    # Do not encourage foot lifting during stand-still commands.
    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.vector_norm(
        command[:, [0, 2]],  # forward velocity and yaw rate
        dim=1,
    ) > 0.05

    return (
        torch.sum(clearance_reward, dim=1)
        * has_support_foot.float()
        * moving_command.float()
    )
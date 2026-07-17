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
    - the episode is not a stage-two stride-training bar episode;
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

    from .wooden_bar import stride_training_bar_active

    return (
        torch.sum(clearance_reward, dim=1)
        * has_support_foot.float()
        * moving_command.float()
        * stride_training_bar_active(env).float()
    )


def feet_stride_length_reward(
    env: ManagerBasedRLEnv,
    foot_length: float,
    target_stride_length: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    """Reward long forward strides when the swing foot touches down.

    A touchdown stride receives a negative reward below one 14 cm foot length,
    zero reward at 14 cm, and a positive reward above 14 cm. The reward reaches
    one at ``target_stride_length`` and is bounded to the range [-1, 1].
    """
    if foot_length <= 0.0:
        raise ValueError("foot_length must be greater than zero.")
    if target_stride_length <= foot_length:
        raise ValueError("target_stride_length must be greater than foot_length.")

    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    if foot_pos_w.shape[1] != 2:
        raise ValueError(
            "Stride-length reward requires exactly two foot bodies, "
            f"but received {foot_pos_w.shape[1]}."
        )

    foot_delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
    foot_delta_b_x = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        torch.cat(
            (foot_delta_w, torch.zeros(env.num_envs, 1, device=env.device)),
            dim=1,
        ),
    )[:, 0]
    stride_length = torch.abs(foot_delta_b_x)
    # Below one foot length:
    #   0 cm  -> -1
    #   7 cm  -> -0.5
    #   14 cm -> 0
    short_stride_reward = stride_length / foot_length - 1.0

    # Above one foot length:
    #   14 cm -> 0
    #   17 cm -> 0.5
    #   20 cm -> 1
    long_stride_reward = (
        (stride_length - foot_length)
        / (target_stride_length - foot_length)
    )

normalized_stride = torch.where(
    stride_length < foot_length,
    short_stride_reward,
    long_stride_reward,
)
normalized_stride = torch.clamp(normalized_stride, min=-1.0, max=1.0)

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    touchdown = torch.any(first_contact, dim=1)
    command = env.command_manager.get_command(command_name)
    moving_command = torch.linalg.vector_norm(command[:, [0, 2]], dim=1) > 0.05

    from .wooden_bar import stride_training_bar_active

    return (
        normalized_stride
        * touchdown.float()
        * moving_command.float()
        * stride_training_bar_active(env).float()
    )

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


def joint_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only torque magnitude above ``threshold`` in N.m.

    Each joint contributes ``max(abs(actual_torque) - threshold, 0)`` so the
    penalty increases linearly as its applied torque approaches the hard limit.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be greater than or equal to zero.")

    asset: Articulation = env.scene[asset_cfg.name]
    actual_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    excess_torque = torch.clamp(torch.abs(actual_torque) - threshold, min=0.0)
    return torch.sum(excess_torque, dim=1)


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
    bar_names: tuple[str, ...],
    activation_distance: float,
    full_weight_distance: float,
) -> torch.Tensor:
    """Reward swing-foot clearance, capped at ``target_height``.

    The grounded foot is used as the ground-height reference. This avoids
    needing to know the vertical offset between the ankle link frame and the
    physical bottom of the foot.

    Reward per swing foot:
        0.00 m clearance -> 0
        0.015 m clearance -> 0.5
        >= 0.03 m clearance -> 1.0

    The reward is disabled outside stride training, before a bar is within
    ``activation_distance``, and after both feet have passed the bar. Its
    effective weight increases linearly to full strength at
    ``full_weight_distance``.
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

    from .wooden_bar import stride_training_reward_scale

    reward_scale = stride_training_reward_scale(
        env,
        bar_names=bar_names,
        feet_cfg=asset_cfg,
        activation_distance=activation_distance,
        full_weight_distance=full_weight_distance,
    )
    return (
        torch.sum(clearance_reward, dim=1)
        * has_support_foot.float()
        * moving_command.float()
        * reward_scale
    )


def feet_stride_length_reward(
    env: ManagerBasedRLEnv,
    foot_length: float,
    target_stride_length: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    bar_names: tuple[str, ...],
    activation_distance: float,
    full_weight_distance: float,
) -> torch.Tensor:
    """Reward long forward strides only when touchdown feet alternate.

    A valid touchdown must involve exactly one foot, place that foot ahead in
    the commanded travel direction, and alternate from the previous valid
    touchdown. Its stride receives a negative reward below one 14 cm foot
    length, zero reward at 14 cm, and a positive reward above 14 cm. The reward
    reaches one at ``target_stride_length`` and is bounded to [-1, 1].

    The reward is active only during stride training. Its effective weight is
    zero until the bar is closer than ``activation_distance``, increases
    linearly, reaches full strength at ``full_weight_distance``, and returns to
    zero after both feet pass the bar.
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
    single_touchdown = torch.sum(first_contact, dim=1) == 1
    touchdown_foot = torch.argmax(first_contact.to(torch.int64), dim=1)

    command = env.command_manager.get_command(command_name)
    forward_command = torch.abs(command[:, 0]) > 0.05
    travel_direction = torch.sign(command[:, 0])

    # A long split stance alone must not earn reward: the landing foot has to
    # be the foot that is ahead in the commanded direction.
    touchdown_separation = torch.where(
        touchdown_foot == 0,
        foot_delta_b_x,
        -foot_delta_b_x,
    )
    touchdown_ahead = touchdown_separation * travel_direction > 0.0

    # Remember the most recent valid landing foot independently for every
    # environment. Reset this history at the start of each episode.
    state_name = "_stride_reward_last_touchdown_foot"
    if not hasattr(env, state_name):
        setattr(
            env,
            state_name,
            torch.full(
                (env.num_envs,),
                -1,
                dtype=torch.long,
                device=env.device,
            ),
        )
    last_touchdown_foot = getattr(env, state_name)
    last_touchdown_foot[env.episode_length_buf == 0] = -1

    valid_touchdown = single_touchdown & forward_command & touchdown_ahead
    alternating_touchdown = valid_touchdown & (
        (last_touchdown_foot == -1)
        | (touchdown_foot != last_touchdown_foot)
    )
    last_touchdown_foot[valid_touchdown] = touchdown_foot[valid_touchdown]

    from .wooden_bar import stride_training_reward_scale

    reward_scale = stride_training_reward_scale(
        env,
        bar_names=bar_names,
        feet_cfg=asset_cfg,
        activation_distance=activation_distance,
        full_weight_distance=full_weight_distance,
    )
    return normalized_stride * alternating_touchdown.float() * reward_scale


def track_lin_vel_xy_yaw_frame_quadratic_relative(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    moving_command_threshold: float = 0.05,
    standing_std: float = 0.20,
) -> torch.Tensor:
    """Track commanded planar velocity without rewarding zero-net rocking.

    For moving commands:
        perfect tracking       -> +1
        stationary robot       ->  0
        opposite-direction     -> -1
        excessive overspeed    -> negative, bounded at -1

    For standing commands, use an exponential penalty on planar velocity.
    """
    if moving_command_threshold <= 0.0:
        raise ValueError("moving_command_threshold must be positive.")
    if standing_std <= 0.0:
        raise ValueError("standing_std must be positive.")

    robot: Articulation = env.scene[asset_cfg.name]

    # World velocity expressed in the gravity-aligned robot yaw frame.
    base_lin_vel_yaw = quat_apply_inverse(
        yaw_quat(robot.data.root_quat_w),
        robot.data.root_lin_vel_w,
    )

    command_xy = env.command_manager.get_command(command_name)[:, :2]
    actual_xy = base_lin_vel_yaw[:, :2]

    command_speed_sq = torch.sum(torch.square(command_xy), dim=1)
    tracking_error_sq = torch.sum(
        torch.square(command_xy - actual_xy),
        dim=1,
    )

    # Normalizing by command speed makes:
    # actual = 0          -> 0
    # actual = command    -> 1
    # actual = -command   -> -1 after clipping
    denominator = torch.clamp(
        command_speed_sq,
        min=moving_command_threshold**2,
    )

    moving_score = 1.0 - tracking_error_sq / denominator
    moving_score = torch.clamp(moving_score, min=-1.0, max=1.0)

    # Standing environments should minimize all planar motion.
    standing_score = torch.exp(
        -torch.sum(torch.square(actual_xy), dim=1) / standing_std**2
    )

    moving_command = (
        torch.sqrt(command_speed_sq) > moving_command_threshold
    )

    return torch.where(
        moving_command,
        moving_score,
        standing_score,
    )
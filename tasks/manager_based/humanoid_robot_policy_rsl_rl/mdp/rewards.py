from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    wrap_to_pi,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


from isaaclab.sensors import ContactSensor


def ground_contact_flatness(
    env: ManagerBasedRLEnv,
    flat_tolerance: float,
    penalty_start_angle: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward flat support feet and penalize tilted ground contact."""
    if flat_tolerance < 0.0:
        raise ValueError("flat_tolerance must be non-negative.")
    if penalty_start_angle <= flat_tolerance:
        raise ValueError("penalty_start_angle must be greater than flat_tolerance.")
    if penalty_start_angle >= 0.5 * math.pi:
        raise ValueError("penalty_start_angle must be less than 90 degrees.")

    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    if foot_quat_w.shape[1] != contact_time.shape[1]:
        raise ValueError("asset_cfg and sensor_cfg must resolve the same number of feet.")

    sole_normal_b = torch.zeros(
        foot_quat_w.shape[0],
        foot_quat_w.shape[1],
        3,
        dtype=foot_quat_w.dtype,
        device=foot_quat_w.device,
    )
    sole_normal_b[..., 2] = 1.0
    sole_normal_w = quat_apply(
        foot_quat_w.reshape(-1, 4),
        sole_normal_b.reshape(-1, 3),
    ).reshape_as(sole_normal_b)

    tilt_angle = torch.atan2(
        torch.linalg.vector_norm(sole_normal_w[..., :2], dim=-1),
        sole_normal_w[..., 2],
    )
    flat_reward = (tilt_angle <= flat_tolerance).to(tilt_angle.dtype)
    tilt_penalty = torch.clamp(
        (tilt_angle - penalty_start_angle)
        / (0.5 * math.pi - penalty_start_angle),
        min=0.0,
        max=1.0,
    )
    foot_score = flat_reward - tilt_penalty

    in_contact = contact_time > 0.0
    contact_count = torch.sum(in_contact, dim=1)
    return torch.sum(
        foot_score * in_contact.to(foot_score.dtype),
        dim=1,
    ) / torch.clamp(contact_count, min=1)


class swing_foot_clearance_reward(ManagerTermBase):
    """Reward swing-foot sole clearance inside a target height range."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        sole_vertices = cfg.params["sole_vertices"]
        if len(sole_vertices) != 2 or any(
            len(vertices) < 3 for vertices in sole_vertices
        ):
            raise ValueError(
                "sole_vertices must contain at least three vertices "
                "for each of two feet."
            )

        self._sole_vertices = torch.tensor(
            sole_vertices,
            dtype=torch.float,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_clearance: float,
        max_clearance: float,
        sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
        command_name: str,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        moving_command_threshold: float = 0.05,
    ) -> torch.Tensor:
        del sole_vertices

        if min_clearance <= 0.0:
            raise ValueError("min_clearance must be positive.")
        if max_clearance <= min_clearance:
            raise ValueError("max_clearance must be greater than min_clearance.")

        robot: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids]
        foot_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
        num_envs, num_feet = foot_pos_w.shape[:2]
        if num_feet != self._sole_vertices.shape[0]:
            raise ValueError(
                "asset_cfg must resolve the same number of feet as sole_vertices."
            )

        num_vertices = self._sole_vertices.shape[1]
        vertices = self._sole_vertices.unsqueeze(0).expand(
            num_envs, -1, -1, -1
        )
        quaternions = foot_quat_w.unsqueeze(2).expand(
            -1, -1, num_vertices, -1
        )
        rotated_vertices = quat_apply(
            quaternions.reshape(-1, 4),
            vertices.reshape(-1, 3),
        ).reshape(num_envs, num_feet, num_vertices, 3)
        sole_z = foot_pos_w[:, :, 2] + torch.amin(
            rotated_vertices[:, :, :, 2],
            dim=2,
        )

        contact_time = contact_sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
        in_contact = contact_time > 0.0
        has_support_foot = torch.any(in_contact, dim=1)
        support_z = torch.amin(
            torch.where(
                in_contact,
                sole_z,
                torch.full_like(sole_z, float("inf")),
            ),
            dim=1,
            keepdim=True,
        )
        support_z = torch.where(
            has_support_foot.unsqueeze(1),
            support_z,
            torch.amin(sole_z, dim=1, keepdim=True),
        )
        clearance = torch.clamp(sole_z - support_z, min=0.0)
        score_below = torch.clamp(
            clearance / min_clearance,
            min=0.0,
            max=1.0,
        )
        score_above = torch.clamp(
            1.0 - (clearance - max_clearance) / min_clearance,
            min=0.0,
            max=1.0,
        )
        clearance_score = torch.minimum(score_below, score_above)

        moving_command = torch.linalg.vector_norm(
            env.command_manager.get_command(command_name)[:, :3],
            dim=1,
        ) > moving_command_threshold
        swing_foot = ~in_contact
        return (
            torch.sum(clearance_score * swing_foot.float(), dim=1)
            * has_support_foot.float()
            * moving_command.float()
        )


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


def joint_torque_over_nominal(
    env: ManagerBasedRLEnv,
    nominal_torque: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only applied joint torque above the nominal actuator torque."""
    if nominal_torque <= 0.0:
        raise ValueError("nominal_torque must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    applied_torque = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    excess_torque = torch.clamp(applied_torque - nominal_torque, min=0.0)
    return torch.sum(excess_torque, dim=1)


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

class feet_clearance_reward(ManagerTermBase):
    """Score both feet's physical sole clearance while they overlap the bar band."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sole_vertices = cfg.params["sole_vertices"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        minimum_clearance_ratio: float,
        band_half_width: float,
        sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del sole_vertices

        if target_height <= 0.0:
            raise ValueError("target_height must be positive.")
        if not 0.0 <= minimum_clearance_ratio < 1.0:
            raise ValueError(
                "minimum_clearance_ratio must be in the range [0, 1)."
            )
        if band_half_width <= 0.0:
            raise ValueError("band_half_width must be positive.")

        from .wooden_bar import foot_bar_geometry

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        sole_vertices_w, _, distance_to_bar_line, bar_active = (
            foot_bar_geometry(
                env,
                feet_cfg=asset_cfg,
                sole_vertices=self._sole_vertices,
            )
        )
        sole_z = torch.amin(sole_vertices_w[..., 2], dim=2)

        contact_time = contact_sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
        in_contact = contact_time > 0.0
        if sole_z.shape != in_contact.shape:
            raise ValueError(
                "asset_cfg and sensor_cfg must resolve the same two feet."
            )

        has_support_foot = torch.any(in_contact, dim=1)
        support_z = torch.amin(
            torch.where(
                in_contact,
                sole_z,
                torch.full_like(sole_z, float("inf")),
            ),
            dim=1,
            keepdim=True,
        )
        nominal_ground_z = env.scene.env_origins[:, 2].unsqueeze(1)
        ground_z = torch.where(
            has_support_foot.unsqueeze(1),
            support_z,
            nominal_ground_z,
        )
        clearance = torch.clamp(sole_z - ground_z, min=0.0)
        clearance = torch.where(in_contact, torch.zeros_like(clearance), clearance)

        zero_score_height = minimum_clearance_ratio * target_height
        score = (clearance - zero_score_height) / (
            target_height - zero_score_height
        )
        score = torch.clamp(score, min=-1.0, max=1.0)

        within_band = distance_to_bar_line <= band_half_width
        active = within_band & bar_active.unsqueeze(1)
        return torch.sum(score * active.to(score.dtype), dim=1)


class stepping_feet_forward_movement_reward(ManagerTermBase):
    """Reward forward velocity of airborne feet while they overlap the bar band."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._sole_vertices = cfg.params["sole_vertices"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_forward_velocity: float,
        band_half_width: float,
        sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del sole_vertices

        if target_forward_velocity <= 0.0:
            raise ValueError("target_forward_velocity must be positive.")
        if band_half_width <= 0.0:
            raise ValueError("band_half_width must be positive.")

        from .wooden_bar import foot_bar_geometry

        robot: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        _, forward_w, distance_to_bar_line, bar_active = foot_bar_geometry(
            env,
            feet_cfg=asset_cfg,
            sole_vertices=self._sole_vertices,
        )

        contact_time = contact_sensor.data.current_contact_time[
            :, sensor_cfg.body_ids
        ]
        stepping_feet = contact_time <= 0.0

        foot_velocity_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        forward_velocity = torch.sum(
            foot_velocity_w * forward_w[:, None, :],
            dim=2,
        )
        velocity_score = torch.clamp(
            forward_velocity / target_forward_velocity,
            min=0.0,
            max=1.0,
        )

        within_band = distance_to_bar_line <= band_half_width
        active = within_band & stepping_feet & bar_active.unsqueeze(1)
        return torch.sum(
            velocity_score * active.to(velocity_score.dtype),
            dim=1,
        )


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

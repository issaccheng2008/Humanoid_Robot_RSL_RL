# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Delayed wooden-bar mechanics and rewards for the humanoid locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


DEFAULT_BAR_DISTANCE = 0.50
NO_BAR_TRAINING_PHASE = 0
WOODEN_BAR_TRAINING_PHASE = 1


def _curriculum_step(env: ManagerBasedRLEnv) -> int:
    """Return the global curriculum step."""
    return int(env.common_step_counter)


class _WoodenBarState:
    """Per-environment state shared by obstacle MDP terms."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.spawned = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self.crossed = torch.zeros_like(self.spawned)
        self.spawn_time_s = torch.zeros(env.num_envs, device=env.device)
        self.spawn_pose_w = torch.zeros(env.num_envs, 7, device=env.device)
        self.spawn_pose_w[:, 3] = 1.0
        self.movement_reference_pose_w = self.spawn_pose_w.clone()
        self.movement_reference_set = torch.zeros_like(self.spawned)
        self.forward_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.forward_w[:, 0] = 1.0
        self.curriculum_phase = NO_BAR_TRAINING_PHASE
        self.episode_phase = torch.full(
            (env.num_envs,),
            NO_BAR_TRAINING_PHASE,
            dtype=torch.long,
            device=env.device,
        )
        self.active_bar_index = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        self.first_time_entering_strip = torch.ones_like(self.spawned)
        self.first_foot_entered = torch.zeros_like(self.spawned)
        self.first_entry_event = torch.zeros(
            (env.num_envs, 2),
            dtype=torch.bool,
            device=env.device,
        )
        self.band_state_update_step = -1
        self.sole_vertices = None
        self.sole_vertices_key = None


def _get_state(env: ManagerBasedRLEnv) -> _WoodenBarState:
    if not hasattr(env, "_wooden_bar_state"):
        env._wooden_bar_state = _WoodenBarState(env)
    return env._wooden_bar_state


def _as_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def _active_bar_pose_w(
    env: ManagerBasedRLEnv,
    bar_names: Sequence[str],
    state: _WoodenBarState,
) -> torch.Tensor:
    """Return the root pose of the active bar in every environment."""
    all_bar_poses_w = torch.stack(
        [env.scene[name].data.root_state_w[:, :7] for name in bar_names],
        dim=1,
    )
    safe_indices = torch.clamp(state.active_bar_index, min=0)
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_pose_w = all_bar_poses_w[env_indices, safe_indices]
    return torch.where(
        state.spawned.unsqueeze(1),
        active_pose_w,
        state.spawn_pose_w,
    )


def _sole_vertices_tensor(
    env: ManagerBasedRLEnv,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
) -> torch.Tensor:
    """Validate and cache the two sole-perimeter vertex sets."""
    if len(sole_vertices) != 2 or any(
        len(vertices) < 3 for vertices in sole_vertices
    ):
        raise ValueError(
            "sole_vertices must contain at least three vertices "
            "for each of two feet."
        )

    state = _get_state(env)
    if state.sole_vertices is None or state.sole_vertices_key != sole_vertices:
        state.sole_vertices = torch.tensor(
            sole_vertices,
            dtype=torch.float,
            device=env.device,
        )
        state.sole_vertices_key = sole_vertices
    return state.sole_vertices


def foot_bar_geometry(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return live sole geometry and exact footprint distance to the bar line.

    The distance is zero when the bar center line intersects the projected
    convex sole footprint. Otherwise it is the shortest longitudinal distance
    from that footprint to the center line.
    """
    state = _get_state(env)
    robot = env.scene[feet_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, feet_cfg.body_ids]
    foot_quat_w = robot.data.body_quat_w[:, feet_cfg.body_ids]
    if foot_pos_w.shape[1] != 2:
        raise ValueError(
            "Wooden-bar geometry requires exactly two foot bodies, "
            f"but received {foot_pos_w.shape[1]}."
        )

    vertices_b = _sole_vertices_tensor(env, sole_vertices)
    num_envs, num_feet = foot_pos_w.shape[:2]
    num_vertices = vertices_b.shape[1]
    vertices = vertices_b.unsqueeze(0).expand(num_envs, -1, -1, -1)
    quaternions = foot_quat_w.unsqueeze(2).expand(
        -1,
        -1,
        num_vertices,
        -1,
    )
    rotated_vertices = quat_apply(
        quaternions.reshape(-1, 4),
        vertices.reshape(-1, 3),
    ).reshape(num_envs, num_feet, num_vertices, 3)
    sole_vertices_w = foot_pos_w.unsqueeze(2) + rotated_vertices

    relative_xy = (
        sole_vertices_w[..., :2] - state.spawn_pose_w[:, None, None, :2]
    )
    longitudinal = torch.sum(
        relative_xy * state.forward_w[:, None, None, :],
        dim=3,
    )
    footprint_min = torch.amin(longitudinal, dim=2)
    footprint_max = torch.amax(longitudinal, dim=2)
    distance_to_bar_line = torch.where(
        footprint_min > 0.0,
        footprint_min,
        torch.where(
            footprint_max < 0.0,
            -footprint_max,
            torch.zeros_like(footprint_min),
        ),
    )
    bar_active = state.spawned & ~state.crossed
    return (
        sole_vertices_w,
        state.forward_w,
        distance_to_bar_line,
        bar_active,
    )


def _minimum_foot_clearance(
    env: ManagerBasedRLEnv,
    sole_vertices_w: torch.Tensor,
    in_contact: torch.Tensor,
) -> torch.Tensor:
    """Estimate each foot's minimum sole height above the local support plane."""
    sole_z = torch.amin(sole_vertices_w[..., 2], dim=2)
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
    return torch.where(in_contact, torch.zeros_like(clearance), clearance)


def _update_foot_band_state(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
) -> tuple[
    _WoodenBarState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Update the first strip-entry/contact state once per control step."""
    if band_half_width <= 0.0:
        raise ValueError("band_half_width must be positive.")

    state = _get_state(env)
    sole_vertices_w, forward_w, _, bar_active = foot_bar_geometry(
        env,
        feet_cfg=feet_cfg,
        sole_vertices=sole_vertices,
    )
    relative_xy = (
        sole_vertices_w[..., :2]
        - state.spawn_pose_w[:, None, None, :2]
    )
    longitudinal = torch.sum(
        relative_xy * forward_w[:, None, None, :],
        dim=3,
    )
    footprint_min = torch.amin(longitudinal, dim=2)
    footprint_max = torch.amax(longitudinal, dim=2)
    overlaps_band = (
        (footprint_max >= -band_half_width)
        & (footprint_min <= band_half_width)
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    )
    if in_contact.shape != footprint_min.shape:
        raise ValueError(
            "feet_cfg and sensor_cfg must resolve the same two feet."
        )

    update_step = _curriculum_step(env)
    if state.band_state_update_step != update_step:
        state.first_entry_event.zero_()
        active_feet = bar_active.unsqueeze(1)
        new_entry_candidates = (
            active_feet
            & overlaps_band
            & ~state.first_foot_entered.unsqueeze(1)
        )
        new_first_entry = torch.any(new_entry_candidates, dim=1)

        candidate_progress = torch.where(
            new_entry_candidates,
            footprint_max,
            torch.full_like(footprint_max, float("-inf")),
        )
        first_foot_ids = torch.argmax(candidate_progress, dim=1)
        entry_env_ids = torch.nonzero(
            new_first_entry,
            as_tuple=False,
        ).squeeze(-1)
        state.first_entry_event[
            entry_env_ids,
            first_foot_ids[entry_env_ids],
        ] = True
        state.first_foot_entered |= new_first_entry

        at_or_beyond_back_edge = footprint_max >= -band_half_width
        ground_contact_in_or_beyond_band = torch.any(
            in_contact & at_or_beyond_back_edge,
            dim=1,
        )
        whole_foot_past_front_edge = torch.any(
            footprint_min > band_half_width,
            dim=1,
        )
        strip_attempt_finished = (
            bar_active
            & (
                ground_contact_in_or_beyond_band
                | whole_foot_past_front_edge
            )
        )
        state.first_time_entering_strip &= ~strip_attempt_finished
        state.band_state_update_step = update_step

    return (
        state,
        sole_vertices_w,
        forward_w,
        footprint_min,
        footprint_max,
        overlaps_band,
        in_contact,
    )


def first_time_entering_strip(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
) -> torch.Tensor:
    """Return 1 until the first strip attempt touches down or clears the band."""
    state, *_ = _update_foot_band_state(
        env,
        feet_cfg=feet_cfg,
        sensor_cfg=sensor_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )
    return state.first_time_entering_strip.float()


def _update_crossed(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
) -> _WoodenBarState:
    """Mark crossed only when every sole point is beyond the front of the band."""
    if band_half_width <= 0.0:
        raise ValueError("band_half_width must be positive.")

    state = _get_state(env)
    pending = state.spawned & ~state.crossed
    if not torch.any(pending):
        return state

    sole_vertices_w, _, _, _ = foot_bar_geometry(
        env,
        feet_cfg=feet_cfg,
        sole_vertices=sole_vertices,
    )
    relative_xy = (
        sole_vertices_w[..., :2] - state.spawn_pose_w[:, None, None, :2]
    )
    longitudinal = torch.sum(
        relative_xy * state.forward_w[:, None, None, :],
        dim=3,
    )
    whole_foot_past_band = torch.amin(longitudinal, dim=2) > band_half_width
    both_feet_past_band = torch.all(whole_foot_past_band, dim=1)
    state.crossed |= pending & both_feet_past_band
    return state


class ObstacleAwareVelocityCommand(UniformVelocityCommand):
    """Command 0.4 m/s before the bar and stop after both feet clear it."""

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = _as_env_ids(self._env, env_ids)
        random_values = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = random_values.uniform_(
            *self.cfg.ranges.lin_vel_x
        )
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = random_values.uniform_(
            *self.cfg.ranges.ang_vel_z
        )

        if self.cfg.heading_command:
            self.heading_target[env_ids] = random_values.uniform_(
                *self.cfg.ranges.heading
            )
            self.is_heading_env[env_ids] = (
                random_values.uniform_(0.0, 1.0)
                <= self.cfg.rel_heading_envs
            )

        state = _get_state(self._env)
        crossed = state.spawned[env_ids] & state.crossed[env_ids]
        self.is_standing_env[env_ids] = crossed
        self.vel_command_b[env_ids[crossed]] = 0.0

    def _update_command(self):
        state = _get_state(self._env)
        bar_active = state.spawned & ~state.crossed
        crossed = state.spawned & state.crossed

        self.is_standing_env[bar_active] = False
        self.is_standing_env[crossed] = True
        super()._update_command()

        self.vel_command_b[bar_active, 0] = self.cfg.ranges.lin_vel_x[0]
        self.vel_command_b[bar_active, 1] = 0.0
        self.vel_command_b[bar_active, 2] = 0.0
        self.vel_command_b[crossed] = 0.0


@configclass
class ObstacleAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    """Forward/yaw-only command configuration."""

    class_type: type = ObstacleAwareVelocityCommand

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING


def forward_yaw_velocity_commands(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return forward velocity and yaw rate, omitting fixed-zero lateral speed."""
    command = env.command_manager.get_command(command_name)
    return command[:, (0, 2)]


def reset_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    bar_names: tuple[str, ...],
    hidden_depth: float,
):
    """Hide both bar variants and prepare an immediate bar spawn."""
    env_ids = _as_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return

    state = _get_state(env)
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, 2] -= hidden_depth
    pose[:, 3] = 1.0
    velocity = torch.zeros(len(env_ids), 6, device=env.device)

    for bar_name in bar_names:
        bar = env.scene[bar_name]
        bar.write_root_pose_to_sim(pose, env_ids=env_ids)
        bar.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    state.spawned[env_ids] = False
    state.crossed[env_ids] = False
    state.episode_phase[env_ids] = state.curriculum_phase
    state.active_bar_index[env_ids] = -1
    state.spawn_time_s[env_ids] = 0.0
    state.spawn_pose_w[env_ids] = pose
    state.movement_reference_pose_w[env_ids] = pose
    state.movement_reference_set[env_ids] = False
    state.forward_w[env_ids, 0] = 1.0
    state.forward_w[env_ids, 1] = 0.0
    state.first_time_entering_strip[env_ids] = True
    state.first_foot_entered[env_ids] = False
    state.first_entry_event[env_ids] = False


def spawn_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    bar_names: tuple[str, ...],
    physical_bar_name: str,
    bar_height: float,
    robot_name: str,
    distance_range: tuple[float, float],
    drop_clearance: float,
    command_name: str,
):
    """Spawn the physical 20 mm bar 15-30 cm ahead after curriculum activation."""
    if bar_height <= 0.0:
        raise ValueError("bar_height must be positive.")
    if (
        distance_range[0] < 0.0
        or distance_range[1] < distance_range[0]
    ):
        raise ValueError("distance_range must be ordered and non-negative.")
    if drop_clearance < 0.0:
        raise ValueError("drop_clearance must be non-negative.")

    state = _get_state(env)
    env_ids = _as_env_ids(env, env_ids)
    env_ids = env_ids[~state.spawned[env_ids]]
    env_ids = env_ids[
        state.episode_phase[env_ids] == WOODEN_BAR_TRAINING_PHASE
    ]
    if len(env_ids) == 0:
        return

    robot = env.scene[robot_name]
    robot_yaw_quat_w = yaw_quat(robot.data.root_quat_w[env_ids])
    local_forward = torch.zeros(len(env_ids), 3, device=env.device)
    local_forward[:, 0] = 1.0
    forward_w = quat_apply(robot_yaw_quat_w, local_forward)
    distance = torch.empty(len(env_ids), device=env.device).uniform_(
        *distance_range
    )

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = (
        robot.data.root_pos_w[env_ids, :2]
        + distance.unsqueeze(1) * forward_w[:, :2]
    )
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + 0.5 * bar_height
    pose[:, 2] += drop_clearance
    pose[:, 3:7] = robot_yaw_quat_w
    velocity = torch.zeros(len(env_ids), 6, device=env.device)
    bar = env.scene[physical_bar_name]
    bar.write_root_pose_to_sim(pose, env_ids=env_ids)
    bar.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    state.spawned[env_ids] = True
    state.active_bar_index[env_ids] = bar_names.index(physical_bar_name)
    state.crossed[env_ids] = False
    state.spawn_time_s[env_ids] = _episode_time_s(env)[env_ids]
    state.spawn_pose_w[env_ids] = pose
    state.movement_reference_pose_w[env_ids] = pose
    state.movement_reference_set[env_ids] = False
    state.forward_w[env_ids] = forward_w[:, :2]

    command_term = env.command_manager.get_term(command_name)
    if isinstance(command_term, ObstacleAwareVelocityCommand):
        command_term.is_standing_env[env_ids] = False
        command_term.vel_command_b[env_ids, 0] = (
            command_term.cfg.ranges.lin_vel_x[0]
        )
        command_term.vel_command_b[env_ids, 1] = 0.0
        command_term.vel_command_b[env_ids, 2] = 0.0


def wooden_bar_distance(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    default_distance: float = DEFAULT_BAR_DISTANCE,
    noise_range: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Return signed forward bar distance, or the default after crossing."""
    state = _update_crossed(
        env,
        feet_cfg=feet_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )
    bar_pose_w = _active_bar_pose_w(env, bar_names, state)
    robot = env.scene[feet_cfg.name]

    relative_bar_xy = bar_pose_w[:, :2] - robot.data.root_pos_w[:, :2]
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, 0] = 1.0
    robot_forward_w = quat_apply(
        yaw_quat(robot.data.root_quat_w),
        local_forward,
    )[:, :2]
    distance = torch.sum(relative_bar_xy * robot_forward_w, dim=1)
    visible = state.spawned & ~state.crossed
    distance += torch.empty_like(distance).uniform_(*noise_range)
    distance = torch.where(
        visible,
        distance,
        torch.full_like(distance, default_distance),
    )
    return distance.unsqueeze(1)


def wooden_bar_moved(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    translation_tolerance: float,
    rotation_tolerance: float,
    settling_time_s: float,
) -> torch.Tensor:
    """Terminate physical-bar episodes when the settled bar is disturbed."""
    state = _update_crossed(
        env,
        feet_cfg=feet_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )
    bar_pose_w = _active_bar_pose_w(env, bar_names, state)

    settled = state.spawned & (
        (_episode_time_s(env) - state.spawn_time_s) >= settling_time_s
    )
    new_references = settled & ~state.movement_reference_set
    state.movement_reference_pose_w[new_references] = bar_pose_w[new_references]
    state.movement_reference_set |= new_references

    translation = torch.linalg.vector_norm(
        bar_pose_w[:, :3] - state.movement_reference_pose_w[:, :3],
        dim=1,
    )
    quat_dot = torch.abs(
        torch.sum(
            bar_pose_w[:, 3:7]
            * state.movement_reference_pose_w[:, 3:7],
            dim=1,
        )
    )
    rotation = 2.0 * torch.acos(
        torch.clamp(quat_dot, min=0.0, max=1.0)
    )
    wooden_bar_training = (
        state.episode_phase == WOODEN_BAR_TRAINING_PHASE
    )
    return (
        wooden_bar_training
        & state.movement_reference_set
        & (
            (translation > translation_tolerance)
            | (rotation > rotation_tolerance)
        )
    )


def wooden_bar_deadline(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    time_limit_s: float,
) -> torch.Tensor:
    """Terminate active bar episodes if crossing exceeds the time limit."""
    state = _update_crossed(
        env,
        feet_cfg=feet_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )
    elapsed = _episode_time_s(env) - state.spawn_time_s
    wooden_bar_training = (
        state.episode_phase == WOODEN_BAR_TRAINING_PHASE
    )
    return (
        wooden_bar_training
        & state.spawned
        & ~state.crossed
        & (elapsed > time_limit_s)
    )


def wooden_bar_step_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
) -> torch.Tensor:
    """Reward height, signed forward speed, and progress during first entry."""
    if height_saturation <= 0.0:
        raise ValueError("height_saturation must be positive.")
    if forward_velocity_saturation <= 0.0:
        raise ValueError("forward_velocity_saturation must be positive.")
    if progress_unit <= 0.0:
        raise ValueError("progress_unit must be positive.")

    (
        state,
        sole_vertices_w,
        forward_w,
        _,
        footprint_max,
        overlaps_band,
        in_contact,
    ) = _update_foot_band_state(
        env,
        feet_cfg=feet_cfg,
        sensor_cfg=sensor_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )

    clearance = _minimum_foot_clearance(
        env,
        sole_vertices_w=sole_vertices_w,
        in_contact=in_contact,
    )
    height_score = torch.clamp(
        clearance / height_saturation,
        min=0.0,
        max=1.0,
    )

    robot = env.scene[feet_cfg.name]
    foot_velocity_w = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2]
    forward_velocity = torch.sum(
        foot_velocity_w * forward_w[:, None, :],
        dim=2,
    )
    velocity_score = torch.clamp(
        forward_velocity / forward_velocity_saturation,
        min=-1.0,
        max=1.0,
    )

    # The frontmost point has made zero progress when it first reaches the
    # band's back edge. Every progress_unit metres thereafter adds one.
    progress_score = torch.clamp(
        (footprint_max + band_half_width) / progress_unit,
        min=0.0,
    )
    bar_active = state.spawned & ~state.crossed
    active = (
        overlaps_band
        & bar_active.unsqueeze(1)
        & state.first_time_entering_strip.unsqueeze(1)
    )
    return torch.sum(
        height_score
        * velocity_score
        * progress_score
        * active.to(height_score.dtype),
        dim=1,
    )


def distance_to_front_edge_of_bar(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    desired_distance: float,
    linear_falloff_distance: float,
) -> torch.Tensor:
    """Reward an airborne foot for reaching a target past the band's front edge.

    The curve is zero while the frontmost sole point has not cleared the front
    edge, rises steeply to one at desired_distance, then falls linearly to zero
    over linear_falloff_distance.
    """
    if desired_distance <= 0.0:
        raise ValueError("desired_distance must be positive.")
    if linear_falloff_distance <= 0.0:
        raise ValueError("linear_falloff_distance must be positive.")

    (
        state,
        _,
        _,
        _,
        footprint_max,
        overlaps_band,
        in_contact,
    ) = _update_foot_band_state(
        env,
        feet_cfg=feet_cfg,
        sensor_cfg=sensor_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )

    distance_past_front_edge = torch.clamp(
        footprint_max - band_half_width,
        min=0.0,
    )
    rapid_rise = torch.sqrt(
        torch.clamp(
            distance_past_front_edge / desired_distance,
            min=0.0,
            max=1.0,
        )
    )
    linear_fall = torch.clamp(
        1.0
        - (distance_past_front_edge - desired_distance)
        / linear_falloff_distance,
        min=0.0,
        max=1.0,
    )
    distance_score = torch.where(
        distance_past_front_edge <= desired_distance,
        rapid_rise,
        linear_fall,
    )

    bar_active = state.spawned & ~state.crossed
    active = (
        overlaps_band
        & ~in_contact
        & bar_active.unsqueeze(1)
        & state.first_time_entering_strip.unsqueeze(1)
    )
    return torch.sum(
        distance_score * active.to(distance_score.dtype),
        dim=1,
    )


def feet_height_entering_band_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
) -> torch.Tensor:
    """Return the first entering foot's minimum clearance for one frame."""
    (
        state,
        sole_vertices_w,
        _,
        _,
        _,
        _,
        in_contact,
    ) = _update_foot_band_state(
        env,
        feet_cfg=feet_cfg,
        sensor_cfg=sensor_cfg,
        sole_vertices=sole_vertices,
        band_half_width=band_half_width,
    )
    clearance = _minimum_foot_clearance(
        env,
        sole_vertices_w=sole_vertices_w,
        in_contact=in_contact,
    )
    return torch.sum(
        clearance * state.first_entry_event.to(clearance.dtype),
        dim=1,
    )


def delayed_wooden_bar_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    wooden_bar_training_start_step: int,
) -> dict[str, float]:
    """Enable physical bars and all bar-specific terms after a global step."""
    del env_ids
    if wooden_bar_training_start_step < 0:
        raise ValueError(
            "wooden_bar_training_start_step must be non-negative."
        )

    state = _get_state(env)
    step = _curriculum_step(env)
    if step < wooden_bar_training_start_step:
        state.curriculum_phase = NO_BAR_TRAINING_PHASE
    else:
        state.curriculum_phase = WOODEN_BAR_TRAINING_PHASE

    return {
        "curriculum_step": float(step),
        "bar_curriculum_phase": float(state.curriculum_phase),
        "wooden_bar_training_enabled": float(
            state.curriculum_phase == WOODEN_BAR_TRAINING_PHASE
        ),
    }

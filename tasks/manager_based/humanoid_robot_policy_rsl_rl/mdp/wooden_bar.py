# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wooden-bar obstacle mechanics for the humanoid locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BAR_DISTANCE = 0.40


class _WoodenBarState:
    """Per-environment state shared by obstacle MDP terms."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.spawned = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.episode_has_bar = torch.zeros_like(self.spawned)
        self.crossed = torch.zeros_like(self.spawned)
        self.crossing_rewarded = torch.zeros_like(self.spawned)
        self.spawn_time_s = torch.zeros(env.num_envs, device=env.device)
        self.spawn_pose_w = torch.zeros(env.num_envs, 7, device=env.device)
        self.spawn_pose_w[:, 3] = 1.0
        self.movement_reference_pose_w = self.spawn_pose_w.clone()
        self.movement_reference_set = torch.zeros_like(self.spawned)
        self.forward_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.forward_w[:, 0] = 1.0
        self.curriculum_enabled = False
        self.current_height_index = 0
        self.active_bar_index = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )


def _get_state(env: ManagerBasedRLEnv) -> _WoodenBarState:
    if not hasattr(env, "_wooden_bar_state"):
        env._wooden_bar_state = _WoodenBarState(env)
    return env._wooden_bar_state


def _as_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def _active_bar_pose_w(
    env: ManagerBasedRLEnv,
    bar_names: Sequence[str],
    state: _WoodenBarState,
) -> torch.Tensor:
    """Return the root pose of the selected height variant in every environment."""
    all_bar_poses_w = torch.stack(
        [env.scene[name].data.root_state_w[:, :7] for name in bar_names],
        dim=1,
    )
    safe_indices = torch.clamp(state.active_bar_index, min=0)
    env_indices = torch.arange(env.num_envs, device=env.device)
    active_pose_w = all_bar_poses_w[env_indices, safe_indices]
    return torch.where(state.spawned.unsqueeze(1), active_pose_w, state.spawn_pose_w)


def _update_crossed(
    env: ManagerBasedRLEnv,
    robot_name: str,
    crossing_margin: float,
    maximum_lateral_offset: float,
) -> _WoodenBarState:
    state = _get_state(env)
    pending = state.spawned & ~state.crossed
    robot = env.scene[robot_name]
    relative_xy = robot.data.root_pos_w[:, :2] - state.spawn_pose_w[:, :2]
    longitudinal = torch.sum(relative_xy * state.forward_w, dim=1)
    side_w = torch.stack((-state.forward_w[:, 1], state.forward_w[:, 0]), dim=1)
    lateral = torch.abs(torch.sum(relative_xy * side_w, dim=1))

    crossed_now = pending & (longitudinal > crossing_margin) & (lateral <= maximum_lateral_offset)
    state.crossed |= crossed_now
    return state


class ObstacleAwareVelocityCommand(UniformVelocityCommand):
    """Sample forward/yaw commands and prevent stopping while a bar is active."""

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = _as_env_ids(self._env, env_ids)
        random_values = torch.empty(len(env_ids), device=self.device)

        # The task has no lateral command. The middle tensor element is retained
        # only because Isaac Lab's locomotion rewards use the standard SE(2)
        # command layout internally; it is not exposed to the policy.
        self.vel_command_b[env_ids, 0] = random_values.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = random_values.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = random_values.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = random_values.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = random_values.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        state = _get_state(self._env)
        bar_active = state.spawned[env_ids] & ~state.crossed[env_ids]
        self.is_standing_env[env_ids[bar_active]] = False

    def _update_command(self):
        state = _get_state(self._env)
        bar_active = state.spawned & ~state.crossed

        # A bar may appear while an environment is executing a standing command.
        # Restore the fixed walking speed before the parent applies standing masks.
        self.is_standing_env[bar_active] = False
        self.vel_command_b[bar_active, 0] = self.cfg.ranges.lin_vel_x[0]
        self.vel_command_b[:, 1] = 0.0
        super()._update_command()


@configclass
class ObstacleAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    """Forward/yaw-only command configuration without a lateral command field."""

    class_type: type = ObstacleAwareVelocityCommand

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        ang_vel_z: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING


def forward_yaw_velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return only forward velocity and yaw rate, omitting the fixed-zero lateral command."""
    command = env.command_manager.get_command(command_name)
    return command[:, (0, 2)]


def reset_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    bar_names: tuple[str, ...],
    hidden_depth: float,
    spawn_probability: float,
):
    """Hide every height variant, clear state, and sample bar episodes."""
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
    state.active_bar_index[env_ids] = -1
    if state.curriculum_enabled:
        state.episode_has_bar[env_ids] = (
            torch.rand(len(env_ids), device=env.device) < spawn_probability
        )
    else:
        state.episode_has_bar[env_ids] = False
    state.crossed[env_ids] = False
    state.crossing_rewarded[env_ids] = False
    state.spawn_time_s[env_ids] = 0.0
    state.spawn_pose_w[env_ids] = pose
    state.movement_reference_pose_w[env_ids] = pose
    state.movement_reference_set[env_ids] = False
    state.forward_w[env_ids, 0] = 1.0
    state.forward_w[env_ids, 1] = 0.0


def spawn_wooden_bar(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    bar_names: tuple[str, ...],
    bar_heights: tuple[float, ...],
    robot_name: str,
    distance_range: tuple[float, float],
    drop_clearance: float,
    command_name: str,
):
    """Place one bar 30-40 cm ahead after the walking-only curriculum phase."""
    state = _get_state(env)
    if not state.curriculum_enabled:
        return

    env_ids = _as_env_ids(env, env_ids)
    env_ids = env_ids[state.episode_has_bar[env_ids] & ~state.spawned[env_ids]]
    if len(env_ids) == 0:
        return

    if len(bar_names) != len(bar_heights):
        raise ValueError("bar_names and bar_heights must have the same length.")

    active_bar_index = state.current_height_index
    bar = env.scene[bar_names[active_bar_index]]
    bar_height = bar_heights[active_bar_index]
    robot = env.scene[robot_name]
    robot_quat_w = robot.data.root_quat_w[env_ids]
    robot_yaw_quat_w = yaw_quat(robot_quat_w)
    local_forward = torch.zeros(len(env_ids), 3, device=env.device)
    local_forward[:, 0] = 1.0
    forward_w = quat_apply(robot_yaw_quat_w, local_forward)

    distance = torch.empty(len(env_ids), device=env.device).uniform_(*distance_range)
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = robot.data.root_pos_w[env_ids, :2] + distance.unsqueeze(1) * forward_w[:, :2]
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + 0.5 * bar_height + drop_clearance
    pose[:, 3:7] = robot_yaw_quat_w
    velocity = torch.zeros(len(env_ids), 6, device=env.device)

    bar.write_root_pose_to_sim(pose, env_ids=env_ids)
    bar.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    state.spawned[env_ids] = True
    state.active_bar_index[env_ids] = active_bar_index
    state.crossed[env_ids] = False
    state.crossing_rewarded[env_ids] = False
    state.spawn_time_s[env_ids] = _episode_time_s(env)[env_ids]
    state.spawn_pose_w[env_ids] = pose
    state.movement_reference_pose_w[env_ids] = pose
    state.movement_reference_set[env_ids] = False
    state.forward_w[env_ids] = forward_w[:, :2]

    # The interval event runs after command generation. Update the command now
    # so a bar never appears during even one control step of a random stop.
    command_term = env.command_manager.get_term(command_name)
    if isinstance(command_term, ObstacleAwareVelocityCommand):
        command_term.is_standing_env[env_ids] = False
        command_term.vel_command_b[env_ids, 0] = command_term.cfg.ranges.lin_vel_x[0]
        command_term.vel_command_b[env_ids, 1] = 0.0


def wooden_bar_distance(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    robot_name: str,
    default_distance: float = DEFAULT_BAR_DISTANCE,
    noise_range: tuple[float, float] = (0.0, 0.0),
    crossing_margin: float = 0.10,
    maximum_lateral_offset: float = 0.225,
) -> torch.Tensor:
    """Return planar robot-to-bar distance, or 40 cm when absent/already crossed."""
    state = _update_crossed(env, robot_name, crossing_margin, maximum_lateral_offset)
    bar_pose_w = _active_bar_pose_w(env, bar_names, state)
    robot = env.scene[robot_name]

    distance = torch.linalg.vector_norm(bar_pose_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    visible = state.spawned & ~state.crossed
    measurement_noise = torch.empty_like(distance).uniform_(*noise_range)
    distance = torch.clamp(distance + measurement_noise, min=0.0)
    distance = torch.where(visible, distance, torch.full_like(distance, default_distance))
    return distance.unsqueeze(1)


def wooden_bar_moved(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    robot_name: str,
    translation_tolerance: float,
    rotation_tolerance: float,
    settling_time_s: float,
    crossing_margin: float = 0.10,
    maximum_lateral_offset: float = 0.225,
) -> torch.Tensor:
    """Terminate when an active bar is translated or rotated beyond tolerance."""
    state = _update_crossed(env, robot_name, crossing_margin, maximum_lateral_offset)
    bar_pose_w = _active_bar_pose_w(env, bar_names, state)

    settled = state.spawned & ((_episode_time_s(env) - state.spawn_time_s) >= settling_time_s)
    new_references = settled & ~state.movement_reference_set
    state.movement_reference_pose_w[new_references] = bar_pose_w[new_references]
    state.movement_reference_set |= new_references

    translation = torch.linalg.vector_norm(bar_pose_w[:, :3] - state.movement_reference_pose_w[:, :3], dim=1)
    quat_dot = torch.abs(
        torch.sum(bar_pose_w[:, 3:7] * state.movement_reference_pose_w[:, 3:7], dim=1)
    )
    rotation = 2.0 * torch.acos(torch.clamp(quat_dot, min=0.0, max=1.0))
    return state.movement_reference_set & (
        (translation > translation_tolerance) | (rotation > rotation_tolerance)
    )


def wooden_bar_deadline(
    env: ManagerBasedRLEnv,
    robot_name: str,
    time_limit_s: float,
    crossing_margin: float = 0.10,
    maximum_lateral_offset: float = 0.225,
) -> torch.Tensor:
    """Terminate if the robot has not crossed within 20 seconds of appearance."""
    state = _update_crossed(env, robot_name, crossing_margin, maximum_lateral_offset)
    elapsed = _episode_time_s(env) - state.spawn_time_s
    return state.spawned & ~state.crossed & (elapsed > time_limit_s)


def wooden_bar_crossing_reward(
    env: ManagerBasedRLEnv,
    robot_name: str,
    crossing_margin: float = 0.10,
    maximum_lateral_offset: float = 0.225,
) -> torch.Tensor:
    """Give a one-step reward when the robot first clears the bar."""
    state = _update_crossed(env, robot_name, crossing_margin, maximum_lateral_offset)
    newly_crossed = state.crossed & ~state.crossing_rewarded
    state.crossing_rewarded |= newly_crossed
    return newly_crossed.float()


def wooden_bar_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    bar_heights: tuple[float, ...],
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Enable bars, then increase their height from the first to last variant."""
    del env_ids
    state = _get_state(env)
    step = env.common_step_counter
    state.curriculum_enabled = step >= start_step

    if not bar_heights:
        raise ValueError("bar_heights must contain at least one height.")

    if end_step <= start_step:
        progress = 1.0
    else:
        progress = (step - start_step) / (end_step - start_step)
        progress = min(max(progress, 0.0), 1.0)

    state.current_height_index = min(
        int(progress * (len(bar_heights) - 1)),
        len(bar_heights) - 1,
    )
    current_height = bar_heights[state.current_height_index]
    return {
        "bar_enabled": float(state.curriculum_enabled),
        "bar_height_m": current_height,
    }


#this curriculm is trying to fix the issue of the robot not really moving 
def fixed_forward_speed_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    initial_speed: float,
    final_speed: float,
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly increase one fixed forward-speed target.

    All non-standing environments receive exactly the same forward speed.
    This is not uniform command sampling.
    """

    del env_ids

    if end_step <= start_step:
        progress = 1.0
    else:
        progress = (
            env.common_step_counter - start_step
        ) / (end_step - start_step)
        progress = min(max(progress, 0.0), 1.0)

    speed = initial_speed + progress * (final_speed - initial_speed)

    command_term = env.command_manager.get_term(command_name)

    # Update future command resampling.
    command_term.cfg.ranges.lin_vel_x = (speed, speed)

    # Also update current commands immediately instead of waiting for the
    # 10-second command-resampling interval.
    moving_envs = ~command_term.is_standing_env
    command_term.vel_command_b[moving_envs, 0] = speed
    command_term.vel_command_b[:, 1] = 0.0

    return {"fixed_forward_speed": speed}

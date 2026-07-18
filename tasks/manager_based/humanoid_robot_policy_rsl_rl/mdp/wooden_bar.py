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
    from isaaclab.managers import SceneEntityCfg


DEFAULT_BAR_DISTANCE = 0.50
NORMAL_WALKING_PHASE = 0
STRIDE_TRAINING_PHASE = 1
OBSTACLE_TRAINING_PHASE = 2

def _curriculum_step(env: ManagerBasedRLEnv) -> int:
    """Return the global curriculum step shared by all curriculum terms."""
    return int(env.common_step_counter)




class _WoodenBarState:
    """Per-environment state shared by obstacle MDP terms."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.spawned = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.episode_has_bar = torch.zeros_like(self.spawned)
        self.crossed = torch.zeros_like(self.spawned)
        self.crossing_rewarded = torch.zeros_like(self.spawned)
        self.spawn_time_s = torch.zeros(env.num_envs, device=env.device)
        self.spawn_delay_s = torch.full(
            (env.num_envs,), float("inf"), device=env.device
        )
        self.spawn_pose_w = torch.zeros(env.num_envs, 7, device=env.device)
        self.spawn_pose_w[:, 3] = 1.0
        self.movement_reference_pose_w = self.spawn_pose_w.clone()
        self.movement_reference_set = torch.zeros_like(self.spawned)
        self.forward_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.forward_w[:, 0] = 1.0
        self.curriculum_phase = NORMAL_WALKING_PHASE
        self.episode_phase = torch.full(
            (env.num_envs,),
            NORMAL_WALKING_PHASE,
            dtype=torch.long,
            device=env.device,
        )
        self.current_height_index = 0
        self.active_bar_index = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )


def _get_state(env: ManagerBasedRLEnv) -> _WoodenBarState:
    if not hasattr(env, "_wooden_bar_state"):
        env._wooden_bar_state = _WoodenBarState(env)
    return env._wooden_bar_state


def stride_training_bar_active(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a mask for stage-two episodes with a visible wooden bar."""
    state = _get_state(env)
    return (
        (state.episode_phase == STRIDE_TRAINING_PHASE)
        & state.spawned
        & ~state.crossed
    )


def _as_env_ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def curriculum_time_out(
    env: ManagerBasedRLEnv,
    normal_training_length_s: float,
    stride_training_length_s: float,
    obstacle_training_length_s: float,
) -> torch.Tensor:
    """Time out episodes using the duration assigned to their curriculum phase."""
    state = _get_state(env)
    episode_limit_s = torch.full(
        (env.num_envs,),
        normal_training_length_s,
        device=env.device,
    )
    episode_limit_s = torch.where(
        state.episode_phase == STRIDE_TRAINING_PHASE,
        stride_training_length_s,
        episode_limit_s,
    )
    episode_limit_s = torch.where(
        state.episode_phase == OBSTACLE_TRAINING_PHASE,
        obstacle_training_length_s,
        episode_limit_s,
    )
    return _episode_time_s(env) >= episode_limit_s - env.step_dt


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
    feet_cfg: SceneEntityCfg,
) -> _WoodenBarState:
    """Mark the bar crossed once the centers of both feet are past it."""
    state = _get_state(env)
    pending = state.spawned & ~state.crossed
    robot = env.scene[feet_cfg.name]
    foot_centers_w = robot.data.body_com_pos_w[:, feet_cfg.body_ids, :2]
    if foot_centers_w.shape[1] != 2:
        raise ValueError(
            "Wooden-bar crossing requires exactly two foot bodies, "
            f"but received {foot_centers_w.shape[1]}."
        )

    relative_xy = foot_centers_w - state.spawn_pose_w[:, None, :2]
    longitudinal = torch.sum(relative_xy * state.forward_w[:, None, :], dim=2)
    both_feet_past_bar = torch.all(longitudinal > 0.0, dim=1)

    crossed_now = pending & both_feet_past_bar
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

        # Do not let the parent command generator treat an active bar
        # environment as a standing environment.
        self.is_standing_env[bar_active] = False

        # Let Isaac Lab perform its normal command update first.
        super()._update_command()

        # Enforce the obstacle-crossing command after the parent update.
        # This runs every control step, so command resampling cannot introduce
        # a nonzero yaw command while the bar is active.
        self.vel_command_b[bar_active, 0] = self.cfg.ranges.lin_vel_x[0]
        self.vel_command_b[bar_active, 1] = 0.0
        self.vel_command_b[bar_active, 2] = 0.0

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
    stride_training_spawn_probability: float,
    obstacle_training_spawn_probability: float,
    stride_training_spawn_delay_s: float,
    obstacle_training_spawn_delay_range_s: tuple[float, float],
):
    """Hide every height variant and schedule a bar for the current phase."""
    if stride_training_spawn_delay_s < 0.0:
        raise ValueError("stride_training_spawn_delay_s must be non-negative.")
    if (
        obstacle_training_spawn_delay_range_s[0] < 0.0
        or obstacle_training_spawn_delay_range_s[1]
        < obstacle_training_spawn_delay_range_s[0]
    ):
        raise ValueError(
            "obstacle_training_spawn_delay_range_s must be ordered and non-negative."
        )

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
    state.episode_phase[env_ids] = state.curriculum_phase
    spawn_probability = torch.zeros(len(env_ids), device=env.device)
    stride_training = state.episode_phase[env_ids] == STRIDE_TRAINING_PHASE
    obstacle_training = state.episode_phase[env_ids] == OBSTACLE_TRAINING_PHASE
    spawn_probability[stride_training] = stride_training_spawn_probability
    spawn_probability[obstacle_training] = obstacle_training_spawn_probability
    state.episode_has_bar[env_ids] = (
        torch.rand(len(env_ids), device=env.device) < spawn_probability
    )

    # Schedule appearance relative to the start of this episode, rather than
    # relative to the EventManager's repeating interval timer.
    state.spawn_delay_s[env_ids] = float("inf")
    stride_training_env_ids = env_ids[stride_training]
    state.spawn_delay_s[stride_training_env_ids] = stride_training_spawn_delay_s
    obstacle_training_env_ids = env_ids[obstacle_training]
    if len(obstacle_training_env_ids) > 0:
        state.spawn_delay_s[obstacle_training_env_ids] = torch.empty(
            len(obstacle_training_env_ids), device=env.device
        ).uniform_(*obstacle_training_spawn_delay_range_s)

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
    stride_bar_name: str,
    bar_heights: tuple[float, ...],
    robot_name: str,
    stride_training_distance_range: tuple[float, float],
    obstacle_training_distance_range: tuple[float, float],
    drop_clearance: float,
    command_name: str,
):
    """Place one bar at the distance selected by the episode's curriculum phase."""
    state = _get_state(env)
    if state.curriculum_phase == NORMAL_WALKING_PHASE:
        return

    env_ids = _as_env_ids(env, env_ids)
    ready_to_spawn = (
        _episode_time_s(env)[env_ids] >= state.spawn_delay_s[env_ids]
    )
    env_ids = env_ids[
        state.episode_has_bar[env_ids]
        & ~state.spawned[env_ids]
        & ready_to_spawn
    ]
    if len(env_ids) == 0:
        return

    if len(bar_names) != len(bar_heights):
        raise ValueError("bar_names and bar_heights must have the same length.")

    obstacle_bar_index = state.current_height_index
    robot = env.scene[robot_name]
    robot_quat_w = robot.data.root_quat_w[env_ids]
    robot_yaw_quat_w = yaw_quat(robot_quat_w)
    local_forward = torch.zeros(len(env_ids), 3, device=env.device)
    local_forward[:, 0] = 1.0
    forward_w = quat_apply(robot_yaw_quat_w, local_forward)

    distance = torch.empty(len(env_ids), device=env.device)
    stride_training = state.episode_phase[env_ids] == STRIDE_TRAINING_PHASE
    obstacle_training = state.episode_phase[env_ids] == OBSTACLE_TRAINING_PHASE
    if torch.any(stride_training):
        distance[stride_training] = torch.empty_like(
            distance[stride_training]
        ).uniform_(*stride_training_distance_range)
    if torch.any(obstacle_training):
        distance[obstacle_training] = torch.empty_like(
            distance[obstacle_training]
        ).uniform_(*obstacle_training_distance_range)

    bar_height = torch.empty(len(env_ids), device=env.device)
    bar_height[stride_training] = bar_heights[0]
    bar_height[obstacle_training] = bar_heights[obstacle_bar_index]

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = robot.data.root_pos_w[env_ids, :2] + distance.unsqueeze(1) * forward_w[:, :2]
    # The collisionless stride bar is kinematic, so place its bottom directly
    # at the nominal terrain height.  Physical obstacle bars keep their drop
    # clearance and settle onto the terrain under gravity.
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + 0.5 * bar_height
    pose[obstacle_training, 2] += drop_clearance
    pose[:, 3:7] = robot_yaw_quat_w
    velocity = torch.zeros(len(env_ids), 6, device=env.device)

    active_bar_indices = torch.empty(
        len(env_ids), dtype=torch.long, device=env.device
    )

    if torch.any(stride_training):
        stride_env_ids = env_ids[stride_training]
        stride_bar = env.scene[stride_bar_name]
        stride_bar.write_root_pose_to_sim(
            pose[stride_training], env_ids=stride_env_ids
        )
        stride_bar.write_root_velocity_to_sim(
            velocity[stride_training], env_ids=stride_env_ids
        )
        # ALL_WOODEN_BAR_NAMES appends the stride bar after the physical bars.
        active_bar_indices[stride_training] = len(bar_names)

    if torch.any(obstacle_training):
        obstacle_env_ids = env_ids[obstacle_training]
        obstacle_bar = env.scene[bar_names[obstacle_bar_index]]
        obstacle_bar.write_root_pose_to_sim(
            pose[obstacle_training], env_ids=obstacle_env_ids
        )
        obstacle_bar.write_root_velocity_to_sim(
            velocity[obstacle_training], env_ids=obstacle_env_ids
        )
        active_bar_indices[obstacle_training] = obstacle_bar_index

    state.spawned[env_ids] = True
    state.active_bar_index[env_ids] = active_bar_indices
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

        # Immediately switch to a straight-forward crossing command.
        command_term.vel_command_b[env_ids, 0] = (
            command_term.cfg.ranges.lin_vel_x[0]
        )
        command_term.vel_command_b[env_ids, 1] = 0.0
        command_term.vel_command_b[env_ids, 2] = 0.0


def wooden_bar_distance(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    feet_cfg: SceneEntityCfg,
    default_distance: float = DEFAULT_BAR_DISTANCE,
    noise_range: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Return signed forward bar distance, or 50 cm when absent/already crossed."""
    state = _update_crossed(env, feet_cfg)
    bar_pose_w = _active_bar_pose_w(env, bar_names, state)
    robot = env.scene[feet_cfg.name]

    relative_bar_xy = bar_pose_w[:, :2] - robot.data.root_pos_w[:, :2]
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, 0] = 1.0
    robot_forward_w = quat_apply(
        yaw_quat(robot.data.root_quat_w), local_forward
    )[:, :2]
    distance = torch.sum(relative_bar_xy * robot_forward_w, dim=1)
    visible = state.spawned & ~state.crossed
    measurement_noise = torch.empty_like(distance).uniform_(*noise_range)
    distance = distance + measurement_noise
    distance = torch.where(visible, distance, torch.full_like(distance, default_distance))
    return distance.unsqueeze(1)


def stride_training_reward_scale(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    feet_cfg: SceneEntityCfg,
    activation_distance: float,
    full_weight_distance: float,
) -> torch.Tensor:
    """Scale stride-training foot rewards from zero at 20 cm to full at 10 cm."""
    if activation_distance <= full_weight_distance:
        raise ValueError(
            "activation_distance must be greater than full_weight_distance."
        )

    distance = wooden_bar_distance(
        env,
        bar_names=bar_names,
        feet_cfg=feet_cfg,
        noise_range=(0.0, 0.0),
    ).squeeze(1)
    scale = torch.clamp(
        (activation_distance - distance)
        / (activation_distance - full_weight_distance),
        min=0.0,
        max=1.0,
    )
    return scale * stride_training_bar_active(env).float()


def wooden_bar_moved(
    env: ManagerBasedRLEnv,
    bar_names: tuple[str, ...],
    feet_cfg: SceneEntityCfg,
    translation_tolerance: float,
    rotation_tolerance: float,
    settling_time_s: float,
) -> torch.Tensor:
    """Terminate when an active bar is translated or rotated beyond tolerance."""
    state = _update_crossed(env, feet_cfg)
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
    obstacle_training = state.episode_phase == OBSTACLE_TRAINING_PHASE
    return obstacle_training & state.movement_reference_set & (
        (translation > translation_tolerance) | (rotation > rotation_tolerance)
    )


def wooden_bar_deadline(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    time_limit_s: float,
) -> torch.Tensor:
    """Terminate if the robot has not crossed within 20 seconds of appearance."""
    state = _update_crossed(env, feet_cfg)
    elapsed = _episode_time_s(env) - state.spawn_time_s
    obstacle_training = state.episode_phase == OBSTACLE_TRAINING_PHASE
    return obstacle_training & state.spawned & ~state.crossed & (elapsed > time_limit_s)


def wooden_bar_crossing_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Give a one-step reward when the robot first clears the bar."""
    state = _update_crossed(env, feet_cfg)
    newly_crossed = state.crossed & ~state.crossing_rewarded
    state.crossing_rewarded |= newly_crossed
    return newly_crossed.float()


def wooden_bar_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    bar_heights: tuple[float, ...],
    stride_training_start_step: int,
    obstacle_training_start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Run normal walking, stride preparation, then the obstacle curriculum."""
    del env_ids
    state = _get_state(env)
    step = _curriculum_step(env)
    if step < stride_training_start_step:
        state.curriculum_phase = NORMAL_WALKING_PHASE
    elif step < obstacle_training_start_step:
        state.curriculum_phase = STRIDE_TRAINING_PHASE
    else:
        state.curriculum_phase = OBSTACLE_TRAINING_PHASE

    if not bar_heights:
        raise ValueError("bar_heights must contain at least one height.")

    if end_step <= obstacle_training_start_step:
        progress = 1.0
    else:
        progress = (step - obstacle_training_start_step) / (
            end_step - obstacle_training_start_step
        )
        progress = min(max(progress, 0.0), 1.0)

    state.current_height_index = min(
        int(progress * (len(bar_heights) - 1)),
        len(bar_heights) - 1,
    )
    current_height = bar_heights[state.current_height_index]
    return {
        "curriculum_step": float(step),
        "bar_enabled": float(state.curriculum_phase != NORMAL_WALKING_PHASE),
        "bar_curriculum_phase": float(state.curriculum_phase),
        "bar_height_m": current_height,
    }


def stride_reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stride_training_start_step: int,
    decay_end_step: int,
    initial_clearance_weight: float,
    final_clearance_weight: float,
    initial_stride_weight: float,
    final_stride_weight: float,
    clearance_term_name: str = "feet_clearance",
    stride_term_name: str = "feet_stride_length",
) -> dict[str, float]:
    """Linearly decay stride-training reward weights to their normal values."""

    del env_ids

    if decay_end_step <= stride_training_start_step:
        raise ValueError(
            "decay_end_step must be greater than stride_training_start_step."
        )

    step = _curriculum_step(env)

    # 0 at the beginning of stride training and 1 at decay_end_step.
    progress = (
        (step - stride_training_start_step)
        / (decay_end_step - stride_training_start_step)
    )
    progress = min(max(progress, 0.0), 1.0)

    clearance_weight = (
        initial_clearance_weight
        + progress * (final_clearance_weight - initial_clearance_weight)
    )
    stride_weight = (
        initial_stride_weight
        + progress * (final_stride_weight - initial_stride_weight)
    )

    clearance_cfg = env.reward_manager.get_term_cfg(clearance_term_name)
    clearance_cfg.weight = clearance_weight
    env.reward_manager.set_term_cfg(clearance_term_name, clearance_cfg)

    stride_cfg = env.reward_manager.get_term_cfg(stride_term_name)
    stride_cfg.weight = stride_weight
    env.reward_manager.set_term_cfg(stride_term_name, stride_cfg)

    # These values should appear in the curriculum statistics/TensorBoard.
    return {
        "curriculum_step": float(step),
        "feet_clearance_weight": clearance_weight,
        "feet_stride_length_weight": stride_weight,
        "stride_reward_weight_progress": progress,
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
        step = _curriculum_step(env)

        progress = (
            step - start_step
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

    return {
        "curriculum_step": float(_curriculum_step(env)),
        "fixed_forward_speed": speed,
    }

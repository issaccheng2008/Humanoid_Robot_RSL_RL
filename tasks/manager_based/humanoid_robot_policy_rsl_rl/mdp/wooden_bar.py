# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swing-cycle-driven step-distance commands and wooden-bar mechanics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from .stop_gate import StopGate

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NORMAL_WALKING_PHASE = 1
VIRTUAL_BAND_PHASE = 2
COLLISIONLESS_BAR_PHASE = 3
PHYSICAL_BAR_PHASE = 4
MIXED_COMMAND_PHASE = 5

NUM_STICKS = 10
FIRST_STICK_X = 0.50
STICK_CLEAR_GAP = 0.20
WOODEN_BAR_WIDTH = 0.03
WOODEN_BAR_LENGTH = 0.80
WOODEN_BAR_HEIGHT = 0.03
STICK_SPACING = STICK_CLEAR_GAP + WOODEN_BAR_WIDTH  # 0.23m pitch
STICK_HALF_WIDTH = 0.5 * WOODEN_BAR_WIDTH
STICK_HALF_LENGTH = 0.5 * WOODEN_BAR_LENGTH  # 0.40m


def _control_step(env: ManagerBasedRLEnv) -> int:
    """Return the shared environment/control-step counter."""
    return int(env.common_step_counter)


def _single_swing_foot(
    in_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return which environments have one swing foot and that foot's index."""
    if in_contact.ndim != 2 or in_contact.shape[1] != 2:
        raise ValueError("Swing-foot tracking requires exactly two feet.")
    swing_mask = ~in_contact
    has_one_swing_foot = torch.sum(swing_mask, dim=1) == 1
    swing_foot_index = torch.argmax(swing_mask.long(), dim=1)
    return has_one_swing_foot, swing_foot_index


def _swing_cycle_touchdown_events(
    in_contact: torch.Tensor,
    tracked_swing_foot_index: torch.Tensor,
    update_envs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finish tracked swing cycles and arm the next unambiguous swing foot."""
    has_one_swing_foot, current_swing_foot_index = _single_swing_foot(
        in_contact
    )
    if tracked_swing_foot_index.shape != in_contact.shape[:1]:
        raise ValueError(
            "tracked_swing_foot_index must have one entry per environment."
        )
    if update_envs.shape != in_contact.shape[:1]:
        raise ValueError("update_envs must have one entry per environment.")

    next_swing_foot_index = tracked_swing_foot_index.clone()
    has_tracked_swing = next_swing_foot_index >= 0
    tracked_index = torch.clamp(next_swing_foot_index, min=0)
    tracked_swing_touched_down = torch.gather(
        in_contact, 1, tracked_index.unsqueeze(1)
    ).squeeze(1)
    touchdown_envs = (
        update_envs & has_tracked_swing & tracked_swing_touched_down
    )

    touchdown_event = torch.zeros_like(in_contact)
    touchdown_event[
        touchdown_envs, next_swing_foot_index[touchdown_envs]
    ] = True
    next_swing_foot_index[touchdown_envs] = -1

    # A gait may transfer support without a sampled double-support frame. Once
    # the tracked foot lands, immediately arm the other foot if it is already
    # the only foot off the ground.
    start_swing = (
        update_envs
        & has_one_swing_foot
        & (next_swing_foot_index < 0)
    )
    next_swing_foot_index[start_swing] = current_swing_foot_index[start_swing]
    return touchdown_event, next_swing_foot_index


class _CrossingState:
    """Per-environment state shared by commands, rewards, and terminations."""

    def __init__(self, env: ManagerBasedRLEnv):
        bools = lambda: torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        longs = lambda value=0: torch.full(
            (env.num_envs,), value, dtype=torch.long, device=env.device
        )

        self.stop = StopGate(
            env.num_envs, env.device, env.step_dt,
            hold_s=getattr(env.cfg, "stop_hold_s", 0.5),
            timeout_s=getattr(env.cfg, "stop_timeout_s", 3.0),
        )
        self.stop_enabled = getattr(env.cfg, "stop_before_crossing", False)
        self.num_sticks = NUM_STICKS
        initial_gap = getattr(env.cfg, "hurdle_initial_gap", STICK_CLEAR_GAP)
        self.course_spacing = torch.full((env.num_envs,), initial_gap + WOODEN_BAR_WIDTH, device=env.device)
        self.course_level = longs()
        gap_levels = getattr(env.cfg, "hurdle_gap_levels", (initial_gap,))
        if initial_gap not in gap_levels or any(a <= b for a, b in zip(gap_levels, gap_levels[1:])):
            raise ValueError("Initial gap must be a curriculum level; levels must strictly decrease.")
        self.curriculum_level = gap_levels.index(initial_gap)
        self.course_level[:] = self.curriculum_level
        self.curriculum_episodes = self.curriculum_completed = 0
        self.curriculum_attempts = self.curriculum_correct = 0
        self.landing_attempts = longs()
        self.correct_landings = longs()
        self.wrong_landing_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.target_approach_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.target_error_previous = torch.zeros((env.num_envs, 2), device=env.device)
        self.target_error_valid = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
        self.current_stick_index = longs(0)
        self.sticks_cleared_count = longs(0)
        self.stick_cleared_event = bools()
        self.stick_failed_event = bools()
        self.stick_collision_event = bools()
        self.stick_collision_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.stick_hit_flags = torch.zeros(
            (env.num_envs, NUM_STICKS), dtype=torch.bool, device=env.device
        )
        self.stick_landing_reward_paid = torch.zeros(
            (env.num_envs, NUM_STICKS, 2), dtype=torch.bool, device=env.device
        )
        self.forward_progress_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.stick_over_clearance_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.stick_landing_center_reward_val = torch.zeros(env.num_envs, device=env.device)
        self.task_success = bools()
        self.task_success_event = bools()
        self.success_support_steps = longs()
        self.initialized = bools()
        self.training_phase = longs(NORMAL_WALKING_PHASE)
        self.step_distance = torch.zeros(env.num_envs, device=env.device)
        self.crossing_command = bools()
        self.following_step_command_stage = longs()
        self.crossing_foot_index = longs(-1)
        self.following_foot_index = longs(-1)
        self.following_foot_touchdown_event = bools()
        self.collisionless_bar_contact_event = bools()
        self.collisionless_bar_contacted = bools()

        self.phase_5_no_bar_episode = bools()
        self.phase_5_stop_command = bools()
        self.phase_5_stop_time_s = torch.zeros(
            env.num_envs, device=env.device
        )
        self.phase_5_stop_step_reward_paid = bools()

        self.touchdown_count = longs()
        self.trigger_touchdown_index = longs()
        self.swing_foot_index = longs(-1)
        self.step_distance_reward_airborne_seen = torch.zeros(
            (env.num_envs, 2), dtype=torch.bool, device=env.device
        )
        self.touchdown_event = torch.zeros(
            (env.num_envs, 2), dtype=torch.bool, device=env.device
        )
        self.touchdown_actual_step = torch.zeros(
            (env.num_envs, 2), device=env.device
        )
        self.touchdown_target_step = torch.zeros_like(
            self.touchdown_actual_step
        )
        self.touchdown_reward_eligible = torch.zeros_like(self.touchdown_event)
        self.step_distance_reward_paid_step = longs(-1)
        self.last_control_update_step = longs(-1)

        self.spawned = bools()
        self.crossed = bools()
        self.crossing_completed = bools()
        self.spawn_time_s = torch.zeros(env.num_envs, device=env.device)
        self.spawn_pose_w = torch.zeros(env.num_envs, 7, device=env.device)
        self.spawn_pose_w[:, 3] = 1.0
        self.forward_w = torch.zeros(env.num_envs, 2, device=env.device)
        self.forward_w[:, 0] = 1.0
        self.crossing_half_width = torch.zeros(
            env.num_envs, device=env.device
        )

        self.movement_reference_pose_w = self.spawn_pose_w.clone()
        self.movement_reference_set = bools()

        self.first_entry_event = torch.zeros_like(self.touchdown_event)
        self.bar_reward_foot_entered = torch.zeros_like(self.touchdown_event)
        self.bar_reward_foot_active = torch.zeros_like(self.touchdown_event)
        self.bar_reward_stepping_foot = torch.zeros_like(self.touchdown_event)
        self.bar_reward_following_foot = torch.zeros_like(self.touchdown_event)
        self.stepping_foot_touchdown_distance_to_band_edge = torch.zeros(
            env.num_envs, device=env.device
        )
        self.stepping_foot_touchdown_distance_cached = bools()
        self.last_band_update_step = longs(-1)

        self.sole_vertices = None
        self.sole_vertices_key = None


def _get_state(env: ManagerBasedRLEnv) -> _CrossingState:
    if not hasattr(env, "_wooden_bar_state"):
        env._wooden_bar_state = _CrossingState(env)
    return env._wooden_bar_state


def _as_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(
            env.num_envs, device=env.device, dtype=torch.long
        )[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _episode_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def _sole_vertices_tensor(
    env: ManagerBasedRLEnv,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
) -> torch.Tensor:
    """Validate and cache the two complete sole-perimeter vertex sets."""
    if len(sole_vertices) != 2 or any(
        len(vertices) < 3 for vertices in sole_vertices
    ):
        raise ValueError(
            "sole_vertices must contain at least three vertices for each "
            "of two feet."
        )

    state = _get_state(env)
    if state.sole_vertices is None or state.sole_vertices_key != sole_vertices:
        state.sole_vertices = torch.tensor(
            sole_vertices, dtype=torch.float, device=env.device
        )
        state.sole_vertices_key = sole_vertices
    return state.sole_vertices


def _sole_geometry_w(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
) -> torch.Tensor:
    """Return every configured sole-perimeter vertex in world coordinates."""
    robot = env.scene[feet_cfg.name]
    foot_pos_w = robot.data.body_pos_w[:, feet_cfg.body_ids]
    foot_quat_w = robot.data.body_quat_w[:, feet_cfg.body_ids]
    if foot_pos_w.shape[1] != 2:
        raise ValueError(
            "Step/crossing geometry requires exactly two foot bodies, "
            f"but received {foot_pos_w.shape[1]}."
        )

    vertices_b = _sole_vertices_tensor(env, sole_vertices)
    num_envs, num_feet = foot_pos_w.shape[:2]
    num_vertices = vertices_b.shape[1]
    vertices = vertices_b.unsqueeze(0).expand(num_envs, -1, -1, -1)
    quaternions = foot_quat_w.unsqueeze(2).expand(
        -1, -1, num_vertices, -1
    )
    rotated = quat_apply(
        quaternions.reshape(-1, 4), vertices.reshape(-1, 3)
    ).reshape(num_envs, num_feet, num_vertices, 3)
    return foot_pos_w.unsqueeze(2) + rotated


def _base_forward_and_sole_front(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return yaw-only base forward, yaw quaternion, and both sole fronts."""
    robot = env.scene[feet_cfg.name]
    robot_yaw_quat_w = yaw_quat(robot.data.root_quat_w.torch)
    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, 0] = 1.0
    forward_w = quat_apply(robot_yaw_quat_w, local_forward)
    relative_xy = (
        sole_vertices_w[..., :2]
        - robot.data.root_pos_w[:, None, None, :2]
    )
    longitudinal = torch.sum(
        relative_xy * forward_w[:, None, None, :2], dim=3
    )
    return forward_w[:, :2], robot_yaw_quat_w, torch.amax(
        longitudinal, dim=2
    )


def _crossing_foot_geometry(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sole geometry relative to the fixed active band/bar pose."""
    state = _get_state(env)
    sole_vertices_w = _sole_geometry_w(env, feet_cfg, sole_vertices)
    relative_xy = (
        sole_vertices_w[..., :2]
        - state.spawn_pose_w[:, None, None, :2]
    )
    longitudinal = torch.sum(
        relative_xy * state.forward_w[:, None, None, :], dim=3
    )
    footprint_min = torch.amin(longitudinal, dim=2)
    footprint_max = torch.amax(longitudinal, dim=2)
    return sole_vertices_w, state.forward_w, footprint_min, footprint_max


def _sole_min_z_over_rectangle(
    vertices: torch.Tensor, half_width: float, half_length: float
) -> torch.Tensor:
    """Minimum height of the planar convex sole over a stick's XY footprint.

    The minimum of a linear height function over the intersection occurs at
    a sole vertex, a sole/rectangle edge intersection, or a rectangle corner.
    All candidates are evaluated in parallel; disjoint footprints return inf.
    """
    xy = vertices[..., :2]
    inside = (
        (xy[..., 0].abs() <= half_width + 1e-7)
        & (xy[..., 1].abs() <= half_length + 1e-7)
    )
    minimum = torch.where(inside, vertices[..., 2], float("inf")).amin(dim=-1)
    edges = torch.roll(vertices, shifts=-1, dims=-2) - vertices
    axes = [0, 0, 1, 1]
    bounds = vertices.new_tensor([-half_width, half_width, -half_length, half_length])
    delta = edges[..., axes]
    nonparallel = delta.abs() > 1e-9
    fraction = (bounds - vertices[..., axes]) / torch.where(
        nonparallel, delta, torch.ones_like(delta)
    )
    intersections = vertices.unsqueeze(-2) + fraction.unsqueeze(-1) * edges.unsqueeze(-2)
    valid = (
        nonparallel & (fraction >= 0.0) & (fraction <= 1.0)
        & (intersections[..., 0].abs() <= half_width + 1e-7)
        & (intersections[..., 1].abs() <= half_length + 1e-7)
    )
    edge_minimum = torch.where(valid, intersections[..., 2], float("inf")).amin(dim=(-2, -1))
    minimum = torch.minimum(minimum, edge_minimum)

    corners = vertices.new_tensor(
        [
            [-half_width, -half_length], [-half_width, half_length],
            [half_width, half_length], [half_width, -half_length],
        ]
    )
    relative = corners - xy.unsqueeze(-2)
    cross = (
        edges[..., 0].unsqueeze(-1) * relative[..., 1]
        - edges[..., 1].unsqueeze(-1) * relative[..., 0]
    )
    corner_inside = (cross >= -1e-8).all(dim=-2) | (cross <= 1e-8).all(dim=-2)
    origin = vertices[..., 0, :]
    fan = vertices[..., 1:, :] - origin.unsqueeze(-2)
    normals = torch.linalg.cross(fan[..., :-1, :], fan[..., 1:, :], dim=-1)
    normal_index = normals[..., 2].abs().argmax(dim=-1)
    normal = normals.gather(
        -2, normal_index[..., None, None].expand(*normal_index.shape, 1, 3)
    ).squeeze(-2)
    planar = normal[..., 2].abs() > 1e-9
    nz = torch.where(planar, normal[..., 2], torch.ones_like(normal[..., 2]))
    corner_z = origin[..., 2, None] - (
        (normal[..., :2].unsqueeze(-2) * (corners - origin[..., :2].unsqueeze(-2))).sum(dim=-1)
    ) / nz.unsqueeze(-1)
    corner_minimum = torch.where(
        corner_inside & planar.unsqueeze(-1), corner_z, float("inf")
    ).amin(dim=-1)
    return torch.minimum(minimum, corner_minimum)



def _course_inside(
    sole_vertices_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    half_length: float,
    course_end_x: float | torch.Tensor,
) -> torch.Tensor:
    """Keep the base and the entire soles inside the world-aligned course."""
    root = root_pos_w - env_origins
    soles = sole_vertices_w - env_origins[:, None, None, :]
    return (
        (root[:, 0] >= -0.1) & (root[:, 0] <= course_end_x)
        & (root[:, 1].abs() <= half_length)
        & (soles[..., 1].amin(dim=(1, 2)) >= -half_length)
        & (soles[..., 1].amax(dim=(1, 2)) <= half_length)
        & (soles[..., 0].amax(dim=(1, 2)) <= course_end_x)
    )


def hurdle_out_of_bounds(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    half_length: float,
    course_end_x: float | torch.Tensor,
) -> torch.Tensor:
    """Terminate sideways escapes and premature runs past the final gap."""
    inside = _course_inside(
        _sole_geometry_w(env, feet_cfg, sole_vertices),
        env.scene[feet_cfg.name].data.root_pos_w,
        env.scene.env_origins, half_length,
        FIRST_STICK_X + NUM_STICKS * _get_state(env).course_spacing - STICK_HALF_WIDTH,
    )
    if bool((~inside)[0]):
        root = env.scene[feet_cfg.name].data.root_pos_w[0] - env.scene.env_origins[0]
        soles = _sole_geometry_w(env, feet_cfg, sole_vertices)[0] - env.scene.env_origins[0, None, None, :]
        print(f"[OOB TRIGGERED] root_x={root[0]:.4f}, root_y={root[1]:.4f}, sole_y_min={soles[..., 1].amin():.4f}, sole_y_max={soles[..., 1].amax():.4f}, sole_x_max={soles[..., 0].amax():.4f}, limit={half_length}", flush=True)
    return ~inside


def _course_failure_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Separate genuine failures from the successful course termination."""
    manager = getattr(env, "termination_manager", None)
    if manager is None or manager.terminated is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active = getattr(manager, "active_terms", [])
    if "hurdle_course_completed" not in active:
        return manager.terminated
    failed = torch.zeros_like(manager.terminated)
    for name in active:
        if name != "hurdle_course_completed" and not manager.get_term_cfg(name).time_out:
            failed |= manager.get_term(name)
    return failed


def hurdle_course_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """End clean and grazed completed runs, without treating success as failure.

    This term must follow the failure terms so the state updater observes the
    current sample's failure buffers before rewards consume its cached events.
    """
    state = _reward_crossing_state(env)
    return (state.current_stick_index >= NUM_STICKS) & ~_course_failure_mask(env)


def _course_gap_mask(
    sole_vertices_w: torch.Tensor,
    stick_x: torch.Tensor,
    env_origins: torch.Tensor,
    half_width: float,
    half_length: float,
    spacing: torch.Tensor | float = STICK_SPACING,
) -> torch.Tensor:
    """Require the whole sole to fit the gap and the course's lateral bounds."""
    minimum = sole_vertices_w.amin(dim=2)
    maximum = sole_vertices_w.amax(dim=2)
    pitch = spacing[:, None] if isinstance(spacing, torch.Tensor) else spacing
    return (
        (minimum[..., 0] >= stick_x[:, None] + half_width)
        & (maximum[..., 0] <= stick_x[:, None] + pitch - half_width)
        & (minimum[..., 1] >= env_origins[:, 1, None] - half_length)
        & (maximum[..., 1] <= env_origins[:, 1, None] + half_length)
    )


def _course_step_targets(sole_vertices_w, gap_center_x, forward_w):
    """Convert a world gap center into the inherited inter-foot toe distance."""
    projection = (sole_vertices_w[..., :2] * forward_w[:, None, None, :]).sum(dim=-1)
    minimum, maximum = projection.amin(dim=2), projection.amax(dim=2)
    center_y = .5 * (sole_vertices_w[..., 1].amin(dim=2) + sole_vertices_w[..., 1].amax(dim=2))
    target_front = (gap_center_x[:, None] * forward_w[:, 0, None]
                    + center_y * forward_w[:, 1, None] + .5 * (maximum - minimum))
    return target_front - maximum.flip(dims=(1,))


def hurdle_skipped_gap(env, feet_cfg, sensor_cfg, sole_vertices):
    """A grounded foot beyond the next stick cannot count as this gap's landing."""
    state = _get_state(env)
    sole = _sole_geometry_w(env, feet_cfg, sole_vertices)
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
    next_edge = (env.scene.env_origins[:, 0] + FIRST_STICK_X
                 + (state.current_stick_index + 1) * state.course_spacing + STICK_HALF_WIDTH)
    beyond = sole[..., 0].amin(dim=2) > next_edge[:, None]
    return (state.initialized & (state.current_stick_index < NUM_STICKS)
            & (beyond & contact).any(dim=1))


def hurdle_target_approach_reward(env):
    return _reward_crossing_state(env).target_approach_reward_val


def hurdle_wrong_landing_penalty(env):
    return _reward_crossing_state(env).wrong_landing_reward_val


def record_hurdle_curriculum(env, env_ids):
    """Accumulate only finished episodes from the current difficulty level."""
    state = _get_state(env)
    ids = env_ids[state.initialized[env_ids] & (state.course_level[env_ids] == state.curriculum_level)]
    if len(ids):
        state.curriculum_episodes += len(ids)
        state.curriculum_completed += int(((state.current_stick_index[ids] >= NUM_STICKS)
                                          & ~_course_failure_mask(env)[ids]).sum().item())
        state.curriculum_attempts += int(state.landing_attempts[ids].sum().item())
        state.curriculum_correct += int(state.correct_landings[ids].sum().item())
    accuracy = state.curriculum_correct / max(1, state.curriculum_attempts)
    completion = state.curriculum_completed / max(1, state.curriculum_episodes)
    metrics = {"Task/landing_accuracy": accuracy, "Task/paired_course_completion_rate": completion}
    gaps = getattr(env.cfg, "hurdle_gap_levels", (.25, .225, .20))
    if (getattr(env.cfg, "hurdle_gap_curriculum", False)
            and state.curriculum_level < len(gaps) - 1
            and state.curriculum_episodes >= getattr(env.cfg, "hurdle_curriculum_min_episodes", 200)
            and state.curriculum_attempts >= 50
            and accuracy >= getattr(env.cfg, "hurdle_curriculum_landing_accuracy", .90)
            and completion >= getattr(env.cfg, "hurdle_curriculum_completion_rate", .80)):
        state.curriculum_level += 1
        state.curriculum_episodes = state.curriculum_completed = 0
        state.curriculum_attempts = state.curriculum_correct = 0
    metrics["Task/curriculum_gap_cm"] = 100 * gaps[state.curriculum_level]
    return metrics


def reset_hurdle_layout(env, env_ids):
    """Move simulator rigid bodies and reward geometry together on reset only."""
    state = _get_state(env)
    gaps = getattr(env.cfg, "hurdle_gap_levels", (getattr(env.cfg, "hurdle_initial_gap", STICK_CLEAR_GAP),))
    state.course_level[env_ids] = state.curriculum_level
    state.course_spacing[env_ids] = gaps[state.curriculum_level] + WOODEN_BAR_WIDTH
    for j in range(NUM_STICKS):
        pose = torch.zeros((len(env_ids), 7), device=env.device)
        pose[:, :3] = env.scene.env_origins[env_ids]
        pose[:, 0] += FIRST_STICK_X + j * state.course_spacing[env_ids]
        pose[:, 2] += .5 * WOODEN_BAR_HEIGHT
        pose[:, 6] = 1.0  # Isaac Lab in this project uses xyzw.
        bar = env.scene[f"course_stick_{j}"]
        bar.write_root_pose_to_sim(pose, env_ids=env_ids)
        bar.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def _sole_overlaps_bar(
    sole_vertices_w: torch.Tensor,
    bar_pose_w: torch.Tensor,
    bar_half_extents: tuple[float, float, float],
) -> torch.Tensor:
    """Return exact 2-D sole-polygon overlap plus vertical bar overlap."""
    num_envs, num_feet, num_vertices = sole_vertices_w.shape[:3]
    relative_w = sole_vertices_w - bar_pose_w[:, None, None, :3]
    bar_quat_w = bar_pose_w[:, None, None, 3:7].expand(
        -1, num_feet, num_vertices, -1
    )
    vertices_b = quat_apply_inverse(
        bar_quat_w.reshape(-1, 4), relative_w.reshape(-1, 3)
    ).reshape(num_envs, num_feet, num_vertices, 3)

    half_width, half_length, half_height = bar_half_extents
    rectangle_xy = torch.tensor(
        (
            (-half_width, -half_length),
            (-half_width, half_length),
            (half_width, half_length),
            (half_width, -half_length),
        ),
        dtype=vertices_b.dtype,
        device=vertices_b.device,
    )

    sole_xy = vertices_b[..., :2]
    edge_xy = torch.roll(sole_xy, shifts=-1, dims=2) - sole_xy
    edge_normals = torch.stack((-edge_xy[..., 1], edge_xy[..., 0]), dim=-1)
    rectangle_axes = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0)),
        dtype=vertices_b.dtype,
        device=vertices_b.device,
    ).expand(num_envs, num_feet, -1, -1)
    axes = torch.cat((rectangle_axes, edge_normals), dim=2)

    sole_projection = torch.einsum("efva,efka->efkv", sole_xy, axes)
    rectangle_projection = torch.einsum(
        "ra,efka->efkr", rectangle_xy, axes
    )
    separated = (
        torch.amax(sole_projection, dim=3)
        < torch.amin(rectangle_projection, dim=3)
    ) | (
        torch.amax(rectangle_projection, dim=3)
        < torch.amin(sole_projection, dim=3)
    )
    footprint_overlap = ~torch.any(separated, dim=2)

    sole_min_z = torch.amin(vertices_b[..., 2], dim=2)
    sole_max_z = torch.amax(vertices_b[..., 2], dim=2)
    vertical_overlap = (sole_min_z <= half_height) & (
        sole_max_z >= -half_height
    )
    return footprint_overlap & vertical_overlap


def configure_collisionless_bar_collisions(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    training_phase: int,
    robot_name: str,
    collisionless_bar_name: str,
    rigid_body_names: Sequence[str],
):
    """Filter the collisionless bar for Phase 3 once."""
    del env_ids
    if training_phase != COLLISIONLESS_BAR_PHASE:
        return

    from pxr import Usd, UsdPhysics

    stage = env.scene.stage
    robot_leaf = env.scene[robot_name].cfg.prim_path.rsplit("/", 1)[-1]
    bar_leaf = env.scene[collisionless_bar_name].cfg.prim_path.rsplit(
        "/", 1
    )[-1]
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim = stage.GetPrimAtPath(f"{env_prim_path}/{robot_leaf}")
        bar_prim = stage.GetPrimAtPath(f"{env_prim_path}/{bar_leaf}")
        if not robot_prim.IsValid() or not bar_prim.IsValid():
            raise RuntimeError(
                "Could not resolve the robot and wooden-bar prims needed "
                "for Phase 3 collision filtering."
            )

        rigid_body_paths = []
        for rigid_body_name in rigid_body_names:
            matches = [
                prim
                for prim in Usd.PrimRange(robot_prim)
                if prim.GetName() == rigid_body_name
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Expected exactly one robot rigid-body prim named "
                    f"{rigid_body_name!r} below {robot_prim.GetPath()}, "
                    f"but found {len(matches)}."
                )
            rigid_body_paths.append(matches[0].GetPath())

        filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(bar_prim)
        filtered_pairs.CreateFilteredPairsRel().SetTargets(rigid_body_paths)


def _hide_bar(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    bar_name: str,
    hidden_depth: float,
) -> torch.Tensor:
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, 2] -= hidden_depth
    pose[:, 3] = 1.0
    velocity = torch.zeros(len(env_ids), 6, device=env.device)
    bar = env.scene[bar_name]
    bar.write_root_pose_to_sim(pose, env_ids=env_ids)
    bar.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    return pose


def reset_crossing_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    collisionless_bar_name: str,
    physical_bar_name: str,
    hidden_depth: float,
    training_phase: int,
    default_step_distance: float,
    trigger_touchdown_range: tuple[int, int],
    phase_5_bar_episode_probability: float,
    phase_5_no_bar_stop_probability: float,
    phase_5_stop_time_range_s: tuple[float, float],
):
    """Reset crossing state and sample each Phase 5 episode type."""
    if training_phase not in (
        NORMAL_WALKING_PHASE,
        VIRTUAL_BAND_PHASE,
        COLLISIONLESS_BAR_PHASE,
        PHYSICAL_BAR_PHASE,
        MIXED_COMMAND_PHASE,
    ):
        raise ValueError("training_phase must be 1, 2, 3, 4, or 5.")
    if default_step_distance <= 0.0:
        raise ValueError("default_step_distance must be positive.")
    if (
        trigger_touchdown_range[0] < 3
        or trigger_touchdown_range[1] < trigger_touchdown_range[0]
    ):
        raise ValueError(
            "trigger_touchdown_range must be ordered and start at 3 or later."
        )
    if hidden_depth <= 0.0:
        raise ValueError("hidden_depth must be positive.")
    if not 0.0 <= phase_5_bar_episode_probability <= 1.0:
        raise ValueError(
            "phase_5_bar_episode_probability must be in [0, 1]."
        )
    if not 0.0 <= phase_5_no_bar_stop_probability <= 1.0:
        raise ValueError(
            "phase_5_no_bar_stop_probability must be in [0, 1]."
        )
    if (
        phase_5_stop_time_range_s[0] < 0.0
        or phase_5_stop_time_range_s[1] < phase_5_stop_time_range_s[0]
    ):
        raise ValueError(
            "phase_5_stop_time_range_s must be ordered and non-negative."
        )

    env_ids = _as_env_ids(env, env_ids)
    if len(env_ids) == 0:
        return
    state = _get_state(env)
    hidden_pose = _hide_bar(
        env, env_ids, collisionless_bar_name, hidden_depth
    )
    _hide_bar(
        env, env_ids, physical_bar_name, hidden_depth
    )

    state.landing_attempts[env_ids] = 0
    state.correct_landings[env_ids] = 0
    state.target_error_valid[env_ids] = False
    state.target_approach_reward_val[env_ids] = 0
    state.wrong_landing_reward_val[env_ids] = 0
    if "course_stick_0" in env.scene.keys():
        reset_hurdle_layout(env, env_ids)
    state.stop.reset(env_ids)
    state.task_success[env_ids] = False
    state.task_success_event[env_ids] = False
    state.success_support_steps[env_ids] = 0
    state.initialized[env_ids] = True
    if training_phase == MIXED_COMMAND_PHASE:
        bar_episode = (
            torch.rand(len(env_ids), device=env.device)
            < phase_5_bar_episode_probability
        )
        no_bar_episode = ~bar_episode
        stop_command = no_bar_episode & (
            torch.rand(len(env_ids), device=env.device)
            < phase_5_no_bar_stop_probability
        )
        effective_phase = torch.where(
            bar_episode,
            torch.full_like(env_ids, PHYSICAL_BAR_PHASE),
            torch.full_like(env_ids, NORMAL_WALKING_PHASE),
        )
        stop_time_s = torch.empty(
            len(env_ids), device=env.device
        ).uniform_(*phase_5_stop_time_range_s)
    else:
        no_bar_episode = torch.zeros(
            len(env_ids), dtype=torch.bool, device=env.device
        )
        stop_command = torch.zeros_like(no_bar_episode)
        effective_phase = torch.full_like(env_ids, training_phase)
        stop_time_s = torch.zeros(len(env_ids), device=env.device)

    state.training_phase[env_ids] = effective_phase
    state.phase_5_no_bar_episode[env_ids] = no_bar_episode
    state.phase_5_stop_command[env_ids] = stop_command
    state.phase_5_stop_time_s[env_ids] = stop_time_s
    state.phase_5_stop_step_reward_paid[env_ids] = False
    state.step_distance[env_ids] = default_step_distance
    state.crossing_command[env_ids] = False
    state.following_step_command_stage[env_ids] = 0
    state.crossing_foot_index[env_ids] = -1
    state.following_foot_index[env_ids] = -1
    state.following_foot_touchdown_event[env_ids] = False
    state.collisionless_bar_contact_event[env_ids] = False
    state.collisionless_bar_contacted[env_ids] = False
    state.touchdown_count[env_ids] = 0
    state.trigger_touchdown_index[env_ids] = torch.randint(
        trigger_touchdown_range[0],
        trigger_touchdown_range[1] + 1,
        (len(env_ids),),
        device=env.device,
    )
    state.swing_foot_index[env_ids] = -1
    state.step_distance_reward_airborne_seen[env_ids] = False
    state.touchdown_event[env_ids] = False
    state.touchdown_actual_step[env_ids] = 0.0
    state.touchdown_target_step[env_ids] = default_step_distance
    state.touchdown_reward_eligible[env_ids] = False
    state.step_distance_reward_paid_step[env_ids] = -1
    state.last_control_update_step[env_ids] = -1

    state.num_sticks = NUM_STICKS
    state.current_stick_index[env_ids] = 0
    state.sticks_cleared_count[env_ids] = 0
    state.stick_cleared_event[env_ids] = False
    state.stick_failed_event[env_ids] = False
    state.stick_collision_event[env_ids] = False
    state.stick_collision_reward_val[env_ids] = 0.0
    state.stick_hit_flags[env_ids, :] = False
    state.stick_landing_reward_paid[env_ids] = False
    state.forward_progress_reward_val[env_ids] = 0.0
    state.stick_over_clearance_reward_val[env_ids] = 0.0
    state.stick_landing_center_reward_val[env_ids] = 0.0
    state.task_success[env_ids] = False
    state.task_success_event[env_ids] = False

    state.spawned[env_ids] = True
    state.crossed[env_ids] = False
    state.crossing_completed[env_ids] = False
    state.spawn_time_s[env_ids] = 0.0
    state.spawn_pose_w[env_ids] = hidden_pose
    state.forward_w[env_ids, 0] = 1.0
    state.forward_w[env_ids, 1] = 0.0
    state.crossing_half_width[env_ids] = 0.0
    state.movement_reference_pose_w[env_ids] = hidden_pose
    state.movement_reference_set[env_ids] = False

    state.first_entry_event[env_ids] = False
    state.bar_reward_foot_entered[env_ids] = False
    state.bar_reward_foot_active[env_ids] = False
    state.bar_reward_stepping_foot[env_ids] = False
    state.bar_reward_following_foot[env_ids] = False
    state.stepping_foot_touchdown_distance_to_band_edge[env_ids] = 0.0
    state.stepping_foot_touchdown_distance_cached[env_ids] = False
    state.last_band_update_step[env_ids] = -1


def _spawn_crossing(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    landing_foot: torch.Tensor,
    feet_cfg: SceneEntityCfg,
    sole_front_x: torch.Tensor,
    forward_w: torch.Tensor,
    robot_yaw_quat_w: torch.Tensor,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    crossing_step_distance: float,
):
    """Fix the obstacle ahead of both feet at a completed swing cycle."""
    state = _get_state(env)
    robot = env.scene[feet_cfg.name]
    placement_front = torch.amax(sole_front_x[env_ids], dim=1)
    phase = state.training_phase[env_ids]
    virtual = phase == VIRTUAL_BAND_PHASE
    bar_phase = (phase == COLLISIONLESS_BAR_PHASE) | (
        phase == PHYSICAL_BAR_PHASE
    )

    forward_offset = torch.empty(len(env_ids), device=env.device)
    forward_offset[virtual] = (
        virtual_band_near_edge_offset + virtual_band_half_width
    )
    placement_error = torch.empty(len(env_ids), device=env.device).uniform_(
        *physical_bar_position_error_range
    )
    forward_offset[bar_phase] = (
        physical_bar_center_distance + placement_error[bar_phase]
    )

    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :2] = robot.data.root_pos_w[env_ids, :2] + (
        placement_front + forward_offset
    ).unsqueeze(1) * forward_w[env_ids]
    pose[:, 2] = env.scene.env_origins[env_ids, 2]
    pose[bar_phase, 2] += (
        0.5 * bar_height + physical_bar_drop_clearance
    )
    pose[:, 3:7] = robot_yaw_quat_w[env_ids]

    collisionless = phase == COLLISIONLESS_BAR_PHASE
    physical = phase == PHYSICAL_BAR_PHASE
    for phase_mask, bar_name in (
        (collisionless, collisionless_bar_name),
        (physical, physical_bar_name),
    ):
        if torch.any(phase_mask):
            bar_env_ids = env_ids[phase_mask]
            velocity = torch.zeros(
                len(bar_env_ids), 6, device=env.device
            )
            bar = env.scene[bar_name]
            bar.write_root_pose_to_sim(
                pose[phase_mask], env_ids=bar_env_ids
            )
            bar.write_root_velocity_to_sim(
                velocity, env_ids=bar_env_ids
            )

    state.spawned[env_ids] = True
    state.crossed[env_ids] = False
    state.crossing_command[env_ids] = True
    state.following_step_command_stage[env_ids] = 0
    state.following_foot_index[env_ids] = landing_foot
    state.crossing_foot_index[env_ids] = 1 - landing_foot
    state.step_distance[env_ids] = crossing_step_distance
    state.spawn_time_s[env_ids] = _episode_time_s(env)[env_ids]
    state.spawn_pose_w[env_ids] = pose
    state.forward_w[env_ids] = forward_w[env_ids]
    state.crossing_half_width[env_ids] = torch.where(
        virtual,
        torch.full_like(placement_front, virtual_band_half_width),
        torch.full_like(placement_front, physical_bar_half_width),
    )
    state.movement_reference_pose_w[env_ids] = pose
    state.movement_reference_set[env_ids] = False
    state.first_entry_event[env_ids] = False
    state.bar_reward_foot_entered[env_ids] = False
    state.bar_reward_foot_active[env_ids] = False
    state.bar_reward_stepping_foot[env_ids] = False
    state.bar_reward_following_foot[env_ids] = False
    state.stepping_foot_touchdown_distance_to_band_edge[env_ids] = 0.0
    state.stepping_foot_touchdown_distance_cached[env_ids] = False
    state.last_band_update_step[env_ids] = -1


def _normal_step_sample(
    env: ManagerBasedRLEnv,
    count: int,
    default_step_distance: float,
    default_probability: float,
    random_step_distance_range: tuple[float, float],
) -> torch.Tensor:
    random_steps = torch.empty(count, device=env.device).uniform_(
        *random_step_distance_range
    )
    use_default = torch.rand(count, device=env.device) < default_probability
    return torch.where(
        use_default,
        torch.full_like(random_steps, default_step_distance),
        random_steps,
    )


def _update_crossing_state_once(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    training_phase: int,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    physical_bar_half_length: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    default_step_distance: float,
    crossing_step_distance: float,
    phase_2_post_crossing_step_distance: float,
    phase_3_post_crossing_step_distance: float,
    phase_4_post_crossing_step_distance: float,
    normal_step_default_probability: float,
    random_step_distance_range: tuple[float, float],
    minimum_air_time_s: float = 0.04,
) -> _CrossingState:
    """Update touchdown, command, spawn, and completion state exactly once."""
    if training_phase not in (
        NORMAL_WALKING_PHASE,
        VIRTUAL_BAND_PHASE,
        COLLISIONLESS_BAR_PHASE,
        PHYSICAL_BAR_PHASE,
        MIXED_COMMAND_PHASE,
    ):
        raise ValueError("training_phase must be 1, 2, 3, 4, or 5.")
    if not 0.0 <= normal_step_default_probability <= 1.0:
        raise ValueError("normal_step_default_probability must be in [0, 1].")
    if minimum_air_time_s < 0.0:
        raise ValueError("minimum_air_time_s must be non-negative.")
    if min(
        default_step_distance,
        crossing_step_distance,
        phase_2_post_crossing_step_distance,
        phase_3_post_crossing_step_distance,
        bar_height,
        physical_bar_half_width,
        physical_bar_half_length,
        virtual_band_half_width,
    ) <= 0.0:
        raise ValueError("Configured distances, widths, and height must be positive.")
    if phase_4_post_crossing_step_distance < 0:
        raise ValueError("Post-crossing step distance must be non-negative.")
    if virtual_band_near_edge_offset < 0.0:
        raise ValueError("virtual_band_near_edge_offset must be non-negative.")
    if physical_bar_center_distance < 0.0:
        raise ValueError("physical_bar_center_distance must be non-negative.")
    if physical_bar_drop_clearance < 0.0:
        raise ValueError("physical_bar_drop_clearance must be non-negative.")
    if (
        physical_bar_position_error_range[1]
        < physical_bar_position_error_range[0]
    ):
        raise ValueError("physical_bar_position_error_range must be ordered.")
    if (
        random_step_distance_range[0] <= 0.0
        or random_step_distance_range[1] < random_step_distance_range[0]
    ):
        raise ValueError(
            "random_step_distance_range must be ordered and positive."
        )
    state = _get_state(env)
    step = _control_step(env)
    update_envs = state.last_control_update_step != step
    if not torch.any(update_envs):
        return state

    state.task_success_event[update_envs] = False
    state.following_foot_touchdown_event[update_envs] = False
    state.collisionless_bar_contact_event[update_envs] = False

    # Reset events initialize the phase per environment. This assignment keeps
    # newly constructed environments safe before their first reset callback.
    uninitialized = update_envs & ~state.initialized
    state.initialized[uninitialized] = True
    state.training_phase[uninitialized] = (
        NORMAL_WALKING_PHASE
        if training_phase == MIXED_COMMAND_PHASE
        else training_phase
    )
    state.step_distance[uninitialized] = default_step_distance

    sole_vertices_w = _sole_geometry_w(env, feet_cfg, sole_vertices)
    forward_w, robot_yaw_quat_w, sole_front_x = _base_forward_and_sole_front(
        env, feet_cfg, sole_vertices_w
    )
    actual_step = torch.stack(
        (
            sole_front_x[:, 0] - sole_front_x[:, 1],
            sole_front_x[:, 1] - sole_front_x[:, 0],
        ),
        dim=1,
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    )
    if in_contact.shape != state.touchdown_event.shape:
        raise ValueError(
            "feet_cfg and sensor_cfg must resolve the same two ordered feet."
        )

    # Restore the old first-contact gate only for the step-distance reward.
    # Obstacle spawning, touchdown_count, and crossing logic continue to use
    # the independent swing-cycle detector below.
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ].bool()
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    other_in_contact = torch.flip(in_contact, dims=(1,))
    other_first_contact = torch.flip(first_contact, dims=(1,))
    valid_reward_touchdown = (
        update_envs.unsqueeze(1)
        & first_contact
        & state.step_distance_reward_airborne_seen
        & (last_air_time >= minimum_air_time_s)
        & other_in_contact
        & ~other_first_contact
    )
    crossing_before_update = state.crossing_command.clone()

    state.touchdown_reward_eligible[update_envs] = False
    state.touchdown_actual_step = torch.where(
        valid_reward_touchdown, actual_step, state.touchdown_actual_step
    )
    current_targets = state.step_distance.unsqueeze(1).expand(-1, 2)
    if not state.stop_enabled:
        gap_center_x = (env.scene.env_origins[:, 0] + FIRST_STICK_X
                        + (state.current_stick_index.float() + .5) * state.course_spacing)
        course_targets = _course_step_targets(sole_vertices_w, gap_center_x, forward_w)
        current_targets = torch.where(crossing_before_update[:, None], course_targets, current_targets)
    state.touchdown_target_step = torch.where(
        valid_reward_touchdown,
        current_targets,
        state.touchdown_target_step,
    )
    state.touchdown_reward_eligible |= (
        valid_reward_touchdown & ~crossing_before_update.unsqueeze(1)
    )
    state.step_distance_reward_airborne_seen[update_envs] |= (
        ~in_contact[update_envs]
    )
    state.step_distance_reward_airborne_seen[valid_reward_touchdown] = False

    collisionless_active = (
        update_envs
        & state.spawned
        & ~state.crossed
        & ~state.collisionless_bar_contacted
        & (state.training_phase == COLLISIONLESS_BAR_PHASE)
    )
    if torch.any(collisionless_active):
        bar_pose_w = env.scene[
            collisionless_bar_name
        ].data.root_state_w[:, :7]
        foot_overlaps_bar = _sole_overlaps_bar(
            sole_vertices_w,
            bar_pose_w,
            (
                physical_bar_half_width,
                physical_bar_half_length,
                0.5 * bar_height,
            ),
        )
        new_contact = collisionless_active & torch.any(
            foot_overlaps_bar, dim=1
        )
        state.collisionless_bar_contact_event[new_contact] = True
        state.collisionless_bar_contacted[new_contact] = True

    touchdown_event, next_swing_foot_index = (
        _swing_cycle_touchdown_events(
            in_contact, state.swing_foot_index, update_envs
        )
    )
    state.swing_foot_index[update_envs] = next_swing_foot_index[update_envs]

    foot_indices = torch.arange(2, device=env.device).unsqueeze(0)
    crossing_foot_touchdown = torch.any(
        touchdown_event
        & (foot_indices == state.crossing_foot_index.unsqueeze(1)),
        dim=1,
    )
    following_foot_touchdown = torch.any(
        touchdown_event
        & (foot_indices == state.following_foot_index.unsqueeze(1)),
        dim=1,
    )

    state.touchdown_event[update_envs] = False
    state.touchdown_event |= touchdown_event
    state.touchdown_count += torch.sum(touchdown_event, dim=1).long()
    state.following_foot_touchdown_event[update_envs] = False
    state.stick_cleared_event[update_envs] = False
    state.stick_failed_event[update_envs] = False
    state.task_success_event[update_envs] = False
    state.forward_progress_reward_val[update_envs] = 0.0
    state.stick_over_clearance_reward_val[update_envs] = 0.0
    state.stick_landing_center_reward_val[update_envs] = 0.0

    if not state.stop_enabled:
        curr_k = state.current_stick_index.clone()
        has_active_stick = curr_k < NUM_STICKS
        active_stick_x = env.scene.env_origins[:, 0] + FIRST_STICK_X + curr_k.float() * state.course_spacing
        state.spawn_pose_w[update_envs, 0] = active_stick_x[update_envs]
        state.spawn_pose_w[update_envs, 1] = env.scene.env_origins[update_envs, 1]
        state.spawn_pose_w[update_envs, 2] = env.scene.env_origins[update_envs, 2] + 0.5 * bar_height
        state.spawn_pose_w[update_envs, 3:6] = 0.0
        state.spawn_pose_w[update_envs, 6] = 1.0
        state.forward_w[update_envs, 0] = 1.0
        state.forward_w[update_envs, 1] = 0.0
        state.crossing_half_width[update_envs] = physical_bar_half_width
        state.spawned[update_envs] = True

        in_gap = _course_gap_mask(
            sole_vertices_w, active_stick_x, env.scene.env_origins,
            physical_bar_half_width, physical_bar_half_length, state.course_spacing,
        )
        both_feet_in_gap = in_gap.all(dim=1) & in_contact.all(dim=1)
        expected_foot = torch.where(state.following_step_command_stage == 0,
                                   state.crossing_foot_index, state.following_foot_index)
        expected_mask = foot_indices == expected_foot[:, None]
        course_active = has_active_stick & state.crossing_command & update_envs
        support_valid = other_in_contact & torch.where(
            (state.following_step_command_stage == 1)[:, None], in_gap.flip(dims=(1,)),
            torch.ones_like(in_gap))
        target_touchdown = touchdown_event & expected_mask & in_gap & support_valid & course_active[:, None]
        # Keep inherited airborne/first-contact constraints, but allow valid course steps.
        state.touchdown_reward_eligible |= valid_reward_touchdown & target_touchdown
        state.landing_attempts += (touchdown_event & course_active[:, None]).sum(dim=1)
        state.correct_landings += target_touchdown.sum(dim=1)
        state.wrong_landing_reward_val[update_envs] = (
            touchdown_event & ~target_touchdown & course_active[:, None]
        ).sum(dim=1)[update_envs].float()
        gap_center_x = active_stick_x + .5 * state.course_spacing
        foot_center_x = .5 * (sole_vertices_w[..., 0].amin(dim=2) + sole_vertices_w[..., 0].amax(dim=2))
        target_error = (foot_center_x - gap_center_x[:, None]).abs()
        approach = (state.target_error_previous - target_error) / env.step_dt
        guided = course_active[:, None] & expected_mask & state.target_error_valid
        state.target_approach_reward_val[update_envs] = (
            approach.clamp(max=.40) * guided.float()
        ).sum(dim=1)[update_envs] / .40
        state.target_error_previous[update_envs] = target_error[update_envs]
        state.target_error_valid[update_envs] = course_active[update_envs, None] & expected_mask[update_envs]

        # 1. Forward progress reward (Layer 1)
        robot = env.scene["robot"]
        vx_tensor = getattr(robot.data.root_lin_vel_w, "torch", robot.data.root_lin_vel_w)
        vx = vx_tensor[:, 0]
        in_course = _course_inside(
            sole_vertices_w, robot.data.root_pos_w, env.scene.env_origins,
            physical_bar_half_length,
            FIRST_STICK_X + NUM_STICKS * state.course_spacing - physical_bar_half_width,
        ) & has_active_stick
        # Keep backward motion signed so returning to the same position cannot
        # collect positive progress. Only the forward contribution saturates.
        fwd_score = torch.clamp(vx / .20, max=1.0)
        state.forward_progress_reward_val[update_envs] = torch.where(
            in_course[update_envs], fwd_score[update_envs],
            torch.zeros_like(fwd_score[update_envs]),
        )

        # 2. Collision detection with all 10 sticks (strictly once per stick!)
        new_collision_step = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        new_collision_count = torch.zeros(env.num_envs, device=env.device)
        for j in range(NUM_STICKS):
            sj_x = env.scene.env_origins[:, 0] + FIRST_STICK_X + j * state.course_spacing
            stick_origin = torch.stack(
                (sj_x, env.scene.env_origins[:, 1], env.scene.env_origins[:, 2]), dim=-1
            )
            relative_sole = sole_vertices_w - stick_origin[:, None, None, :]
            overlap_min_z = _sole_min_z_over_rectangle(
                relative_sole, physical_bar_half_width, physical_bar_half_length
            )
            hit_j = (overlap_min_z <= bar_height + 0.002).any(dim=1)
            # Only trigger on FIRST contact per stick per environment!
            new_hit_j = update_envs & hit_j & ~state.stick_hit_flags[:, j]
            state.stick_hit_flags[:, j] |= new_hit_j
            new_collision_step |= new_hit_j
            new_collision_count += new_hit_j.float()
        state.stick_collision_event[update_envs] = new_collision_step[update_envs]
        state.stick_collision_reward_val[update_envs] = new_collision_count[update_envs]

        # 3. Over-Stick Clearance Shaping (Layer 2)
        clearance_scores = torch.zeros(env.num_envs, device=env.device)
        if torch.any(has_active_stick & update_envs):
            dx_active = sole_vertices_w[..., 0] - active_stick_x[:, None, None]
            dx_max = torch.amax(dx_active, dim=2)
            dx_min = torch.amin(dx_active, dim=2)
            in_band = (dx_max >= -0.05) & (dx_min <= 0.05)
            dy_active = sole_vertices_w[..., 1] - env.scene.env_origins[:, 1, None, None]
            in_lat = (torch.amax(dy_active, dim=2) >= -physical_bar_half_length - 0.05) & (torch.amin(dy_active, dim=2) <= physical_bar_half_length + 0.05)
            is_swing = ~in_contact
            sole_min_z_active = torch.amin(sole_vertices_w[..., 2], dim=2) - env.scene.env_origins[:, 2, None]
            height_score = torch.clamp((sole_min_z_active - bar_height) / 0.035, min=0.0, max=1.0)
            if hasattr(robot.data, "body_lin_vel_w"):
                foot_vel_x = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, 0]
            else:
                foot_vel_x = vx_tensor[:, 0, None].expand(-1, len(feet_cfg.body_ids))
            # Reward feet being safely above the stick without penalizing slow apex crossing
            fwd_not_backward = (foot_vel_x >= -0.05).float()
            foot_score = height_score * fwd_not_backward * (is_swing & in_band & in_lat & (expected_mask | ~state.crossing_command[:, None])).float()
            clearance_scores = torch.sum(foot_score, dim=1)
        state.stick_over_clearance_reward_val[update_envs] = torch.where(
            (has_active_stick & in_course)[update_envs], clearance_scores[update_envs],
            torch.zeros_like(clearance_scores[update_envs]),
        )

        # 4. Landing Centering Reward (Layer 3)
        centering_scores = torch.zeros(env.num_envs, device=env.device)
        has_touchdown = torch.any(touchdown_event, dim=1)
        if torch.any(has_touchdown & has_active_stick & update_envs):
            gap_k_center = active_stick_x + 0.5 * state.course_spacing
            foot_center_x = 0.5 * (torch.amin(sole_vertices_w[..., 0], dim=2) + torch.amax(sole_vertices_w[..., 0], dim=2))
            dx_center = foot_center_x - gap_k_center[:, None]
            gaussian_score = torch.exp(-0.5 * torch.square(dx_center / 0.03))
            active_index = curr_k.clamp(max=NUM_STICKS - 1)
            env_index = torch.arange(env.num_envs, device=env.device)
            paid = state.stick_landing_reward_paid[env_index, active_index]
            eligible = (
                target_touchdown & ~paid
                & (has_active_stick & update_envs & in_course).unsqueeze(1)
            )
            td_score = gaussian_score * eligible.float()
            centering_scores = torch.sum(td_score, dim=1)
            state.stick_landing_reward_paid[env_index, active_index] |= eligible
        state.stick_landing_center_reward_val[update_envs] = centering_scores[update_envs]

        # Stage 0: crossing foot in flight
        stage_0_active = (
            update_envs
            & has_active_stick
            & state.crossing_command
            & (state.following_step_command_stage == 0)
        )
        state.step_distance[stage_0_active] = crossing_step_distance
        start_following_step = stage_0_active & (target_touchdown.any(dim=1))
        if torch.any(start_following_step):
            state.step_distance[start_following_step] = phase_4_post_crossing_step_distance
            state.following_step_command_stage[start_following_step] = 1

        # Stage 1: following foot in flight
        stage_1_active = (
            update_envs
            & has_active_stick
            & state.crossing_command
            & (state.following_step_command_stage == 1)
            & ~start_following_step
        )
        state.step_distance[stage_1_active] = phase_4_post_crossing_step_distance

        cleared = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        finish_following_step = stage_1_active & target_touchdown.any(dim=1)
        if torch.any(finish_following_step):
            state.following_foot_touchdown_event[finish_following_step] = True
            cleared = finish_following_step & both_feet_in_gap
            not_cleared = finish_following_step & ~both_feet_in_gap

            if torch.any(cleared):
                # Rule 1 & 2: Check whether active stick curr_k was hit
                hit_curr_k = torch.gather(state.stick_hit_flags, 1, curr_k.clamp(max=NUM_STICKS - 1).unsqueeze(1)).squeeze(1)
                clean_clear = cleared & ~hit_curr_k
                # Rule 1: Clean Clear (+20.0 reward triggered once for 1 step)
                state.stick_cleared_event[clean_clear] = True
                # Rule 2: Grazed Clear (0.0 reward, no event triggered)

                # Both clean and grazed advance the progression
                state.sticks_cleared_count[cleared] += 1
                state.current_stick_index[cleared] += 1

                all_done = cleared & (state.current_stick_index >= NUM_STICKS)
                more_to_go = cleared & (state.current_stick_index < NUM_STICKS)

                if torch.any(all_done):
                    # Grand Prize: 10 sticks cleared with ZERO hits across the whole course!
                    clean_all_done = all_done & (~torch.any(state.stick_hit_flags, dim=1))
                    state.task_success[clean_all_done] = True
                    state.task_success_event[clean_all_done] = True
                    state.crossed[all_done] = True
                    state.crossing_completed[all_done] = True
                    state.crossing_command[all_done] = False
                    state.step_distance[all_done] = default_step_distance
                    state.following_step_command_stage[all_done] = 2

                if torch.any(more_to_go):
                    prev_c = state.crossing_foot_index[more_to_go].clone()
                    state.crossing_foot_index[more_to_go] = state.following_foot_index[more_to_go]
                    state.following_foot_index[more_to_go] = prev_c
                    state.crossing_command[more_to_go] = True
                    state.following_step_command_stage[more_to_go] = 0
                    state.step_distance[more_to_go] = crossing_step_distance

            if torch.any(not_cleared):
                state.step_distance[not_cleared] = phase_4_post_crossing_step_distance

        # Rule 3: Hit stick and failed to pass (terminated on active stick)
        if hasattr(env, "termination_manager") and env.termination_manager.terminated is not None:
            term = env.termination_manager.terminated
            time_outs = getattr(env.termination_manager, "time_outs", torch.zeros_like(term))
            done_envs = term | time_outs
            hit_active_k = torch.gather(state.stick_hit_flags, 1, curr_k.clamp(max=NUM_STICKS - 1).unsqueeze(1)).squeeze(1)
            failed_hit = update_envs & done_envs & has_active_stick & hit_active_k & ~cleared
            state.stick_failed_event[failed_hit] = True

        # Triggering crossing for Stick 0
        approaching = (
            update_envs
            & (state.current_stick_index == 0)
            & ~state.crossing_command
        )
        touchdown_any = torch.any(touchdown_event, dim=1)
        front_x_w = sole_vertices_w[..., 0].amax(dim=(1, 2))
        dist_to_stick0 = active_stick_x - front_x_w - physical_bar_half_width
        trigger_first = approaching & touchdown_any & (dist_to_stick0 <= 0.14) & (dist_to_stick0 >= -0.05)
        if torch.any(trigger_first):
            landing_foot = torch.argmax(touchdown_event[trigger_first].long(), dim=1)
            state.following_foot_index[trigger_first] = landing_foot
            state.crossing_foot_index[trigger_first] = 1 - landing_foot
            state.crossing_command[trigger_first] = True
            state.following_step_command_stage[trigger_first] = 0
            state.step_distance[trigger_first] = crossing_step_distance

        # Normal walking steps before stick 0 or after all sticks
        normal_walk = (
            update_envs
            & ~state.crossing_command
            & touchdown_any
            & ~trigger_first
        )
        if torch.any(normal_walk):
            state.step_distance[normal_walk] = default_step_distance
            state.following_step_command_stage[normal_walk] = 2

        current_center = (env.scene.env_origins[:, 0] + FIRST_STICK_X
                          + (state.current_stick_index.float() + .5) * state.course_spacing)
        targets = _course_step_targets(sole_vertices_w, current_center, forward_w)
        next_foot = torch.where(state.following_step_command_stage == 0,
                               state.crossing_foot_index, state.following_foot_index)
        selected = targets.gather(1, next_foot.clamp(min=0)[:, None]).squeeze(1)
        commanded = update_envs & state.crossing_command
        state.step_distance[commanded] = selected[commanded]
        changed_target = start_following_step | cleared | trigger_first
        state.target_error_valid[changed_target] = False
        state.target_approach_reward_val[~in_course & update_envs] = 0

    else:
        # Give the post-crossing distance to exactly one step: the cached
        # following foot.
        stage_0_active = (
            update_envs
            & state.spawned
            & (state.following_step_command_stage == 0)
        )
        state.step_distance[stage_0_active] = crossing_step_distance
        start_following_step = stage_0_active & crossing_foot_touchdown
        if torch.any(start_following_step):
            phase_2_start = start_following_step & (
                state.training_phase == VIRTUAL_BAND_PHASE
            )
            phase_3_start = start_following_step & (
                state.training_phase == COLLISIONLESS_BAR_PHASE
            )
            phase_4_start = start_following_step & (
                state.training_phase == PHYSICAL_BAR_PHASE
            )
            state.step_distance[phase_2_start] = (
                phase_2_post_crossing_step_distance
            )
            state.step_distance[phase_3_start] = (
                phase_3_post_crossing_step_distance
            )
            state.step_distance[phase_4_start] = (
                phase_4_post_crossing_step_distance
            )
            state.following_step_command_stage[start_following_step] = 1

        stage_1_active = (
            update_envs
            & state.spawned
            & (state.following_step_command_stage == 1)
            & ~start_following_step
        )
        phase_2_stage_1 = stage_1_active & (
            state.training_phase == VIRTUAL_BAND_PHASE
        )
        phase_3_stage_1 = stage_1_active & (
            state.training_phase == COLLISIONLESS_BAR_PHASE
        )
        phase_4_stage_1 = stage_1_active & (
            state.training_phase == PHYSICAL_BAR_PHASE
        )
        state.step_distance[phase_2_stage_1] = (
            phase_2_post_crossing_step_distance
        )
        state.step_distance[phase_3_stage_1] = (
            phase_3_post_crossing_step_distance
        )
        state.step_distance[phase_4_stage_1] = (
            phase_4_post_crossing_step_distance
        )

        finish_following_step = stage_1_active & following_foot_touchdown
        if torch.any(finish_following_step):
            state.following_foot_touchdown_event[finish_following_step] = True
            finish_env_ids = torch.nonzero(
                finish_following_step, as_tuple=False
            ).squeeze(1)
            state.step_distance[finish_env_ids] = _normal_step_sample(
                env,
                len(finish_env_ids),
                default_step_distance,
                normal_step_default_probability,
                random_step_distance_range,
            )
            state.crossing_command[finish_env_ids] = False
            state.following_step_command_stage[finish_env_ids] = 2

        active = update_envs & state.spawned & ~state.crossed
        if torch.any(active):
            relative_xy = (
                sole_vertices_w[..., :2]
                - state.spawn_pose_w[:, None, None, :2]
            )
            longitudinal = torch.sum(
                relative_xy * state.forward_w[:, None, None, :], dim=3
            )
            foot_past_far_edge = (
                torch.amin(longitudinal, dim=2)
                > state.crossing_half_width.unsqueeze(1)
            )
            both_feet_past_far_edge = torch.all(foot_past_far_edge, dim=1)
            completed = active & both_feet_past_far_edge
        else:
            completed = torch.zeros_like(active)

        if torch.any(completed):
            state.crossed[completed] = True
            state.crossing_completed[completed] = True
            state.crossing_command[completed] = False
            state.bar_reward_foot_active[completed] = False

        completed_swing = torch.any(touchdown_event, dim=1)
        trigger_candidate = (
            update_envs
            & (state.training_phase != NORMAL_WALKING_PHASE)
            & ~state.spawned
            & ~state.crossing_completed
            & (state.touchdown_count >= state.trigger_touchdown_index)
            & completed_swing
        )
        if state.stop_enabled:
            robot = env.scene[feet_cfg.name]
            trigger_candidate = state.stop.update(
                trigger_candidate, update_envs, in_contact,
                robot.data.root_lin_vel_w.torch, robot.data.root_ang_vel_w.torch[:, 2],
            ) & update_envs
            relative = sole_vertices_w[..., :2] - state.spawn_pose_w[:, None, None, :2]
            longitudinal_now = (relative * state.forward_w[:, None, None, :]).sum(dim=3)
            still_clear = (longitudinal_now.amin(dim=2) > state.crossing_half_width[:, None]).all(dim=1)
            supported = update_envs & state.crossed & still_clear & in_contact.all(dim=1)
            state.success_support_steps[update_envs & ~supported] = 0
            state.success_support_steps[supported] += 1
            state.task_success_event = (
                update_envs & state.stop.completed & ~state.task_success
                & (state.success_support_steps >= 2)
            )
            state.task_success |= state.task_success_event
        if torch.any(trigger_candidate):
            trigger_env_ids = torch.nonzero(
                trigger_candidate, as_tuple=False
            ).squeeze(1)
            landing_foot = torch.argmax(
                touchdown_event[trigger_env_ids].long(),
                dim=1,
            )
            if state.stop_enabled:
                landing_foot = torch.argmax(sole_front_x[trigger_env_ids], dim=1)
            _spawn_crossing(
                env,
                trigger_env_ids,
                landing_foot,
                feet_cfg,
                sole_front_x,
                forward_w,
                robot_yaw_quat_w,
                collisionless_bar_name,
                physical_bar_name,
                bar_height,
                physical_bar_half_width,
                virtual_band_half_width,
                virtual_band_near_edge_offset,
                physical_bar_center_distance,
                physical_bar_position_error_range,
                physical_bar_drop_clearance,
                crossing_step_distance,
            )

        normal_touchdown = (
            update_envs
            & torch.any(touchdown_event, dim=1)
            & ~state.crossing_command
            & (~state.spawned | (state.following_step_command_stage == 2))
            & ~finish_following_step
            & ~completed
            & ~trigger_candidate
            & ~(state.stop.active | state.stop.failed)
        )
        if torch.any(normal_touchdown):
            normal_env_ids = torch.nonzero(
                normal_touchdown, as_tuple=False
            ).squeeze(1)
            state.step_distance[normal_env_ids] = _normal_step_sample(
                env,
                len(normal_env_ids),
                default_step_distance,
                normal_step_default_probability,
                random_step_distance_range,
            )
            state.following_step_command_stage[normal_env_ids] = 2

    # Phase 5 no-bar episodes use one fixed target instead of the normal
    # per-touchdown step-distance sampling.
    phase_5_no_bar = update_envs & state.phase_5_no_bar_episode
    state.step_distance[phase_5_no_bar] = default_step_distance

    state.last_control_update_step[update_envs] = step
    if state.stop_enabled or hasattr(env.cfg, "hurdle_follow_speed"):
        # Terminations/rewards run before CommandManager.compute in Isaac Lab.
        # Synchronize here so the same sample uses the same command in rewards/observations.
        env.command_manager.get_term("base_velocity")._update_command()
    return state


def update_crossing_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    training_phase: int,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    physical_bar_half_length: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    default_step_distance: float,
    crossing_step_distance: float,
    phase_2_post_crossing_step_distance: float,
    phase_3_post_crossing_step_distance: float,
    phase_4_post_crossing_step_distance: float,
    normal_step_default_probability: float,
    random_step_distance_range: tuple[float, float],
    minimum_air_time_s: float | None = None,
):
    """Interval-event wrapper for the shared once-per-step state update."""
    del env_ids
    reward_minimum_air_time_s = (
        0.04 if minimum_air_time_s is None else minimum_air_time_s
    )
    _update_crossing_state_once(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        training_phase,
        collisionless_bar_name,
        physical_bar_name,
        bar_height,
        physical_bar_half_width,
        physical_bar_half_length,
        virtual_band_half_width,
        virtual_band_near_edge_offset,
        physical_bar_center_distance,
        physical_bar_position_error_range,
        physical_bar_drop_clearance,
        default_step_distance,
        crossing_step_distance,
        phase_2_post_crossing_step_distance,
        phase_3_post_crossing_step_distance,
        phase_4_post_crossing_step_distance,
        normal_step_default_probability,
        random_step_distance_range,
        reward_minimum_air_time_s,
    )


class ObstacleAwareVelocityCommand(UniformVelocityCommand):
    """Forward/yaw command with step-target and virtual-band visualization."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cfg.feet_cfg.resolve(env.scene)
        self.cfg.sensor_cfg.resolve(env.scene)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = _as_env_ids(self._env, env_ids)
        random_values = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = random_values.uniform_(
            *self.cfg.ranges.lin_vel_x
        )
        self.vel_command_b[env_ids, 1] = 0.0
        sampled_ang_vel_z = random_values.uniform_(
            *self.cfg.ranges.ang_vel_z
        )
        if self.cfg.phase_5_enabled:
            state = _get_state(self._env)
            no_bar_episode = state.phase_5_no_bar_episode[env_ids]
            self.vel_command_b[env_ids, 2] = torch.where(
                no_bar_episode,
                sampled_ang_vel_z,
                torch.zeros_like(sampled_ang_vel_z),
            )
        else:
            self.vel_command_b[env_ids, 2] = sampled_ang_vel_z
        self.is_standing_env[env_ids] = False

        if self.cfg.heading_command:
            self.heading_target[env_ids] = random_values.uniform_(
                *self.cfg.ranges.heading
            )
            self.is_heading_env[env_ids] = (
                random_values.uniform_(0.0, 1.0)
                <= self.cfg.rel_heading_envs
            )

    def _update_command(self):
        super()._update_command()
        state = _get_state(self._env)

        if self.cfg.phase_5_enabled:
            bar_episode = state.initialized & ~state.phase_5_no_bar_episode
            self.is_standing_env[bar_episode] = False
            self.vel_command_b[bar_episode, 0] = self.cfg.ranges.lin_vel_x[0]
            self.vel_command_b[bar_episode, 1] = 0.0
            self.vel_command_b[bar_episode, 2] = 0.0

            stop_active = (
                state.phase_5_no_bar_episode
                & state.phase_5_stop_command
                & (_episode_time_s(self._env) >= state.phase_5_stop_time_s)
            )
            self.is_standing_env[stop_active] = True
            self.vel_command_b[stop_active] = 0.0
            state.step_distance[stop_active] = 0.0

        active = state.crossing_command
        self.is_standing_env[active] = False
        self.vel_command_b[active, 0] = self.cfg.ranges.lin_vel_x[0]
        if not state.stop_enabled:
            following = active & (state.following_step_command_stage == 1)
            self.vel_command_b[following, 0] = getattr(self._env.cfg, "hurdle_follow_speed", .10)
        self.vel_command_b[active, 1] = 0.0
        self.vel_command_b[active, 2] = 0.0

        # Stop has final priority over Phase 5's forced forward command.
        if state.stop_enabled:
            stopping = state.stop.active | state.stop.failed
            self.is_standing_env[stopping] = True
            self.vel_command_b[stopping] = 0.0
            state.step_distance[stopping] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create and toggle the additional playback markers."""
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "step_target_visualizer"):
                self.step_target_visualizer = VisualizationMarkers(
                    self.cfg.step_target_visualizer_cfg
                )
                self.virtual_band_visualizer = VisualizationMarkers(
                    self.cfg.virtual_band_visualizer_cfg
                )
            self.step_target_visualizer.set_visibility(True)
            self.virtual_band_visualizer.set_visibility(True)
        elif hasattr(self, "step_target_visualizer"):
            self.step_target_visualizer.set_visibility(False)
            self.virtual_band_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Draw the commanded swing-foot front and the Phase 2 band."""
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return

        state = _get_state(self._env)
        sole_vertices_w = _sole_geometry_w(
            self._env, self.cfg.feet_cfg, self.cfg.sole_vertices
        )
        forward_w, robot_yaw_quat_w, sole_front_x = (
            _base_forward_and_sole_front(
                self._env, self.cfg.feet_cfg, sole_vertices_w
            )
        )

        contact_sensor: ContactSensor = self._env.scene.sensors[
            self.cfg.sensor_cfg.name
        ]
        in_contact = (
            contact_sensor.data.current_contact_time[
                :, self.cfg.sensor_cfg.body_ids
            ]
            > 0.0
        )
        has_one_swing_foot, stepping_foot = _single_swing_foot(in_contact)
        support_foot = 1 - stepping_foot

        support_front = torch.gather(
            sole_front_x, 1, support_foot.unsqueeze(1)
        ).squeeze(1)
        target_longitudinal = support_front + state.step_distance
        target_position_w = self.robot.data.root_pos_w.clone()
        target_position_w[:, :2] += (
            target_longitudinal.unsqueeze(1) * forward_w
        )

        num_vertices = sole_vertices_w.shape[2]
        support_vertices_z = torch.gather(
            sole_vertices_w[..., 2],
            1,
            support_foot[:, None, None].expand(-1, 1, num_vertices),
        ).squeeze(1)
        target_position_w[:, 2] = (
            torch.amin(support_vertices_z, dim=1)
            + 0.5 * self.cfg.step_target_marker_height
        )

        sole_vertices_b = _sole_vertices_tensor(
            self._env, self.cfg.sole_vertices
        )
        sole_widths = (
            torch.amax(sole_vertices_b[..., 1], dim=1)
            - torch.amin(sole_vertices_b[..., 1], dim=1)
        )
        target_scales = torch.ones(
            (self.num_envs, 3), device=self.device
        )
        target_scales[:, 0] = self.cfg.step_target_marker_depth
        target_scales[:, 1] = sole_widths[stepping_foot]
        target_scales[:, 2] = self.cfg.step_target_marker_height
        target_visible = state.initialized & has_one_swing_foot
        self.step_target_visualizer.visualize(
            target_position_w,
            robot_yaw_quat_w,
            target_scales,
            marker_indices=(~target_visible).long(),
        )

        band_position_w = state.spawn_pose_w[:, :3].clone()
        band_position_w[:, 2] = (
            self._env.scene.env_origins[:, 2]
            + 0.5 * self.cfg.virtual_band_height
        )
        band_scales = torch.ones(
            (self.num_envs, 3), device=self.device
        )
        band_scales[:, 0] = 2.0 * state.crossing_half_width
        band_scales[:, 1] = self.cfg.virtual_band_length
        band_scales[:, 2] = self.cfg.virtual_band_height
        band_visible = (
            (state.training_phase == VIRTUAL_BAND_PHASE)
            & state.spawned
        )
        self.virtual_band_visualizer.visualize(
            band_position_w,
            state.spawn_pose_w[:, 3:7],
            band_scales,
            marker_indices=(~band_visible).long(),
        )


@configclass
class ObstacleAwareVelocityCommandCfg(UniformVelocityCommandCfg):
    """Forward/yaw-only command configuration."""

    class_type: type = ObstacleAwareVelocityCommand

    feet_cfg: SceneEntityCfg = MISSING
    sensor_cfg: SceneEntityCfg = MISSING
    sole_vertices: tuple[
        tuple[tuple[float, float, float], ...], ...
    ] = MISSING
    virtual_band_length: float = MISSING
    virtual_band_height: float = MISSING
    phase_5_enabled: bool = False

    # The landing target is a short vertical plane at the commanded sole-front
    # position. Keeping it 1.5 cm tall avoids obscuring the foot.
    step_target_marker_depth: float = 0.006
    step_target_marker_height: float = 0.015

    step_target_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/step_distance_target",
            markers={
                "target": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.75, 0.0),
                        emissive_color=(0.15, 0.08, 0.0),
                        opacity=0.50,
                    ),
                ),
                "hidden": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visible=False,
                ),
            },
        )
    )
    virtual_band_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/virtual_band",
            markers={
                "band": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.65, 1.0),
                        emissive_color=(0.0, 0.05, 0.10),
                        opacity=0.25,
                    ),
                ),
                "hidden": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visible=False,
                ),
            },
        )
    )

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


def step_distance_command(
    env: ManagerBasedRLEnv,
    default_step_distance: float,
) -> torch.Tensor:
    """Return the per-environment signed longitudinal touchdown target."""
    state = _get_state(env)
    values = torch.where(
        state.initialized,
        state.step_distance,
        torch.full_like(state.step_distance, default_step_distance),
    )
    return values.unsqueeze(1)


def crossing_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return zero for normal walking and one during the crossing action."""
    return _get_state(env).crossing_command.float().unsqueeze(1)


def physical_bar_crossing_completion_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    training_phase: int,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    physical_bar_half_length: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    default_step_distance: float,
    crossing_step_distance: float,
    phase_2_post_crossing_step_distance: float,
    phase_3_post_crossing_step_distance: float,
    phase_4_post_crossing_step_distance: float,
    normal_step_default_probability: float,
    random_step_distance_range: tuple[float, float],
    minimum_air_time_s: float | None = None,
) -> torch.Tensor:
    """Reward a successful Phase 3/4 following-foot touchdown once."""
    del minimum_air_time_s
    state = _update_crossing_state_once(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        training_phase,
        collisionless_bar_name,
        physical_bar_name,
        bar_height,
        physical_bar_half_width,
        physical_bar_half_length,
        virtual_band_half_width,
        virtual_band_near_edge_offset,
        physical_bar_center_distance,
        physical_bar_position_error_range,
        physical_bar_drop_clearance,
        default_step_distance,
        crossing_step_distance,
        phase_2_post_crossing_step_distance,
        phase_3_post_crossing_step_distance,
        phase_4_post_crossing_step_distance,
        normal_step_default_probability,
        random_step_distance_range,
    )
    if hasattr(env, "termination_manager") and env.termination_manager.terminated is not None:
        failed_now = state.task_success_event & _course_failure_mask(env)
        state.task_success[failed_now] = False
        return (state.task_success_event & ~failed_now).float()
    return state.task_success_event.float()


def _reward_crossing_state(env: ManagerBasedRLEnv) -> _CrossingState:
    """Refresh this control sample before reading any cached course reward."""
    params = env.event_manager.get_term_cfg("update_crossing_state").params
    return _update_crossing_state_once(env, **params)


def stick_cleared_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward awarded on the step each stick is cleared."""
    state = _reward_crossing_state(env)
    return state.stick_cleared_event.float()


def all_sticks_completed_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward awarded when all 10 sticks are cleared."""
    state = _reward_crossing_state(env)
    if hasattr(env, "termination_manager") and env.termination_manager.terminated is not None:
        failed_now = state.task_success_event & _course_failure_mask(env)
        state.task_success[failed_now] = False
        return (state.task_success_event & ~failed_now).float()
    return state.task_success_event.float()


def stick_collision_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty applied strictly once per stick upon first collision."""
    state = _reward_crossing_state(env)
    return state.stick_collision_reward_val


def stick_failed_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty awarded when the robot collides with a stick and fails to clear it."""
    state = _reward_crossing_state(env)
    return state.stick_failed_event.float()


def hurdle_forward_progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Continuous forward progress reward towards completing the 10-stick hurdle course."""
    state = _reward_crossing_state(env)
    return state.forward_progress_reward_val


def stick_over_clearance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense dynamic shaping: foot height * forward velocity while swinging over the active stick."""
    state = _reward_crossing_state(env)
    return state.stick_over_clearance_reward_val


def stick_landing_center_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gaussian precision reward for landing in the current curriculum gap."""
    state = _reward_crossing_state(env)
    return state.stick_landing_center_reward_val


def collisionless_bar_contact_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize only the first Phase 3 foot/bar contact frame per episode."""
    state = _get_state(env)
    return (
        (state.training_phase == COLLISIONLESS_BAR_PHASE)
        & state.collisionless_bar_contact_event
    ).float()


def step_distance_tracking_reward(
    env: ManagerBasedRLEnv,
    gaussian_std: float,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    training_phase: int,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    physical_bar_half_length: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    default_step_distance: float,
    crossing_step_distance: float,
    phase_2_post_crossing_step_distance: float,
    phase_3_post_crossing_step_distance: float,
    phase_4_post_crossing_step_distance: float,
    normal_step_default_probability: float,
    random_step_distance_range: tuple[float, float],
    minimum_air_time_s: float | None = None,
) -> torch.Tensor:
    """Score an old-style valid touchdown using its cached step distance."""
    reward_minimum_air_time_s = (
        0.04 if minimum_air_time_s is None else minimum_air_time_s
    )
    if gaussian_std <= 0.0:
        raise ValueError("gaussian_std must be positive.")
    state = _update_crossing_state_once(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        training_phase,
        collisionless_bar_name,
        physical_bar_name,
        bar_height,
        physical_bar_half_width,
        physical_bar_half_length,
        virtual_band_half_width,
        virtual_band_near_edge_offset,
        physical_bar_center_distance,
        physical_bar_position_error_range,
        physical_bar_drop_clearance,
        default_step_distance,
        crossing_step_distance,
        phase_2_post_crossing_step_distance,
        phase_3_post_crossing_step_distance,
        phase_4_post_crossing_step_distance,
        normal_step_default_probability,
        random_step_distance_range,
        reward_minimum_air_time_s,
    )
    error = (
        state.touchdown_actual_step - state.touchdown_target_step
    ) / gaussian_std
    scores = torch.exp(-0.5 * torch.square(error))
    stop_active = (
        state.phase_5_no_bar_episode
        & state.phase_5_stop_command
        & (_episode_time_s(env) >= state.phase_5_stop_time_s)
    )
    eligible = (
        state.touchdown_reward_eligible
        & ~(state.stop.active | state.stop.failed | state.stop.ready).unsqueeze(1)
        & (~state.crossing_command | (not state.stop_enabled)).unsqueeze(1)
        & (
            state.step_distance_reward_paid_step != _control_step(env)
        ).unsqueeze(1)
        & (
            ~stop_active | ~state.phase_5_stop_step_reward_paid
        ).unsqueeze(1)
    )
    reward = torch.sum(scores * eligible.to(scores.dtype), dim=1)
    rewarded_envs = torch.any(eligible, dim=1)
    state.step_distance_reward_paid_step[rewarded_envs] = _control_step(env)
    state.phase_5_stop_step_reward_paid |= stop_active & rewarded_envs
    return reward


def _minimum_foot_clearance(
    env: ManagerBasedRLEnv,
    sole_vertices_w: torch.Tensor,
    in_contact: torch.Tensor,
) -> torch.Tensor:
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
        has_support_foot.unsqueeze(1), support_z, nominal_ground_z
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
    _CrossingState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Cache each virtual-band entry independently once per control step."""
    if band_half_width <= 0.0:
        raise ValueError("band_half_width must be positive.")
    state = _get_state(env)
    sole_vertices_w, forward_w, footprint_min, footprint_max = (
        _crossing_foot_geometry(env, feet_cfg, sole_vertices)
    )
    overlaps_band = (
        (footprint_max >= -band_half_width)
        & (footprint_min <= band_half_width)
    )
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    )

    step = _control_step(env)
    fresh = state.last_band_update_step != step
    if torch.any(fresh):
        state.first_entry_event[fresh] = False
        active = (
            fresh
            & state.spawned
            & ~state.crossed
            & state.crossing_command
            & (state.training_phase == VIRTUAL_BAND_PHASE)
        )
        new_entry = (
            active.unsqueeze(1)
            & overlaps_band
            & ~state.bar_reward_foot_entered
        )
        had_entered = torch.any(
            state.bar_reward_foot_entered, dim=1, keepdim=True
        )
        frontmost_new_foot = torch.argmax(
            torch.where(
                new_entry,
                footprint_max,
                torch.full_like(footprint_max, -torch.inf),
            ),
            dim=1,
            keepdim=True,
        )
        first_role = torch.zeros_like(new_entry)
        first_role.scatter_(1, frontmost_new_foot, True)
        stepping_entry = new_entry & ~had_entered & first_role
        following_entry = new_entry & ~stepping_entry

        state.first_entry_event |= new_entry
        state.bar_reward_stepping_foot |= stepping_entry
        state.bar_reward_following_foot |= following_entry
        state.bar_reward_foot_entered |= new_entry
        state.bar_reward_foot_active |= new_entry

        stepping_foot_touchdown = (
            active.unsqueeze(1)
            & state.bar_reward_stepping_foot
            & state.touchdown_event
            & ~state.stepping_foot_touchdown_distance_cached.unsqueeze(1)
        )
        touchdown_envs = torch.any(stepping_foot_touchdown, dim=1)
        touchdown_distance = torch.sum(
            (footprint_max - band_half_width)
            * stepping_foot_touchdown.to(footprint_max.dtype),
            dim=1,
        )
        state.stepping_foot_touchdown_distance_to_band_edge = torch.where(
            touchdown_envs,
            touchdown_distance,
            state.stepping_foot_touchdown_distance_to_band_edge,
        )
        state.stepping_foot_touchdown_distance_cached |= touchdown_envs

        whole_foot_past_far_edge = footprint_min > band_half_width
        finished = (
            (state.bar_reward_foot_active & in_contact)
            | whole_foot_past_far_edge
            | ~active.unsqueeze(1)
        )
        state.bar_reward_foot_active &= ~finished
        state.last_band_update_step[fresh] = step

    return (
        state,
        sole_vertices_w,
        forward_w,
        footprint_min,
        footprint_max,
        overlaps_band,
        in_contact,
    )


def _wooden_bar_step_score_components(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
) -> tuple[
    _CrossingState,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the shared per-foot inputs for role-specific step scores."""
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
        env, feet_cfg, sensor_cfg, sole_vertices, band_half_width
    )
    clearance = _minimum_foot_clearance(env, sole_vertices_w, in_contact)
    height_score = torch.clamp(
        clearance / height_saturation, min=0.0, max=1.0
    )
    robot = env.scene[feet_cfg.name]
    foot_velocity_w = robot.data.body_lin_vel_w[:, feet_cfg.body_ids, :2]
    forward_velocity = torch.sum(
        foot_velocity_w * forward_w[:, None, :], dim=2
    )
    velocity_score = torch.clamp(
        forward_velocity / forward_velocity_saturation,
        min=-1.0,
        max=1.0,
    )
    active = overlaps_band & state.bar_reward_foot_active
    return state, height_score, velocity_score, footprint_max, active


def _stepping_wooden_bar_step_score(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
) -> tuple[_CrossingState, torch.Tensor]:
    """Return the original increasing-progress score for the stepping foot."""
    state, height_score, velocity_score, footprint_max, active = (
        _wooden_bar_step_score_components(
            env,
            feet_cfg,
            sensor_cfg,
            sole_vertices,
            band_half_width,
            height_saturation,
            forward_velocity_saturation,
            progress_unit,
        )
    )
    progress_score = torch.clamp(
        (footprint_max + band_half_width) / progress_unit, min=0.0
    )
    return state, (
        height_score
        * velocity_score
        * progress_score
        * active.to(height_score.dtype)
    )


def _following_wooden_bar_step_score(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
) -> tuple[_CrossingState, torch.Tensor]:
    """Return a decreasing-progress score for the following foot."""
    state, height_score, velocity_score, footprint_max, active = (
        _wooden_bar_step_score_components(
            env,
            feet_cfg,
            sensor_cfg,
            sole_vertices,
            band_half_width,
            height_saturation,
            forward_velocity_saturation,
            progress_unit,
        )
    )
    current_progress_score = torch.clamp(
        (footprint_max + band_half_width) / progress_unit, min=0.0, max=1.0
    )
    progress_score = 1.0 - current_progress_score
    return state, (
        height_score
        * velocity_score
        * progress_score
        * active.to(height_score.dtype)
    )


def stepping_wooden_bar_step_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
) -> torch.Tensor:
    """Reward the first foot that enters the virtual band."""
    state, step_score = _stepping_wooden_bar_step_score(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        band_half_width,
        height_saturation,
        forward_velocity_saturation,
        progress_unit,
    )
    return torch.sum(
        step_score
        * state.bar_reward_stepping_foot.to(step_score.dtype),
        dim=1,
    )


def following_wooden_bar_step_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
    forward_velocity_saturation: float,
    progress_unit: float,
    stepping_foot_distance_to_band_edge: float,
) -> torch.Tensor:
    """Reward the following foot, scaled by the stepping foot's progress."""
    if stepping_foot_distance_to_band_edge <= 0.0:
        raise ValueError(
            "stepping_foot_distance_to_band_edge must be positive."
        )
    state, step_score = _following_wooden_bar_step_score(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        band_half_width,
        height_saturation,
        forward_velocity_saturation,
        progress_unit,
    )
    current_reward = torch.sum(
        step_score
        * state.bar_reward_following_foot.to(step_score.dtype),
        dim=1,
    )
    stepping_foot_distance_score = torch.clamp(
        state.stepping_foot_touchdown_distance_to_band_edge
        / stepping_foot_distance_to_band_edge,
        min=0.0,
        max=1.0,
    )
    stepping_foot_distance_score *= (
        state.stepping_foot_touchdown_distance_cached.to(
            stepping_foot_distance_score.dtype
        )
    )
    return current_reward * stepping_foot_distance_score


def feet_height_entering_band_reward(
    env: ManagerBasedRLEnv,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    band_half_width: float,
    height_saturation: float,
) -> torch.Tensor:
    """Reward each foot's first airborne entry into the virtual band once."""
    if height_saturation <= 0.0:
        raise ValueError("height_saturation must be positive.")
    (
        state,
        sole_vertices_w,
        _,
        _,
        _,
        _,
        in_contact,
    ) = _update_foot_band_state(
        env, feet_cfg, sensor_cfg, sole_vertices, band_half_width
    )
    clearance = _minimum_foot_clearance(env, sole_vertices_w, in_contact)
    saturated = torch.clamp(clearance, max=height_saturation)
    active_entry = state.first_entry_event & state.bar_reward_foot_active
    return torch.sum(
        saturated * active_entry.to(saturated.dtype), dim=1
    )


def wooden_bar_moved(
    env: ManagerBasedRLEnv,
    translation_tolerance: float,
    rotation_tolerance: float,
    settling_time_s: float,
    feet_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_vertices: tuple[tuple[tuple[float, float, float], ...], ...],
    training_phase: int,
    collisionless_bar_name: str,
    physical_bar_name: str,
    bar_height: float,
    physical_bar_half_width: float,
    physical_bar_half_length: float,
    virtual_band_half_width: float,
    virtual_band_near_edge_offset: float,
    physical_bar_center_distance: float,
    physical_bar_position_error_range: tuple[float, float],
    physical_bar_drop_clearance: float,
    default_step_distance: float,
    crossing_step_distance: float,
    phase_2_post_crossing_step_distance: float,
    phase_3_post_crossing_step_distance: float,
    phase_4_post_crossing_step_distance: float,
    normal_step_default_probability: float,
    random_step_distance_range: tuple[float, float],
    minimum_air_time_s: float | None = None,
) -> torch.Tensor:
    """Terminate Phase 4 when the settled physical bar is disturbed."""
    # Kept as an ignored keyword so previously exported YAML configs still load.
    del minimum_air_time_s
    state = _update_crossing_state_once(
        env,
        feet_cfg,
        sensor_cfg,
        sole_vertices,
        training_phase,
        collisionless_bar_name,
        physical_bar_name,
        bar_height,
        physical_bar_half_width,
        physical_bar_half_length,
        virtual_band_half_width,
        virtual_band_near_edge_offset,
        physical_bar_center_distance,
        physical_bar_position_error_range,
        physical_bar_drop_clearance,
        default_step_distance,
        crossing_step_distance,
        phase_2_post_crossing_step_distance,
        phase_3_post_crossing_step_distance,
        phase_4_post_crossing_step_distance,
        normal_step_default_probability,
        random_step_distance_range,
    )
    bar_pose_w = env.scene[physical_bar_name].data.root_state_w[:, :7]
    physical = (
        state.training_phase == PHYSICAL_BAR_PHASE
    ) & state.spawned
    settled = physical & (
        (_episode_time_s(env) - state.spawn_time_s) >= settling_time_s
    )
    new_reference = settled & ~state.movement_reference_set
    state.movement_reference_pose_w[new_reference] = bar_pose_w[
        new_reference
    ]
    state.movement_reference_set |= new_reference

    translation = torch.linalg.vector_norm(
        bar_pose_w[:, :3] - state.movement_reference_pose_w[:, :3], dim=1
    )
    quat_dot = torch.abs(
        torch.sum(
            bar_pose_w[:, 3:7]
            * state.movement_reference_pose_w[:, 3:7],
            dim=1,
        )
    )
    rotation = 2.0 * torch.acos(torch.clamp(quat_dot, 0.0, 1.0))
    return (
        physical
        & state.movement_reference_set
        & (
            (translation > translation_tolerance)
            | (rotation > rotation_tolerance)
        )
    )


def is_any_terminated_term(
    env: ManagerBasedRLEnv,
    term_keys: str | list[str],
) -> torch.Tensor:
    """Return one when any selected termination term is active."""
    terminated = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    for term_name in env.termination_manager.find_terms(term_keys):
        terminated |= env.termination_manager.get_term(term_name).bool()
    return terminated.float()


def step_distance_gaussian_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    initial_std: float,
    final_std: float,
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Narrow the touchdown step-distance Gaussian over training."""
    del env_ids
    if initial_std <= 0.0 or final_std <= 0.0:
        raise ValueError("Gaussian standard deviations must be positive.")
    if start_step < 0 or end_step <= start_step:
        raise ValueError("Gaussian curriculum step range is invalid.")
    step = _control_step(env)
    progress = min(
        max((step - start_step) / (end_step - start_step), 0.0), 1.0
    )
    gaussian_std = initial_std * (final_std / initial_std) ** progress
    term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    term_cfg.params["gaussian_std"] = gaussian_std
    env.reward_manager.set_term_cfg(reward_term_name, term_cfg)
    return {
        "step_distance_gaussian_progress": progress,
        "step_distance_gaussian_std": gaussian_std,
    }


def phase_5_ang_vel_z_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    initial_range: tuple[float, float],
    final_range: tuple[float, float],
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Linearly widen Phase 5 yaw-rate commands over training."""
    del env_ids
    if start_step < 0 or end_step <= start_step:
        raise ValueError("Angular-velocity curriculum step range is invalid.")
    if initial_range[1] < initial_range[0]:
        raise ValueError("initial_range must be ordered.")
    if final_range[1] < final_range[0]:
        raise ValueError("final_range must be ordered.")

    step = _control_step(env)
    progress = min(
        max((step - start_step) / (end_step - start_step), 0.0), 1.0
    )
    ang_vel_z_range = (
        initial_range[0] * (final_range[0] / initial_range[0]) ** progress,
        initial_range[1] * (final_range[1] / initial_range[1]) ** progress,
    )
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.ranges.ang_vel_z = ang_vel_z_range
    return {
        "phase_5_ang_vel_z_progress": progress,
        "phase_5_ang_vel_z_min": ang_vel_z_range[0],
        "phase_5_ang_vel_z_max": ang_vel_z_range[1],
    }


def wooden_bar_reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_weight_ranges: dict[str, tuple[float, float]],
    pre_start_reward_weights: dict[str, float],
    start_step: int,
    end_step: int,
) -> dict[str, float]:
    """Retain the Phase 2 virtual-band reward-weight curriculum."""
    del env_ids
    if start_step < 0 or end_step <= start_step:
        raise ValueError("Reward curriculum step range is invalid.")
    if not reward_weight_ranges:
        raise ValueError("reward_weight_ranges must not be empty.")
    if set(pre_start_reward_weights) != set(reward_weight_ranges):
        raise ValueError(
            "pre_start_reward_weights and reward_weight_ranges must match."
        )

    step = _control_step(env)
    progress = min(
        max((step - start_step) / (end_step - start_step), 0.0), 1.0
    )
    metrics = {"wooden_bar_reward_weight_progress": progress}
    for term_name, weight_range in reward_weight_ranges.items():
        if len(weight_range) != 2:
            raise ValueError(
                f"Reward {term_name!r} must define exactly two weights."
            )
        initial_weight, final_weight = map(float, weight_range)
        if step < start_step:
            weight = float(pre_start_reward_weights[term_name])
        else:
            weight = initial_weight + progress * (
                final_weight - initial_weight
            )
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        metrics[f"{term_name}_weight"] = weight
    return metrics


def policy_observation_shape_check(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    group_name: str,
    expected_dim: int,
) -> dict[str, float]:
    """Fail explicitly if the concatenated policy observation is not expected."""
    del env_ids
    group_shape = env.observation_manager.group_obs_dim[group_name]
    if isinstance(group_shape, int):
        actual_dim = group_shape
    elif group_shape and isinstance(group_shape[0], (tuple, list)):
        actual_dim = sum(math.prod(term_shape) for term_shape in group_shape)
    else:
        actual_dim = math.prod(group_shape)
    if actual_dim != expected_dim:
        raise ValueError(
            f"Observation group {group_name!r} must be {expected_dim}-D, "
            f"but the configured shape is {group_shape}."
        )
    return {"policy_observation_dim": float(actual_dim)}



def stop_before_crossing_timeout(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Update before all rewards, and fail a stop that never settles."""
    _update_crossing_state_once(env, **env.event_manager.get_term_cfg("update_crossing_state").params)
    state = _get_state(env)
    return state.stop.failed if state.stop_enabled else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def stop_stability_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    state = _get_state(env)
    if not state.stop_enabled:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    velocity = robot.data.root_lin_vel_w.torch
    yaw_rate = robot.data.root_ang_vel_w.torch[:, 2]
    contact = env.scene.sensors[sensor_cfg.name].data.current_contact_time[:, sensor_cfg.body_ids] > 0
    score = torch.exp(-velocity[:, :2].square().sum(dim=1) / 0.05**2 - yaw_rate.square() / 0.10**2)
    return state.stop.active.float() * contact.all(dim=1).float() * score


def stop_completion_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _get_state(env)
    return state.stop.ready.float() if state.stop_enabled else torch.zeros(env.num_envs, device=env.device)


def align_wooden_bars_to_env_origins(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None = None,
) -> None:
    """Initialize the movable course layout at each environment's origin."""
    reset_hurdle_layout(env, _as_env_ids(env, env_ids))

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Obstacle-free variable-speed locomotion commands for Phase 6."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class Phase6VelocityCommand(UniformVelocityCommand):
    """Sample episode commands, then independently reconsider yaw and speed.

    Reset samples both velocities and one stop-episode flag. Integer episode
    steps drive the one-second checks, avoiding floating-point countdown drift
    and keeping asynchronously reset environments on their own clocks.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._stop_episode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_update_tick = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._update_interval_steps = max(
            1, math.ceil(cfg.command_update_interval_s / env.step_dt - 1.0e-9)
        )
        self._stop_step = math.ceil(cfg.stop_time_s / env.step_dt - 1.0e-9)

    def _env_ids(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            all_ids = torch.arange(self.num_envs, device=self.device)
            return all_ids if env_ids is None else all_ids[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        return super().reset(self._env_ids(env_ids))

    def _update_metrics(self):
        # The inherited metric divides by the very long command timer.
        # Normalize by episode length to retain useful tracking errors.
        episode_steps = self._env.max_episode_length
        self.metrics["error_vel_xy"] += torch.norm(
            self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1
        ) / episode_steps
        self.metrics["error_vel_yaw"] += torch.abs(
            self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]
        ) / episode_steps

    def _resample_command(self, env_ids: Sequence[int]):
        """Initialize only reset episodes; periodic resampling is disabled."""
        env_ids = self._env_ids(env_ids)
        samples = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = samples.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = 0.0
        self.vel_command_b[env_ids, 2] = samples.uniform_(*self.cfg.ranges.ang_vel_z)
        self._stop_episode[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.stop_probability
        )
        self._last_update_tick[env_ids] = 0
        self.is_standing_env[env_ids] = False

    def _update_command(self):
        episode_steps = self._env.episode_length_buf
        stop_active = self._stop_episode & (episode_steps >= self._stop_step)
        tick = episode_steps // self._update_interval_steps
        due = tick > self._last_update_tick
        self._last_update_tick[due] = tick[due]
        moving_ids = (due & ~stop_active).nonzero(as_tuple=False).flatten()

        # Independent Bernoulli draws; an unchanged command keeps its old value.
        yaw_ids = moving_ids[
            torch.rand(len(moving_ids), device=self.device)
            < self.cfg.yaw_change_probability
        ]
        speed_ids = moving_ids[
            torch.rand(len(moving_ids), device=self.device)
            < self.cfg.speed_change_probability
        ]
        self.vel_command_b[yaw_ids, 2] = torch.empty(
            len(yaw_ids), device=self.device
        ).uniform_(*self.cfg.ranges.ang_vel_z)
        self.vel_command_b[speed_ids, 0] = torch.empty(
            len(speed_ids), device=self.device
        ).uniform_(*self.cfg.ranges.lin_vel_x)

        # The stop flag is sampled once per episode and stays active to timeout.
        self.is_standing_env[:] = stop_active
        self.vel_command_b[:, 1] = 0.0
        super()._update_command()


@configclass
class Phase6VelocityCommandCfg(UniformVelocityCommandCfg):
    """Phase 6 defaults; the base timer is unused between episode resets."""

    class_type: type = Phase6VelocityCommand
    # Isaac Lab samples this timer using Tensor.uniform_, which requires finite
    # bounds. Keep it longer than an episode; episode steps drive actual updates.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    command_update_interval_s: float = 1.0
    yaw_change_probability: float = 0.20
    speed_change_probability: float = 0.10
    stop_probability: float = 0.30
    stop_time_s: float = 7.0


def phase_6_step_distance_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_step_distance: float,
    reference_speed: float,
) -> torch.Tensor:
    """Keep the old observation slot, scaling its target with commanded speed."""
    forward_speed = env.command_manager.get_command(command_name)[:, 0:1]
    return forward_speed * (reference_step_distance / reference_speed)


def phase_6_crossing_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the checkpoint's crossing slot permanently zero without bar state."""
    return torch.zeros((env.num_envs, 1), device=env.device)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment with a configurable global curriculum clock."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from .mdp.wooden_bar import record_hurdle_curriculum


class CurriculumAwareManagerBasedRLEnv(ManagerBasedRLEnv):
    """Start Isaac Lab's global step counter at a configured resume step.

    RSL-RL restores the policy, optimizer, and learning iteration from a
    checkpoint, but a newly-created Isaac Lab environment normally initializes
    ``common_step_counter`` to zero.  Time-based curriculum terms read that
    counter, so they would otherwise restart even though the policy resumed.

    Setting the actual environment counter here gives every curriculum term --
    including Isaac Lab built-ins and future custom terms -- the same resumed
    clock.  The value is applied before Gym/RSL-RL performs the first reset.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        curriculum_start_step = int(getattr(cfg, "curriculum_start_step", 0))
        if curriculum_start_step < 0:
            raise ValueError(
                "curriculum_start_step must be greater than or equal to zero, "
                f"but received {curriculum_start_step}."
            )

        self.common_step_counter = curriculum_start_step
        print(
            "[INFO] Curriculum global step starts at "
            f"{self.common_step_counter:,}."
        )

    def _reset_idx(self, env_ids):
        # Capture completed episodes before the reset event clears crossing state.
        metrics = {}
        if hasattr(self, "_fixed_stick_state"):
            state = self._fixed_stick_state
            # reset_buf does not exist until the first step in IsaacLab 3.0.
            terminated = getattr(self, 'reset_buf', state.initialized & False)
            ids = env_ids[state.initialized[env_ids] & (self.episode_length_buf[env_ids] > 0) & terminated[env_ids]]
            if len(ids):
                # Keep terminal outcomes available after Isaac's same-step autoreset.
                self.extras['fixed_stick_outcome'] = {
                    'env_ids': ids.clone(), 'success': state.success[ids].clone(),
                    'curriculum_level': state.episode_level[ids].clone(),
                    'hit': state.hit[ids].clone(), 'clean_success': state.clean_success[ids].clone(),
                    'stage': state.stage[ids].clone(),
                    'command_stage': state.command_stage[ids].clone(),
                    'control_steps': state.control_steps[ids].clone(),
                    'sequence_finished': state.sequence_finished[ids].clone(),
                    'forward_command': state.forward_command[ids].clone(),
                    'walk_completed': state.walk_completed[ids].clone(),
                    'step_command': state.step_distance[ids].clone(),
                    'cross_command': state.crossing_command[ids].clone(),
                    'touchdown': state.touchdown[ids].clone(),
                    'root_xyz': self.scene['robot'].data.root_pos_w[ids].clone(),
                    'joint_pos': self.scene['robot'].data.joint_pos[ids].clone(),
                    'termination_terms': {name: self.termination_manager.get_term(name)[ids].clone()
                                          for name in self.termination_manager.active_terms},
                }
                state.clean_stats.record(state.clean_success[ids].detach().cpu().tolist())
                curriculum = getattr(state, 'curriculum', None)
                if curriculum is not None:
                    curriculum.record(state.episode_level[ids].detach().cpu().tolist(),
                                      state.clean_success[ids].detach().cpu().tolist())
                    walk, gap = curriculum.parameters
                    metrics.update({
                        'Task/direct_curriculum_level': float(curriculum.level),
                        'Task/direct_curriculum_clean_success_rate': curriculum.rate,
                        'Task/direct_curriculum_episodes': len(curriculum.outcomes),
                        'Task/direct_curriculum_walk_step_m': walk,
                        'Task/direct_curriculum_initial_gap_m': gap,
                    })
                metrics.update({
                    'Task/clean_crossing_success_rate': state.clean_success[ids].float().mean(),
                    'Task/clean_crossing_success_rate_window': state.clean_stats.rate,
                    'Task/clean_crossing_window_episodes': len(state.clean_stats.outcomes),
                    'Task/first_step_completion_rate': state.walk_completed[ids].float().mean(),
                    'Task/crossing_command_rate': (state.command_stage[ids] >= state.LEAD).float().mean(),
                    'Task/command_sequence_finished_rate': state.sequence_finished[ids].float().mean(),
                    'Task/stick_contact_rate': state.hit[ids].float().mean(),
                    'Task/full_sequence_success_rate': state.success[ids].float().mean(),
                })
        elif hasattr(self, "_wooden_bar_state"):
            state = self._wooden_bar_state
            if not state.stop_enabled:
                metrics.update(record_hurdle_curriculum(self, env_ids))
            bar_ids = env_ids[~state.phase_5_no_bar_episode[env_ids] & state.initialized[env_ids]]
            if len(bar_ids):
                metrics.update({
                    "Task/stop_success_rate": state.stop.completed[bar_ids].float().mean(),
                    "Task/stop_timeout_rate": state.stop.failed[bar_ids].float().mean(),
                    "Task/full_sequence_success_rate": state.task_success[bar_ids].float().mean(),
                    "Task/geometric_crossing_rate": state.crossed[bar_ids].float().mean(),
                })
        super()._reset_idx(env_ids)
        self.extras["log"].update(metrics)

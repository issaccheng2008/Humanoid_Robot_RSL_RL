# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment with a configurable global curriculum clock."""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


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
        if hasattr(self, "_wooden_bar_state"):
            state = self._wooden_bar_state
            bar_ids = env_ids[~state.phase_5_no_bar_episode[env_ids] & state.initialized[env_ids]]
            if len(bar_ids):
                metrics = {
                    "Task/stop_success_rate": state.stop.completed[bar_ids].float().mean(),
                    "Task/stop_timeout_rate": state.stop.failed[bar_ids].float().mean(),
                    "Task/full_sequence_success_rate": state.task_success[bar_ids].float().mean(),
                    "Task/geometric_crossing_rate": state.crossed[bar_ids].float().mean(),
                }
        super()._reset_idx(env_ids)
        self.extras["log"].update(metrics)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-local piecewise entropy scheduling for RSL-RL PPO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from rsl_rl.algorithms import PPO


EntropySchedule = tuple[tuple[int, float], ...]
_CHECKPOINT_KEY = "entropy_schedule_state"


def _validate_entropy_schedule(
    schedule: Sequence[Sequence[Real]],
) -> EntropySchedule:
    """Validate and normalize a piecewise-constant entropy schedule."""
    if not schedule:
        raise ValueError("entropy_schedule must contain at least one entry.")

    normalized: list[tuple[int, float]] = []
    previous_iteration = -1
    for entry in schedule:
        if len(entry) != 2:
            raise ValueError(
                "Each entropy_schedule entry must be "
                "(start_iteration, entropy_coef)."
            )
        iteration, coefficient = entry
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise TypeError("Entropy schedule iterations must be integers.")
        if iteration < 0:
            raise ValueError(
                "Entropy schedule iterations must be non-negative."
            )
        if iteration <= previous_iteration:
            raise ValueError(
                "Entropy schedule iterations must be strictly increasing."
            )
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
            raise TypeError("Entropy coefficients must be real numbers.")
        coefficient = float(coefficient)
        if coefficient < 0.0 or not math.isfinite(coefficient):
            raise ValueError(
                "Entropy coefficients must be finite and non-negative."
            )
        normalized.append((iteration, coefficient))
        previous_iteration = iteration

    if normalized[0][0] != 0:
        raise ValueError(
            "The first entropy_schedule entry must start at iteration 0."
        )
    return tuple(normalized)


def entropy_coef_for_iteration(
    schedule: EntropySchedule,
    iteration: int,
) -> float:
    """Return the coefficient active at a phase-local PPO iteration."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")

    coefficient = schedule[0][1]
    for start_iteration, scheduled_coefficient in schedule[1:]:
        if iteration < start_iteration:
            break
        coefficient = scheduled_coefficient
    return coefficient


class EntropyScheduledPPO(PPO):
    """PPO with one reusable piecewise entropy schedule for every phase."""

    def __init__(
        self,
        *args,
        training_phase: int,
        entropy_schedule: Sequence[Sequence[Real]],
        **kwargs,
    ):
        if training_phase not in (1, 2, 3, 4, 5):
            raise ValueError(
                "training_phase must be 1, 2, 3, 4, or 5, but received "
                f"{training_phase}."
            )
        self.training_phase = int(training_phase)
        self.entropy_schedule = _validate_entropy_schedule(entropy_schedule)
        self.entropy_schedule_iteration = 0

        # Make the schedule authoritative even if entropy_coef is also present
        # in an older Hydra/YAML configuration.
        kwargs["entropy_coef"] = entropy_coef_for_iteration(
            self.entropy_schedule, self.entropy_schedule_iteration
        )
        super().__init__(*args, **kwargs)

    def update(self):
        """Apply the scheduled value before each PPO optimization update."""
        self.entropy_coef = entropy_coef_for_iteration(
            self.entropy_schedule, self.entropy_schedule_iteration
        )
        loss_dict = super().update()
        loss_dict["entropy_coef"] = self.entropy_coef
        self.entropy_schedule_iteration += 1
        return loss_dict

    def save(self) -> dict:
        """Save the phase-local schedule position with the PPO state."""
        saved_dict = super().save()
        saved_dict[_CHECKPOINT_KEY] = {
            "training_phase": self.training_phase,
            "phase_iteration": self.entropy_schedule_iteration,
        }
        return saved_dict

    def load(self, loaded_dict: dict, *args, **kwargs):
        """Continue same-phase schedules and reset schedules between phases."""
        load_iteration = super().load(loaded_dict, *args, **kwargs)
        schedule_state = loaded_dict.get(_CHECKPOINT_KEY)

        if (
            isinstance(schedule_state, dict)
            and schedule_state.get("training_phase") == self.training_phase
        ):
            phase_iteration = int(
                schedule_state.get("phase_iteration", 0)
            )
            if phase_iteration < 0:
                raise ValueError(
                    "Checkpoint entropy phase_iteration must be non-negative."
                )
            self.entropy_schedule_iteration = phase_iteration
        else:
            self.entropy_schedule_iteration = 0

        self.entropy_coef = entropy_coef_for_iteration(
            self.entropy_schedule, self.entropy_schedule_iteration
        )
        return load_iteration

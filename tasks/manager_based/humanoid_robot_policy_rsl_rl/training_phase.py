# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single switch for the independently resumed wooden-bar training phases."""

# Set this manually to 1, 2, or 3 before starting the corresponding run.
WOODEN_BAR_TRAINING_PHASE = 1

if WOODEN_BAR_TRAINING_PHASE not in (1, 2, 3):
    raise ValueError(
        "WOODEN_BAR_TRAINING_PHASE must be 1, 2, or 3, but received "
        f"{WOODEN_BAR_TRAINING_PHASE}."
    )

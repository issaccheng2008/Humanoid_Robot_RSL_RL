# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single switch for the independently resumed wooden-bar training phases."""

# 1: normal walking
# 2: virtual band
# 3: robot-collisionless bar with geometric contact penalty
# 4: physical movable bar
# Set this manually before starting the corresponding independently resumed run.
WOODEN_BAR_TRAINING_PHASE = 2

if WOODEN_BAR_TRAINING_PHASE not in (1, 2, 3, 4):
    raise ValueError(
        "WOODEN_BAR_TRAINING_PHASE must be 1, 2, 3, or 4, but received "
        f"{WOODEN_BAR_TRAINING_PHASE}."
    )


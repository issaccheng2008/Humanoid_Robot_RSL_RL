# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single switch for the independently resumed wooden-bar training phases."""

# 1: normal walking
# 2: virtual band
# 3: robot-collisionless bar with geometric contact penalty
# 4: physical movable bar
# 5: Phase 3 obstacle episodes mixed with turning/no-bar/stop episodes
# Resume Phase 5 from a Phase 3 checkpoint.
# Set this manually before starting the corresponding independently resumed run.
WOODEN_BAR_TRAINING_PHASE = 1

if WOODEN_BAR_TRAINING_PHASE not in (1, 2, 3, 4, 5):
    raise ValueError(
        "WOODEN_BAR_TRAINING_PHASE must be 1, 2, 3, 4, or 5, but received "
        f"{WOODEN_BAR_TRAINING_PHASE}."
    )


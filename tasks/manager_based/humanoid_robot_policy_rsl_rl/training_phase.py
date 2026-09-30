# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single switch for the independently resumed locomotion training phases."""

# 1: normal walking
# 2: virtual band
# 3: robot-collisionless bar with geometric contact penalty
# 4: physical movable bar
# 5: Phase 4 obstacle episodes mixed with turning/no-bar/stop episodes
# 6: variable-speed walking, turning, and timed stopping without obstacles
# Resume Phase 5 from a Phase 4 checkpoint.
# Resume Phase 6 from a completed Phase 5 checkpoint.
# Set this manually before starting the corresponding independently resumed run.
WOODEN_BAR_TRAINING_PHASE = 6

if WOODEN_BAR_TRAINING_PHASE not in (1, 2, 3, 4, 5, 6):
    raise ValueError(
        "WOODEN_BAR_TRAINING_PHASE must be 1, 2, 3, 4, 5, or 6, but received "
        f"{WOODEN_BAR_TRAINING_PHASE}."
    )


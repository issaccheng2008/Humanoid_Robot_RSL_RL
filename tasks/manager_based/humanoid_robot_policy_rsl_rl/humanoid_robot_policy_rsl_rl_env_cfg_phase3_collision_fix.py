# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-3-safe environment configuration.

The Phase 3 bar remains a normal dynamic rigid object with gravity and collision
enabled so it can fall onto and settle on the terrain.  The only difference is
that its robot collision pairs are authored during ``prestartup`` -- before the
simulation is started -- instead of mutating USD collision relationships during
``startup`` after PhysX has already initialized.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from . import mdp
from .humanoid_robot_policy_rsl_rl_env_cfg import (
    EventCfg as _BaseEventCfg,
    HumanoidRobotPolicyEnvCfg as _BaseEnvCfg,
    HumanoidRobotPolicyEnvCfg_PLAY as _BasePlayCfg,
    PHYSICAL_WOODEN_BAR_NAME,
)
from .training_phase import WOODEN_BAR_TRAINING_PHASE


@configclass
class EventCfg(_BaseEventCfg):
    """Move Phase 3 robot/bar collision filtering before PhysX startup."""

    configure_collisionless_bar_collisions = EventTerm(
        func=mdp.configure_collisionless_bar_collisions,
        mode="prestartup",
        params={
            "training_phase": WOODEN_BAR_TRAINING_PHASE,
            "robot_name": "robot",
            "physical_bar_name": PHYSICAL_WOODEN_BAR_NAME,
        },
    )


@configclass
class HumanoidRobotPolicyEnvCfg(_BaseEnvCfg):
    """Training configuration with prestartup Phase 3 collision filtering."""

    events: EventCfg = EventCfg()


@configclass
class HumanoidRobotPolicyEnvCfg_PLAY(_BasePlayCfg):
    """Playback configuration with the same prestartup collision filtering."""

    events: EventCfg = EventCfg()

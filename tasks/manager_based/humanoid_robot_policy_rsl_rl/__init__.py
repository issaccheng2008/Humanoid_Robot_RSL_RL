# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium environments for the custom humanoid RSL-RL task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Humanoid-Robot-RSLRL-v0",
    entry_point=(
        f"{__name__}.curriculum_aware_env:"
        "CurriculumAwareManagerBasedRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.fixed_stick_env_cfg:"
            "FixedStickEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "HumanoidRobotRoughPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Humanoid-Robot-RSLRL-Play-v0",
    entry_point=(
        f"{__name__}.curriculum_aware_env:"
        "CurriculumAwareManagerBasedRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.fixed_stick_env_cfg:"
            "FixedStickEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "HumanoidRobotRoughPPORunnerCfg"
        ),
    },
)

# Historical entry point for the v3.1 recording utilities.
gym.register(
    id="Humanoid-Robot-RSLRL-Legacy-Play-v0",
    entry_point=f"{__name__}.curriculum_aware_env:CurriculumAwareManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_robot_policy_rsl_rl_env_cfg:HumanoidRobotPolicyEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidRobotRoughPPORunnerCfg",
    },
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import compute_symmetric_states


@configclass
class HumanoidRobotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the custom humanoid."""

    # Equivalent to the 24-step rollout used by the old SKRL config.
    num_steps_per_env = 24

    # Old SKRL setting:
    #     trainer.timesteps = 72000
    #
    # 72000 / 24 rollout steps = 3000 PPO iterations.
    max_iterations = 5000

    save_interval = 50
    experiment_name = "humanoid_robot_rsl_rl_rough"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,

        # Keep normalization disabled initially to match the SKRL setup.
        actor_obs_normalization=False,
        critic_obs_normalization=False,

        # Same network dimensions as the existing SKRL configuration.
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,

        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=0.5,
            data_augmentation_func=compute_symmetric_states,
        ),
    )
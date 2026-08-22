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
from ..training_phase import WOODEN_BAR_TRAINING_PHASE


# Each entry is (phase-local PPO iteration, entropy_coef). Iteration 0 is
# required. Add later changes in ascending order, for example:
#
#     2: ((0, 0.008), (1000, 0.001), (2500, 0.0005))
#
# A phase-local iteration counts completed PPO updates in that phase. The
# counter is restored only when resuming a checkpoint from the same phase.
ENTROPY_COEF_SCHEDULES: dict[int, tuple[tuple[int, float], ...]] = {
    1: ((0, 0.008),),
    2: ((0, 0.008),),
    3: ((0, 0.008),),
    4: ((0, 0.002),),
    5: ((0, 0.002),(5000, 0.0005)),
}
ENTROPY_COEF_SCHEDULE = ENTROPY_COEF_SCHEDULES[
    WOODEN_BAR_TRAINING_PHASE
]


@configclass
class EntropyScheduledPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO configuration with a phase-local piecewise entropy schedule."""

    class_name: str = (
        f"{__package__}.entropy_schedule:EntropyScheduledPPO"
    )
    training_phase: int = WOODEN_BAR_TRAINING_PHASE
    entropy_schedule: tuple[tuple[int, float], ...] = ENTROPY_COEF_SCHEDULE


@configclass
class HumanoidRobotRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the custom humanoid."""

    # Equivalent to the 24-step rollout used by the old SKRL config.
    num_steps_per_env = 24

    # Old SKRL setting:
    #     trainer.timesteps = 72000
    #
    # 72000 / 24 rollout steps = 3000 PPO iterations.
    if WOODEN_BAR_TRAINING_PHASE == 1:
        max_iterations = 5000 
    if WOODEN_BAR_TRAINING_PHASE == 2:
        max_iterations = 10000
    if WOODEN_BAR_TRAINING_PHASE == 3:
        max_iterations = 10000
    if WOODEN_BAR_TRAINING_PHASE == 4:
        max_iterations = 5000
    if WOODEN_BAR_TRAINING_PHASE == 5:
        max_iterations = 10000

    save_interval = 50
    experiment_name = "humanoid_robot_rsl_rl_rough"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,

        # Keep normalization disabled initially to match the SKRL setup.
        actor_obs_normalization=False,
        critic_obs_normalization=False,

        # Same network dimensions as the existing SKRL configuration.
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
    )

    algorithm = EntropyScheduledPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=ENTROPY_COEF_SCHEDULE[0][1],
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

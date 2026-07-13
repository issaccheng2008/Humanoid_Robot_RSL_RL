"""Left-right symmetry transformations for the custom humanoid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NUM_JOINTS = 12
POLICY_OBS_DIM = 48


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap the right and left legs and convert joint-coordinate signs.

    Expected ordering:
        0:5   = right leg
        6:11  = left leg
    """
    if joint_data.shape[-1] != NUM_JOINTS:
        raise ValueError(
            f"Expected {NUM_JOINTS} joint values, "
            f"but received {joint_data.shape[-1]}."
        )

    right = joint_data[..., 0:6]
    left = joint_data[..., 6:12]

    # All corresponding axes in this robot's URDF have opposite coordinate
    # directions after left-right reflection.
    return -torch.cat((left, right), dim=-1)


def _mirror_policy_observation(obs: torch.Tensor) -> torch.Tensor:
    """Mirror one policy-observation batch across the robot's sagittal plane."""

    if obs.shape[-1] != POLICY_OBS_DIM:
        raise ValueError(
            f"Expected a {POLICY_OBS_DIM}-D policy observation, "
            f"but received {obs.shape[-1]} dimensions. "
            "Update symmetry.py if the observation configuration changed."
        )

    mirrored = obs.clone()
    device = obs.device
    dtype = obs.dtype

    # Observation layout:
    # 0:3    imu_linear_acceleration
    # 3:6    imu_angular_velocity
    # 6:9    projected_gravity
    # 9:12   velocity_commands
    # 12:24  joint_pos
    # 24:36  joint_vel
    # 36:48  previous actions

    # IMU linear acceleration is a polar vector:
    # [ax, ay, az] -> [ax, -ay, az]
    mirrored[..., 0:3] *= torch.tensor(
        [1.0, -1.0, 1.0], device=device, dtype=dtype
    )

    # Angular velocity is an axial vector:
    # [wx, wy, wz] -> [-wx, wy, -wz]
    mirrored[..., 3:6] *= torch.tensor(
        [-1.0, 1.0, -1.0], device=device, dtype=dtype
    )

    # Projected gravity:
    # [gx, gy, gz] -> [gx, -gy, gz]
    mirrored[..., 6:9] *= torch.tensor(
        [1.0, -1.0, 1.0], device=device, dtype=dtype
    )

    # Velocity commands:
    # [vx, vy, wz] -> [vx, -vy, -wz]
    mirrored[..., 9:12] *= torch.tensor(
        [1.0, -1.0, -1.0], device=device, dtype=dtype
    )

    mirrored[..., 12:24] = _mirror_joint_data(obs[..., 12:24])
    mirrored[..., 24:36] = _mirror_joint_data(obs[..., 24:36])
    mirrored[..., 36:48] = _mirror_joint_data(obs[..., 36:48])

    return mirrored


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror the policy's raw joint-position actions."""
    return _mirror_joint_data(actions)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and left-right-mirrored samples."""

    if obs is not None:
        if "policy" not in obs.keys():
            raise KeyError(
                f"Expected observation group 'policy', but found {list(obs.keys())}."
            )

        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _mirror_policy_observation(
            obs["policy"]
        )
    else:
        obs_aug = None

    if actions is not None:
        actions_aug = torch.cat(
            (actions, _mirror_actions(actions)),
            dim=0,
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug
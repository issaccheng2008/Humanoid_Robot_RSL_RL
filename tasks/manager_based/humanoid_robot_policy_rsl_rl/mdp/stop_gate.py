"""Vectorized, simulator-independent continuous stop gate."""
import math
import torch


class StopGate:
    def __init__(self, num_envs, device, dt, hold_s=0.5, timeout_s=3.0,
                 speed_limit=0.05, yaw_limit=0.10):
        if not 0 < dt <= hold_s < timeout_s or min(speed_limit, yaw_limit) <= 0:
            raise ValueError('Require 0 < dt <= hold_s < timeout_s and positive limits')
        self.hold_steps = math.ceil(hold_s / dt)
        self.timeout_steps = math.ceil(timeout_s / dt)
        self.speed_limit, self.yaw_limit = speed_limit, yaw_limit
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.completed = torch.zeros_like(self.active)
        self.failed = torch.zeros_like(self.active)
        self.ready = torch.zeros_like(self.active)
        self.stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.elapsed_steps = torch.zeros_like(self.stable_steps)

    def reset(self, env_ids):
        for value in (self.active, self.completed, self.failed, self.ready,
                      self.stable_steps, self.elapsed_steps):
            value[env_ids] = 0

    def update(self, trigger, update_envs, in_contact, linear_velocity, yaw_rate):
        self.ready[update_envs] = False
        start = trigger & update_envs & ~self.active & ~self.completed & ~self.failed
        self.active |= start
        # Do not count the sample that first issued the stop command.
        tick = self.active & update_envs & ~start
        stable = (in_contact.all(dim=1)
                  & (linear_velocity[:, :2].square().sum(dim=1) < self.speed_limit**2)
                  & (yaw_rate.abs() < self.yaw_limit))
        self.elapsed_steps[tick] += 1
        self.stable_steps[tick & ~stable] = 0
        self.stable_steps[tick & stable] += 1
        self.ready |= tick & (self.stable_steps >= self.hold_steps)
        self.failed |= tick & ~self.ready & (self.elapsed_steps >= self.timeout_steps)
        self.completed |= self.ready
        self.active &= ~self.completed & ~self.failed
        return self.ready

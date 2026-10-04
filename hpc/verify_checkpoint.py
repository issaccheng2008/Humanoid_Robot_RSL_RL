"""Run a short, single-environment checkpoint smoke test on the HPC."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# The project is a Python package in its own directory, not an installed wheel.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import tasks  # noqa: E402,F401  Register Humanoid-Robot-RSLRL-Play-v0.


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Humanoid-Robot-RSLRL-Legacy-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--steps", type=int, default=250)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *remaining_args]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive")
    for label, filename in (("checkpoint", args_cli.checkpoint), ("USD", args_cli.usd)):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{label} missing: {filename}")

    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn.usd_path = os.path.abspath(args_cli.usd)
    env_cfg.commands.base_velocity.debug_vis = False
    # Every episode in this check uses the physical-bar branch of Phase 5.
    env_cfg.events.reset_crossing_state.params["phase_5_bar_episode_probability"] = 1.0
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        import importlib.metadata

        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, importlib.metadata.version("rsl-rl-lib")
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            if agent_cfg.class_name != "OnPolicyRunner":
                raise ValueError(f"Unexpected runner: {agent_cfg.class_name}")
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(os.path.abspath(args_cli.checkpoint))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            obs = env.get_observations()
            # RSL-RL 5.x wraps observation groups in a TensorDict. Its shape is
            # the batch size (1 here); the feature dimension belongs to "policy".
            policy_obs = obs["policy"]
            print(f"[VERIFY] policy observation shape: {tuple(policy_obs.shape)}", flush=True)
            if policy_obs.shape[-1] != 49:
                raise RuntimeError(f"Expected 49 policy observations, got {policy_obs.shape[-1]}")
            robot = env.unwrapped.scene["robot"]
            print(f"[VERIFY] joints: {robot.data.joint_names}", flush=True)
            print(f"[VERIFY] USD: {env_cfg.scene.robot.spawn.usd_path}", flush=True)

            for step in range(1, args_cli.steps + 1):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                if step == 1 or step % 50 == 0:
                    pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                    print(f"[VERIFY] step={step} root_xyz={pos} done={bool(dones[0])}", flush=True)
            print(f"[VERIFY] completed {args_cli.steps} control steps", flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()

"""Fine-tune the v3.2 walk-stop-cross task; defaults to model_33999.pt warm-start."""
from __future__ import annotations
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import gymnasium as gym
    import torch
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab.utils.io import dump_yaml
    from rsl_rl.runners import OnPolicyRunner
    from fine_tune_checkpoint import load_fine_tune_state
    import tasks  # register only this checkout's task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Humanoid-Robot-RSLRL-v0')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'model_33999.pt')
    parser.add_argument('--usd', type=Path, default=ROOT/'v3.2/v3.2.usd')
    parser.add_argument('--resume', action='store_true', help='Restore a NEW fine-tune checkpoint including optimizer')
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--max-iterations', type=int, default=3000, help='Additional updates, not an absolute checkpoint index')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--run-name', default='finetune')
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *remaining]
    if args.num_envs < 1 or args.max_iterations < 1:
        raise ValueError('num-envs and max-iterations must be positive')
    for path in (args.checkpoint, args.usd):
        if not path.is_file(): raise FileNotFoundError(path)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if tuple(checkpoint['actor_state_dict']['mlp.0.weight'].shape) != (512,49):
        raise ValueError('Checkpoint must match the existing 49-dimensional policy')

    @hydra_task_config(args.task, 'rsl_rl_cfg_entry_point')
    def run(env_cfg, agent_cfg):
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.scene.robot.spawn.usd_path = str(args.usd.resolve())
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.seed = agent_cfg.seed
        if args.device is not None:
            env_cfg.sim.device = args.device
            agent_cfg.device = args.device
        env_cfg.curriculum_start_step = (
            int(checkpoint['entropy_schedule_state']['phase_iteration']) * agent_cfg.num_steps_per_env
            if args.resume else 0
        )
        log_dir = ROOT/'logs/rsl_rl'/agent_cfg.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')+'_'+args.run_name)
        log_dir.mkdir(parents=True, exist_ok=False)
        env_cfg.log_dir = str(log_dir)
        agent_cfg.algorithm.learning_rate = args.learning_rate
        agent_cfg.max_iterations = args.max_iterations
        with launch_simulation(env_cfg, args):
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version('rsl-rl-lib'))
            env = gym.make(args.task, cfg=env_cfg)
            try:
                env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
                load_fine_tune_state(runner, checkpoint, args.resume, args.learning_rate)
                obs = env.get_observations()
                if obs['policy'].shape[-1] != 49 or env.num_actions != 12:
                    raise RuntimeError('Observation/action interface changed')
                metadata = {'checkpoint': str(args.checkpoint.resolve()), 'usd': str(args.usd.resolve()),
                            'resume': args.resume, 'learning_rate': runner.alg.learning_rate,
                            'optimizer_lrs': [g['lr'] for g in runner.alg.optimizer.param_groups],
                            'stage': runner.alg.training_stage, 'start_iteration': runner.current_learning_iteration,
                            'additional_iterations': args.max_iterations,
                            'joint_names': list(env.unwrapped.scene['robot'].data.joint_names)}
                (log_dir/'fine_tune.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
                dump_yaml(str(log_dir/'params/env.yaml'),env_cfg)
                dump_yaml(str(log_dir/'params/agent.yaml'),agent_cfg)
                print('[FINETUNE]', json.dumps(metadata), flush=True)
                print('[FINETUNE] logs:', log_dir, flush=True)
                runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
            finally:
                env.close()
    run()


if __name__ == '__main__':
    main()

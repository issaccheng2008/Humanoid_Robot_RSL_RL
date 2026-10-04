"""Final-level evaluation and an explicit, checkpointed success stop condition."""
import copy
import json
from pathlib import Path


class SuccessBatch:
    """Count each environment's first episode; unfinished episodes are failures."""
    def __init__(self, count):
        self.count = count
        self.seen = set()
        self.successes = 0

    def record(self, env_ids, clean_success):
        for env_id, success in zip(env_ids, clean_success):
            if 0 <= env_id < self.count and env_id not in self.seen:
                self.seen.add(env_id)
                self.successes += int(bool(success))

    @property
    def complete(self):
        return len(self.seen) == self.count

    @property
    def rate(self):
        return self.successes / self.count


def evaluate_deterministic(runner, episodes=200):
    """Use mean actions, clean terminal outcomes, and the fixed initial pose.

    Evaluation resets the shared scene between training blocks. Training window
    statistics, RNG, observation noise and curriculum clock are restored; no
    evaluation transitions enter PPO storage or training reward logs.
    """
    import torch
    env = runner.env
    raw = env.unwrapped
    state = raw._fixed_stick_state
    if state.curriculum.level != 5 or state.curriculum.parameters != (0., .03):
        raise ValueError('Final evaluation requires zero preparation and a 3 cm toe gap')
    stats, curriculum = copy.deepcopy(state.clean_stats), copy.deepcopy(state.curriculum)
    counter = raw.common_step_counter
    policy = runner.get_inference_policy(device=runner.device)
    # Isaac's observation manager owns the live group configuration.
    group = raw.observation_manager._group_obs_term_cfgs.get('policy', [])
    saved_noise = [(term, term.noise) for term in group]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    successes = 0
    evaluated = 0
    try:
        for term, _ in saved_noise:
            term.noise = None
        state.curriculum.auto_advance = False
        while evaluated < episodes:
            batch = SuccessBatch(min(env.num_envs, episodes - evaluated))
            obs, _ = env.reset()
            policy.reset()
            max_steps = int(raw.max_episode_length) + 1
            with torch.inference_mode():
                for _ in range(max_steps):
                    obs, _, dones, infos = env.step(policy(obs.to(runner.device)))
                    policy.reset(dones)
                    outcome = infos.get('fixed_stick_outcome', {})
                    if outcome:
                        ids = outcome['env_ids']
                        valid = dones[ids].bool()
                        batch.record(ids[valid].detach().cpu().tolist(),
                                     outcome['clean_success'][valid].detach().cpu().tolist())
                    if batch.complete:
                        break
            successes += batch.successes
            evaluated += batch.count
        return {'mode': 'deterministic', 'rate': successes / evaluated,
                'successes': successes, 'episodes': evaluated}
    finally:
        for term, noise in saved_noise:
            term.noise = noise
        state.clean_stats, state.curriculum = copy.deepcopy(stats), copy.deepcopy(curriculum)
        raw.common_step_counter = counter
        env.reset()
        policy.reset()
        # Reset can itself clear/record episode windows: restore them afterwards.
        state.clean_stats, state.curriculum = stats, curriculum
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        runner.alg.train_mode()


def learn_until_success(runner, iterations, log_dir, threshold=.8, min_episodes=200,
                        interval=50, mode='deterministic', evaluate=None):
    if iterations < 1 or interval < 1 or min_episodes < 1 or not 0 < threshold <= 1:
        raise ValueError('Invalid success-stop settings')
    if mode not in ('deterministic', 'training'):
        raise ValueError('Unknown success-stop mode')
    log_dir = Path(log_dir)
    logger = runner.logger
    init_writer, stop_writer = logger.init_logging_writer, logger.stop_logging_writer
    init_writer()
    logger.init_logging_writer = lambda: None
    logger.stop_logging_writer = lambda: None
    result = {'target_reached': False, 'threshold': threshold, 'mode': mode}
    remaining = iterations
    try:
        while remaining:
            count = min(interval, remaining)
            runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
            remaining -= count
            state = runner.env.unwrapped._fixed_stick_state
            if state.curriculum.level == 5:
                if evaluate is not None:
                    metrics = evaluate()
                elif mode == 'deterministic':
                    metrics = evaluate_deterministic(runner, min_episodes)
                else:
                    metrics = {'rate': state.curriculum.rate,
                               'episodes': len(state.curriculum.outcomes)}
                result.update(metrics, iteration=runner.current_learning_iteration)
                result['target_reached'] = metrics['episodes'] >= min_episodes and metrics['rate'] >= threshold
                print('[FINAL EVAL]', json.dumps(result), flush=True)
                if logger.writer is not None:
                    logger.writer.add_scalar(f'Evaluation/{mode}_clean_success_rate', metrics['rate'],
                                             runner.current_learning_iteration)
                runner.final_evaluation = dict(result)
                runner.save(str(log_dir / 'model_final_latest.pt'))
                (log_dir / 'final_evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                if result['target_reached']:
                    runner.save(str(log_dir / 'model_success_80.pt'))
                    print('[STOP] Final-level clean success target reached; checkpoint saved.', flush=True)
                    break
            # RSL's runner stores the last executed zero-based update index.
            if remaining:
                runner.current_learning_iteration += 1
        return result
    finally:
        logger.init_logging_writer, logger.stop_logging_writer = init_writer, stop_writer
        stop_writer()

"""Explicit warm-start/resume behavior for RSL-RL 5.x."""
import math


def load_fine_tune_state(runner, checkpoint, resume=False, learning_rate=5e-5):
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 5e-5:
        raise ValueError('Fine-tune learning rate must be in (0, 5e-5]')
    state = checkpoint.get('entropy_schedule_state', {})
    if resume and (state.get('training_stage') != runner.alg.training_stage
                   or state.get('training_phase') != runner.alg.training_phase):
        raise ValueError('Resume requires a checkpoint from this fine-tune stage; use warm-start for model_33999.pt')
    runner.alg.load(checkpoint, load_cfg={'actor': True, 'critic': True,
                    'optimizer': resume, 'iteration': resume}, strict=True)
    # Checkpoints record the last executed (zero-based) iteration.
    runner.current_learning_iteration = int(checkpoint['iter']) + 1 if resume else 0
    runner.alg.learning_rate = learning_rate
    runner.alg.schedule = 'fixed'
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = learning_rate
    if any(group['lr'] != learning_rate for group in runner.alg.optimizer.param_groups):
        raise RuntimeError('Optimizer learning rate differs from fine-tune configuration')

"""Checkpoint stage selection shared by training and replay (no simulator imports)."""
import warnings


def checkpoint_stage(checkpoint):
    label = checkpoint.get('entropy_schedule_state', {}).get('training_stage', '')
    return {f'fixed_stick_stage{i}_v32': i for i in (1, 2)}.get(label)


def validate_stage_checkpoint(checkpoint, stage, resume=False):
    source = checkpoint_stage(checkpoint)
    if resume and source != stage:
        raise ValueError('Resume requires a checkpoint from the same stage; omit --resume to warm-start')
    return source

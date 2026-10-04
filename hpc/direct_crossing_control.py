"""Configure curriculum before simulation reset; freeze levels during replay."""
def configure_direct_crossing(cfg, checkpoint, training, resume=False, level=None):
    saved = (checkpoint.get('infos') or {}).get('direct_crossing_curriculum')
    if resume and saved is None:
        raise ValueError('Old checkpoint has no direct curriculum: warm-start without --resume')
    if resume and level is not None and level != saved['level']:
        raise ValueError('Changing curriculum level requires warm-start without --resume')
    cfg.direct_curriculum_enabled = True
    cfg.direct_curriculum_auto_advance = bool(training)
    cfg.direct_curriculum_saved = saved if resume or (not training and level is None) else None
    cfg.direct_curriculum_level = int(level if level is not None else
                                     saved['level'] if saved and (resume or not training) else 0)
    if cfg.direct_curriculum_enabled:
        if cfg.fixed_command_mode != 'touchdown':
            raise ValueError('Curriculum uses touchdown mode; explicitly choose --command-mode touchdown')
        # Shared clock contract remains valid even though its boundaries are unused.
        cfg.fixed_walk_step = .10
        cfg.fixed_initial_gap = .08
    return cfg


def ensure_direct_crossing_initialized(env, curriculum_factory=None):
    """Ensure curriculum exists before the first actor action; never disable it silently."""
    cfg = env.cfg
    if not getattr(cfg, 'direct_curriculum_enabled', False):
        raise RuntimeError('Direct curriculum is disabled or missing. Sync hpc/direct_crossing_control.py '
                           'and tasks/manager_based/humanoid_robot_policy_rsl_rl/fixed_stick_env_cfg.py.')
    state = getattr(env, '_fixed_stick_state', None)
    if state is None:
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp.fixed_stick import get_fixed_state
        state = get_fixed_state(env)
    if not hasattr(state, 'episode_level'):
        raise RuntimeError('Stale fixed_stick_state.py: missing episode_level. Sync the entire hpc/ and '
                           'tasks/ folders, especially mdp/fixed_stick_state.py, mdp/fixed_stick.py '
                           'and curriculum_aware_env.py before training.')
    if state.command_mode != 'touchdown' or state.collisionless_mode:
        raise RuntimeError('Direct curriculum requires physical Stage 2 and touchdown mode')
    if getattr(state, 'curriculum', None) is not None:
        return False
    if curriculum_factory is None:
        from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp.fixed_stick_state import DirectCrossingCurriculum
        curriculum_factory = DirectCrossingCurriculum
    state.curriculum = curriculum_factory(
        level=cfg.direct_curriculum_level,
        auto_advance=cfg.direct_curriculum_auto_advance,
        saved=cfg.direct_curriculum_saved,
    )
    # Initial reset may already have placed the bar using the old geometry.
    # Reset the robot, geometry and first observation together before policy inference.
    env.reset()
    print('[DIRECT CURRICULUM] initialized missing curriculum and reset before first action', flush=True)
    return True

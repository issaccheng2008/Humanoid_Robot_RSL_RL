# Fixed-stick phase clock implementation

Approved design: initial toe/bar gap 0.08 m; WALK (0.20, 0, 0.10, 0), LEAD (0.20, 0, 0.23, 1), FOLLOW (0.20, 0, 0, 1). Candidate transitions at 0.14/0.44/0.76 s. Keep 49 policy observations and 12 joint actions. Only IMU/encoders plus one start cue on hardware. Completion means command sequence ended; physical success remains a separate simulation score.

1. Add `tests/test_phase_clock.py`, run it red. Cover boundaries, host start latch, observation/action conversion, per-environment reset, independence from simulator touchdown, and absence of success on clock timeout alone.
2. Add shared `deployment/phase_clock.py` and `deployment/policy_interface.py`; integrate command-only clock into `mdp/fixed_stick_state.py`, `mdp/fixed_stick.py` and `fixed_stick_env_cfg.py`. Preserve legacy touchdown scheduling as an explicit comparison mode.
3. Add `hpc/fixed_stick_control.py`; unify train/play CLI settings, terminal snapshots, checkpoint metadata and resume compatibility. Store clock experiments separately. Keep scoring/collision curriculum unchanged.
4. Add CPU ONNX exporter and deployment contract. Verify native vs exported actor numerically. Document exact 49/12 order, units, IMU frame/gravity semantics, action scaling, reset, phase handoff, launch commands and which training files the upper-computer developer should reference.
5. Run old/new unit tests, real Kit phase-clock replay of model_850, small clock training smoke, and checkpoint resume/replay. Record actual results and limits in the handoff MD; request independent code review and fix substantive findings.

All project edits stay in the user-requested `cross_stick_p3` directory. Existing weights are a warm-start prototype until phase-clock training and physical Stage 2 evaluation pass; old touchdown-scheduled success rates cannot certify the new schedule.

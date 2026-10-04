# Fixed-stick moving crossing implementation plan

> Execute inline in the user-selected p3 checkout. Preserve local changes and legacy task modules.

**Goal:** Train v3.2 to walk one 10 cm step then cross one fixed stick initially 8 cm ahead without stopping or observing live obstacle distance. The user specified a 10 cm walking step and an 8 cm initial gap; all other task settings remain unchanged.

**Architecture:** A vectorized state machine owns commands and events. An Isaac adapter places the bar after reset FK refresh, measures contacts, and updates once per control step. A dedicated default config replaces task-specific scene, command, reward and termination terms.

**Tech stack:** Existing IsaacLab 3.0 launch API, Isaac Sim 6.0.1, PyTorch, RSL-RL 5.4.1; 49D/12D MLP and existing checkpoint loader.

1. [x] Add `tests/test_fixed_stick.py`; run `python -B -m unittest discover -s tests -p test_fixed_stick.py -q`; verify missing-state failures.
2. [x] Implement `mdp/fixed_stick_state.py`: partial reset, WALK→LEAD→FOLLOW→DONE, swing touchdown gates, fixed bar coordinates, failure priority, fixed commands and one-shot events. Run tests again.
3. [x] Add `mdp/fixed_stick.py` and `fixed_stick_env_cfg.py`, register the new default config. Test exact reset, one kinematic colliding stick, observation order, fixed commands, success and retained stability rewards.
4. [x] Update runner stage/log identity, metrics, terminal outcome capture, training/play/HPC entry points and runtime metadata. Test strict warm start/resume and original suites.
5. [x] Run the complete unittest suite; run `hpc/train_walk_stop_cross.py --num-envs 4 --max-iterations 2 --headless`; inspect errors and actual initial gap/command switches.
6. [x] Write Chinese training/deployment instructions with the 8 cm initial gap and first-touchdown command trigger, runtime requirements, 49D observation/action mapping, warm start/resume and validation limits. Review final diff.

Verification: 98 tests in `tests/` pass (20 new fixed-stick checks); the final 4-environment, 2-update Isaac/PPO run exits successfully. The separate legacy recording suite has 6 passes and 2 stale camera-expectation failures against pre-existing camera changes. Full crossing success is still 0 in these short runs; HPC and hardware were not executed.

Latest correction: issue crossing on the first completed swing touchdown without stride/displacement gates. 117 tests pass; real Kit replay issues cross=1 at step 8 (0.16 s), with an 8 cm measured initial gap. Historical verification above remains unchanged.

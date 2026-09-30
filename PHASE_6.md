# Phase 6: walk, turn, and stop

Phase 6 fine-tunes a completed Phase 5 checkpoint. The default switch in
`tasks/manager_based/humanoid_robot_policy_rsl_rl/training_phase.py` is now
`WOODEN_BAR_TRAINING_PHASE = 6`. Selecting 1–5 still uses the previous behavior.

| Setting | Phase 6 behavior |
| --- | --- |
| Episode duration | 10 seconds, unless the robot falls earlier |
| Initial forward speed | Uniform random sample from 0.05–0.4 m/s |
| Initial yaw rate | Uniform random sample from −1–1 rad/s |
| Lateral speed | Always zero |
| Command updates | At episode ages 1, 2, 3, … seconds |
| Yaw update | Independent 20% chance at each one-second check |
| Speed update | Independent 10% chance at each one-second check |
| Stop episodes | Independent 30% selection at each episode reset |
| Stop timing | At 7 seconds; all velocities and step distance stay zero until reset |
| Step-distance observation | `DEFAULT_STEP_DISTANCE * forward_speed / 0.4` |
| Crossing observation | Always zero |
| Obstacles | No physical/collisionless bars, virtual bands, or crossing callbacks |
| Step-distance reward | Removed in Phase 6; its Gaussian curriculum is also removed |

The current `DEFAULT_STEP_DISTANCE` is 0.08 m, so the target is 0.01 m at
0.05 m/s and 0.08 m at 0.4 m/s. The step command remains in the observations
even though its tracking reward is disabled. The observation order and size
remain 49, and the action size and network architecture are unchanged.

Command changes and stop selection are independent per environment. A failed
one-second probability check leaves that command unchanged. Stops are sampled
once per episode, so later command checks cannot restart a stopped robot.
The Play task also uses the Phase 6 ranges, schedule, and 10-second duration.

The inherited command timer uses finite bounds `(1.0e9, 1.0e9)` because Isaac
Lab samples it with PyTorch's `uniform_()`. It cannot expire during a 10-second
episode. The one-second command changes and seven-second stops use episode
steps independently of this timer.

## Resume training

Use your usual Isaac Lab RSL-RL training command and the same task
`Humanoid-Robot-RSLRL-v0`, specifying `--resume` and your completed Phase 5
checkpoint. For example, from the Isaac Lab directory:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Humanoid-Robot-RSLRL-v0 --headless --resume \
  --load_run <phase_5_run_directory> --checkpoint <model_N.pt>
```

Replace the bracketed placeholders with your run and checkpoint names. Use the
updated installed task source rather than an old Phase 5 configuration override.
Keep `curriculum_start_step = 0` for this new phase. The entropy scheduler resets
its phase-local counter when loading a Phase 5 checkpoint and uses a constant
0.0005 coefficient in Phase 6, matching Phase 5's final scheduled value. A
same-phase Phase 6 resume retains its saved schedule counter. The Phase 6 runner
collects 24 steps per update and defaults to 10,000 additional PPO iterations.

## Validation

Run the CPU logic checks with:

```bash
python -m unittest discover -s tests -v
```

The tests stub Isaac Lab interfaces so they do not launch Isaac Sim. When PyTorch
is unavailable, a small NumPy tensor adapter exercises the same command code;
the test output identifies that backend. These checks cover command sampling,
timing, independent updates, persistent stops, partial resets, observation
scaling, disabled obstacle/reward wiring, and checkpoint schedule transitions.
A full Isaac Sim training smoke test still requires your simulator environment
and robot assets.

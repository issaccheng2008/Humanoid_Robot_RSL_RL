# Historical walk → stop → cross notes

The p3 default task is now **moving crossing of one fixed stick**. Use
`walk_stop_cross_hpc_zh.md` for current training, HPC and deployment commands.
The stop gate, dynamically spawned bar, ten-stick course and paths below describe
older experiments. The v3.1 tools use `Humanoid-Robot-RSLRL-Legacy-Play-v0`.

This checkout fine-tunes `model_33999.pt` on the v3.2 robot. The Phase 5 obstacle episode first walks, then commands zero speed. After both feet remain in contact and planar speed is below 0.05 m/s with yaw rate below 0.10 rad/s for 0.5 continuous seconds, the physical bar appears ahead of the *current* sole geometry and crossing starts. The stop fails after 3 seconds. The 49-observation, 12-action policy interface is unchanged. Completion requires both feet still past the bar and two double-support control samples; fall or bar-move termination on that frame cannot earn completion reward.

The v3.2 sole contour was regenerated from the local USD collision meshes because the toe and heel were each shortened by approximately 1 cm. PPO is configured for fixed learning rate `5e-5`, entropy `5e-4`, clip `0.1`, 8-second episodes, and 80% obstacle episodes. New fine-tuning runs retain actor/critic weights and noise from `model_33999.pt` while resetting the optimizer and schedule; `--resume` only accepts a checkpoint written by this fine-tune stage and restores optimizer and iteration.

## HPC

Copy the updated `cross_stick` directory, including `model_33999.pt` and `v3.2/`, to `$HOME/cross_stick` on the HPC. The Slurm wrapper reuses the existing `biped-sandbox.sif` and `IsaacLab/source` layout from this project's previous HPC scripts. From the project root:

```bash
sbatch hpc/train_walk_stop_cross.sh
```

`PROJECT` and `CONTAINER` can override the default host paths. `NUM_ENVS` (default 512) and `ITERATIONS` (default 3000) can override the job size. If resuming a *new* fine-tune checkpoint, pass its path inside the container:

```bash
sbatch hpc/train_walk_stop_cross.sh --resume \
  --checkpoint /workspace/cross_stick/logs/rsl_rl/cross_stick_walk_stop_cross_v32/RUN/model_N.pt \
  --max-iterations 1000
```

`--max-iterations` means additional PPO updates. Runs are saved under `logs/rsl_rl/cross_stick_walk_stop_cross_v32/`, each with `fine_tune.json`, config YAML, TensorBoard data, and `model_N.pt`. Check `Task/stop_success_rate`, `Task/stop_timeout_rate`, `Task/geometric_crossing_rate`, and `Task/full_sequence_success_rate` on obstacle episodes. The first runs should be assessed against the original walking behavior and recordings; 90% full-sequence success was a proposed target, not a verified result.

For a single-episode recording on the HPC, use `hpc/play_walk_stop_cross.py --checkpoint <fine-tune-model.pt> --steps 400 --headless`. It defaults to the v3.2 USD and a software-rendered video. The older `play_obstacle_v31.py` and its Slurm wrapper remain legacy v3.1 entry points.

## Local validation already performed

- 19 new unit/integration tests passed, including continuous stopping, bar spawn exactly once, retreat rejection, timeout penalty, checkpoint warm-start/resume, and v3.2 sole extents. A final 1-environment, 20-control-step inference check using v3.2 completed after the sole change with 49 policy observations and all 12 joints.
- 8 existing video/recording tests passed; Python syntax compilation passed.
- An 8-environment, 2-update simulator smoke run loaded `model_33999.pt`, v3.2, and used actual optimizer learning rate `5e-5`.
- A 32-environment, 25-update short run completed without task crashes. It did **not** learn the full new behavior at this short budget; the measured full-sequence success rate was 0. Training is intended to continue on the HPC.

Do not interpret Isaac Lab's generic `Metrics/success_rate` as this task's crossing success. Use `Task/full_sequence_success_rate` for obstacle episodes.


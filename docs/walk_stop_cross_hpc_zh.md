# 行走—停稳—跨越（v3.2）

本项目基于 v3.2 机器人和 `model_33999.pt` 进行微调。Phase 5 的障碍物回合先行走，再将速度指令设为零。双脚保持接地、水平速度低于 0.05 m/s、偏航角速度低于 0.10 rad/s，并连续满足 0.5 秒后，才会在当前脚底几何前方生成实体木棍并开始跨越。若 3 秒内未能停稳，该回合会失败。策略接口保持不变：49 维观测、12 维动作。成功跨越要求双脚仍处于木棍远侧，并连续两个控制步保持双脚支撑；若同一控制步触发跌倒或木棍移动终止，不会获得成功奖励。

v3.2 的脚尖和脚跟各缩短约 1 cm，因此足底轮廓已依据项目内 USD 碰撞网格重新生成。PPO 使用固定学习率 `5e-5`、熵系数 `5e-4`、裁剪参数 `0.1`、8 秒回合，障碍物回合占 80%。首次微调会加载 `model_33999.pt` 的 actor、critic 和策略噪声参数，并重新初始化优化器和微调计数。`--resume` 仅接受本微调阶段生成的检查点，并恢复优化器状态和训练迭代。

## 在 HPC 上训练

将更新后的 `cross_stick` 文件夹复制到 HPC 的 `$HOME/cross_stick`，并确保其中包含 `model_33999.pt` 和 `v3.2/`。Slurm 脚本默认使用此前项目脚本采用的 `biped-sandbox.sif` 容器和 `IsaacLab/source` 目录结构。进入项目目录后提交：

```bash
sbatch hpc/train_walk_stop_cross.sh
```

可通过环境变量 `PROJECT` 和 `CONTAINER` 指定项目目录与容器路径。也可用 `NUM_ENVS`（默认 512）和 `ITERATIONS`（默认 3000）调整并行环境数和新增 PPO 更新次数。续训时，检查点路径须使用容器内路径：

```bash
sbatch hpc/train_walk_stop_cross.sh --resume \
  --checkpoint /workspace/cross_stick/logs/rsl_rl/cross_stick_walk_stop_cross_v32/RUN/model_N.pt \
  --max-iterations 1000
```

`--max-iterations` 表示本次新增的 PPO 更新次数。训练结果保存在 `logs/rsl_rl/cross_stick_walk_stop_cross_v32/`，其中包含 `fine_tune.json`、环境和算法 YAML 配置、TensorBoard 数据及 `model_N.pt` 检查点。请重点查看障碍物回合的以下指标：

- `Task/stop_success_rate`：成功停稳比例
- `Task/stop_timeout_rate`：停车超时比例
- `Task/geometric_crossing_rate`：双脚在几何上越过木棍的比例
- `Task/full_sequence_success_rate`：完成“行走—停稳—跨越”全过程的比例

首轮训练应结合原行走能力和录制视频一并评估。90% 的全过程成功率是预期目标，目前尚未通过训练验证。

在 HPC 上录制单个回合，可运行 `hpc/play_walk_stop_cross.py --checkpoint <微调检查点路径> --steps 400 --headless`。该脚本默认使用 v3.2 USD 和软件视频渲染。旧版 `play_obstacle_v31.py` 及其 Slurm 脚本仍是 v3.1 的回放入口。

## 已完成的本地验证

- 19 项新增单元与集成测试通过，覆盖连续停稳、木棍仅生成一次、越杆后退回时拒绝判成功、停车超时惩罚、检查点冷启动与续训、v3.2 足底尺寸等情况。更新足底轮廓后，还完成了单环境 20 个控制步的 v3.2 推理检查，确认策略观测为 49 维且包含全部 12 个关节。
- 8 项原有视频与录制测试通过；Python 语法编译检查通过。
- 完成 8 环境、2 次更新的 Isaac Sim 冒烟训练：成功加载 `model_33999.pt` 和 v3.2，优化器实际学习率为 `5e-5`。
- 完成 32 环境、25 次更新的短程训练，未发生任务崩溃。但这段短训练尚未学会完整的新动作，当时测得的全过程成功率为 0。后续训练应在 HPC 上进行。

不要把 Isaac Lab 通用指标 `Metrics/success_rate` 当作跨越成功率。请使用 `Task/full_sequence_success_rate` 评估障碍物回合的完整任务成功率。

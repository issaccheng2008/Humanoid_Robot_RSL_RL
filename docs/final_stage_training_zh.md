# 最后一级直接跨棍训练（2026-10-04）

本次训练直接从最后一级开始：Stage 2 实体木棍、零预备步、初始最前脚尖到木棍近侧净距离 3 cm。3 cm 已是原课程最后一级的距离，本次没有把它改成别的参考点。机器人从原微屈膝姿态启动，第一帧即输入跨棍指令，跨前腿目标仍为 23 cm；前腿落地后切换收后腿。

默认暖启动检查点为 `logs/rsl_rl/fixed_stick_stage2_v32/2026-10-04_18-10-03-535593_finetune/model_350.pt`。它是当前已下载文件中首次保存到最后一级的检查点。只继承策略和价值网络，不使用 `--resume`，不继承之前零成功率窗口，也不再从第一课程级重新升级。新训练的更新序号从 0 开始，不代表原模型的第 0 轮。

## 运行与停止

HPC 同步整个 `hpc/` 后，在项目目录执行：

```bash
sbatch hpc/train_walk_stop_cross.sh
```

需要服务器上存在上述 `model_350.pt`；脚本默认传入容器挂载路径 `/workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage2_v32/2026-10-04_18-10-03-535593_finetune/model_350.pt`。

默认每训练 50 轮，以当前策略确定性动作（不加策略动作采样噪声）评估 200 个回合。观测噪声在评估期间关闭，物理碰撞与电机限制保留。只有完整跨越且全程未碰棍才算成功。每个环境只计本批次的第一个回合；超时未完成计失败。使用当前训练环境的物理参数，不把它当作真机成功率。

200 个回合至少 160 个成功（成功率 >= 80%）时，保存 `model_success_80.pt` 并正常退出训练。未达到则继续，最多执行 `--max-iterations` 指定的新增轮数；达到最大轮数退出不代表达标。

每次评估均输出 `[FINAL EVAL]`，写入 `Evaluation/deterministic_clean_success_rate` 和运行目录的 `final_evaluation.json`，并保存 `model_final_latest.pt`。达标打印 `[STOP]`。评估结果也写入检查点 `infos.final_evaluation`。

训练日志里的采样成功率与这里的确定性评估成功率不同，不能用前者宣称默认回放已达到 80%。此前 model_300 在采样模式能成功，而确定性回放失败。

可覆盖：

```bash
# 修改每次评估之间的训练轮数
sbatch hpc/train_walk_stop_cross.sh --evaluation-interval 25

# 如果明确选择训练采样窗口作为停止口径
sbatch hpc/train_walk_stop_cross.sh --success-stop-mode training
```

评估在训练块之间重置场景，重新开始训练回合；不会将评估动作加入 PPO 训练数据，并恢复训练成功率窗口、随机状态和课程计数。奖励函数、50 RPM 速度限制、扭矩上限与惩罚均沿用当前版本。没有修改上位机部署。

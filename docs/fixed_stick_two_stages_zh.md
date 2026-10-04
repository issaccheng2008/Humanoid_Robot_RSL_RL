当前任务已于2026-10-04切换为直接Stage 2六级原地跨棍课程；以下旧两阶段流程和10cm参数仅为历史说明。当前执行规则见 [Stage 2原地跨棍课程](direct_crossing_curriculum_zh.md)。

# p3 固定单棍两阶段训练（v3.2）

当前普通步长 **10 cm**，初始最前足底/脚尖到木棍近侧边缘的净距离 **8 cm**。木棍长80 cm、宽3 cm、高3 cm；回合内世界坐标固定。使用原微屈膝初始姿态和v3.2 USD。策略仍为49维观测、12维动作，不输入实时木棍距离。

当前默认使用内部相位时钟：一次启动后，0.14秒切LEAD（23cm、cross=1），0.44秒切FOLLOW（0cm、cross=1），0.76秒发出序列结束。执行中前进速度0.20m/s；结束后由上层站立控制接管。触地和障碍几何只用于训练奖励及真实成功评分。旧的首次落地切指令方式保留为显式 `--command-mode touchdown` 对照。完整49/12接口、源码参考及新时钟训练命令见 [上位机交接文档](phase_clock_upper_computer_handoff_zh.md)。10cm是双脚脚尖相对纵向步长输入，8cm是初始脚尖到棍近边净间隙，两者参考点不同。

## Stage 1：先学完整动作和净空

默认启动Stage 1。木棍可见，但 `collision_enabled=False`，不会产生绊脚反作用力。`FixedStickState.collisionless_mode=True`：检测到穿透时记录累计 `hit`、当控制步 `current_hit`，施加软惩罚，允许继续完成动作。跌倒、低基座、足底横向绕出木棍跨度仍会失败；时钟模式到序列结束尚未真实完成也记失败。旧触地模式保留8秒期限。

穿透检查计算足底真正落在木棍投影内的部分，不会把棍后接地的脚跟算作碰棍。每个5 ms物理子步检查几何并锁存，四个子步中的任一个穿透都会在20 ms控制步计入 `hit` 和软罚；离开木棍后停止软罚，回合累计 `hit` 保留到复位。Stage 1 的纯净判定针对足底穿透；Stage 2 加上机器人13个刚体与实体木棍的接触检测。

几何奖励沿用原版“进入时抬脚、跨越时抬脚并向前推进”的思路，适配固定木棍及当前指定跨棍脚，替换原静态净空奖励。以下是这次涉及的奖励；速度、姿态、滑移、关节限制、动作平滑等基础奖励继续沿用。

| 奖励 | 权重 | 简单含义 |
|---|---:|---|
| 普通步长跟踪 | 50 | 摆动脚落地时，实际脚尖步长接近当前指令 |
| 普通步完成 | 25 | 完成首次有效摆动并落地，一次发放25分 |
| 前腿完整跨越 | 30 | 指定前腿完整足底过棍并落地，一次发放30分 |
| 纯净完整成功 | 100 | 两脚完整过棍、连续双脚支撑且整回合无hit，一次发放100分 |
| 躯干前进进度 | 6 | 向前推进 |
| `stick_entry_clearance_reward` | 75 | 指定摆动脚首次进入木棍投影时，按棍上净空给分，每脚一次 |
| `stick_crossing_progress_reward` | 15 | 棍上净空×脚向前速度×前进进度；静止悬脚没有该奖励 |
| `collisionless_hit_penalty` | −100 | Stage 1中本控制步含任何子步穿透则扣分；Stage 2关闭这项软罚 |

几何高度分为 `clamp(足底棍上净空 / 0.03 m, 0, 1)`；前进速度分为 `clamp(脚x速度 / 0.5 m/s, -1, 1)`，后退可扣分；进度按脚尖越过棍近边的距离除以0.23 m并截断到0～1。几何奖励只给当前指定的跨棍脚，并要求离地及与木棍投影重叠。

Isaac的RewardManager会把奖励乘控制周期0.02 s：75和15是奖励率权重，入障高度满分事件实际给1.5分；−100软罚实际为每个含穿透的控制步−2分。25/30/100里程碑已在函数内除以dt，实际分数就是表中的值。穿透后仍可获得普通步/前腿完成奖励并记录完整动作，但不会获得最终100分。

## 准出与Stage 2

| 日志项 | 含义 |
|---|---|
| `Task/full_sequence_success_rate` | 全序列完成率，Stage 1可以包括穿透回合 |
| `Task/stick_contact_rate` | 已终止回合中曾发生hit的比例 |
| `Task/clean_crossing_success_rate` | 全序列完成且整个回合无hit的比例 |
| `Task/clean_crossing_success_rate_window` | 最近最多1000个真实终止回合的纯净成功率 |
| `Task/clean_crossing_window_episodes` | 当前窗口回合数，检查点中保存实际整数统计 |
| `Task/stage2_ready` | 窗口至少200回合且纯净成功率≥60%时为1 |

启动复位和手动中途复位不计入窗口；失败、超时和穿透完成回合按0计入。准出统计随每个训练检查点保存在 `infos.fixed_stick_training`。同阶段 `--resume` 会恢复窗口，跨阶段暖启动会开始新窗口。TensorBoard对同一次PPO更新内收到的日志取平均，窗口回合数显示可能是小数，准出判断使用真实整数窗口。

达到预期后，结束Stage 1，明确启动Stage 2并加载Stage 1检查点。不会在一次运行中突然修改碰撞。Stage 2设置 `collisionless_mode=False`、`collision_enabled=True`，启用13刚体接触传感器和四个5 ms子步历史；碰棍立即失败。保留几何奖励，载入actor/critic及策略噪声，重置优化器和阶段计数，固定学习率默认5e-5快速微调。

Stage 2入口要求新Stage 1或Stage 2检查点。尚未达到所记录准出条件的Stage 1检查点会输出提示，仍允许显式Stage 2测试；充分训练应按窗口指标达到预期后切换。60%是切换课程的参考，不能替代Stage 2的实体碰撞评估。

## 本地启动

在 `F:\RobotProject\cross_stick_p3` 中运行：

```powershell
# 默认从 model_33999.pt 暖启动Stage 1
.\hpc\train_walk_stop_cross.ps1 -Stage 1 -NumEnvs 1024 -Iterations 3000

# 同阶段、同调度模式/边界的新模型续训；旧触地模型改时钟应不加Resume暖启动
.\hpc\train_walk_stop_cross.ps1 -Stage 1 -Resume -Checkpoint 'Stage 1模型绝对路径' -Iterations 1000

# 达标后，加载Stage 1权重进行实体木棍微调
.\hpc\train_walk_stop_cross.ps1 -Stage 2 -Checkpoint 'Stage 1模型绝对路径' -Iterations 1000

# Stage 2续训
.\hpc\train_walk_stop_cross.ps1 -Stage 2 -Resume -Checkpoint 'Stage 2模型绝对路径' -Iterations 1000
```

需要修改准出阈值时使用Python入口，默认窗口1000回合：

```powershell
& 'F:\isaacsim\env_isaacsim\Scripts\python.exe' -B hpc/train_walk_stop_cross.py --stage 1 --clean-success-threshold 0.60 --clean-success-min-episodes 200 --num-envs 1024 --max-iterations 3000 --headless
```

录像自动根据新检查点选择阶段；旧模型默认Stage 1。需要观察实体木棍时明确加 `--stage 2`：

```powershell
& 'F:\isaacsim\env_isaacsim\Scripts\python.exe' -B hpc/play_walk_stop_cross.py --checkpoint '模型绝对路径' --headless --video-backend kit --video-stride 4

& 'F:\isaacsim\env_isaacsim\Scripts\python.exe' -B hpc/play_walk_stop_cross.py --stage 2 --checkpoint '模型绝对路径' --headless --video-backend kit --video-stride 4
```

视频使用3D机械模型Kit后端，输出 `obstacle-3d.mp4`、`trajectory.csv`和`summary.json`。summary包含实际初始间隙、阶段、碰撞模式、hit及clean_success。3D画面左上角显示当前阶段、cross标志和步长；CSV的step_command/cross_command为本帧物理步结束后的指令，applied_step_command/applied_cross_command记录刚执行动作时使用的上一帧指令，避免把落地帧旧指令误当作新指令。`--no-video`只记录数值。

## HPC

代码和v3.2资产上传至 `$HOME/cross_stick_p3`。HPC启动、容器绑定与版本检查继续使用现有参考格式：Python3.12.x、IsaacSim6.0.x、IsaacLab框架3.0.x、RSL-RL5.4.x。

```bash
sbatch hpc/train_walk_stop_cross.sh --stage 1

sbatch hpc/train_walk_stop_cross.sh --stage 1 --resume \
  --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage1_v32/RUN/model_N.pt

sbatch hpc/train_walk_stop_cross.sh --stage 2 \
  --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage1_v32/RUN/model_N.pt \
  --max-iterations 1000

bash hpc/play_fixed_stick.sh --stage 2 \
  --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage2_v32/RUN/model_N.pt
```

时钟模式日志分别写入 `logs/rsl_rl/fixed_stick_stage1_v32_phase_clock/` 和 `logs/rsl_rl/fixed_stick_stage2_v32_phase_clock/`；旧触地模式使用不带 `_phase_clock` 的目录。`fine_tune.json`与导出的env.yaml记录实际碰撞开关、距离、步长、阶段、准出条件和运行时版本。

## 最新落地触发修复与验证（8 cm）

此前WALK切换额外要求实际步长10±1 cm且脚尖前移至少9 cm，导致第一次落地没有准确达标时一直不给跨棍命令。已去除两个位置门槛，保留至少40 ms离地的有效摆动检测、单脚落地以及失败优先；10 cm目标继续用于步长奖励。第一脚落地后，另一只脚立即收到跨棍标志1和23 cm步长。相关完成率改名为`Task/first_step_completion_rate`，不再把首次落地误称为实际走满10 cm。

2026-10-02使用原`model_33999.pt`进行Stage 1真实Kit录像：实测初始净距离`0.079999998 m`、木棍中心`[0.187600106,0,0.015] m`。第8步（0.16 s）左脚首次有效落地，同帧观测跨棍标志0→1、步长10→23 cm；第9步策略实际使用cross=1。第25步前腿落地进入FOLLOW，第46步完成全序列。hit=True、clean_success=False，存在穿透，不能作为实体无碰撞成功。

最终`tests/`共117项通过，包含左右脚首次实际步长3/7/15 cm仍同帧发出命令、初始化接地与短时接触闪烁不触发、重复调用不改变指令、8 cm净距离等回归。

证据：`first_touchdown_full_tests.log`、`first_touchdown_gap08_play.log`、`recordings/first_touchdown_gap08_model33999/trajectory.csv`、`summary.json`、`obstacle-3d.mp4`和`first_cross_command.png`。参数和触发逻辑更改后，建议暖启动新训练（省略Resume），避免旧任务的准出统计混入新窗口。

## 两阶段原验证记录（历史13 cm配置）

2026-10-02最终验证：`tests/`全部115项通过，包含两阶段状态、软罚、子步hit锁存、纯净100分、启动/手动reset统计、阶段检查点和窗口恢复测试；代码复查未发现待修复阻塞问题。

真实Isaac（Python3.12.10/IsaacSim6.0.1.0/IsaacLab3.0.0/RSL-RL5.4.1）完成：

- Stage 1四环境6次PPO更新：允许穿透后完整动作结束，软罚非零，完整动作与纯净指标分别记录；窗口4个真实终止回合，纯净成功率0。
- 同Stage 1检查点续训2次：从iteration 6开始，保存到iteration 7，phase_iteration为8；4个窗口回合正确恢复。
- Stage 2从Stage 1检查点暖启动四环境2次：优化器学习率5e-5，阶段从iteration 0开始，实体碰撞与碰棍失败生效，软罚权重0。
- 两阶段各一次单环境数值回放：自动识别阶段，实测净间隙`0.129999995 m`，木棍中心`[0.237600103, 0, 0.015] m`，观测49维/动作12维。Stage 1回放171步后任务失败，Stage 2回放18步碰棍终止，均未获得纯净成功。

证据：`two_stage_final_tests.log`、`two_stage_stage1_verified.log`、`two_stage_stage1_resume.log`、`two_stage_stage2_verified.log`，以及`recordings/two_stage_stage1_verified/summary.json`与`recordings/two_stage_stage2_verified/summary.json`。

短程验证证明任务、奖励、阶段切换和检查点流程可运行；**当前未训练出纯净成功模型**。HPC容器没有实际运行；充分训练、实体木棍评估和真机验证仍需完成。


### Stage 1 扭矩惩罚（2026-10-03）

按每个关节实际施加扭矩的绝对值 t（N·m）计算，再对12个关节求和。
连续分段线性惩罚为 `p(t) = max(t - 1.5, 0) + 9 * max(t - 3, 0)`。

| 扭矩范围 | 惩罚增加斜率 | 奖励权重 |
|---|---:|---:|
| ≤ 1.5 N·m | 0 | -0.1 |
| 1.5～3 N·m | 1 | -0.1 |
| > 3 N·m | 10 | -0.1 |

奖励项为 `-0.1 * sum(p(abs(torque)))`，环境还会乘控制时间步长。
例如单关节4 N·m时，惩罚原值11.5，乘权重后为-1.15；20 ms一步为-0.023。
这两个门槛是奖励门槛，物理扭矩硬上限仍为5 N·m。
Stage 2同步相同的1.5/3 N·m分段线性惩罚。
已有检查点加载后，新奖励需要继续训练才会影响策略；直接回放不会改变已学动作。

当前固定木棍任务所有12个关节的输出转速上限为50 rpm（5.235987755982989 rad/s），由FixedStickEnvCfg覆盖执行器配置，Stage 1/2及回放均生效。扭矩硬上限仍为5 N·m，扭矩惩罚不变。旧检查点需要在此速度限制下继续训练，动作相位时间尚未调整。


## 当前教师训练默认方式（2026-10-03）

训练入口默认使用touchdown：首次有效摆动并落地后切换跨前腿指令，前腿完整过棍并有效落地后切换后腿跟进。无固定相位时间限制，沿用8秒回合期限。保留50 rpm关节转速上限、5 N·m扭矩硬上限及Stage 1分段扭矩惩罚。上文相位时钟说明仅作为显式phase_clock对照方式，部署文件未改。若从phase_clock检查点切换教师训练，加载权重暖启动且不要加--resume。后续蒸馏尚未实现，当前教师仍接收阶段指令。

Stage 1新增fixed_joint_overspeed：每关节实际速度绝对值超过25 rpm（2.617993878 rad/s）才计算平方惩罚，对12关节求和，权重-0.1。50 rpm硬上限保留。公式为-0.1 × sum(max(abs(qd)-2.617993878,0)^2)，环境再乘20 ms；单关节50 rpm时每步约-0.01371。Stage 2同步启用该超速惩罚，原动作平滑与成功奖励保持。

HPC训练SH当前默认Stage 2、touchdown，从2026-10-02_17-07-45-315815_finetune/model_850.pt暖启动，不带--resume。给定主机检查点位于/share/home/2242211591/cross_stick_p3/logs/，脚本使用其项目挂载后的/workspace/cross_stick_p3/logs/容器路径。两阶段电机限制及扭矩/超速惩罚相同，Stage 2仍启用实体木棍、碰棍终止，关闭Stage 1穿透软罚。

## 当前惩罚权重更新（2026-10-03）

Stage 1/2的dof_torque_over_nominal及fixed_joint_overspeed权重统一为-1.0（原-0.1增强10倍，以上旧权重示例仅为历史记录）。门槛保持1.5/3 N·m（第二段斜率10倍）与25 rpm。单关节持续1秒：50 rpm超速惩罚约-6.85分，4 N·m扭矩惩罚-11.5分，5 N·m扭矩惩罚-21.5分。两项独立叠加并跨关节求和；里程碑25/30/100分不变，50 rpm和5 N·m硬上限不变。

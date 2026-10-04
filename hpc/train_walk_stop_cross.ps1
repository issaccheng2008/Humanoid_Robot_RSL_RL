param(
    [string]$Python = 'F:\isaacsim\env_isaacsim\Scripts\python.exe',
    [string]$Checkpoint = '',
    [ValidateSet(2)][int]$Stage = 2,
    [int]$NumEnvs = 1024,
    [int]$Iterations = 3000,
    [ValidateSet('phase_clock','touchdown')][string]$CommandMode = '',
    [switch]$Resume
)
$projectRoot = Split-Path -Parent $PSScriptRoot
if (!$Checkpoint) { $Checkpoint = Join-Path $projectRoot 'logs/rsl_rl/fixed_stick_stage2_v32/2026-10-04_18-10-03-535593_finetune/model_350.pt' }
$trainArgs = @((Join-Path $PSScriptRoot 'train_walk_stop_cross.py'), '--checkpoint', $Checkpoint,
              '--stage', $Stage, '--curriculum-level', 5, '--clean-success-threshold', 0.8,
              '--success-stop-mode', 'deterministic', '--num-envs', $NumEnvs, '--max-iterations', $Iterations, '--headless')
if ($CommandMode) { $trainArgs += @('--command-mode', $CommandMode) }
if ($Resume) { $trainArgs += '--resume' }
& $Python @trainArgs
exit $LASTEXITCODE

param(
    [string]$Python = 'F:\isaacsim\env_isaacsim\Scripts\python.exe',
    [string]$Checkpoint = '',
    [int]$NumEnvs = 512,
    [int]$Iterations = 3000,
    [switch]$Resume
)
$projectRoot = Split-Path -Parent $PSScriptRoot
if (!$Checkpoint) { $Checkpoint = Join-Path $projectRoot 'model_33999.pt' }
$trainArgs = @((Join-Path $PSScriptRoot 'train_walk_stop_cross.py'), '--checkpoint', $Checkpoint,
              '--num-envs', $NumEnvs, '--max-iterations', $Iterations, '--headless')
if ($Resume) { $trainArgs += '--resume' }
& $Python @trainArgs
exit $LASTEXITCODE

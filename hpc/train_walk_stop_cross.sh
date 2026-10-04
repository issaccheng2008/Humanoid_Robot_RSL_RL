#!/bin/bash
#SBATCH -p gpu4090
#SBATCH -A 2242211591
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J fixed_stick_p3
#SBATCH -o fixed_stick_p3_%j.log
#SBATCH -e fixed_stick_p3_%j.log
set -euo pipefail
PROJECT="${PROJECT:-$HOME/cross_stick_p3}"
CONTAINER="${CONTAINER:-$HOME/biped-sandbox.sif}"
KIT_CACHE="$HOME/.cache/cross-stick-p3/kit-cache"
mkdir -p "$KIT_CACHE"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
test -f "$PROJECT/v3.2/v3.2.usd"
test -f "$ISAACLAB_ROOT/VERSION"
ISAACLAB_PATHS=$(find "$ISAACLAB_ROOT/source" -maxdepth 1 -mindepth 1 -type d -printf '/opt/isaaclab/source/%f:' )
export SINGULARITYENV_PYTHONPATH="/workspace/cross_stick_p3:$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"
singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind "$KIT_CACHE:/isaac-sim/kit/cache" \
    --bind "$PROJECT:/workspace/cross_stick_p3" \
    --bind "$ISAACLAB_ROOT:/opt/isaaclab" \
    --bind "$ISAACLAB_ROOT/source:/opt/isaaclab_source" \
    --env DISPLAY= --env QT_QPA_PLATFORM=offscreen \
    "$CONTAINER" /isaac-sim/python.sh /workspace/cross_stick_p3/hpc/train_walk_stop_cross.py \
    --stage 2 --command-mode touchdown --curriculum-level 5 \
    --clean-success-threshold 0.8 --clean-success-min-episodes 200 \
    --success-stop-mode deterministic --evaluation-interval 50 \
    --checkpoint /workspace/cross_stick_p3/logs/rsl_rl/fixed_stick_stage2_v32/2026-10-04_18-10-03-535593_finetune/model_350.pt \
    --num-envs "${NUM_ENVS:-1024}" --max-iterations "${ITERATIONS:-3000}" --headless "$@"

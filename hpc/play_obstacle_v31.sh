#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -A 2242211591
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J humanoid_v31_obstacle
#SBATCH -o humanoid_v31_obstacle_%j.log
#SBATCH -e humanoid_v31_obstacle_%j.log

set -euo pipefail

PROJECT="$HOME/Humanoid_Robot_RSL_RL"
VIDEO_BACKEND="${VIDEO_BACKEND:-software}"
KIT_CACHE="$HOME/.cache/isaacsim-v31-obstacle/kit-cache"
mkdir -p "$KIT_CACHE"
echo "[RECORD] Kit cache: $KIT_CACHE"
echo "[RECORD] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[RECORD] video backend: $VIDEO_BACKEND"
ISAACLAB_PATHS=$(find "$HOME/IsaacLab/source" -maxdepth 1 -mindepth 1 -type d | tr '\n' ':')
export SINGULARITYENV_PYTHONPATH="/workspace/Humanoid_Robot_RSL_RL:$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind "$KIT_CACHE:/isaac-sim/kit/cache" \
    --bind "$PROJECT:/workspace/Humanoid_Robot_RSL_RL" \
    --bind "$HOME/IsaacLab/source:/opt/isaaclab_source" \
    --env DISPLAY= \
    --env QT_QPA_PLATFORM=offscreen \
    "$HOME/biped-sandbox.sif" \
    /isaac-sim/python.sh /workspace/Humanoid_Robot_RSL_RL/hpc/play_obstacle_v31.py \
    --checkpoint /workspace/Humanoid_Robot_RSL_RL/model_33999.pt \
    --usd /workspace/Humanoid_Robot_RSL_RL/v3.1/v3.1.usd \
    --output-dir "/workspace/Humanoid_Robot_RSL_RL/recordings/v31_obstacle_${SLURM_JOB_ID}" \
    --steps 250 \
    --video-backend "$VIDEO_BACKEND" \
    --headless

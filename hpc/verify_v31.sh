#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -A 2242211591
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -J humanoid_v31_check
#SBATCH -o humanoid_v31_check_%j.log
#SBATCH -e humanoid_v31_check_%j.log

set -euo pipefail

PROJECT="$HOME/Humanoid_Robot_RSL_RL"
ISAACLAB_PATHS=$(find "$HOME/IsaacLab/source" -maxdepth 1 -mindepth 1 -type d | tr '\n' ':')
export SINGULARITYENV_PYTHONPATH="/workspace/Humanoid_Robot_RSL_RL:$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"

singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind "$PROJECT:/workspace/Humanoid_Robot_RSL_RL" \
    --bind "$HOME/IsaacLab/source:/opt/isaaclab_source" \
    --env DISPLAY= \
    --env QT_QPA_PLATFORM=offscreen \
    "$HOME/biped-sandbox.sif" \
    /isaac-sim/python.sh /workspace/Humanoid_Robot_RSL_RL/hpc/verify_checkpoint.py \
    --task Humanoid-Robot-RSLRL-Legacy-Play-v0 \
    --checkpoint /workspace/Humanoid_Robot_RSL_RL/model_33999.pt \
    --usd /workspace/Humanoid_Robot_RSL_RL/v3.1/v3.1.usd \
    --steps 250 \
    --viz none

#!/bin/bash
#SBATCH -p gpu4090,gpu,gput4
#SBATCH -A 2242211591
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -J cross_stick_v32
#SBATCH -o cross_stick_v32_%j.log
#SBATCH -e cross_stick_v32_%j.log
set -euo pipefail
PROJECT="${PROJECT:-$HOME/cross_stick}"
CONTAINER="${CONTAINER:-$HOME/biped-sandbox.sif}"
KIT_CACHE="$HOME/.cache/cross-stick-v32/kit-cache"
mkdir -p "$KIT_CACHE"
ISAACLAB_PATHS=$(find "$HOME/IsaacLab/source" -maxdepth 1 -mindepth 1 -type d | tr '\n' ':')
export SINGULARITYENV_PYTHONPATH="/workspace/cross_stick:$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"
singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind "$KIT_CACHE:/isaac-sim/kit/cache" \
    --bind "$PROJECT:/workspace/cross_stick" \
    --bind "$HOME/IsaacLab/source:/opt/isaaclab_source" \
    --env DISPLAY= --env QT_QPA_PLATFORM=offscreen \
    "$CONTAINER" /isaac-sim/python.sh /workspace/cross_stick/hpc/train_walk_stop_cross.py \
    --num-envs "${NUM_ENVS:-512}" --max-iterations "${ITERATIONS:-3000}" --headless "$@"

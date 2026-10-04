#!/bin/bash
# Run inside an allocated HPC GPU job: bash hpc/play_fixed_stick.sh --checkpoint ...
set -euo pipefail
PROJECT="${PROJECT:-$HOME/cross_stick_p3}"
CONTAINER="${CONTAINER:-$HOME/biped-sandbox.sif}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"
KIT_CACHE="$HOME/.cache/cross-stick-p3/kit-cache"
mkdir -p "$KIT_CACHE"
ISAACLAB_PATHS=$(find "$ISAACLAB_ROOT/source" -maxdepth 1 -mindepth 1 -type d -printf '/opt/isaaclab/source/%f:')
export SINGULARITYENV_PYTHONPATH="/workspace/cross_stick_p3:$HOME/.local/lib/python3.12/site-packages:${ISAACLAB_PATHS}"
singularity exec --nv \
    --bind /dev/shm:/dev/shm \
    --bind "$KIT_CACHE:/isaac-sim/kit/cache" \
    --bind "$PROJECT:/workspace/cross_stick_p3" \
    --bind "$ISAACLAB_ROOT:/opt/isaaclab" \
    --env DISPLAY= --env QT_QPA_PLATFORM=offscreen \
    "$CONTAINER" /isaac-sim/python.sh /workspace/cross_stick_p3/hpc/play_walk_stop_cross.py \
    --stage 2 --command-mode touchdown --headless "$@"

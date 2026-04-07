#!/bin/bash
# =============================================================================
# Meluxina Slurm: 3-D prediction visualisation (1× A100, single node)
# =============================================================================
# Submit:
#   sbatch mmdetection3d/tools/sbatch/meluxina_vis3d.sbatch
#
# Override defaults:
#   CONFIG=... CKPT=... OUT_DIR=... NUM_SAMPLES=16 SCORE_THR=0.3 sbatch ...
#
#SBATCH --job-name=vis3d
#SBATCH --account=p201222
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out
#SBATCH --error=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
MOE_ROOT=${MOE_ROOT:-$HOME/projects/multimodal-MoE}
PYTHON=${PYTHON:-$HOME/miniconda3/envs/multimodal-moe/bin/python}

CONFIG=${CONFIG:-$MOE_ROOT/mmdetection3d/configs/zod/zod_lidar_only.py}
CKPT=${CKPT:-$MOE_ROOT/outputs/runs/zod_lidar_only/zod-lidar-only_4452913/best_mAP_0.50_epoch_18.pth}
OUT_DIR=${OUT_DIR:-$(dirname "$CKPT")/vis3d}
NUM_SAMPLES=${NUM_SAMPLES:-8}
SCORE_THR=${SCORE_THR:-0.25}
SPLIT=${SPLIT:-val}

echo "Using PYTHON=$PYTHON"
echo "python $($PYTHON -c 'import sys; print(sys.executable)')"
echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="

module load CUDA 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd "$MOE_ROOT"

$PYTHON scripts/bev_scripts/visualize_3d_predictions.py \
    --config     "$CONFIG" \
    --ckpt       "$CKPT" \
    --out-dir    "$OUT_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --score-thr  "$SCORE_THR" \
    --split      "$SPLIT" \
    --device     cuda:0

echo "Done. Output in $OUT_DIR"

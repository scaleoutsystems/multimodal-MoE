#!/bin/bash
# =============================================================================
# Meluxina Slurm: Multi-split eval — MOE Fusion-then (best: ep69, mAP_0.50=0.5756)
# =============================================================================
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/eval_moe_fusion_then.sbatch
#
# =============================================================================

#SBATCH --job-name=eval_moe_fusion_then
#SBATCH --account=p201392
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out
#SBATCH --error=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out

set -euo pipefail

MOE_ROOT=${MOE_ROOT:-$HOME/projects/multimodal-MoE}
PYTHON=${PY:-$HOME/miniconda3/envs/multimodal-moe/bin/python}

CONFIG=mmdetection3d/configs/zod/zod_moe_fusion_then.py
CKPT=outputs/runs/zod_moe_fusion_then/moe-fusion-then_4758769/best_mAP_0.50_epoch_69.pth
SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits
ANN_KEY=test_dataloader.dataset.ann_file

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CONFIG      : $CONFIG"
echo "CKPT        : $CKPT"
echo "SPLITS_ROOT : $SPLITS_ROOT"

module load CUDA 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd "$MOE_ROOT"
export PYTHONPATH="$MOE_ROOT/mmdetection3d:${PYTHONPATH:-}"

"$PYTHON" scripts/bev_scripts/run_multi_split_eval.py \
    "$CONFIG" \
    "$CKPT" \
    --splits-root "$SPLITS_ROOT" \
    --ann-key "$ANN_KEY" \
    --launcher none

echo ""
echo "=== Done. Results in outputs/runs/moe-fusion-then_4758769/eval/ ==="

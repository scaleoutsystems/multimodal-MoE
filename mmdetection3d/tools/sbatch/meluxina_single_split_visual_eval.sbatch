#!/bin/bash
# =============================================================================
# Meluxina Slurm: Single-split evaluation with visualizations (1× A100)
# =============================================================================
#
# Runs scripts/bev_scripts/run_single_split_visual_eval.py for one split with
# MMDetection3D visualization enabled (--show-dir).
#
# The exact visual output depends on the VisualizationHook in the config and
# the --task flag.  This script saves visuals to disk; it does NOT use --show
# (interactive display) which would block in a batch context.
#
# SUBMIT:
#   CONFIG=mmdetection3d/configs/zod/zod_lidar_only.py \
#   CKPT=outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \
#   SPLIT=lighting_test_night.txt \
#   SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits \
#   TASK=lidar_det \
#     sbatch mmdetection3d/tools/sbatch/meluxina_single_split_visual_eval.sbatch
#
# ENVIRONMENT VARIABLES
#   CONFIG        (required) path to MMDet3D config file
#   CKPT          (required) path to checkpoint .pth
#   SPLIT         (required) split filename, e.g. lighting_test_night.txt
#   SPLITS_ROOT   (required) directory containing split .txt files
#   TASK          (required) visualization task type:
#                   mono_det | multi-view_det | lidar_det | lidar_seg | multi-modality_det
#   SCORE_THR     (optional) score threshold for visualization; default: 0.1
#   INFOS_DIR     (optional) directory containing annotation pkls; default: {SPLITS_ROOT}/../infos
#   ANN_KEY       (optional) dotted config key to override; default: test_dataloader.dataset.ann_file
#   PY            (optional) path to python executable
#   EXTRA_ARGS    (optional) additional flags passed verbatim to run_single_split_visual_eval.py
# =============================================================================

#SBATCH --job-name=eval_vis
#SBATCH --account=p201392
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out
#SBATCH --error=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/%x_%j.out

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
MOE_ROOT=${MOE_ROOT:-$HOME/projects/multimodal-MoE}
PYTHON=${PY:-$HOME/miniconda3/envs/multimodal-moe/bin/python}

# ── Required env var guards ────────────────────────────────────────────────────
: "${CONFIG:?CONFIG env var is required (path to MMDet3D config file)}"
: "${CKPT:?CKPT env var is required (path to checkpoint .pth)}"
: "${SPLIT:?SPLIT env var is required (e.g. lighting_test_night.txt)}"
: "${SPLITS_ROOT:?SPLITS_ROOT env var is required (directory of split .txt files)}"
: "${TASK:?TASK env var is required (e.g. lidar_det, multi-modality_det)}"

# ── Optional env vars ─────────────────────────────────────────────────────────
SCORE_THR=${SCORE_THR:-0.1}
ANN_KEY=${ANN_KEY:-test_dataloader.dataset.ann_file}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CONFIG      : $CONFIG"
echo "CKPT        : $CKPT"
echo "SPLIT       : $SPLIT"
echo "SPLITS_ROOT : $SPLITS_ROOT"
echo "TASK        : $TASK"
echo "SCORE_THR   : $SCORE_THR"
echo "ANN_KEY     : $ANN_KEY"
echo "INFOS_DIR   : ${INFOS_DIR:-<inferred from SPLITS_ROOT/../infos>}"
echo "EXTRA_ARGS  : ${EXTRA_ARGS:-<none>}"

module load CUDA 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd "$MOE_ROOT"

# Add mmdetection3d to PYTHONPATH so mmdet3d imports work inside test.py
export PYTHONPATH="$MOE_ROOT/mmdetection3d:${PYTHONPATH:-}"

# ── Build command ──────────────────────────────────────────────────────────────
CMD=(
    "$PYTHON"
    scripts/bev_scripts/run_single_split_visual_eval.py
    "$CONFIG"
    "$CKPT"
    "$SPLIT"
    --splits-root "$SPLITS_ROOT"
    --task "$TASK"
    --score-thr "$SCORE_THR"
    --ann-key "$ANN_KEY"
    --launcher none
)

# Append --infos-dir only if explicitly set
if [[ -n "${INFOS_DIR:-}" ]]; then
    CMD+=(--infos-dir "$INFOS_DIR")
fi

# Append any passthrough args
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    CMD+=(--extra-test-args ${EXTRA_ARGS})
fi

echo ""
echo "Command: $(printf '%q ' "${CMD[@]}")"
echo ""

"${CMD[@]}"

echo ""
echo "=== Done. ==="

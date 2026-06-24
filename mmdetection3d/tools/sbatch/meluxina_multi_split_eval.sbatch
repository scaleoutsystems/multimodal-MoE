#!/bin/bash
# =============================================================================
# Meluxina Slurm: Multi-split quantitative evaluation (1× A100, single node)
# =============================================================================
#
# Runs scripts/bev_scripts/run_multi_split_eval.py for all specified test splits.
# No visualization is produced; this is for quantitative metric collection only.
#
# SUBMIT (all default splits):
#   CONFIG=mmdetection3d/configs/zod/zod_lidar_only.py \
#   CKPT=outputs/runs/zod_lidar_only/zod-lidar-only_4454825/best_mAP_0.50_epoch_18.pth \
#   SPLITS_ROOT=/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes/splits \
#     sbatch mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch
#
# SUBMIT (specific splits only, space-separated inside quotes):
#   CONFIG=... CKPT=... SPLITS_ROOT=... \
#   SPLITS="lighting_test_day.txt lighting_test_night.txt lighting_test_twilight.txt" \
#     sbatch mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch
#
# ENVIRONMENT VARIABLES
#   CONFIG        (required) path to MMDet3D config file
#   CKPT          (required) path to checkpoint .pth
#   SPLITS_ROOT   (required) directory containing split .txt files
#   SPLITS        (optional) space-separated split filenames; defaults to all DEFAULT_SPLITS
#   INFOS_DIR     (optional) directory containing annotation pkls; defaults to {SPLITS_ROOT}/../infos
#   ANN_KEY       (optional) dotted config key to override; default: test_dataloader.dataset.ann_file
#   PY            (optional) path to python executable
#   OUT_DIR       (optional) override the eval output directory (default: outputs/runs/<run_name>/eval/)
#   EXTRA_ARGS    (optional) additional flags passed verbatim to run_multi_split_eval.py
# =============================================================================

#SBATCH --job-name=eval_multi
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

# ── Paths ─────────────────────────────────────────────────────────────────────
MOE_ROOT=${MOE_ROOT:-$HOME/projects/multimodal-MoE}
PYTHON=${PY:-$HOME/miniconda3/envs/multimodal-moe/bin/python}

# ── Required env var guards ────────────────────────────────────────────────────
: "${CONFIG:?CONFIG env var is required (path to MMDet3D config file)}"
: "${CKPT:?CKPT env var is required (path to checkpoint .pth)}"
: "${SPLITS_ROOT:?SPLITS_ROOT env var is required (directory of split .txt files)}"

# ── Optional env vars ─────────────────────────────────────────────────────────
ANN_KEY=${ANN_KEY:-test_dataloader.dataset.ann_file}
OUT_DIR=${OUT_DIR:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "CONFIG      : $CONFIG"
echo "CKPT        : $CKPT"
echo "SPLITS_ROOT : $SPLITS_ROOT"
echo "SPLITS      : ${SPLITS:-<default – all DEFAULT_SPLITS>}"
echo "ANN_KEY     : $ANN_KEY"
echo "OUT_DIR     : ${OUT_DIR:-<inferred from checkpoint path>}"
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
    scripts/bev_scripts/run_multi_split_eval.py
    "$CONFIG"
    "$CKPT"
    --splits-root "$SPLITS_ROOT"
    --ann-key "$ANN_KEY"
    --launcher none
)

# Append --out-dir only if explicitly set
if [[ -n "${OUT_DIR:-}" ]]; then
    CMD+=(--out-dir "$OUT_DIR")
fi

# Append --infos-dir only if explicitly set
if [[ -n "${INFOS_DIR:-}" ]]; then
    CMD+=(--infos-dir "$INFOS_DIR")
fi

# Append --splits only if SPLITS is non-empty; otherwise DEFAULT_SPLITS is used
if [[ -n "${SPLITS:-}" ]]; then
    # Word-split intentional: SPLITS is a space-separated list of filenames
    # shellcheck disable=SC2206
    CMD+=(--splits ${SPLITS})
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
echo "=== Done. Results in outputs/runs/$(basename $(dirname "$CKPT"))/eval/ ==="

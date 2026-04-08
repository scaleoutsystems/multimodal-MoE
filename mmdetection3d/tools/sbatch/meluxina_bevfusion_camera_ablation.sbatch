#!/bin/bash
# =============================================================================
# Meluxina Slurm: BEVFusion camera-contribution ablation eval (1× A100)
# =============================================================================
#
# Evaluates the best BEVFusion finetune checkpoint on the main test split
# under two conditions:
#   1. Full model   — normal camera + LiDAR forward pass
#   2. Camera-zero  — camera BEV replaced with zeros before fusion
#
# Outputs comparison markdown to:
#   outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/camera_ablation.md
#
# SUBMIT (defaults — runs the finetune best checkpoint):
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_bevfusion_camera_ablation.sbatch
#
# OVERRIDE checkpoint / configs:
#   CKPT=<path> FULL_CONFIG=<path> ZERO_CONFIG=<path> sbatch ...
#
# SKIP one condition (e.g. if eval already ran):
#   EXTRA_ARGS="--skip-full" sbatch ...   # only run camera-zero + markdown
#   EXTRA_ARGS="--skip-zero" sbatch ...   # only run full-model + markdown
# =============================================================================

#SBATCH --job-name=bev_cam_ablation
#SBATCH --account=p201222
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.err

set -euo pipefail

# ── Repo roots ────────────────────────────────────────────────────────────────
MMDET3D_DEFAULT="${HOME}/projects/multimodal-MoE/mmdetection3d"
MOE_DEFAULT="${HOME}/projects/multimodal-MoE"
if [[ -z "${MMDET3D_ROOT:-}" ]] || [[ ! -d "${MMDET3D_ROOT}" ]]; then
  MMDET3D_ROOT="${MMDET3D_DEFAULT}"
fi
if [[ -z "${MOE_ROOT:-}" ]] || [[ ! -d "${MOE_ROOT}" ]]; then
  MOE_ROOT="${MOE_DEFAULT}"
fi
if [[ ! -d "${MMDET3D_ROOT}" ]]; then
  echo "ERROR: mmdetection3d not found at ${MMDET3D_ROOT}" >&2; exit 1
fi
mkdir -p "${MOE_ROOT}/outputs/runs/slurm_logs"
cd "${MOE_ROOT}"

# ── Python env ────────────────────────────────────────────────────────────────
CONDA_ENV_NAME="${CONDA_ENV_NAME:-multimodal-moe}"
CONDA_ROOT="${CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]]; then
  for d in "${HOME}/miniforge3" "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/mambaforge"; do
    if [[ -x "${d}/bin/conda" ]] || [[ -d "${d}/envs/${CONDA_ENV_NAME}/bin" ]]; then
      CONDA_ROOT="${d}"; break
    fi
  done
fi
if [[ -z "${CONDA_ROOT}" ]] || [[ ! -d "${CONDA_ROOT}" ]]; then
  echo "ERROR: CONDA_ROOT not found" >&2; exit 1
fi
ENV_PREFIX="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
PYTHON="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: python not found at ${PYTHON}" >&2; exit 1
fi
unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT CONDA_SHLVL
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${MMDET3D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Using PYTHON=${PYTHON}"

# ── Defaults (override with env vars) ─────────────────────────────────────────
FULL_CONFIG="${FULL_CONFIG:-mmdetection3d/configs/zod/zod_bevfusion_finetune.py}"
ZERO_CONFIG="${ZERO_CONFIG:-mmdetection3d/configs/zod/zod_bevfusion_finetune_camzero.py}"
CKPT="${CKPT:-outputs/runs/zod_bevfusion_finetune/bevfusion-finetune_4456392/best_mAP_0.50_epoch_12.pth}"
SPLITS_ROOT="${SPLITS_ROOT:-/mnt/tier2/project/p201222/u103958/zod_moe/zod_nuscenes/splits}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "FULL_CONFIG  : ${FULL_CONFIG}"
echo "ZERO_CONFIG  : ${ZERO_CONFIG}"
echo "CKPT         : ${CKPT}"
echo "SPLITS_ROOT  : ${SPLITS_ROOT}"
echo "EXTRA_ARGS   : ${EXTRA_ARGS:-<none>}"
nvidia-smi -L || true

# shellcheck disable=SC2086
"${PYTHON}" scripts/bev_scripts/run_camera_ablation_eval.py \
    "${FULL_CONFIG}" \
    "${ZERO_CONFIG}" \
    "${CKPT}" \
    --splits-root "${SPLITS_ROOT}" \
    --launcher none \
    --python "${PYTHON}" \
    ${EXTRA_ARGS}

echo ""
echo "=== Done. Comparison markdown:"
echo "    ${CKPT%/*}/camera_ablation.md"

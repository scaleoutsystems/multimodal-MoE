#!/bin/bash
# =============================================================================
# Meluxina Slurm: Camera-only BEVFusion / ZOD — Experiment B
#   gentler fine-tune LR + dbound 60 m + splat_radius=0 (1×1 write only)
# =============================================================================
# Changes vs meluxina_train_zod_camera_only_ftlr_d60.sbatch (Experiment A):
#   - CONFIG  → configs/zod/zod_camera_only_ftlr_d60_splat0.py
#   - WORK_DIR → outputs/runs/zod_camera_only_ftlr_d60_splat0/
#   - job-name → zod-cam-ftlr-d60-sp0
#   - log filenames follow job-name (via %x-%j Slurm substitution)
#
# Architecture, resource requests, env setup, and launcher are UNCHANGED.
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_train_zod_camera_only_ftlr_d60_splat0.sbatch
#
# Watch progress:
#   tail -f ~/projects/multimodal-MoE/outputs/runs/slurm_logs/zod-cam-ftlr-d60-sp0-<JOBID>.out
#
# Override config / work-dir:
#   CONFIG=configs/zod/zod_camera_only_ftlr_d60_splat0.py WORK_DIR=/path/to/run sbatch ...
# =============================================================================

#SBATCH --job-name=zod-cam-ftlr-d60-sp0
#SBATCH --account=p201222
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=40ga100
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.err

set -euo pipefail

MMDET3D_DEFAULT="${HOME}/projects/multimodal-MoE/mmdetection3d"
MOE_DEFAULT="${HOME}/projects/multimodal-MoE"
if [[ -z "${MMDET3D_ROOT:-}" ]] || [[ ! -d "${MMDET3D_ROOT}" ]]; then
  if [[ -n "${MMDET3D_ROOT:-}" ]]; then
    echo "WARN: MMDET3D_ROOT='${MMDET3D_ROOT}' is missing or not a directory; using ${MMDET3D_DEFAULT}" >&2
  fi
  MMDET3D_ROOT="${MMDET3D_DEFAULT}"
fi
if [[ -z "${MOE_ROOT:-}" ]] || [[ ! -d "${MOE_ROOT}" ]]; then
  if [[ -n "${MOE_ROOT:-}" ]]; then
    echo "WARN: MOE_ROOT='${MOE_ROOT}' is missing or not a directory; using ${MOE_DEFAULT}" >&2
  fi
  MOE_ROOT="${MOE_DEFAULT}"
fi
if [[ ! -d "${MMDET3D_ROOT}" ]]; then
  echo "ERROR: mmdetection3d not found at ${MMDET3D_ROOT} (set MMDET3D_ROOT to your clone)." >&2
  exit 1
fi
if [[ ! -d "${MOE_ROOT}" ]]; then
  echo "ERROR: multimodal-MoE not found at ${MOE_ROOT} (set MOE_ROOT to your clone)." >&2
  exit 1
fi
mkdir -p "${MOE_ROOT}/outputs/runs/slurm_logs"
cd "${MMDET3D_ROOT}"

export PYTHONPATH="${MMDET3D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ── Python env ────────────────────────────────────────────────────────────────
CONDA_ENV_NAME="${CONDA_ENV_NAME:-multimodal-moe}"
CONDA_ROOT="${CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]]; then
  for d in "${HOME}/miniforge3" "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/mambaforge"; do
    if [[ -x "${d}/bin/conda" ]] || [[ -d "${d}/envs/${CONDA_ENV_NAME}/bin" ]]; then
      CONDA_ROOT="${d}"
      break
    fi
  done
fi
if [[ -z "${CONDA_ROOT}" ]] || [[ ! -d "${CONDA_ROOT}" ]]; then
  echo "ERROR: CONDA_ROOT not set or missing: ${CONDA_ROOT:-<empty>}" >&2
  exit 1
fi
ENV_PREFIX="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
PYTHON="${ENV_PREFIX}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: training env python missing or not executable: ${PYTHON}" >&2
  exit 1
fi
unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT CONDA_SHLVL
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
echo "Using PYTHON=${PYTHON}"

# ── Sanity check ──────────────────────────────────────────────────────────────
"${PYTHON}" -c "
import sys
print('python', sys.executable)
import torch
print('torch', torch.__version__)
import projects.BEVFusion.bevfusion  # noqa: F401
from mmdet3d.registry import MODELS
assert 'CameraOnlyBEVFusion' in MODELS.module_dict, (
    'CameraOnlyBEVFusion not registered — check that '
    'camera_only_bevfusion.py is imported in projects/BEVFusion/bevfusion/__init__.py '
    'and that bev_pool_ext CUDA extension is built.'
)
print('CameraOnlyBEVFusion registered OK')
for cls_name in ('DepthLSSTransform', 'GeneralizedLSSFPN'):
    assert cls_name in MODELS.module_dict, f'{cls_name} not registered'
print('Camera-side modules registered OK')
" || {
  echo "ERROR: env sanity check failed." >&2
  exit 1
}

# ── Config / work-dir ─────────────────────────────────────────────────────────
CONFIG="${CONFIG:-configs/zod/zod_camera_only_ftlr_d60_splat0.py}"
DEFAULT_WORK_PARENT="${MOE_ROOT}/outputs/runs/zod_camera_only_ftlr_d60_splat0"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_PARENT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "MMDET3D_ROOT=${MMDET3D_ROOT}"
echo "CONFIG=${CONFIG}"
echo "WORK_DIR=${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun --gpus-per-task=1 --kill-on-bad-exit=1 \
  env -u SLURM_LOCALID \
  "${PYTHON}" -u tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher slurm \
  ${EXTRA_ARGS:-}

echo "=== Finished. Checkpoints and logs under: ${WORK_DIR} ==="

#!/bin/bash
# =============================================================================
# Meluxina Slurm: Camera+LiDAR BEVFusion / ZOD — frozen LiDAR branch (4× A100)
# =============================================================================
# Identical to meluxina_train_zod_bevfusion.sbatch except:
#   - job-name: zod-bevfusion-fixlidar
#   - CONFIG defaults to configs/zod/zod_bevfusion_finetune_fixedLidar.py
#   - WORK_DIR defaults to multimodal-MoE/outputs/runs/zod_bevfusion_fixedlidar/
#
# Cluster facts (from sinfo / scontrol on Meluxina):
#   - Partition: gpu
#   - GPU nodes: 4 GPUs per node, feature 40ga100 (NVIDIA A100 40GB)
#
# Submit from anywhere:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_fixedLidar.sbatch
#
# ParaStation/Slurm may run a *copy* of this script from /var/spool/... — do not rely on
# BASH_SOURCE to find repo paths. Monorepo layout: mmdetection3d is inside multimodal-MoE.
# Defaults: $HOME/projects/multimodal-MoE/mmdetection3d and $HOME/projects/multimodal-MoE.
# Only set overrides if your trees live elsewhere:
#   MMDET3D_ROOT=$HOME/other/mmdetection3d MOE_ROOT=$HOME/other/multimodal-MoE sbatch ...
#
# Slurm account + QOS: default Slurm account is "nocredit" here, which does not
# combine with the gpu partition. We set account p201392 and QOS explicitly.
# Override: sbatch -A other_proj --qos=long ...
#
# Override defaults when submitting:
#   WORK_DIR=/mnt/tier2/.../my_run sbatch .../meluxina_train_zod_bevfusion_fixedLidar.sbatch
#   CONFIG=configs/zod/zod_bevfusion_finetune_fixedLidar.py EXTRA_ARGS="--amp" sbatch ...
#
# Slurm stdout/stderr (see tail -f below) — must live on a filesystem writable from
# gpu nodes; do not use mmdetection3d/slurm_logs if that tree is read-only on compute.
# Default training work_dir: multimodal-MoE/outputs/runs/zod_bevfusion_fixedlidar/<jobname>_<id>/
#
# Watch progress (use the JOBID printed by sbatch, not an old one):
#   tail -f ~/projects/multimodal-MoE/outputs/runs/slurm_logs/zod-bevfusion-fixlidar-<JOBID>.out
# =============================================================================

#SBATCH --job-name=zod-bevfusion-fixlidar
#SBATCH --account=p201392
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=40ga100
#SBATCH --time=40:00:00
#SBATCH --output=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.err

set -euo pipefail

# Repo roots: BASH_SOURCE points at a spool copy on Meluxina (ParaStation), so paths
# like dirname "${BASH_SOURCE[0]}"/../.. break. Use $HOME defaults.
# If MMDET3D_ROOT/MOE_ROOT are exported but wrong (e.g. copied "/path/to/..." from a doc),
# Slurm still forwards them — fall back to defaults instead of failing.
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
# Avoid CPU oversubscription: 4 tasks × many OpenMP threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# -----------------------------------------------------------------------------
# Python env: do NOT use conda activate / conda.sh here. Slurm exports your login
# environment; broken CONDA_EXE= /tmp/... or a stale conda.sh breaks sourcing.
# Use the env's python directly and a clean PATH.
# -----------------------------------------------------------------------------
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
# Strip broken /tmp relocation leftovers Slurm forwards from the login shell
unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT CONDA_SHLVL
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
echo "Using PYTHON=${PYTHON}"
"${PYTHON}" -c "
import sys
print('python', sys.executable)
import torch
print('torch', torch.__version__)
import projects.BEVFusion.bevfusion  # noqa: F401
from mmdet3d.registry import MODELS
assert 'BEVFusion' in MODELS.module_dict, (
    'BEVFusion not in mmdet3d registry (custom import chain failed — often bev_pool_ext missing; '
    'camera+LiDAR BEVFusion needs bev_pool_ext CUDA extension built)'
)
print('BEVFusion registered OK')
# Quick check that camera-side classes are available
for cls_name in ('DepthLSSTransform', 'ConvFuser', 'GeneralizedLSSFPN'):
    assert cls_name in MODELS.module_dict, f'{cls_name} not registered'
print('Camera-side modules registered OK')
" || {
  echo "ERROR: env sanity check failed (torch / BEVFusion import)." >&2
  exit 1
}

CONFIG="${CONFIG:-configs/zod/zod_bevfusion_finetune_fixedLidar.py}"
# Default: MMEngine work_dir under multimodal-MoE/outputs/runs (override: WORK_DIR=... sbatch ...)
DEFAULT_WORK_PARENT="${MOE_ROOT}/outputs/runs/zod_bevfusion_fixedlidar"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_PARENT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "MMDET3D_ROOT=${MMDET3D_ROOT}"
echo "CONFIG=${CONFIG}"
echo "WORK_DIR=${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES (step 0): ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# MMEngine maps LOCAL_RANK from SLURM_LOCALID then torch.cuda.set_device(local_rank).
# Slurm GPU cgroups usually expose exactly ONE GPU per task as cuda:0, but SLURM_LOCALID
# is still 0..3 → ranks 1–3 call set_device(1/2/3) → "invalid device ordinal".
# Dropping SLURM_LOCALID makes MMEngine use local_rank = proc_id % device_count (0 when count==1).
# shellcheck disable=SC2086
srun --gpus-per-task=1 --kill-on-bad-exit=1 \
  env -u SLURM_LOCALID \
  "${PYTHON}" -u tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher slurm \
  ${EXTRA_ARGS:-}

echo "=== Finished. Checkpoints and logs under: ${WORK_DIR} ==="

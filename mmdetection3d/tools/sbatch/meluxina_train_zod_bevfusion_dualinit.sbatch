#!/bin/bash
# =============================================================================
# Meluxina Slurm: BEVFusion dual-init controlled baseline / ZOD (4× A100, single node)
# =============================================================================
# Controlled dual-init baseline — differs from meluxina_train_bev_fusion_finetune.sbatch
# only in the initialisation strategy (no load_from; DualCheckpointInitHook instead):
#   - LiDAR branch (pts_middle_encoder, pts_backbone, pts_neck, bbox_head)
#       loaded from best ZOD LiDAR-only checkpoint (best_mAP_0.50_epoch_18.pth).
#   - Camera branch (img_backbone, img_neck, view_transform)
#       loaded from best ZOD camera-only checkpoint (best_mAP_0.50_epoch_11.pth).
#   - fusion_layer (ConvFuser) randomly initialised — same as finetune.
#   - No load_from in config (intentional; see config docstring).
#   - Schedule, epochs (12), and all other hyperparameters identical to finetune.
#   - LiDAR and camera remain fully trainable throughout (no branch freezing).
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_dualinit.sbatch
#
# Watch progress:
#   tail -f ~/projects/multimodal-MoE/outputs/runs/slurm_logs/bevfusion-dualinit-<JOBID>.out
# =============================================================================

#SBATCH --job-name=bevfusion-dualinit
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
"${PYTHON}" -c "
import sys
print('python', sys.executable)
import torch
print('torch', torch.__version__)
import projects.BEVFusion.bevfusion  # noqa: F401
import mmdet3d.engine.hooks  # ensure all custom hooks are registered  # noqa: F401
from mmdet3d.registry import MODELS, HOOKS
assert 'BEVFusion' in MODELS.module_dict, (
    'BEVFusion not in mmdet3d registry (custom import chain failed — often bev_pool_ext missing; '
    'camera+LiDAR BEVFusion needs bev_pool_ext CUDA extension built)'
)
print('BEVFusion registered OK')
for cls_name in ('DepthLSSTransform', 'ConvFuser', 'GeneralizedLSSFPN'):
    assert cls_name in MODELS.module_dict, f'{cls_name} not registered'
print('Camera-side modules registered OK')
assert 'DualCheckpointInitHook' in HOOKS.module_dict, (
    'DualCheckpointInitHook not registered — check mmdet3d/engine/hooks/__init__.py')
print('DualCheckpointInitHook registered OK')
" || {
  echo "ERROR: env sanity check failed (torch / BEVFusion / DualCheckpointInitHook import)." >&2
  exit 1
}

CONFIG="${CONFIG:-configs/zod/zod_bevfusion_dualinit.py}"
DEFAULT_WORK_PARENT="${MOE_ROOT}/outputs/runs/zod_bevfusion_dualinit"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_PARENT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"

echo "=== Slurm job ${SLURM_JOB_ID} on $(hostname) ==="
echo "MMDET3D_ROOT=${MMDET3D_ROOT}"
echo "CONFIG=${CONFIG}"
echo "WORK_DIR=${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES (step 0): ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

srun --gpus-per-task=1 --kill-on-bad-exit=1 \
  env -u SLURM_LOCALID \
  "${PYTHON}" -u tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher slurm \
  ${EXTRA_ARGS:-}

echo "=== Finished. Checkpoints and logs under: ${WORK_DIR} ==="

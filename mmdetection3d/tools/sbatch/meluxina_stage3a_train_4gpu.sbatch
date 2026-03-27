#!/bin/bash
# =============================================================================
# Stage 3A — 4×A100 distributed training (DDP, 1 node, --launcher slurm)
# =============================================================================
# Full training run using all 4 GPUs on a single MeluXina GPU node via
# PyTorch DistributedDataParallel (DDP). Each Slurm task owns exactly 1 GPU.
# MMEngine's --launcher slurm reads SLURM_PROCID/SLURM_NTASKS to set up
# torch.distributed.init_process_group (NCCL backend) automatically.
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_stage3a_train_4gpu.sbatch
#
# Override examples:
#   CONFIG=configs/zod/zod_lidar_only.py sbatch ...
#   WORK_DIR=/mnt/tier2/.../my_run sbatch ...
#   EXTRA_ARGS="--amp" sbatch ...    # (Stage 3B — see docs)
#
# Logs:
#   Slurm stdout/err → multimodal-MoE/outputs/runs/slurm_logs/stage3a-4gpu-<jobid>.{out,err}
#   MMEngine work_dir → multimodal-MoE/outputs/runs/zod_lidar_only/<jobname>_<jobid>/
# =============================================================================

# ─── Slurm directives ───────────────────────────────────────────────────────

#SBATCH --job-name=stage3a-4gpu
#     ↳ Short label visible in squeue; used in log filenames via %x.

#SBATCH --account=p201222
#     ↳ Billing account. MeluXina's default "nocredit" is rejected by the gpu partition.

#SBATCH --qos=default
#     ↳ Quality of Service. Must be set explicitly on MeluXina.

#SBATCH --partition=gpu
#     ↳ GPU partition — the only partition with A100 nodes.

#SBATCH --nodes=1
#     ↳ Single-node DDP. All 4 GPUs are on the same physical node.

#SBATCH --ntasks=4
#     ↳ 4 MPI/srun tasks — one per GPU. Each becomes a DDP rank.

#SBATCH --ntasks-per-node=4
#     ↳ Redundant with nodes=1 + ntasks=4, but makes the intent explicit
#       and prevents accidental cross-node scattering.

#SBATCH --gres=gpu:4
#     ↳ Request 4 GPUs total on the node.

#SBATCH --gpus-per-task=1
#     ↳ Slurm cgroups bind exactly 1 GPU to each task. Each task sees its
#       GPU as cuda:0. Critical for the SLURM_LOCALID workaround below.

#SBATCH --cpus-per-task=8
#     ↳ 8 CPU cores per task → 32 total. MeluXina GPU nodes have 128 cores,
#       so this leaves headroom. Feeds DataLoader workers.

#SBATCH --constraint=40ga100
#     ↳ Target NVIDIA A100 40GB SXM4 nodes specifically.

#SBATCH --time=24:00:00
#     ↳ 24-hour wall clock. Config has max_epochs=20 with ~40k iters/epoch
#       at batch_size=2/GPU. Adjust if needed.

#SBATCH --output=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.err
#     ↳ %u = username, %x = job-name, %j = jobid. Directory must exist before submit.

# ─── Shell safety ────────────────────────────────────────────────────────────

set -eo pipefail

# ─── Repository paths ────────────────────────────────────────────────────────
# Monorepo layout: mmdetection3d lives inside multimodal-MoE.
# ParaStation copies the script to /var/spool/... so BASH_SOURCE is unreliable.

MMDET3D_DEFAULT="${HOME}/projects/multimodal-MoE/mmdetection3d"
MOE_DEFAULT="${HOME}/projects/multimodal-MoE"

if [[ -z "${MMDET3D_ROOT:-}" ]] || [[ ! -d "${MMDET3D_ROOT}" ]]; then
  [[ -n "${MMDET3D_ROOT:-}" ]] && echo "WARN: MMDET3D_ROOT='${MMDET3D_ROOT}' invalid; using ${MMDET3D_DEFAULT}" >&2
  MMDET3D_ROOT="${MMDET3D_DEFAULT}"
fi
if [[ -z "${MOE_ROOT:-}" ]] || [[ ! -d "${MOE_ROOT}" ]]; then
  [[ -n "${MOE_ROOT:-}" ]] && echo "WARN: MOE_ROOT='${MOE_ROOT}' invalid; using ${MOE_DEFAULT}" >&2
  MOE_ROOT="${MOE_DEFAULT}"
fi
[[ -d "${MMDET3D_ROOT}" ]] || { echo "ERROR: ${MMDET3D_ROOT} not found" >&2; exit 1; }
[[ -d "${MOE_ROOT}" ]]     || { echo "ERROR: ${MOE_ROOT} not found" >&2; exit 1; }

mkdir -p "${MOE_ROOT}/outputs/runs/slurm_logs"
cd "${MMDET3D_ROOT}"

# ─── Python environment (REQUIRED for distributed training) ──────────────────
# Bypass conda activate — Slurm inherits stale CONDA_* vars from login shell.
# Call the env's python binary directly and put it first on PATH.

CONDA_ENV_NAME="${CONDA_ENV_NAME:-multimodal-moe}"
CONDA_ROOT="${CONDA_ROOT:-}"
if [[ -z "${CONDA_ROOT}" ]]; then
  for d in "${HOME}/miniforge3" "${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/mambaforge"; do
    [[ -d "${d}/envs/${CONDA_ENV_NAME}/bin" ]] && { CONDA_ROOT="${d}"; break; }
  done
fi
[[ -d "${CONDA_ROOT:-}" ]] || { echo "ERROR: CONDA_ROOT not found" >&2; exit 1; }
ENV_PREFIX="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}"
PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${PYTHON}" ]] || { echo "ERROR: ${PYTHON} missing" >&2; exit 1; }

unset CONDA_EXE _CONDA_EXE CONDA_PYTHON_EXE _CONDA_ROOT CONDA_SHLVL
export CONDA_PREFIX="${ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

# ─── PYTHONPATH, buffering, threading ────────────────────────────────────────

export PYTHONPATH="${MMDET3D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# REQUIRED for distributed: avoid CPU oversubscription.
# Without these, each of the 4 tasks spawns OMP_NUM_THREADS=<all cores> threads,
# leading to 4×128 = 512 threads fighting over 128 cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ─── NCCL configuration (OPTIONAL robustness) ───────────────────────────────
# NCCL_DEBUG=INFO prints the full transport negotiation on startup — very
# useful for first-time verification, but verbose. Set to WARN once stable.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# ─── Sanity checks (run ONCE on rank 0 before srun) ─────────────────────────

echo "=== Stage 3A: 4×A100 DDP training ==="
echo "host=$(hostname)  date=$(date -Is)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}  SLURM_NTASKS=${SLURM_NTASKS}"
echo "MMDET3D_ROOT=${MMDET3D_ROOT}"
echo "PYTHON=${PYTHON}"

"${PYTHON}" -c "
import sys, torch
print('python', sys.executable)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),
      'devices', torch.cuda.device_count())
import projects.BEVFusion.bevfusion  # noqa: F401
from mmdet3d.registry import MODELS
assert 'BEVFusion' in MODELS.module_dict, 'BEVFusion not registered'
print('BEVFusion registered OK')
" || { echo "ERROR: env sanity check failed" >&2; exit 1; }

# ─── Config and work directory ───────────────────────────────────────────────

CONFIG="${CONFIG:-configs/zod/zod_lidar_only.py}"
DEFAULT_WORK_PARENT="${MOE_ROOT}/outputs/runs/zod_lidar_only"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_PARENT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"

echo "CONFIG=${CONFIG}"
echo "WORK_DIR=${WORK_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>/dev/null || true

# ─── Launch distributed training ────────────────────────────────────────────
#
# srun launches 4 copies of tools/train.py (one per --ntasks).
#
#   --gpus-per-task=1       Slurm cgroups isolate 1 GPU per task.
#   --kill-on-bad-exit=1    If any rank crashes, kill all ranks immediately
#                           instead of leaving orphans.
#   env -u SLURM_LOCALID   CLUSTER-SPECIFIC WORKAROUND (MeluXina/ParaStation):
#                           Slurm sets SLURM_LOCALID=0..3, but cgroups make
#                           each task see only 1 GPU (cuda:0). MMEngine does
#                           torch.cuda.set_device(SLURM_LOCALID), so ranks
#                           1-3 get "invalid device ordinal". Unsetting
#                           SLURM_LOCALID makes MMEngine fall back to
#                           local_rank = proc_id % device_count = 0, which
#                           is correct when each task sees exactly 1 GPU.
#   --launcher slurm        Tells MMEngine to read SLURM_PROCID (global rank)
#                           and SLURM_NTASKS (world size) and call
#                           init_process_group(backend='nccl') automatically.
#
# No --cfg-options overrides: the config's defaults (max_epochs=20,
# val_interval=1, custom_hooks with BEV visualizations) run as-is.

# shellcheck disable=SC2086
srun --gpus-per-task=1 --kill-on-bad-exit=1 \
  env -u SLURM_LOCALID \
  "${PYTHON}" -u tools/train.py "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher slurm \
  ${EXTRA_ARGS:-}

echo "=== Stage 3A finished. Artifacts: ${WORK_DIR} ==="

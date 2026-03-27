#!/bin/bash
# =============================================================================
# One-off MeluXina job: compile BEVFusion CUDA ops (bev_pool_ext, voxel_layer)
# =============================================================================
# GPU nodes: add MUSE to MODULEPATH (`module use …/modules/all`), then CUDA/12.8.0 — or we
# fall back to the same path under /apps/USE/easybuild/.../software/CUDA/12.8.0.
# After success, *.so files live under mmdetection3d/projects/BEVFusion/bevfusion/ops/
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_build_bevfusion_ops.sbatch
#
# Logs: multimodal-MoE/outputs/runs/slurm_logs/build-bevfusion-ops-<jobid>.{out,err}
# =============================================================================

#SBATCH --job-name=build-bevfusion-ops
#SBATCH --account=p201222
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=40ga100
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.out
#SBATCH --error=/home/users/%u/projects/multimodal-MoE/outputs/runs/slurm_logs/%x-%j.err

# Do not use `set -u` here: Lmod and /etc/profile.d/00-modulepath.sh touch MODULEPATH
# before it is defined; nounset makes sourcing fail on some nodes.
set -eo pipefail

MMDET3D_ROOT="${MMDET3D_ROOT:-${HOME}/projects/multimodal-MoE/mmdetection3d}"
# Must export: build_inplace.sh runs in a child bash and reads PY from the environment.
export PY="${PY:-${HOME}/miniconda3/envs/multimodal-moe/bin/python}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/modules.sh
fi

# MUSE modules are not on MODULEPATH until you `module use …/modules/all` (MeluXina docs).
if declare -F module &>/dev/null; then
  module use /apps/USE/easybuild/release/2025.1/modules/all 2>/dev/null || true
  module use /apps/USE/easybuild/release/2024.1/modules/all 2>/dev/null || true
  for _cuda in CUDA/12.8.0 CUDA/12.6.0; do
    if module load "${_cuda}" 2>/dev/null; then
      echo "Loaded module: ${_cuda}"
      break
    fi
  done
fi

# If Lmod still did not put nvcc on PATH, use the EasyBuild install tree directly (same as the CUDA module).
if ! command -v nvcc &>/dev/null; then
  for _cuda_root in \
    /apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0 \
    /apps/USE/easybuild/release/2024.1/software/CUDA/12.6.0; do
    if [[ -x "${_cuda_root}/bin/nvcc" ]]; then
      export CUDA_HOME="${_cuda_root}"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "Using CUDA_HOME=${CUDA_HOME} (direct MUSE software path, no module)"
      break
    fi
  done
fi

export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: PY not executable: ${PY}" >&2
  exit 1
fi

bash "${MMDET3D_ROOT}/projects/BEVFusion/scripts/build_inplace.sh"

echo "=== Build job finished ==="

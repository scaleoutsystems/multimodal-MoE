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
#SBATCH --time=48:00:00
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

# PyTorch 2.4 requires GCC >= 9.  MeluXina RHEL8 ships GCC 8 in /usr/bin,
# so we must use a newer GCCcore from EasyBuild.
_gcc_found=false
if declare -F module &>/dev/null; then
  for _gcc in GCCcore/13.3.0 GCCcore/12.3.0 GCC/13.3.0 GCC/12.3.0; do
    if module load "${_gcc}" 2>/dev/null; then
      echo "Loaded module: ${_gcc}"
      _gcc_found=true
      break
    fi
  done
fi
if [[ "${_gcc_found}" != true ]]; then
  for _gcc_root in \
    /apps/USE/easybuild/release/2024.1/software/GCCcore/13.3.0 \
    /apps/USE/easybuild/release/2025.1/software/GCCcore/14.2.0 \
    /apps/USE/easybuild/release/2024.1/software/GCCcore/12.3.0; do
    if [[ -x "${_gcc_root}/bin/gcc" ]]; then
      export PATH="${_gcc_root}/bin:${PATH}"
      export LD_LIBRARY_PATH="${_gcc_root}/lib64:${LD_LIBRARY_PATH:-}"
      echo "Using GCC from ${_gcc_root}/bin/gcc"
      _gcc_found=true
      break
    fi
  done
fi
if [[ "${_gcc_found}" != true ]]; then
  echo "ERROR: Could not find GCC >= 9. PyTorch 2.4 CUDA extensions require it." >&2
  exit 1
fi
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
echo "CC=${CC}  ($(gcc --version | head -1))"

export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: PY not executable: ${PY}" >&2
  exit 1
fi

bash "${MMDET3D_ROOT}/projects/BEVFusion/scripts/build_inplace.sh"

echo "=== Build job finished ==="

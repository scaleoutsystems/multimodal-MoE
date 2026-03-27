#!/usr/bin/env bash
# Build BEVFusion CUDA extensions (bev_pool_ext, voxel_layer) in-tree.
#
# Must run from a context where nvcc works (GPU node + `module load CUDA/...` on MeluXina,
# or a conda env with cuda-toolkit, or system CUDA in CUDA_HOME).
#
# Usage (from anywhere):
#   PY=$HOME/miniconda3/envs/multimodal-moe/bin/python \
#     bash /path/to/mmdetection3d/projects/BEVFusion/scripts/build_inplace.sh
#
# Cross-compile on a CPU node (needs nvcc + CUDA headers):
#   FORCE_CUDA=1 CUDA_HOME=/path/to/cuda PY=... bash .../build_inplace.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET3D_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ ! -f "${MMDET3D_ROOT}/projects/BEVFusion/setup.py" ]]; then
  echo "ERROR: MMDET3D_ROOT mis-resolved: ${MMDET3D_ROOT}" >&2
  exit 1
fi

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${HOME}/miniconda3/envs/multimodal-moe/bin/python" ]]; then
    PY="${HOME}/miniconda3/envs/multimodal-moe/bin/python"
  else
    PY="$(command -v python3 || command -v python)"
  fi
fi
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Set PY= to your training interpreter (e.g. conda env multimodal-moe)." >&2
  exit 1
fi

need_force=false
if ! "${PY}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  need_force=true
fi
if [[ "${FORCE_CUDA:-0}" == "1" ]] || [[ "${need_force}" == true ]]; then
  export FORCE_CUDA=1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc &>/dev/null; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    # MeluXina MUSE: toolkit under EasyBuild (same trees the CUDA module prepends to PATH).
    for _root in \
      /apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0 \
      /apps/USE/easybuild/release/2024.1/software/CUDA/12.6.0; do
      if [[ -x "${_root}/bin/nvcc" ]]; then
        export CUDA_HOME="${_root}"
        break
      fi
    done
  fi
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: CUDA_HOME is not set and nvcc is not on PATH." >&2
  echo "  On MeluXina: sbatch tools/sbatch/meluxina_build_bevfusion_ops.sbatch" >&2
  echo "  Or: module use .../release/2025.1/modules/all && module load CUDA/12.8.0" >&2
  echo "  Or: export CUDA_HOME=/apps/USE/easybuild/release/2025.1/software/CUDA/12.8.0" >&2
  echo "  Or: conda install -c nvidia cuda-toolkit (match PyTorch CUDA major)." >&2
  exit 1
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found under CUDA_HOME=${CUDA_HOME}/bin" >&2
  exit 1
fi

cd "${MMDET3D_ROOT}"
echo "=== BEVFusion build_ext --inplace ==="
echo "MMDET3D_ROOT=${MMDET3D_ROOT}"
echo "PYTHON=${PY}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FORCE_CUDA=${FORCE_CUDA:-0}"
nvcc --version | head -n 1

# Parallel compile (optional)
export MAX_JOBS="${MAX_JOBS:-4}"

"${PY}" projects/BEVFusion/setup.py build_ext --inplace

echo "=== Done. Look for *.so under projects/BEVFusion/bevfusion/ops/bev_pool and .../ops/voxel ==="

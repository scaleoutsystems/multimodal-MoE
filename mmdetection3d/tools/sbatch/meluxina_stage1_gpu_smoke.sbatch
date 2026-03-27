#!/bin/bash
# =============================================================================
# Stage 1 — smallest GPU smoke test (1 node, 1 task, 1× A100, no distributed)
# =============================================================================

#SBATCH --job-name=stage1-gpu-smoke
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

set -eo pipefail

# Root of the MMDetection3D clone (code + configs live here); exported for the Python heredoc below.
export MMDET3D_ROOT="${MMDET3D_ROOT:-${HOME}/projects/multimodal-MoE/mmdetection3d}"
# Interpreter with torch / mmcv / mmdet3d (explicit path avoids conda activate).
export PY="${PY:-${HOME}/miniconda3/envs/multimodal-moe/bin/python}"
# So `import projects.BEVFusion...` matches a local `PYTHONPATH=. python tools/train.py` run.
export PYTHONPATH="${MMDET3D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${MMDET3D_ROOT}"

echo "=== host ==="
hostname
date
echo "=== Slurm GPU binding (if set) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo "=== python ==="
"${PY}" -c "import sys; print(sys.executable)"

echo "=== torch + cuda ==="
"${PY}" -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count());
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))"

echo "=== nvidia-smi ==="
nvidia-smi -L || true

echo "=== training stack: Config + Runner.from_cfg (same as train.py before runner.train()) ==="
# tools/train.py always calls runner.train() at the end; that runs a full epoch. For stage 1 we only
# build the runner (model, dataloaders, hooks) so import/registry/CUDA errors show up without a long train.
export WORK_SMOKE="${HOME}/projects/multimodal-MoE/outputs/runs/stage1_smoke_workdir/${SLURM_JOB_ID}"
mkdir -p "${WORK_SMOKE}"
"${PY}" <<'PY'
import os
import sys

root = os.environ["MMDET3D_ROOT"]
sys.path.insert(0, root)
os.chdir(root)

from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile("configs/zod/zod_lidar_only.py")
cfg.launcher = "none"
cfg.work_dir = os.environ["WORK_SMOKE"]
cfg.merge_from_dict(
    {
        "train_dataloader": {
            "batch_size": 1,
            "num_workers": 0,
            "dataset": {"indices": [0]},
        },
    }
)
Runner.from_cfg(cfg)
print("OK: Runner.from_cfg succeeded (ready for runner.train() — that is Stage 2).")
PY

echo "=== stage1 smoke finished OK ==="

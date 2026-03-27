#!/bin/bash
# =============================================================================
# Stage 2 — real 1-GPU training (no distributed): same Slurm box as Stage 1
# =============================================================================
# Runs tools/train.py with --launcher none for one short epoch. Use Stage 3 for 4×A100 + slurm launcher.
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/meluxina_stage2_train_1gpu.sbatch
# Optional:
#   EXTRA_ARGS="--amp" sbatch .../meluxina_stage2_train_1gpu.sbatch
#
# Logs: multimodal-MoE/outputs/runs/slurm_logs/stage2-train-1gpu-<jobid>.{out,err}
# Work dir: multimodal-MoE/outputs/runs/zod_lidar_only_stage2/<jobid>/
# =============================================================================

#SBATCH --job-name=stage2-train-1gpu
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

export MMDET3D_ROOT="${MMDET3D_ROOT:-${HOME}/projects/multimodal-MoE/mmdetection3d}"
export PY="${PY:-${HOME}/miniconda3/envs/multimodal-moe/bin/python}"
export PYTHONPATH="${MMDET3D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

MOE_ROOT="${MOE_ROOT:-${HOME}/projects/multimodal-MoE}"
# Checkpoints, logs, and MMEngine artifacts land here (same idea as the old full train sbatch).
export WORK_DIR="${WORK_DIR:-${MOE_ROOT}/outputs/runs/zod_lidar_only_stage2/${SLURM_JOB_ID}}"
mkdir -p "${WORK_DIR}"

cd "${MMDET3D_ROOT}"

echo "=== Stage 2: 1-GPU train ==="
echo "host=$(hostname) date=$(date -Is)"
echo "WORK_DIR=${WORK_DIR}"
echo "PY=${PY}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PY}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# One epoch, validate once, checkpoint once; drop BEV vis hooks for less I/O (re-enable in cfg when stable).
# Quote custom_hooks=[] so bash does not treat [] as a glob.
# shellcheck disable=SC2086
"${PY}" -u tools/train.py configs/zod/zod_lidar_only.py \
  --work-dir "${WORK_DIR}" \
  --launcher none \
  --cfg-options \
    train_cfg.max_epochs=1 \
    val_interval=1 \
    default_hooks.checkpoint.interval=1 \
    'custom_hooks=[]' \
  ${EXTRA_ARGS:-}

echo "=== Stage 2 finished ==="

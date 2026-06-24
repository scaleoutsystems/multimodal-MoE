#!/bin/bash
# =============================================================================
# Migration: p201222 -> p201392
# =============================================================================
# Does three things in sequence:
#   1. rsync /project/home/p201222/u103958/ -> /project/home/p201392/u103958/
#      (preserves all data, skips already-transferred files on re-run)
#   2. Fixes CAM_FRONT symlinks inside zod_nuscenes/samples/CAM_FRONT/
#      (symlinks still point to old p201222 path after rsync; updates them)
#   3. Replaces every p201222 reference in the multimodal-MoE codebase
#      with p201392 (idempotent: safe to re-run if already done)
#
# Submit:
#   sbatch ~/projects/multimodal-MoE/mmdetection3d/tools/sbatch/migrate_p201222_to_p201392.sbatch
#
# Watch progress:
#   tail -f ~/projects/multimodal-MoE/outputs/runs/slurm_logs/migrate_<JOBID>.out
# =============================================================================

#SBATCH --job-name=migrate_p201222
#SBATCH --account=p201392
#SBATCH --qos=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/migrate_%j.out
#SBATCH --error=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/migrate_%j.err

set -euo pipefail

SRC=/project/home/p201222/u103958/
DST=/project/home/p201392/u103958/
REPO=/home/users/u103958/projects/multimodal-MoE
LOG_PREFIX="[migrate $(date '+%H:%M:%S')]"

echo "============================================================"
echo " Migration: p201222 -> p201392"
echo " Started: $(date)"
echo "============================================================"

# ── STEP 1: rsync ────────────────────────────────────────────────────────────
echo ""
echo "$LOG_PREFIX STEP 1: rsync $SRC -> $DST"
echo ""

mkdir -p "$DST"

rsync -av \
    --progress \
    --stats \
    --checksum \
    "$SRC" "$DST"

RSYNC_EXIT=$?
echo ""
echo "$LOG_PREFIX rsync finished with exit code $RSYNC_EXIT"
if [ $RSYNC_EXIT -ne 0 ]; then
    echo "ERROR: rsync failed. Aborting."
    exit $RSYNC_EXIT
fi

# ── STEP 1b: Verify file counts ───────────────────────────────────────────────
echo ""
echo "$LOG_PREFIX Verifying file counts..."
SRC_COUNT=$(find "$SRC" -type f | wc -l)
DST_COUNT=$(find "$DST" -type f | wc -l)
echo "  Source files: $SRC_COUNT"
echo "  Dest   files: $DST_COUNT"
if [ "$SRC_COUNT" -eq "$DST_COUNT" ]; then
    echo "  VERIFICATION PASSED: file counts match."
else
    echo "  WARNING: counts differ (src=$SRC_COUNT dst=$DST_COUNT)."
    echo "  This may be expected if dst already had extra files. Continuing."
fi

# ── STEP 2: Fix CAM_FRONT symlinks ───────────────────────────────────────────
# rsync copies symlinks as-is, so CAM_FRONT/*.png still point to the old
# /project/home/p201222/... path. Update them to p201392.
echo ""
echo "$LOG_PREFIX STEP 2: Updating CAM_FRONT symlinks in zod_nuscenes..."
CAM_DIR="$DST/zod_moe/zod_nuscenes/samples/CAM_FRONT"

if [ -d "$CAM_DIR" ]; then
    UPDATED=0
    SKIPPED=0
    while IFS= read -r -d '' link; do
        target=$(readlink "$link")
        if echo "$target" | grep -q 'p201222'; then
            new_target="${target/p201222/p201392}"
            ln -sf "$new_target" "$link"
            UPDATED=$((UPDATED + 1))
        else
            SKIPPED=$((SKIPPED + 1))
        fi
    done < <(find "$CAM_DIR" -maxdepth 1 -type l -print0)
    echo "  Symlinks updated: $UPDATED  (already-correct: $SKIPPED)"
else
    echo "  WARNING: $CAM_DIR not found — skipping symlink fix."
fi

# ── STEP 3: Update p201222 references in codebase ────────────────────────────
echo ""
echo "$LOG_PREFIX STEP 3: Replacing p201222 -> p201392 in codebase files..."

FILES=(
    # configs
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_taskdriven_30ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_30ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_dualinit_28ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_dualinit_40ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_taskdriven_40ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_40ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_camera_only_40ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_30ep.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_z004.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4_taskdriven.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only_moe_dense4.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_dualinit.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_baseline.py"
    "$REPO/mmdetection3d/configs/zod/zod_lidar_only.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_finetune.py"
    "$REPO/mmdetection3d/configs/zod/zod_camera_only_crop.py"
    "$REPO/mmdetection3d/configs/zod/zod_camera_only.py"
    "$REPO/mmdetection3d/configs/zod/zod_camera_only_ftlr_d60_splat0.py"
    "$REPO/mmdetection3d/configs/zod/zod_camera_only_ftlr_d60.py"
    "$REPO/mmdetection3d/configs/zod/zod_bevfusion_finetune_fixedLidar.py"
    "$REPO/mmdetection3d/configs/zod/zod_moe_fusion_then.py"
    "$REPO/mmdetection3d/configs/zod/zod_moe_joint_modality.py"
    "$REPO/mmdetection3d/configs/zod/zod_moe_modality_specific.py"
    # sbatch scripts
    "$REPO/mmdetection3d/tools/sbatch/meluxina_multi_split_eval.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/run_moe_modality_specific.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/run_moe_joint_modality.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/run_moe_fusion_then.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_dualinit_28ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_taskdriven_30ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_dualinit_40ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_taskdriven_40ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_camera_only_40ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_30ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_30ep.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_z004.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4_taskdriven.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe_dense4.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_dualinit.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only_moe.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion_fixedLidar.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_vis3d.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_lidar_only.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_bevfusion.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_bev_fusion_finetune.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_build_bevfusion_ops.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_camera_only_crop.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_camera_only_ftlr_d60_splat0.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_zod_camera_only_ftlr_d60.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_train_camera_only.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_single_split_visual_eval.sbatch"
    "$REPO/mmdetection3d/tools/sbatch/meluxina_bevfusion_camera_ablation.sbatch"
    # tools and scripts
    "$REPO/mmdetection3d/tools/debug_heatmap_coords.py"
    "$REPO/mmdetection3d/tools/benchmark_inference.py"
    "$REPO/scripts/bev_scripts/run_multi_split_eval.py"
    "$REPO/scripts/bev_scripts/run_single_split_visual_eval.py"
    "$REPO/scripts/bev_scripts/run_camera_ablation_eval.py"
    "$REPO/scripts/bev_scripts/plot_bev.py"
    "$REPO/scripts/bev_dataset/build_infos.py"
    "$REPO/scripts/bev_dataset/build_zod_moe_dataset.py"
    "$REPO/scripts/bev_dataset/create_splits.py"
    "$REPO/scripts/bev_dataset/sanity_check.py"
    "$REPO/scripts/bev_dataset/filter_parquet_to_bev_range.py"
    "$REPO/scripts/add_context_bins.py"
    "$REPO/scripts/pkl_to_txt.py"
    # docs / analysis
    "$REPO/third_party/mmdetection3d_changes.md"
    "$REPO/outputs/analysis/bev_dataset/zod_nuscenes_infos_val_first_item.txt"
    "$REPO/outputs/analysis/bev_dataset/zod_nuscenes_infos_train_first_item.txt"
    "$REPO/outputs/analysis/bev_dataset/zod_nuscenes_infos_test_first_item.txt"
)

REPLACED=0
SKIPPED=0
for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  SKIP (not found): $f"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if grep -q 'p201222' "$f" 2>/dev/null; then
        COUNT=$(grep -c 'p201222' "$f" || true)
        sed -i 's/p201222/p201392/g' "$f"
        echo "  UPDATED ($COUNT occurrences): ${f#$REPO/}"
        REPLACED=$((REPLACED + 1))
    else
        SKIPPED=$((SKIPPED + 1))
    fi
done

echo ""
echo "  Files updated: $REPLACED  |  Already clean / not found: $SKIPPED"

# ── STEP 3b: Sanity check — confirm no p201222 remains ───────────────────────
echo ""
echo "$LOG_PREFIX Checking for any remaining p201222 references in codebase..."
REMAINING=$(grep -rl 'p201222' \
    "$REPO/mmdetection3d/configs" \
    "$REPO/mmdetection3d/tools" \
    "$REPO/scripts" \
    2>/dev/null | grep -v '.git' || true)
if [ -z "$REMAINING" ]; then
    echo "  CLEAN: no p201222 references remain in configs/tools/scripts."
else
    echo "  WARNING: p201222 still found in:"
    echo "$REMAINING" | sed 's/^/    /'
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Migration complete: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Verify a training run: check one config references p201392 paths"
echo "  2. Submit a job with the new account: --account=p201392"
echo "  3. If CAM_FRONT symlinks still broken, re-run build_infos.py:"
echo "       python \$REPO/scripts/bev_dataset/build_infos.py \\"
echo "           --source-root /project/home/p201392/u103958/zod_moe \\"
echo "           --final-root  /project/home/p201392/u103958/zod_moe/zod_nuscenes"

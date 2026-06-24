#!/bin/bash
#SBATCH --job-name=migrate_zod_moe
#SBATCH --partition=cpu
#SBATCH --account=p201392
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/migrate_zod_moe_%j.out
#SBATCH --error=/home/users/u103958/projects/multimodal-MoE/outputs/runs/slurm_logs/migrate_zod_moe_%j.err

SRC=/project/home/p201222/u103958/zod_moe/
DST=/project/home/p201392/u103958/zod_moe/

echo "=== rsync migration: zod_moe ==="
echo "Source:      $SRC"
echo "Destination: $DST"
echo "Started at:  $(date)"
echo ""

mkdir -p "$DST"

rsync -av --progress --stats \
    "$SRC" "$DST"

STATUS=$?
echo ""
echo "=== rsync finished at $(date) with exit code $STATUS ==="

if [ $STATUS -eq 0 ]; then
    echo "--- File count verification ---"
    SRC_COUNT=$(find "$SRC" -type f | wc -l)
    DST_COUNT=$(find "$DST" -type f | wc -l)
    echo "Source files: $SRC_COUNT"
    echo "Dest   files: $DST_COUNT"
    if [ "$SRC_COUNT" -eq "$DST_COUNT" ]; then
        echo "VERIFICATION PASSED: file counts match."
    else
        echo "VERIFICATION FAILED: counts differ! (src=$SRC_COUNT dst=$DST_COUNT)"
        exit 1
    fi
fi

exit $STATUS

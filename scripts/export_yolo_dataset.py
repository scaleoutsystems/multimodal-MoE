"""
Export parquet + split CSVs into Ultralytics dataset format.

Role in the pipeline
--------------------
This is the "prepare data for Ultralytics detectors" entrypoint.
It does not contain core conversion logic itself; it is intentionally thin:
parse CLI args -> call reusable functions in `src/data/index.py` and `src/data/exports.py`.

Inputs
------
- Frame-level parquet with canonical fields (frame_id, resized image path, xyxy boxes, etc.)
- Split CSV files (train_ids.csv / val_ids.csv / test_ids.csv)
- Export policy flags (currently unclear-policy)

What it does
------------
- Loads each split using canonical frame_id matching/normalization.
- Converts canonical xyxy boxes to YOLO normalized xywh labels.
- Writes YOLO images/labels folders and dataset.yaml.
- Prints per-split summary counts (written images/labels/boxes, dropped unclear, empty labels).

Ultralytics compatibility note
------------------------------
This export format is usable by both Ultralytics YOLO and Ultralytics RT-DETR
models. So this script can be reused for RT-DETR experiments in this project
without changing dataset annotation format.

Why this script exists
----------------------
- Keeps YOLO export reproducible as a single command.
- Keeps script-level orchestration separate from reusable library logic.
- Makes it easy to add future exporters (COCO/DINO/etc.) with the same architecture.

EXPECTED ULTRALYTICS DATASET STRUCTURE:
<out_dir>/
  dataset.yaml
  images/
    train/
      000000.jpg
      000001.jpg
      ...
    val/
      ...
    test/
      ...
  labels/
    train/
      000000.txt
      000001.txt
      ...
    val/
      ...
    test/
      ...

000000.txt contains one line per box: <class_id> <x_center> <y_center> <width> <height>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running as either:
# - python -m scripts.export_yolo_dataset
# - python scripts/export_yolo_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.index import load_split_frames
from src.data.exports import export_yolo_split, write_yolo_dataset_yaml
from src.paths import (
    EXPORTS_DIR, # /home/edgelab/multimodal-moe/outputs/exports
    TEST_SPLIT_CSV, # /home/edgelab/zod_moe/splits/test_ids.csv
    TRAIN_SPLIT_CSV, # /home/edgelab/zod_moe/splits/train_ids.csv
    VAL_SPLIT_CSV, # /home/edgelab/zod_moe/splits/val_ids.csv
    ZODMOE_FRAMES_WITH_BOXES_PARQUET, # /home/edgelab/multimodal-moe/outputs/index/ZODmoe_frames_with_xyxy_bboxes.parquet
)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for YOLO dataset export.

    Input:
        CLI flags from the command line.

    Output:
        argparse.Namespace with export configuration values.

    Why:
        Keeps script entrypoint thin and explicit while delegating logic to src/.
    """
    parser = argparse.ArgumentParser(description="Export YOLO dataset from parquet index.")
    parser.add_argument(
        "--frames-parquet",
        type=str,
        default=str(ZODMOE_FRAMES_WITH_BOXES_PARQUET),
        help="Parquet with frame metadata + boxes.",
    )
    parser.add_argument("--train-split-csv", type=str, default=str(TRAIN_SPLIT_CSV))
    parser.add_argument("--val-split-csv", type=str, default=str(VAL_SPLIT_CSV))
    parser.add_argument("--test-split-csv", type=str, default=str(TEST_SPLIT_CSV))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(EXPORTS_DIR / "yolo" / "pedestrian_v1_exclude_unclear"),
        help="Output YOLO dataset root (contains images/, labels/, dataset.yaml).",
    )
    parser.add_argument(
        "--unclear-policy",
        choices=["keep_all", "exclude_unclear"],
        default="exclude_unclear",
        help="How to treat boxes marked unclear.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build YOLO train/val/test files from parquet + split CSVs.

    Input:
        None directly (reads parsed CLI args).

    Output:
        None (writes dataset files and prints per-split summaries).

    Why:
        Provides a reproducible one-command export step before training.
    """
    args = parse_args()

    frames_parquet = Path(args.frames_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required columns for exporting YOLO labels.
    required_cols = [
        "frame_id",
        "resized_image_path",
        "xyxy_bboxes",
        "ped_unclear_list",
        "new_w",
        "new_h",
    ]

    split_cfg = [
        ("train", args.train_split_csv),
        ("val", args.val_split_csv),
        ("test", args.test_split_csv),
    ]

    summaries = []
    for split_name, split_csv in split_cfg:
        split_df = load_split_frames(
            frames_parquet=frames_parquet,
            split_csv=split_csv,
            frame_id_col="frame_id",
            required_columns=required_cols,
        )
        summary = export_yolo_split(
            split_name=split_name,
            frames_df=split_df,
            out_dataset_dir=out_dir,
            image_path_col="resized_image_path",
            frame_id_col="frame_id",
            boxes_col="xyxy_bboxes",
            unclear_col="ped_unclear_list",
            img_w_col="new_w",
            img_h_col="new_h",
            unclear_policy=args.unclear_policy,
            class_id=0,  # single-class pedestrian detection
        )
        summaries.append(summary)

    yaml_path = write_yolo_dataset_yaml(
        out_dataset_dir=out_dir,
        class_names={0: "pedestrian"},
    )

    print(f"Wrote dataset.yaml -> {yaml_path}")
    for s in summaries:
        print(
            f"[{s.split}] frames={s.n_frames} images={s.n_images_written} "
            f"labels={s.n_label_files_written} boxes={s.n_boxes_written} "
            f"dropped_unclear={s.n_boxes_dropped_unclear} empty_labels={s.n_empty_label_files}"
        )


if __name__ == "__main__":
    main()

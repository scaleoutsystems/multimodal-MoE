"""
Export parquet + split CSVs into COCO detection format.

This is the exporter used for the third-party RT-DETRv2 PyTorch pipeline,
which expects COCO-style annotation JSON files.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Literal

import numpy as np

# Allow running as either:
# - python -m scripts.export_coco_dataset
# - python scripts/export_coco_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.index import load_split_frames
from src.paths import (
    EXPORTS_DIR,
    TEST_SPLIT_CSV,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    ZODMOE_FRAMES_WITH_BOXES_AND_SOLAR_BINS_PARQUET,
)


UnclearPolicy = Literal["keep_all", "exclude_unclear"]


@dataclass
class CocoExportSummary:
    split: str
    n_frames: int
    n_images_written: int
    n_annotations_written: int
    n_boxes_dropped_unclear: int
    n_images_without_boxes: int
    ann_json_path: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _symlink_image(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def _safe_iter_boxes(xyxy_bboxes) -> list[np.ndarray]:
    if xyxy_bboxes is None:
        return []
    arr = np.asarray(xyxy_bboxes)
    if arr.size == 0:
        return []
    if arr.dtype == object:
        out: list[np.ndarray] = []
        for item in xyxy_bboxes:
            item_arr = np.asarray(item, dtype=np.float32)
            if item_arr.shape == (4,):
                out.append(item_arr)
        return out
    if arr.ndim == 2 and arr.shape[1] == 4:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 1 and arr.shape[0] == 4:
        return [arr]
    return []


def _xyxy_to_coco_xywh(box: np.ndarray, img_w: float, img_h: float) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    w = x2 - x1
    h = y2 - y1
    if w <= 0.0 or h <= 0.0:
        return None
    return x1, y1, w, h


def export_coco_split(
    *,
    split_name: str,
    frames_df,
    out_dataset_dir: str | Path,
    image_path_col: str = "resized_image_path",
    frame_id_col: str = "frame_id",
    boxes_col: str = "xyxy_bboxes",
    unclear_col: str = "ped_unclear_list",
    img_w_col: str = "new_w",
    img_h_col: str = "new_h",
    solar_bin_col: str = "solar_context_bin",
    unclear_policy: UnclearPolicy = "exclude_unclear",
    category_id: int = 1,
) -> CocoExportSummary:
    out_dataset_dir = Path(out_dataset_dir)
    images_dir = out_dataset_dir / "images" / split_name
    ann_dir = out_dataset_dir / "annotations"
    _ensure_dir(images_dir)
    _ensure_dir(ann_dir)

    required = [frame_id_col, image_path_col, boxes_col, unclear_col, img_w_col, img_h_col]
    for col in required:
        if col not in frames_df.columns:
            raise ValueError(f"frames_df missing required column '{col}'")

    images = []
    annotations = []
    ann_id = 1
    n_images_written = 0
    n_annotations_written = 0
    n_boxes_dropped_unclear = 0
    n_images_without_boxes = 0

    for image_id, (_, row) in enumerate(frames_df.iterrows(), start=1):
        frame_id = str(row[frame_id_col]).zfill(6)
        src_img = Path(row[image_path_col])
        if not src_img.exists():
            continue

        dst_img = images_dir / f"{frame_id}.jpg"
        _symlink_image(src=src_img, dst=dst_img)
        n_images_written += 1

        img_w = float(row[img_w_col])
        img_h = float(row[img_h_col])
        image_entry = {
            "id": int(image_id),
            "file_name": f"{frame_id}.jpg",
            "width": int(round(img_w)),
            "height": int(round(img_h)),
        }
        # Keep context available for analysis/debug while staying COCO-compatible.
        if solar_bin_col in frames_df.columns:
            solar_value = row[solar_bin_col]
            image_entry["solar_context_bin"] = None if solar_value is None else str(solar_value)
        images.append(image_entry)

        boxes = _safe_iter_boxes(row[boxes_col])
        unclear_flags = np.asarray(row[unclear_col]) if row[unclear_col] is not None else np.asarray([])
        image_box_count = 0

        for i, box in enumerate(boxes):
            is_unclear = bool(unclear_flags[i]) if i < len(unclear_flags) else False
            if unclear_policy == "exclude_unclear" and is_unclear:
                n_boxes_dropped_unclear += 1
                continue

            coco_box = _xyxy_to_coco_xywh(box=box, img_w=img_w, img_h=img_h)
            if coco_box is None:
                continue
            x, y, w, h = coco_box

            annotations.append(
                {
                    "id": int(ann_id),
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
            n_annotations_written += 1
            image_box_count += 1

        if image_box_count == 0:
            n_images_without_boxes += 1

    coco = {
        "info": {
            "description": "ZOD pedestrian detection export",
            "version": "1.0",
            "year": 2026,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": int(category_id), "name": "pedestrian", "supercategory": "person"}],
    }
    ann_json_path = ann_dir / f"instances_{split_name}.json"
    ann_json_path.write_text(json.dumps(coco, indent=2))

    return CocoExportSummary(
        split=split_name,
        n_frames=int(len(frames_df)),
        n_images_written=int(n_images_written),
        n_annotations_written=int(n_annotations_written),
        n_boxes_dropped_unclear=int(n_boxes_dropped_unclear),
        n_images_without_boxes=int(n_images_without_boxes),
        ann_json_path=str(ann_json_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export COCO dataset from parquet index.")
    parser.add_argument(
        "--frames-parquet",
        type=str,
        default=str(ZODMOE_FRAMES_WITH_BOXES_AND_SOLAR_BINS_PARQUET),
        help="Parquet with frame metadata + boxes + context (solar bins).",
    )
    parser.add_argument("--train-split-csv", type=str, default=str(TRAIN_SPLIT_CSV))
    parser.add_argument("--val-split-csv", type=str, default=str(VAL_SPLIT_CSV))
    parser.add_argument("--test-split-csv", type=str, default=str(TEST_SPLIT_CSV))
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(EXPORTS_DIR / "coco" / "pedestrian_v1_exclude_unclear"),
        help="Output COCO dataset root (images/<split>/ + annotations/instances_*.json).",
    )
    parser.add_argument(
        "--unclear-policy",
        choices=["keep_all", "exclude_unclear"],
        default="exclude_unclear",
        help="How to treat boxes marked unclear.",
    )
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke export (debug only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_parquet = Path(args.frames_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "frame_id",
        "resized_image_path",
        "xyxy_bboxes",
        "ped_unclear_list",
        "new_w",
        "new_h",
        "solar_context_bin",
    ]

    split_cfg = [
        ("train", args.train_split_csv),
        ("val", args.val_split_csv),
        ("test", args.test_split_csv),
    ]

    summaries: list[CocoExportSummary] = []
    for split_name, split_csv in split_cfg:
        split_df = load_split_frames(
            frames_parquet=frames_parquet,
            split_csv=split_csv,
            frame_id_col="frame_id",
            required_columns=required_cols,
        )
        if args.max_frames_per_split is not None:
            split_df = split_df.head(int(args.max_frames_per_split)).copy()

        summary = export_coco_split(
            split_name=split_name,
            frames_df=split_df,
            out_dataset_dir=out_dir,
            image_path_col="resized_image_path",
            frame_id_col="frame_id",
            boxes_col="xyxy_bboxes",
            unclear_col="ped_unclear_list",
            img_w_col="new_w",
            img_h_col="new_h",
            solar_bin_col="solar_context_bin",
            unclear_policy=args.unclear_policy,
            category_id=1,
        )
        summaries.append(summary)

    manifest = {
        "frames_parquet": str(frames_parquet.resolve()),
        "unclear_policy": args.unclear_policy,
        "splits": [asdict(s) for s in summaries],
    }
    manifest_path = out_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote export manifest -> {manifest_path}")
    for s in summaries:
        print(
            f"[{s.split}] frames={s.n_frames} images={s.n_images_written} "
            f"annotations={s.n_annotations_written} dropped_unclear={s.n_boxes_dropped_unclear} "
            f"images_without_boxes={s.n_images_without_boxes} ann_json={s.ann_json_path}"
        )


if __name__ == "__main__":
    main()


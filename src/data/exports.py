"""
Dataset export helpers.

Right now this module provides YOLO export while keeping the rest of the
pipeline framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import shutil

import numpy as np
import pandas as pd


UnclearPolicy = Literal["keep_all", "exclude_unclear"]


@dataclass
class YoloExportSummary:
    split: str
    n_frames: int
    n_images_written: int
    n_label_files_written: int
    n_boxes_written: int
    n_boxes_dropped_unclear: int
    n_empty_label_files: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _xyxy_to_yolo_xywh(box: np.ndarray, img_w: float, img_h: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + (w / 2.0)
    yc = y1 + (h / 2.0)

    # Normalize to [0,1]
    xc_n = xc / float(img_w)
    yc_n = yc / float(img_h)
    w_n = w / float(img_w)
    h_n = h / float(img_h)
    return xc_n, yc_n, w_n, h_n


def _link_or_copy_image(src: Path, dst: Path, mode: Literal["symlink", "copy"] = "symlink") -> None:
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    # symlink mode
    dst.symlink_to(src)


def _safe_iter_boxes(xyxy_bboxes) -> list[np.ndarray]:
    if xyxy_bboxes is None:
        return []
    arr = np.asarray(xyxy_bboxes)
    if arr.size == 0:
        return []
    # Expected shape (N, 4), but handle weird object containers too.
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


def export_yolo_split(
    split_name: str,
    frames_df: pd.DataFrame,
    out_dataset_dir: str | Path,
    image_path_col: str = "resized_image_path",
    frame_id_col: str = "frame_id",
    boxes_col: str = "xyxy_bboxes",
    unclear_col: str = "ped_unclear_list",
    img_w_col: str = "new_w",
    img_h_col: str = "new_h",
    unclear_policy: UnclearPolicy = "exclude_unclear",
    class_id: int = 0,
    image_write_mode: Literal["symlink", "copy"] = "symlink",
) -> YoloExportSummary:
    """
    Export one split to YOLO image/label directories.
    """
    out_dataset_dir = Path(out_dataset_dir)
    images_dir = out_dataset_dir / "images" / split_name
    labels_dir = out_dataset_dir / "labels" / split_name
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)

    needed = [frame_id_col, image_path_col, boxes_col, unclear_col, img_w_col, img_h_col]
    for col in needed:
        if col not in frames_df.columns:
            raise ValueError(f"frames_df missing required column '{col}'")

    n_images_written = 0
    n_label_files_written = 0
    n_boxes_written = 0
    n_boxes_dropped_unclear = 0
    n_empty_label_files = 0

    for _, row in frames_df.iterrows():
        frame_id = str(row[frame_id_col]).zfill(6)
        src_image_path = Path(row[image_path_col])
        if not src_image_path.exists():
            # keep going, but skip this sample.
            continue

        dst_image_path = images_dir / f"{frame_id}.jpg"
        _link_or_copy_image(src=src_image_path, dst=dst_image_path, mode=image_write_mode)
        n_images_written += 1

        boxes = _safe_iter_boxes(row[boxes_col])
        unclear_flags = np.asarray(row[unclear_col]) if row[unclear_col] is not None else np.asarray([])

        img_w = float(row[img_w_col])
        img_h = float(row[img_h_col])
        label_lines: list[str] = []

        for i, box in enumerate(boxes):
            is_unclear = bool(unclear_flags[i]) if i < len(unclear_flags) else False
            if unclear_policy == "exclude_unclear" and is_unclear:
                n_boxes_dropped_unclear += 1
                continue

            xc, yc, w, h = _xyxy_to_yolo_xywh(box=box, img_w=img_w, img_h=img_h)

            # final guardrails: skip degenerate/out-of-range boxes
            if w <= 0.0 or h <= 0.0:
                continue
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue

            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            n_boxes_written += 1

        label_path = labels_dir / f"{frame_id}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        n_label_files_written += 1
        if not label_lines:
            n_empty_label_files += 1

    return YoloExportSummary(
        split=split_name,
        n_frames=len(frames_df),
        n_images_written=n_images_written,
        n_label_files_written=n_label_files_written,
        n_boxes_written=n_boxes_written,
        n_boxes_dropped_unclear=n_boxes_dropped_unclear,
        n_empty_label_files=n_empty_label_files,
    )


def write_yolo_dataset_yaml(
    out_dataset_dir: str | Path,
    class_names: dict[int, str] | list[str],
    yaml_path: str | Path | None = None,
) -> Path:
    """
    Write dataset.yaml in standard Ultralytics format.
    """
    out_dataset_dir = Path(out_dataset_dir)
    _ensure_dir(out_dataset_dir)
    if yaml_path is None:
        yaml_path = out_dataset_dir / "dataset.yaml"
    yaml_path = Path(yaml_path)

    if isinstance(class_names, dict):
        names = [name for _, name in sorted(class_names.items(), key=lambda kv: kv[0])]
    else:
        names = list(class_names)

    yaml_lines = [
        f"path: {str(out_dataset_dir.resolve())}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        yaml_lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    return yaml_path

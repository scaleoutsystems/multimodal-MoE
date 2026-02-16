"""
Dataset export helpers for detector-ready datasets.
Adapts canonical parquet data into YOLO/DINO/fusion dataset files.

What this module takes as input
-------------------------------
- Split-filtered frame DataFrames (produced by
  `src.data.index.load_split_frames(...)`)
- Canonical box column(s) from parquet (xyxy format)
- Per-box flags (for example unclear annotations)

What this module does
---------------------
- Converts canonical boxes into detector-specific export format (currently YOLO)
- Writes image/label files for each split
- Writes dataset metadata files required by training frameworks (dataset.yaml)
- Returns export summaries (counts of written/dropped/empty labels)

Current behavior (YOLO)
--------------------------------
Given a DataFrame from `src.data.index.load_split_frames(...)`, this module:
- reads each frame's `xyxy_bboxes` (+ optional `ped_unclear_list` filtering),
- converts each box to YOLO label format:
  `<class_id> <x_center> <y_center> <width> <height>` (normalized to [0,1]),
- writes one label file per image under `labels/<split>/<frame_id>.txt`,
- writes matching images under `images/<split>/`,
- writes `dataset.yaml` with train/val/test paths and class names.

YOLO expects this layout and label syntax
-----------------------------------------
<out_dataset_dir>/
  dataset.yaml
  images/
    train/*.jpg
    val/*.jpg
    test/*.jpg
  labels/
    train/*.txt
    val/*.txt
    test/*.txt

Per-image labels:
- For image `images/train/000123.jpg`, label file is `labels/train/000123.txt`.
- Each line in `000123.txt` is one object:
  `class_id x_center y_center width height`
- Values are normalized by image width/height (range [0,1]).
- For this project (single pedestrian class), `class_id` is always `0`.
- If an image has no kept boxes after filtering, its label file is empty.

-yaml file is a config file for the YOLO dataset and includes:
path: /home/edgelab/multimodal-MoE/outputs/exports/yolo/pedestrian_v1_exclude_unclear
train: images/train (path to the train images)
val: images/val (path to the val images)
test: images/test (path to the test images)
nc: 1 (number of classes)
names:
  0: pedestrian (class name)

Why this module exists
----------------------
Model adapters (YOLO now, DINO later) should not own low-level file export logic.
Keeping export logic here makes data preparation reusable and consistent across
backends, while still allowing backend-specific export functions by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from src.data.bboxes import xyxy_to_yolo


UnclearPolicy = Literal["keep_all", "exclude_unclear"]


@dataclass
class YoloExportSummary:
    """
    Compact summary of what was written for one split export.
    We return this summary to the user so they can see how many images, labels, boxes, etc. were written
    in creating the yolo dataset. 
    """
    split: str
    n_frames: int
    n_images_written: int
    n_label_files_written: int
    n_boxes_written: int
    n_boxes_dropped_unclear: int
    n_empty_label_files: int


def _ensure_dir(path: Path) -> None:
    """
    Create directory (including parents) if it does not exist.

    Input:
        path: Target directory path.

    Output:
        None.

    Why:
        Export code should not fail just because output folders are missing.
    """
    path.mkdir(parents=True, exist_ok=True)


def _symlink_image(src: Path, dst: Path) -> None:
    """
    Place one image into the YOLO export folder as a symlink.

    Input:
        src: Source image file path from parquet.
             Example: /home/edgelab/zod_moe/resized_images/000123.jpg
        dst: Destination image path inside exported YOLO dataset.
             Example: outputs/exports/yolo/.../images/train/000123.jpg

    Output:
        None.

    Why:
        For our current workflow, symlinks are the simplest and most storage-
        efficient way to build YOLO exports without duplicating the full image corpus.
    """
    if dst.exists():
        return
    # create a symlink at dst pointing to src
    dst.symlink_to(src)


def _safe_iter_boxes(xyxy_bboxes) -> list[np.ndarray]:
    """
    Normalize different box container shapes into a list of [x1,y1,x2,y2] arrays.

    Input:
        xyxy_bboxes: Box value from parquet row (can be ndarray/list/object-array).

    Output:
    - normalizes input into a predictable list of (4,) boxes before conversion. 
    --> output is a list[np.ndarray], each array shape (4,).

    Why:
        In our current parquet, boxes are typically consistent ndarray values.
        This helper still acts as a guardrail so export does not break if a
        future parquet version (or preprocessing step) returns a different
        but equivalent container shape.
    """
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
    frames_df: pd.DataFrame, # we get this from the src.data.index.load_split_frames(...) function.
    out_dataset_dir: str | Path, # the path to the output directory for the yolo dataset.
    image_path_col: str = "resized_image_path",
    frame_id_col: str = "frame_id",
    boxes_col: str = "xyxy_bboxes",
    unclear_col: str = "ped_unclear_list", # FORMAT: [True, False, True, False, ...]
    img_w_col: str = "new_w", # width of the resized image (in pixels)
    img_h_col: str = "new_h", # height of the resized image (in pixels)
    unclear_policy: UnclearPolicy = "exclude_unclear", # after assessing the unclear boxes, we decided to exclude them from the training set.
    class_id: int = 0, # class_id=0 just means “this box is pedestrian,” and there are no other classes.
) -> YoloExportSummary:
    """
    Export one split to YOLO image/label directories (contents: .txt label files and .jpg images).

    Input:
        split_name: "train" | "val" | "test" label for output subdirs.
        frames_df: Split-filtered frame DataFrame (from src.data.index.load_split_frames(...)).
        out_dataset_dir: YOLO dataset root path (e.g. /home/edgelab/zod_moe/exports/yolo/train).
        image_path_col/frame_id_col/boxes_col/...: parquet column names.
        unclear_policy: Whether unclear boxes are kept or dropped.
        class_id: YOLO class ID (0 for pedestrian-only).

    Output:
        YoloExportSummary with counts for images/labels/boxes/dropped boxes.

    note:
        This is the single adapter step that converts canonical parquet data
        into detector-specific YOLO files.
    """
    # create the output directories for the images and labels
    out_dataset_dir = Path(out_dataset_dir)
    images_dir = out_dataset_dir / "images" / split_name
    labels_dir = out_dataset_dir / "labels" / split_name
    # ensure the directories exist
    _ensure_dir(images_dir)
    _ensure_dir(labels_dir)
    # check if the required columns are in the DataFrame
    needed = [frame_id_col, image_path_col, boxes_col, unclear_col, img_w_col, img_h_col]
    # We check to ensure that the required columns are in the DataFrame obtained from the src.data.index.load_split_frames(...) function.
    for col in needed:
        if col not in frames_df.columns: # raise an error if the column is not in the DataFrame
            raise ValueError(f"frames_df missing required column '{col}'")
    # initialize the counters
    n_images_written = 0
    n_label_files_written = 0
    n_boxes_written = 0
    n_boxes_dropped_unclear = 0
    n_empty_label_files = 0

    # iterate over the frames in the DataFrame
    for _, row in frames_df.iterrows():
        frame_id = str(row[frame_id_col]).zfill(6)
        src_image_path = Path(row[image_path_col])
        if not src_image_path.exists(): 
            # keep going, but skip this sample.
            continue
        # create the destination image path
        dst_image_path = images_dir / f"{frame_id}.jpg"
        # create a symlink at dst_image_path pointing to src_image_path
        _symlink_image(src=src_image_path, dst=dst_image_path)
        # increment the counter
        n_images_written += 1
        # get the boxes and unclear flags from the DataFrame
        boxes = _safe_iter_boxes(row[boxes_col])
        unclear_flags = np.asarray(row[unclear_col]) if row[unclear_col] is not None else np.asarray([])
        
        img_w = float(row[img_w_col])
        img_h = float(row[img_h_col])
        
        # initialize list of label lines for this keyframe (for .txt label file)
        label_lines: list[str] = []
        
        # iterate over the boxes (iterable within each keyframe row in the DataFrame)
        for i, box in enumerate(boxes):
            is_unclear = bool(unclear_flags[i]) if i < len(unclear_flags) else False
            # if the box is unclear and we are excluding them, skip it
            if unclear_policy == "exclude_unclear" and is_unclear:
                n_boxes_dropped_unclear += 1
                continue

            # Reuse canonical conversion utility so YOLO conversion logic stays in one place.
            xc, yc, w, h = xyxy_to_yolo(
                box=box.tolist(),
                img_w=int(img_w),
                img_h=int(img_h),
            )

            # final guardrails: skip degenerate/out-of-range boxes
            if w <= 0.0 or h <= 0.0:
                continue
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue
            # add the label line to the list for this keyframe
            label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            n_boxes_written += 1

        # write the label file for this keyframe
        label_path = labels_dir / f"{frame_id}.txt"
        # one box per line in the label file
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

    Input:
        out_dataset_dir: YOLO dataset root containing images/labels.
        class_names: Class mapping as dict or ordered list.
        yaml_path: Optional explicit output yaml path.

    Output:
        Path to the written dataset.yaml file.

    Why:
        Ultralytics training/eval expects a dataset YAML descriptor.
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

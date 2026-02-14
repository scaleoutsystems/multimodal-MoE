# scripts/vis_boxes.py
"""
Visual sanity-check for derived pedestrian bounding boxes.

Reads:
  outputs/index/ZODmoe_frames_with_xyxy_bboxes.parquet

Writes:
  outputs/vis/boxes/*.jpg

Goal:
  Quickly confirm that xyxy_bboxes actually wraps pedestrians.
  If boxes look wrong here, do NOT export to YOLO/COCO yet.

note: green boxes are "clear" pedestrians, red boxes are "unclear" pedestrians.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from src.paths import (
    OUTPUTS_DIR,
    ZODMOE_FRAMES_WITH_BOXES_PARQUET,
)


def draw_boxes_on_image(
    img: Image.Image,
    boxes_xyxy: list,
    unclear_flags=None,
    max_boxes: int | None = None,
) -> Image.Image:
    """
    Draw bounding boxes on a PIL image.
    boxes_xyxy is expected to be a list of [x1, y1, x2, y2] in pixel coords.
    unclear_flags is expected to be a per-box boolean sequence, where:
        True  -> unclear pedestrian (red box)
        False -> clear pedestrian (green box)
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)

    boxes_to_draw = boxes_xyxy if max_boxes is None else boxes_xyxy[:max_boxes]
    flags_to_use = unclear_flags if max_boxes is None else unclear_flags[:max_boxes] if unclear_flags is not None else None

    for i, b in enumerate(boxes_to_draw):
        x1, y1, x2, y2 = b
        # Ensure ints for drawing
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

        is_unclear = False
        if flags_to_use is not None and i < len(flags_to_use):
            is_unclear = bool(flags_to_use[i])
        color = "red" if is_unclear else "lime"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=1)

    return out


def main():
    random.seed(0)

    parquet_path = Path(ZODMOE_FRAMES_WITH_BOXES_PARQUET)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Keep only rows with at least one box.
    df = df[df["xyxy_bboxes"].apply(lambda x: x is not None and len(x) > 0)]
    print("frames with >=1 bbox:", len(df))

    if df.empty:
        raise ValueError("No frames with non-empty xyxy_bboxes found.")

    # Keep only frames that contain at least one unclear and at least one clear pedestrian.
    def has_both_clear_and_unclear(row) -> bool:
        boxes = row["xyxy_bboxes"]
        unclear_flags = row.get("ped_unclear_list", None)
        if unclear_flags is None:
            return False

        n = min(len(boxes), len(unclear_flags))
        if n == 0:
            return False

        flags = [bool(unclear_flags[i]) for i in range(n)]
        return any(flags) and any(not f for f in flags)

    df = df[df.apply(has_both_clear_and_unclear, axis=1)]
    print("frames with both clear and unclear pedestrians:", len(df))

    if df.empty:
        raise ValueError("No frames found with both clear and unclear pedestrians.")

    # Sample a small set for visualization
    n_samples = min(20, len(df))
    sample_rows = df.sample(n=n_samples, random_state=0)

    out_dir = OUTPUTS_DIR / "analysis" / "camera" / "detection" / "bbox_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in sample_rows.iterrows():
        frame_id = str(row["frame_id"])
        img_path = Path(row["resized_image_path"])

        if not img_path.exists():
            print(f"[warn] missing image for frame_id={frame_id}: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        boxes = row["xyxy_bboxes"]
        unclear_flags = row.get("ped_unclear_list", None)

        vis = draw_boxes_on_image(img, boxes, unclear_flags=unclear_flags)

        out_path = out_dir / f"{frame_id}_boxes.jpg"
        vis.save(out_path)

    print(f"Saved {n_samples} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()

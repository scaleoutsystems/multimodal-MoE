"""
Add canonical pedestrian bounding boxes to ZODMoE parquet.

Reads:
    outputs/index/ZODmoe_frames.parquet

Writes:
    outputs/index/ZODmoe_frames_with_boxes.parquet
"""

import pandas as pd
from tqdm import tqdm

from src.data.boxes import points_to_xyxy, clamp_xyxy, is_valid_box
from src.paths import OUTPUTS_DIR  # adjust if needed

#run from project root: python scripts/add_bboxes.py
INPUT_PARQUET = "outputs/index/ZODmoe_frames.parquet"
OUTPUT_PARQUET = "outputs/index/ZODmoe_frames_with_boxes.parquet"


def main():
    df = pd.read_parquet(INPUT_PARQUET)

    new_boxes_column = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_w = row["new_w"]
        img_h = row["new_h"]

        frame_boxes = []

        # ped_points_xy_resized should be list of multipoints per pedestrian
        for ped_points in row["ped_points_xy_resized"]:
            box = points_to_xyxy(ped_points)
            if box is None:
                continue

            box = clamp_xyxy(box, img_w, img_h)

            if is_valid_box(box):
                frame_boxes.append(box)

        new_boxes_column.append(frame_boxes)

    df["ped_bboxes_xyxy_resized"] = new_boxes_column

    df.to_parquet(OUTPUT_PARQUET)
    print(f"Saved updated parquet to: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()

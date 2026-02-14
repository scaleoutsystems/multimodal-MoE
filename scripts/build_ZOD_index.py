#!/usr/bin/env python3
"""
Build a frame-level index for ZOD (original) with resized camera_front_dnat images
and resized pedestrian MultiPoint coordinates stored directly in the parquet.

Output:
  outputs/index/zod_moe_frames.parquet

Optionally writes resized images to:
  /home/edgelab/zod_moe/resized_images/<frame_id>.jpg
"""

from pathlib import Path
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
ZOD_ROOT = Path("/home/edgelab/zod_dino_data/train2017")
OUT_PATH = Path("/home/edgelab/multimodal-moe/outputs/index/zod_moe_frames.parquet")

WRITE_RESIZED_IMAGES = True

#this is where we'll store the resized images.
RESIZED_IMG_ROOT = Path("/home/edgelab/zod_moe/resized_images")

ORIG_W, ORIG_H = 3848, 2168
NEW_W, NEW_H = 1248, 704

#scaling factors for resizing the images and the multipoint coordinates.
sx = NEW_W / ORIG_W
sy = NEW_H / ORIG_H


# -----------------------------
# Helpers
# -----------------------------
def read_json(path: Path) -> Any:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

# derives bins for multiclass classification of pedestrian count.
def ped_bin_4(ped_count_clear: int) -> int:
    if ped_count_clear == 0:
        return 0
    if ped_count_clear <= 5:
        return 1
    if ped_count_clear <= 15:
        return 2
    return 3


def find_front_dnat_image(frame_dir: Path) -> Optional[Path]:
    """
    Find the front dnat image for a given frame directory.
    Input: 
      frame_dir: Path to the frame directory
    Output:
      Path to the front dnat image
    """
    cam_dir = frame_dir / "camera_front_dnat"
    if not cam_dir.exists():
        return None

    candidates = list(cam_dir.glob("*.jpg"))

    if not candidates:
        return None

    non_resized = [p for p in candidates if "resized" not in p.name.lower()]

    if len(non_resized) == 1:
        return non_resized[0]

    if len(non_resized) == 0:
        # If everything is "resized", fall back to the first jpg deterministically
        # (or return None if you prefer)
        return None

    # Multiple non-resized images: pick deterministically OR fail loudly
    # I recommend failing loudly so you notice data inconsistencies.
    raise ValueError(
        f"Expected exactly one non-resized DNAT jpg in {cam_dir}, found {len(non_resized)}: "
        f"{[p.name for p in non_resized]}"
    )


def normalize_multipoint_coords(coords: Any) -> List[List[float]]:
    """
    ZOD MultiPoint often looks like:
      [[x,y], [x,y], [x,y], [x,y]]
    but sometimes:
      [[[x,y], ...]]
    Returns a flat list of [x,y] points.
    """
    if coords is None:
        return []
    if not isinstance(coords, list) or len(coords) == 0:
        return []

    # unwrap one level if [[[x,y], ...]]
    if (
        isinstance(coords, list)
        and len(coords) == 1
        and isinstance(coords[0], list)
        and len(coords[0]) > 0
        and isinstance(coords[0][0], (list, tuple))
    ):
        coords = coords[0]

    pts: List[List[float]] = []
    for pt in coords:
        if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
            continue
        x, y = pt[0], pt[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            pts.append([float(x), float(y)])
    return pts


def resize_points_xy(points_xy: List[List[float]]) -> List[List[float]]:
    if not points_xy:
        return []
    arr = np.asarray(points_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    arr2 = np.zeros((arr.shape[0], 2), dtype=np.float32)
    arr2[:, 0] = arr[:, 0] * sx
    arr2[:, 1] = arr[:, 1] * sy
    return arr2.tolist()


def occlusion_bucket(occ: Any) -> str:
    """
    According to observed ZOD distribution:
      (missing), Heavy, Light, Medium, None, VeryHeavy
    Normalize to: missing|none|light|medium|heavy|veryheavy|unknown
    """
    if occ is None:
        return "missing"

    s = str(occ).strip().lower()

    if s == "none":
        return "none"
    if s == "light":
        return "light"
    if s == "medium":
        return "medium"
    if s == "heavy":
        return "heavy"
    if s == "veryheavy":
        return "veryheavy"

    return "unknown"


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    

    if WRITE_RESIZED_IMAGES:
        # Create directory to store resized images. 
        RESIZED_IMG_ROOT.mkdir(parents=True, exist_ok=True)

    # Discover frames via metadata.json
    # list of paths to all metadata.json files in the ZOD_ROOT directory.
    # this is how we iterate over all frames in the dataset.
    # since each single_frame directory contains a metadata.json file, this is how we iterate over all frames in the dataset.
    metadata_files = list(
        tqdm(
            ZOD_ROOT.rglob("metadata.json"),
            desc="Discovering frames",
        )
    )



    # we create a list of dictionaries
    # this will later be converted to a pandas dataframe and saved to the parquet file.
    rows: List[Dict[str, Any]] = []
    
    for meta_path in tqdm(metadata_files, desc="Indexing frames", total=len(metadata_files)):
        # get the frame directory from the metadata file path.
        frame_dir = meta_path.parent
        metadata = read_json(meta_path)
        if metadata is None or not isinstance(metadata, dict):
            continue

        # get the frame id from the metadata file.
        frame_id = metadata.get("frame_id", frame_dir.name)

        # image
        image_path = find_front_dnat_image(frame_dir)
        if image_path is None:
            continue

        resized_image_path = None
        if WRITE_RESIZED_IMAGES:
            # if we're writing resized images, we need to store the path to the resized image.
            resized_image_path = RESIZED_IMG_ROOT / f"{frame_id}.jpg"
            if not resized_image_path.exists():
                try:
                    img = Image.open(image_path).convert("RGB")
                    # resize the image to the new width and height. We use bilinear interpolation for better quality.
                    img_resized = img.resize((NEW_W, NEW_H), resample=Image.BILINEAR)
                    # save the resized image to the resized image directory.
                    img_resized.save(resized_image_path, quality=95)
                except Exception:
                    continue

        # annotations
        obj_path = frame_dir / "annotations" / "object_detection.json"
        # annotations is a list of dictionaries, each representing an object in the frame.
        annotations = read_json(obj_path) if obj_path.exists() else []
        if annotations is None:
            annotations = []
        if isinstance(annotations, dict):
            annotations = annotations.get("annotations", [])

        # counts + lists for pedestrians
        ped_clear = 0
        ped_unclear = 0

        ped_occ_none = 0
        ped_occ_light = 0
        ped_occ_medium = 0
        ped_occ_heavy = 0
        ped_occ_veryheavy = 0
        ped_occ_missing = 0
        ped_occ_unknown = 0

        ped_points_xy_resized: List[List[List[float]]] = []  # [ped][4][2]
        ped_uuid: List[str] = [] # unique identifier for each pedestrian annotation.
        ped_unclear_list: List[bool] = [] # list of booleans indicating whether each pedestrian annotation is unclear.
        ped_occlusion_list: List[str] = [] # list of strings indicating the occlusion level of each pedestrian annotation.

        for obj in annotations:
            # if the object is not a dictionary, skip it.
            if not isinstance(obj, dict):
                continue
            props = obj.get("properties", {}) or {}
            if props.get("class") != "Pedestrian":
                continue

            unclear = bool(props.get("unclear", False))
            if unclear:
                ped_unclear += 1
            else:
                ped_clear += 1

            occ_bucket = occlusion_bucket(props.get("occlusion_ratio", None))
            if occ_bucket == "none":
                ped_occ_none += 1
            elif occ_bucket == "light":
                ped_occ_light += 1
            elif occ_bucket == "medium":
                ped_occ_medium += 1
            elif occ_bucket == "heavy":
                ped_occ_heavy += 1
            elif occ_bucket == "veryheavy":
                ped_occ_veryheavy += 1
            elif occ_bucket == "missing":
                ped_occ_missing += 1
            else:
                ped_occ_unknown += 1

            geom = obj.get("geometry", {}) or {}
            # the normalization is just a formatting safeguard to ensure the coordinates are always a list of lists.
            coords = normalize_multipoint_coords(geom.get("coordinates", None))
            if len(coords) != 4:
                # skip malformed
                continue

            coords_resized = resize_points_xy(coords)
            if len(coords_resized) != 4:
                continue

            ped_points_xy_resized.append(coords_resized)
            ped_uuid.append(str(props.get("annotation_uuid", "")))
            ped_unclear_list.append(unclear)
            ped_occlusion_list.append(occ_bucket)

        ped_bin = ped_bin_4(ped_clear)
        ped_present = int(ped_bin > 0)

        rows.append(
            dict(
                frame_id=frame_id,
                time=metadata.get("time"),
                image_path=str(image_path),
                resized_image_path=str(resized_image_path) if resized_image_path is not None else None,
                orig_w=ORIG_W,
                orig_h=ORIG_H,
                new_w=NEW_W,
                new_h=NEW_H,
                sx=float(sx),
                sy=float(sy),
                scraped_weather=metadata.get("scraped_weather"),
                time_of_day=metadata.get("time_of_day"),
                solar_angle_elevation=metadata.get("solar_angle_elevation"),
                country_code=metadata.get("country_code"),
                road_type=metadata.get("road_type"),
                road_condition=metadata.get("road_condition"),
                ped_count_clear=int(ped_clear),
                ped_count_unclear=int(ped_unclear),
                ped_occ_none=int(ped_occ_none),
                ped_occ_light=int(ped_occ_light),
                ped_occ_medium=int(ped_occ_medium),
                ped_occ_heavy=int(ped_occ_heavy),
                ped_occ_veryheavy=int(ped_occ_veryheavy),
                ped_occ_missing=int(ped_occ_missing),
                ped_occ_unknown=int(ped_occ_unknown),
                ped_uuid=ped_uuid,
                ped_unclear_list=ped_unclear_list,
                ped_occlusion_list=ped_occlusion_list,
                # scaled multipoint coordinates (list[ped][4][2])
                ped_points_xy_resized=ped_points_xy_resized,
                # derived labels
                ped_bin_4=int(ped_bin),
                ped_present=int(ped_present),
            )
        )

    df = pd.DataFrame(rows)
    # Ensure pyarrow handles nested lists cleanly (recommended)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"Saved {len(df)} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()

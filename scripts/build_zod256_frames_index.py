#!/usr/bin/env python3
"""
Build a frame-level index for ZOD256 single_frames.

Output:
  outputs/index/zod256_frames.parquet

One row per keyframe with:
- image path
- selected metadata
- pedestrian counts (clear / unclear)
- derived 4-bin pedestrian label
"""

from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

# create Pathlib objects for ZOD256 single frames root and output path
ZOD_ROOT = Path("/mnt/pr_2018_scaleout_workdir/ZOD256/single_frames")
OUT_PATH = Path("outputs/index/zod256_frames.parquet")

# helper function to read json files
def read_json(path: Path):
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


# helper function to derive 4-bin pedestrian label from "clear" count
# ie from the count of pedestrians that are not labeled as "unclear"
def ped_bin_4(ped_count_clear: int) -> int:
    """
    Derive 4-bin pedestrian label from "clear" count.
    - 0: no pedestrians
    - 1: 1-5 pedestrians
    - 2: 6-15 pedestrians
    - 3: >15 pedestrians
    """
    if ped_count_clear == 0:
        return 0
    if ped_count_clear <= 5:
        return 1
    if ped_count_clear <= 15:
        return 2
    return 3


def main():
    """
    Main function to build the frame-level index.
    """
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # list to store rows for the dataframe (output)
    # each row is a dictionary of the frame-level metadata
    # the dataframe will be saved as a parquet file
    rows = []

    # Discover frames via metadata.json
    # this is a list of all the metadata.json files in the ZOD256 dataset
    metadata_files = ZOD_ROOT.rglob("metadata.json")

    # iterate over the metadata files
    for meta_path in tqdm(metadata_files, desc="Indexing frames"):
        # get the path to the parent directory of the metadata file
        # e.g. 000000/metadata.json -> 000000
        frame_dir = meta_path.parent
        # read the metadata from the metadata file
        metadata = read_json(meta_path)
        if metadata is None:
            continue

        # fixed choice: front DNAT camera
        # first we grab the dnat image from the frame. 
        # although each from should only have one, we grab the first one just in case.
        image_candidates = list(frame_dir.glob("**/camera_front_dnat/*.jpg"))
        if not image_candidates:
            continue
        image_path = image_candidates[0]

        # next we grab the object detection annotations from the frame
        # these are in the annotations/object_detection.json file
        obj_path = frame_dir / "annotations" / "object_detection.json"
        # note that annotations is a list of objects, each a dictionary
        # with keys "properties" and "geometry" where 
        # properties contains things like "class", "unclear", "occlusion ratio", etc. 
        annotations = read_json(obj_path) if obj_path.exists() else []


        # INITIALIZE COUNTS
        ped_clear = 0
        ped_unclear = 0

        # pedestrian occlusion counts (from properties["occlusion_ratio"])
        ped_occ_none = 0
        ped_occ_light = 0
        ped_occ_medium = 0
        ped_occ_heavy = 0
        ped_occ_veryheavy = 0
        ped_occ_unknown = 0


        for obj in annotations:
            props = obj.get("properties", {})
            # only look at pedestrians
            if props.get("class") != "Pedestrian":
                continue
            # counts for clear and unclear pedestrians
            if props.get("unclear", False):
                ped_unclear += 1
            else:
                ped_clear += 1

            # count occlusion
            occ = props.get("occlusion_ratio", None)
            if occ is None:
                ped_occ_unknown += 1
            else:
                occ_s = str(occ).strip().lower()
                if occ_s == "none":
                    ped_occ_none += 1
                elif occ_s == "light":
                    ped_occ_light += 1
                elif occ_s == "medium":
                    ped_occ_medium += 1
                elif occ_s == "heavy":
                    ped_occ_heavy += 1
                elif occ_s == "veryheavy":
                    ped_occ_veryheavy += 1
                else:
                    ped_occ_unknown += 1


        rows.append(
            dict(
                frame_id=metadata.get("frame_id"),
                time=metadata.get("time"),
                image_path=str(image_path),

                scraped_weather=metadata.get("scraped_weather"),
                time_of_day=metadata.get("time_of_day"),
                solar_angle_elevation=metadata.get("solar_angle_elevation"),
                country_code=metadata.get("country_code"),
                road_type=metadata.get("road_type"),
                road_condition=metadata.get("road_condition"),

                ped_count_clear=ped_clear,
                ped_count_unclear=ped_unclear,

                ped_occ_none=ped_occ_none,
                ped_occ_light=ped_occ_light,
                ped_occ_medium=ped_occ_medium,
                ped_occ_heavy=ped_occ_heavy,
                ped_occ_veryheavy=ped_occ_veryheavy,
                ped_occ_unknown=ped_occ_unknown,
                
                ped_bin_4=ped_bin_4(ped_clear), # this is the 4-bin pedestrian label
            )
        )

    # convert the list of rows (dicts) to a pandas dataframe and save to parquet
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Saved {len(df)} frames to {OUT_PATH}")


if __name__ == "__main__":
    main()

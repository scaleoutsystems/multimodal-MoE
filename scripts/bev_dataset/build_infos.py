#!/usr/bin/env python3
"""
Build MMDetection3D-compatible info pickles from pre-processed ZOD-MoE artifacts.

Position in pipeline
--------------------
This is the *second* dataset script.  The *first* is
``scripts/build_zod_moe_dataset.py``, which does all heavy lifting:

    ZOD raw data
        ↓  (undistort, deskew, FoV-filter, extract labels)
    /mnt/.../zod_moe/{bev_images, lidar, calibs, labels, index}

This script reads those finished artifacts and reshapes them into the
NuScenes-style directory layout and info pickle format expected by
OpenMMLab MMDetection3D / BEVFusion:

    /mnt/.../zod_moe/zod_nuscenes/
    ├── samples/
    │   ├── CAM_FRONT/   (symlinks to source images)
    │   └── LIDAR_TOP/   (symlinks to source LiDAR bins)
    ├── infos/
    │   ├── zod_nuscenes_infos_train.pkl
    │   ├── zod_nuscenes_infos_val.pkl
    │   └── zod_nuscenes_infos_test.pkl
    ├── splits/
    │   ├── train.txt
    │   ├── val.txt
    │   └── test.txt
    └── index/
        ├── zod_moe_dataset_with_weather_group.parquet
        └── build_infos_failures.csv

What it reads
-------------
* The parquet index produced by the build pipeline.
* Pre-existing train/val/test split text files (one frame_id per line).
* Per-frame ``.npz`` calibration files  (K_final, camera2ego, …).
* Per-frame ``.json`` label files       (instances with box_3d, …).

What it writes
--------------
* Symlinks (or copies) of images and point clouds into samples/.
* Three pickle files whose schema matches ``mmdet3d`` ``data_list`` format.
* Copies of the parquet and split files for self-contained reproducibility.
* A CSV failure log for any frames that could not be processed.

Geometry note
-------------
Both the saved point clouds and the saved 3D boxes are already in the
**ego frame** (ISO-8855: X forward, Y left, Z up).  We treat ego as the
working "LiDAR" reference frame throughout, so the metadata field called
``lidar2cam`` is actually ``ego2cam = inverse(camera2ego)``.

Box convention
--------------
The build script stores ``box_3d`` as
``[x, y, z_bottom, dx, dy, dz, yaw]`` (z = bottom-centre of the box).
MMDetection3D expects centre-z, so we convert:
``z_center = z_bottom + dz / 2``.

Example
-------
python scripts/dataset/build_infos.py

python scripts/dataset/build_infos.py \\
    --source-root /mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe \\
    --final-root  /mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/zod_nuscenes \\
    --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================================
# Defaults — change these if your mount point differs
# ==================================================================
_MNT = Path("/mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe")

DEFAULT_SOURCE_ROOT = _MNT
DEFAULT_FINAL_ROOT = _MNT / "zod_nuscenes"
DEFAULT_INDEX_PARQUET = _MNT / "index" / "zod_moe_dataset_with_weather_group.parquet"
DEFAULT_TRAIN_SPLIT = _MNT / "index" / "train.txt"
DEFAULT_VAL_SPLIT = _MNT / "index" / "val.txt"
DEFAULT_TEST_SPLIT = _MNT / "index" / "test.txt"
DEFAULT_TRAIN_OUT = DEFAULT_FINAL_ROOT / "infos" / "zod_nuscenes_infos_train.pkl"
DEFAULT_VAL_OUT = DEFAULT_FINAL_ROOT / "infos" / "zod_nuscenes_infos_val.pkl"
DEFAULT_TEST_OUT = DEFAULT_FINAL_ROOT / "infos" / "zod_nuscenes_infos_test.pkl"


# ==================================================================
# CLI
# ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build MMDetection3D info pickles from ZOD-MoE artifacts.",
    )
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    p.add_argument("--index-parquet", type=Path, default=DEFAULT_INDEX_PARQUET)
    p.add_argument("--train-split", type=Path, default=DEFAULT_TRAIN_SPLIT)
    p.add_argument("--val-split", type=Path, default=DEFAULT_VAL_SPLIT)
    p.add_argument("--test-split", type=Path, default=DEFAULT_TEST_SPLIT)
    p.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN_OUT)
    p.add_argument("--val-out", type=Path, default=DEFAULT_VAL_OUT)
    p.add_argument("--test-out", type=Path, default=DEFAULT_TEST_OUT)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: process at most N frames per split (smoke test).",
    )
    p.add_argument(
        "--force-copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )
    return p.parse_args()


# ==================================================================
# Helper utilities
# ==================================================================
def ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink_or_copy(src: Path, dst: Path, *, force_copy: bool = False) -> None:
    """Create a symlink *dst → src*, falling back to copy on failure.

    If *dst* already exists (symlink or file) it is left untouched.
    """
    if dst.exists() or dst.is_symlink():
        return

    if force_copy:
        shutil.copy2(src, dst)
        return

    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def read_split_ids(txt_path: Path) -> list[str]:
    """Read a split file and return a list of frame_id strings."""
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------
def load_calib(npz_path: str | Path) -> dict:
    """Load a calibration ``.npz`` and return the arrays we need.

    Returns a dict with:
        K_final       — (3, 3) float32  camera intrinsic (after undistort+resize)
        camera2ego    — (4, 4) float32  extrinsic transform
        image_size    — (2,)   int32    [height, width]
    """
    d = np.load(npz_path, allow_pickle=True)
    return {
        "K_final": d["K_final"].astype(np.float32),
        "camera2ego": d["camera2ego"].astype(np.float32),
        "image_size": d["image_size"],
    }


# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------
def load_label(json_path: str | Path) -> list[dict]:
    """Load a label JSON and return the list of instance dicts."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("instances", [])


def convert_box_bottom_to_center(box_3d: list[float]) -> list[float]:
    """Convert [x, y, z_bottom, dx, dy, dz, yaw] → [x, y, z_center, …].

    The build script stores z as the bottom-centre of the 3D box.
    MMDetection3D expects the geometric centre.
    """
    x, y, z_bottom, dx, dy, dz, yaw = box_3d
    z_center = z_bottom + dz / 2.0
    return [x, y, z_center, dx, dy, dz, yaw]


def make_instance_entry(inst: dict) -> dict:
    """Convert one raw label instance into the MMDet3D format.

    Input keys (from build script):
        label_3d   — int (always 0 for Pedestrian)
        box_3d     — [x, y, z_bottom, dx, dy, dz, yaw]

    Output keys (MMDet3D convention):
        bbox_label_3d  — int, hardcoded to 7 (the nuScenes class index
                          for "pedestrian") so NuScenesDataset's label
                          mapping works out of the box
        bbox_3d        — [x, y, z_center, dx, dy, dz, yaw]
        bbox_3d_isvalid — bool, always True so NuScenesDataset works
                          with ``use_valid_flag=True``
    """
    return {
        "bbox_label_3d": 7,
        "bbox_3d": convert_box_bottom_to_center(inst["box_3d"]),
        "bbox_3d_isvalid": True,
        "num_lidar_pts": 1,
    }


# ------------------------------------------------------------------
# Context metadata (for later MoE routing)
# ------------------------------------------------------------------
def make_context_entry(row: pd.Series) -> dict:
    """Extract context fields from a parquet row as raw strings."""
    return {
        "solar_context_bin": str(row.get("solar_context_bin", "")),
        "weather_group": str(row.get("weather_group", "")),
        "road_type": str(row.get("road_type", "")),
    }


# ------------------------------------------------------------------
# Per-sample info dict
# ------------------------------------------------------------------
def make_sample_entry(
    frame_id: str,
    row: pd.Series,
    final_img_rel: str,
    final_lidar_rel: str,
) -> dict:
    """Build one MMDetection3D sample dict.

    Parameters
    ----------
    frame_id : str
        The frame identifier (passed explicitly because the DataFrame is
        indexed by frame_id, so it's no longer a regular column).
    row : pd.Series
        One row from the parquet (must contain calib/label paths + context).
    final_img_rel : str
        Relative path inside the final dataset root, e.g.
        ``"samples/CAM_FRONT/050000.png"``.
    final_lidar_rel : str
        Relative path, e.g. ``"samples/LIDAR_TOP/050000.bin"``.

    Returns
    -------
    dict   with keys: sample_idx, token, lidar_points, images, instances, context
    """

    # --- calibration ---
    calib = load_calib(row["calib_file_path"])
    K_final = calib["K_final"]                     # (3, 3) float32
    camera2ego = calib["camera2ego"]               # (4, 4) float32

    # lidar2cam = ego2cam = inverse(camera2ego)
    # because our saved LiDAR & boxes are already in ego frame
    lidar2cam = np.linalg.inv(camera2ego).astype(np.float32)

    assert K_final.shape == (3, 3), f"Expected cam2img shape (3,3), got {K_final.shape}"
    assert lidar2cam.shape == (4, 4), f"Expected lidar2cam shape (4,4), got {lidar2cam.shape}"

    # --- labels ---
    raw_instances = load_label(row["label_file_path"])
    instances = [make_instance_entry(inst) for inst in raw_instances]

    return {
        "sample_idx": frame_id,
        "token": frame_id,

        "lidar_points": {
            "lidar_path": final_lidar_rel,
            "num_pts_feats": 4,
        },

        "images": {
            "CAM_FRONT": {
                "img_path": final_img_rel,
                "lidar2cam": lidar2cam,
                "cam2img": K_final,
            }
        },

        "instances": instances,

        "context": make_context_entry(row),
    }


# ------------------------------------------------------------------
# Top-level info object
# ------------------------------------------------------------------
METAINFO = {
    "classes": ("Pedestrian",),
    "version": "zod_pedestrian_singlecam_singlelidar",
}


def build_info_object(data_list: list[dict]) -> dict:
    """Wrap a list of sample dicts into the MMDet3D pickle schema."""
    return {
        "metainfo": METAINFO,
        "data_list": data_list,
    }


def save_pickle(obj: dict, out_path: Path) -> None:
    """Write *obj* to a pickle file."""
    ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_failure_log(records: list[dict], out_path: Path) -> None:
    """Write failure records to CSV (or create an empty file)."""
    ensure_dir(out_path.parent)
    df = pd.DataFrame(records, columns=["frame_id", "split", "error"])
    df.to_csv(out_path, index=False)


# ==================================================================
# Main
# ==================================================================
def main() -> None:
    args = parse_args()

    # --- Validate inputs ---
    assert args.index_parquet.exists(), f"Parquet not found: {args.index_parquet}"
    for sp, name in [
        (args.train_split, "train"),
        (args.val_split, "val"),
        (args.test_split, "test"),
    ]:
        assert sp.exists(), f"{name} split not found: {sp}"

    # ----------------------------------------------------------
    # 1. Read parquet
    # ----------------------------------------------------------
    df = pd.read_parquet(args.index_parquet)
    df["frame_id"] = df["frame_id"].astype(str)
    parquet_ids = set(df["frame_id"])
    print(f"Parquet: {len(df)} rows,  {len(parquet_ids)} unique frame_ids")

    # ----------------------------------------------------------
    # 2. Read splits
    # ----------------------------------------------------------
    split_map: dict[str, list[str]] = {
        "train": read_split_ids(args.train_split),
        "val": read_split_ids(args.val_split),
        "test": read_split_ids(args.test_split),
    }
    for name, ids in split_map.items():
        missing = set(ids) - parquet_ids
        if missing:
            print(f"  WARNING: {len(missing)} {name} IDs not found in parquet")
        if args.limit:
            split_map[name] = ids[: args.limit]
        print(f"  {name}: {len(split_map[name])} frame_ids")

    # ----------------------------------------------------------
    # 3. Create final directory structure
    # ----------------------------------------------------------
    final = args.final_root
    cam_dir = final / "samples" / "CAM_FRONT"
    lid_dir = final / "samples" / "LIDAR_TOP"
    info_dir = final / "infos"
    split_dir = final / "splits"
    idx_dir = final / "index"

    for d in (cam_dir, lid_dir, info_dir, split_dir, idx_dir):
        ensure_dir(d)

    # ----------------------------------------------------------
    # 4. Copy parquet and split files into final tree
    # ----------------------------------------------------------
    dst_pq = idx_dir / args.index_parquet.name
    if not dst_pq.exists():
        shutil.copy2(args.index_parquet, dst_pq)
        print(f"Copied parquet → {dst_pq}")

    for name in ("train", "val", "test"):
        src_split = getattr(args, f"{name}_split")
        dst_split = split_dir / f"{name}.txt"
        if not dst_split.exists():
            shutil.copy2(src_split, dst_split)
            print(f"Copied {name}.txt → {dst_split}")

    # ----------------------------------------------------------
    # 5. Build index for fast row lookup
    # ----------------------------------------------------------
    df_indexed = df.set_index("frame_id")

    # ----------------------------------------------------------
    # 6. Process each split
    # ----------------------------------------------------------
    out_paths = {
        "train": args.train_out,
        "val": args.val_out,
        "test": args.test_out,
    }

    failures: list[dict] = []
    summary: dict[str, dict[str, int]] = {}

    for split_name, frame_ids in split_map.items():
        print(f"\nProcessing {split_name} ({len(frame_ids)} frames) …")
        data_list: list[dict] = []
        n_skip = 0
        n_missing = 0

        for fid in frame_ids:
            # ---- check existence in parquet ----
            if fid not in df_indexed.index:
                n_missing += 1
                failures.append({
                    "frame_id": fid,
                    "split": split_name,
                    "error": "frame_id not found in parquet",
                })
                continue

            row = df_indexed.loc[fid]
            # If duplicate frame_ids exist, take the first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            try:
                # ---- source file paths from parquet ----
                src_img = Path(str(row["final_img_file_path"]))
                src_lid = Path(str(row["final_points_file_path"]))

                if not src_img.exists():
                    raise FileNotFoundError(f"Image not found: {src_img}")
                if not src_lid.exists():
                    raise FileNotFoundError(f"LiDAR not found: {src_lid}")

                # ---- create symlinks / copies ----
                dst_img = cam_dir / f"{fid}.png"
                dst_lid = lid_dir / f"{fid}.bin"

                safe_symlink_or_copy(src_img, dst_img, force_copy=args.force_copy)
                safe_symlink_or_copy(src_lid, dst_lid, force_copy=args.force_copy)

                # ---- build sample entry ----
                final_img_rel = f"samples/CAM_FRONT/{fid}.png"
                final_lid_rel = f"samples/LIDAR_TOP/{fid}.bin"

                entry = make_sample_entry(fid, row, final_img_rel, final_lid_rel)
                data_list.append(entry)

            except Exception as exc:
                n_skip += 1
                failures.append({
                    "frame_id": fid,
                    "split": split_name,
                    "error": traceback.format_exception_only(type(exc), exc)[0].strip(),
                })

        # ---- save pickle ----
        info_obj = build_info_object(data_list)
        out_path = out_paths[split_name]
        save_pickle(info_obj, out_path)
        print(f"  → {out_path}  ({len(data_list)} samples)")

        summary[split_name] = {
            "requested": len(frame_ids),
            "written": len(data_list),
            "missing": n_missing,
            "skipped": n_skip,
        }

    # ----------------------------------------------------------
    # 7. Write failure log
    # ----------------------------------------------------------
    fail_path = idx_dir / "build_infos_failures.csv"
    write_failure_log(failures, fail_path)
    print(f"\nFailure log: {fail_path}  ({len(failures)} entries)")

    # ----------------------------------------------------------
    # 8. Final summary
    # ----------------------------------------------------------
    link_mode = "copy" if args.force_copy else "symlink"
    print("\n" + "=" * 60)
    print("BUILD_INFOS SUMMARY")
    print("=" * 60)
    print(f"Source root:        {args.source_root}")
    print(f"Final root:         {args.final_root}")
    print(f"Parquet rows:       {len(df)}")
    print(f"File mode:          {link_mode}")
    total_written = 0
    total_fail = 0
    for name in ("train", "val", "test"):
        s = summary[name]
        print(
            f"  {name:6s}  requested={s['requested']:>6d}  "
            f"written={s['written']:>6d}  "
            f"missing={s['missing']:>4d}  "
            f"skipped={s['skipped']:>4d}"
        )
        total_written += s["written"]
        total_fail += s["missing"] + s["skipped"]
    print(f"Total written:      {total_written}")
    print(f"Total failures:     {total_fail}")
    if args.limit:
        print(f"(--limit {args.limit} was active)")

    # Quick sanity: confirm new per-instance fields are present
    for name in ("train", "val", "test"):
        pkl = out_paths[name]
        if pkl.exists():
            with open(pkl, "rb") as f:
                check = pickle.load(f)
            dl = check.get("data_list", [])
            for sample in dl:
                insts = sample.get("instances", [])
                if insts:
                    ex = insts[0]
                    print(f"\n  Instance field check ({name}, sample {sample['sample_idx']}):")
                    print(f"    bbox_3d_isvalid : {ex.get('bbox_3d_isvalid', 'MISSING')}")
                    print(f"    num_lidar_pts   : {ex.get('num_lidar_pts', 'MISSING')}")
                    break

    print("=" * 60)


if __name__ == "__main__":
    main()

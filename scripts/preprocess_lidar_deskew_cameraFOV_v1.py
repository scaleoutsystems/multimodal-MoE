from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm
from zod.constants import Camera, Lidar
from zod.data_classes.frame import ZodFrame
from zod.data_classes.info import Information
from zod.data_classes.sensor import LidarData
from zod.utils.compensation import motion_compensate_pointwise
from zod.utils.geometry import get_points_in_camera_fov, transform_points
from zod.visualization.lidar_on_image import get_3d_transform_camera_lidar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import paths  # noqa: E402

CACHE_VERSION = "lidar_deskewed_cameraFOV_v1"
META_HEADER = [
    "frame_id",
    "keyframe_time_iso",
    "camera_timestamp_epoch_s",
    "nearest_lidar_filename",
    "lidar_sweep_time_iso",
    "abs_dt_to_keyframe_s",
    "raw_lidar_path",
    "out_npy_path",
    "n_points_raw",
    "n_points_positive_depth",
    "n_points_fov",
    "dtype",
    "cache_version",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def nearest_lidar_filename(frame_dir: Path) -> tuple[str | None, str | None, float | None]:
    """Match notebook helper: pick the single nearest sweep to keyframe_time."""
    info = _read_json(frame_dir / "info.json")
    lidar_dir = frame_dir / "lidar_velodyne"

    if not isinstance(info, dict) or "keyframe_time" not in info or not lidar_dir.exists():
        return None, None, None

    keyframe_timestamp = _iso_to_dt(info["keyframe_time"])
    best_name: str | None = None
    best_timestamp_str: str | None = None
    best_abs_dt = float("inf")

    for p in lidar_dir.glob("*.npy"):
        timestamp_str = p.stem.rsplit("_", 1)[-1]
        try:
            delta_time_s = (_iso_to_dt(timestamp_str) - keyframe_timestamp).total_seconds()
        except Exception:
            continue
        abs_dt = abs(delta_time_s)
        if abs_dt < best_abs_dt:
            best_abs_dt = abs_dt
            best_name = p.name
            best_timestamp_str = timestamp_str

    if best_name is None:
        return None, None, None
    return best_name, best_timestamp_str, float(best_abs_dt)


def make_zodframe_from_raw(frame_dir: Path) -> ZodFrame:
    """Notebook-equivalent helper to create ZodFrame from raw folder layout."""
    info_dict = json.loads((frame_dir / "info.json").read_text())
    frame_id = info_dict.get("id", frame_dir.name)
    prefix = f"single_frames/{frame_id}/"

    def fix_path(p: str | None) -> str | None:
        if p is None:
            return None
        if p.startswith(prefix):
            return p[len(prefix) :]
        if p.startswith("single_frames/"):
            return p.split("/", 2)[-1]
        return p

    for k in ["calibration_path", "ego_motion_path", "metadata_path", "oxts_path", "vehicle_data_path"]:
        if info_dict.get(k) is not None:
            info_dict[k] = fix_path(info_dict[k])

    for ann in info_dict.get("annotations", {}).values():
        if ann.get("filepath") is not None:
            ann["filepath"] = fix_path(ann["filepath"])

    for arr in info_dict.get("camera_frames", {}).values():
        for x in arr:
            if x.get("filepath") is not None:
                x["filepath"] = fix_path(x["filepath"])

    for arr in info_dict.get("lidar_frames", {}).values():
        for x in arr:
            if x.get("filepath") is not None:
                x["filepath"] = fix_path(x["filepath"])

    info = Information.from_dict(info_dict)
    info.convert_paths_to_absolute(str(frame_dir))
    return ZodFrame(info)


def discover_frames(dataset_root: Path) -> list[Path]:
    frame_dirs: list[Path] = []
    with os.scandir(dataset_root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            p = Path(entry.path)
            if (p / "info.json").exists() and (p / "lidar_velodyne").exists():
                frame_dirs.append(p)
    return sorted(frame_dirs, key=lambda x: x.name)


def write_config_yaml(out_root: Path, dataset_root: Path) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    config_path = out_root / "config.yaml"
    yaml_text = "\n".join(
        [
            f"cache_version: {CACHE_VERSION}",
            "output_fields: [x, y, z, intensity]",
            "dtype: float32",
            "coordinate_frame: ego_at_camera_timestamp",
            "lidar_sweep_selection: nearest_to_keyframe_time",
            "motion_compensation: motion_compensate_pointwise(target_timestamp=camera_timestamp)",
            "fov_filter: positive_depth + angular_fov (get_points_in_camera_fov)",
            "camera: FRONT",
            "lidar: VELODYNE",
            f"dataset_root: {dataset_root}",
            f"created_at: {created_at}",
            "code_ref: scripts/preprocess_lidar_deskew_cameraFOV_v1.py",
            "",
        ]
    )
    config_path.write_text(yaml_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset-wide deskewed LiDAR camera-FOV cache for BEV fusion.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=paths.ZOD_DINO_DATA / "train2017",
        help="Root directory containing frame folders (default: paths.ZOD_DINO_DATA / train2017).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/edgelab/zod_moe/lidar/lidar_deskewed_cameraFOV_v1"),
        help="Output root for cache, config, and metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-frame npy outputs.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for quick testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    velodyne_out = out_root / "velodyne"
    velodyne_out.mkdir(parents=True, exist_ok=True)

    write_config_yaml(out_root, dataset_root)
    meta_path = out_root / "meta.csv"

    print(f"Discovering frame directories under: {dataset_root}", flush=True)
    frame_dirs = discover_frames(dataset_root)
    print(f"Discovered {len(frame_dirs)} valid frame directories.", flush=True)
    if args.max_frames is not None:
        frame_dirs = frame_dirs[: args.max_frames]

    processed = 0
    skipped = 0
    failed = 0
    fov_counts: list[int] = []
    rows: list[dict[str, Any]] = []

    for idx, frame_dir in enumerate(tqdm(frame_dirs, desc="Building LiDAR cache", unit="frame"), start=1):
        try:
            zod_frame = make_zodframe_from_raw(frame_dir)
            # Tie cache naming to keyframe folder ids (e.g., 000000.npy).
            frame_id = frame_dir.name

            nearest_name, lidar_sweep_time_iso, abs_dt_to_keyframe_s = nearest_lidar_filename(frame_dir)
            if nearest_name is None or lidar_sweep_time_iso is None or abs_dt_to_keyframe_s is None:
                raise ValueError("Could not select nearest LiDAR sweep.")

            front_camera_frame = zod_frame.get_camera_frame()
            camera_timestamp = float(front_camera_frame.time.timestamp())

            raw_lidar_path = frame_dir / "lidar_velodyne" / nearest_name
            raw_lidar = LidarData.from_npy(raw_lidar_path)

            ego_motion = zod_frame.ego_motion
            calib = zod_frame.calibration
            lidar_calib = calib.lidars[Lidar.VELODYNE]

            aligned_lidar = motion_compensate_pointwise(
                raw_lidar,
                ego_motion,
                lidar_calib,
                target_timestamp=camera_timestamp,
            )
            points_lidar = aligned_lidar.points

            aligned_lidar_ego = aligned_lidar.copy()
            aligned_lidar_ego.transform(lidar_calib.extrinsics)
            points_ego = aligned_lidar_ego.points

            t_lidar_to_cam = get_3d_transform_camera_lidar(
                calib, lidar=Lidar.VELODYNE, camera=Camera.FRONT
            ).transform
            points_cam = transform_points(points_lidar, t_lidar_to_cam)

            positive_depth_mask = points_cam[:, 2] > 0
            points_cam_positive = points_cam[positive_depth_mask]
            _, fov_mask_positive = get_points_in_camera_fov(
                calib.cameras[Camera.FRONT].field_of_view,
                points_cam_positive,
            )

            fov_mask_full = np.zeros(points_cam.shape[0], dtype=bool)
            fov_mask_full[positive_depth_mask] = fov_mask_positive

            fov_xyz_ego = points_ego[fov_mask_full]
            fov_intensity = aligned_lidar.intensity[fov_mask_full]
            xyzi_ego = np.hstack(
                [fov_xyz_ego.astype(np.float32), fov_intensity.astype(np.float32)[:, None]]
            )

            out_path = velodyne_out / f"{frame_id}.npy"
            if out_path.exists() and not args.overwrite:
                skipped += 1
            else:
                np.save(out_path, xyzi_ego.astype(np.float32))

            rows.append(
                {
                    "frame_id": frame_id,
                    "keyframe_time_iso": zod_frame.info.keyframe_time.isoformat(),
                    "camera_timestamp_epoch_s": f"{camera_timestamp:.6f}",
                    "nearest_lidar_filename": nearest_name,
                    "lidar_sweep_time_iso": lidar_sweep_time_iso,
                    "abs_dt_to_keyframe_s": f"{abs_dt_to_keyframe_s:.6f}",
                    "raw_lidar_path": str(raw_lidar_path),
                    "out_npy_path": str(out_path),
                    "n_points_raw": int(points_lidar.shape[0]),
                    "n_points_positive_depth": int(points_cam_positive.shape[0]),
                    "n_points_fov": int(xyzi_ego.shape[0]),
                    "dtype": "float32",
                    "cache_version": CACHE_VERSION,
                }
            )
            fov_counts.append(int(xyzi_ego.shape[0]))
            processed += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"[FAIL] frame={frame_dir.name}: {exc}")

        if idx % 100 == 0:
            mean_fov = float(np.mean(fov_counts)) if fov_counts else 0.0
            tqdm.write(
                f"[PROGRESS] processed={processed} skipped={skipped} failed={failed} mean_n_points_fov={mean_fov:.2f}"
            )

    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    saved_paths = [Path(row["out_npy_path"]) for row in rows if Path(row["out_npy_path"]).exists()]
    if saved_paths:
        sample_n = min(5, len(saved_paths))
        for npy_path in random.sample(saved_paths, sample_n):
            arr = np.load(npy_path)
            assert arr.ndim == 2 and arr.shape[1] == 4, f"Invalid shape for {npy_path}: {arr.shape}"
            assert arr.dtype == np.float32, f"Invalid dtype for {npy_path}: {arr.dtype}"
        print(f"Sanity check passed on {sample_n} random saved files.")
    else:
        print("No saved files found for sanity check.")

    if rows:
        k_vals = np.array([int(r["n_points_fov"]) for r in rows], dtype=np.int64)
        print(
            f"K stats (n_points_fov): min={int(k_vals.min())}, median={float(np.median(k_vals)):.1f}, max={int(k_vals.max())}"
        )
    else:
        print("No successful rows in meta.csv.")

    print(
        f"Done. processed={processed}, skipped={skipped}, failed={failed}, "
        f"meta_csv={meta_path}, config={out_root / 'config.yaml'}"
    )


if __name__ == "__main__":
    main()

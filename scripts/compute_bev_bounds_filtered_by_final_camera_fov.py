#!/usr/bin/env python3
"""Compute BEV pedestrian bounds after final camera-FoV filtering.

Pipeline per keyframe:
1) Load frame calibration with ZOD SDK.
2) Build frame-specific transforms (LiDAR->camera and LiDAR->ego).
3) Compute frame-specific final intrinsics:
   raw K,D -> OpenCV fisheye new_K (balance) -> resize to final WxH.
4) Keep only pedestrian 3D centers that project inside final image:
   z_cam > 0 and (u,v) in [0,W) x [0,H).
5) Convert kept centers LiDAR->ego and aggregate dataset-wide.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from zod.data_classes.frame import ZodFrame
    from zod.data_classes.info import Information
    from zod.constants import Camera, Lidar
    from zod.constants import AnnotationProject
    from zod.data_classes.calibration import Calibration
    from zod.utils.geometry import transform_points
    from zod.visualization.lidar_on_image import get_3d_transform_camera_lidar

    ZOD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment specific
    Camera = None  # type: ignore[assignment]
    Lidar = None  # type: ignore[assignment]
    AnnotationProject = None  # type: ignore[assignment]
    Calibration = None  # type: ignore[assignment]
    Information = None  # type: ignore[assignment]
    ZodFrame = None  # type: ignore[assignment]
    transform_points = None  # type: ignore[assignment]
    get_3d_transform_camera_lidar = None  # type: ignore[assignment]
    ZOD_IMPORT_ERROR = exc


DEFAULT_DATASET_ROOT = Path("/home/edgelab/zod_dino_data/train2017")
DEFAULT_OUTPUT_DIR = Path("/home/edgelab/multimodal-MoE/outputs/analysis")

BALANCE_DEFAULT = 0.3
FINAL_WIDTH_DEFAULT = 1248
FINAL_HEIGHT_DEFAULT = 704
GRID_RESOLUTION_DEFAULT = 0.5
MARGIN_METERS_DEFAULT = 2.0
X_MAX_FIXED_DEFAULT = 250.0
MAX_FRAMES_DEFAULT = 10000

EPS_Z = 1e-6
PRINT_EVERY = 500


@dataclass
class Counters:
    total_frames_discovered: int = 0
    frames_processed: int = 0
    failed_frames: int = 0
    total_objects: int = 0
    pedestrian_objects: int = 0
    ped_objects_with_3d: int = 0
    ped_centers_inside_final_fov: int = 0
    ped_centers_outside_final_fov: int = 0
    ped_centers_behind_camera: int = 0


@dataclass
class Timings:
    total_sec: float = 0.0
    frame_loop_sec: float = 0.0
    read_calib_sec: float = 0.0
    compute_intrinsics_sec: float = 0.0
    load_annotations_sec: float = 0.0
    obj_loop_sec: float = 0.0


def iter_frame_dirs(dataset_root: Path, max_frames: Optional[int]) -> list[Path]:
    """Return unsorted valid frame directories only."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    frame_dirs: list[Path] = []
    for p in dataset_root.iterdir():
        if not p.is_dir():
            continue
        if (
            (p / "info.json").exists()
            and (p / "calibration.json").exists()
            and (p / "annotations" / "object_detection.json").exists()
        ):
            frame_dirs.append(p)
            if max_frames is not None and len(frame_dirs) >= max_frames:
                break
    return frame_dirs


def make_zodframe_from_raw(frame_dir: Path) -> Any:
    """Build ZodFrame from single-frame raw directory layout."""
    info_dict = json.loads((frame_dir / "info.json").read_text())
    frame_id = info_dict.get("id", frame_dir.name)
    prefix = f"single_frames/{frame_id}/"

    def fix_path(path_str: Optional[str]) -> Optional[str]:
        if path_str is None:
            return None
        if path_str.startswith(prefix):
            return path_str[len(prefix) :]
        if path_str.startswith("single_frames/"):
            return path_str.split("/", 2)[-1]
        return path_str

    for key in [
        "calibration_path",
        "ego_motion_path",
        "metadata_path",
        "oxts_path",
        "vehicle_data_path",
    ]:
        if info_dict.get(key) is not None:
            info_dict[key] = fix_path(info_dict[key])

    for ann in info_dict.get("annotations", {}).values():
        if ann.get("filepath") is not None:
            ann["filepath"] = fix_path(ann["filepath"])

    for frames in info_dict.get("camera_frames", {}).values():
        for item in frames:
            if item.get("filepath") is not None:
                item["filepath"] = fix_path(item["filepath"])

    for frames in info_dict.get("lidar_frames", {}).values():
        for item in frames:
            if item.get("filepath") is not None:
                item["filepath"] = fix_path(item["filepath"])

    info = Information.from_dict(info_dict)  # type: ignore[operator]
    info.convert_paths_to_absolute(str(frame_dir))
    return ZodFrame(info)  # type: ignore[operator]


def _get_raw_image_size(cam_calib: Any) -> tuple[int, int]:
    """Extract raw image size (w,h) from SDK camera calibration object."""
    dims = getattr(cam_calib, "image_dimensions", None)
    if dims is None:
        raise ValueError("Camera calibration object missing image_dimensions.")
    if len(dims) < 2:
        raise ValueError(f"Invalid image_dimensions: {dims}")
    raw_w, raw_h = int(dims[0]), int(dims[1])
    if raw_w <= 0 or raw_h <= 0:
        raise ValueError(f"Invalid raw image size: {(raw_w, raw_h)}")
    return raw_w, raw_h


def compute_final_camera_intrinsics(
    cam_calib: Any,
    final_width: int = FINAL_WIDTH_DEFAULT,
    final_height: int = FINAL_HEIGHT_DEFAULT,
    balance: float = BALANCE_DEFAULT,
) -> np.ndarray:
    """Compute frame-specific K_final = S @ new_K."""
    k_raw = np.asarray(cam_calib.intrinsics[:3, :3], dtype=np.float64)
    d_raw = np.asarray(cam_calib.distortion, dtype=np.float64).reshape(-1)
    if d_raw.shape[0] != 4:
        raise ValueError(f"Expected 4 fisheye distortion coefficients, got {d_raw.shape[0]}")
    d_raw = d_raw.reshape(4, 1)
    raw_w, raw_h = _get_raw_image_size(cam_calib)

    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k_raw,
        d_raw,
        (raw_w, raw_h),
        np.eye(3, dtype=np.float64),
        balance=balance,
    )
    sx = final_width / float(raw_w)
    sy = final_height / float(raw_h)
    scale = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return scale @ new_k


def get_transforms(calib: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return frame-specific (T_lidar_to_cam, T_lidar_to_ego) using SDK."""
    t_lidar_to_cam = get_3d_transform_camera_lidar(  # type: ignore[misc]
        calib,
        lidar=Lidar.VELODYNE,  # type: ignore[union-attr]
        camera=Camera.FRONT,  # type: ignore[union-attr]
    ).transform
    t_lidar_to_ego = calib.lidars[Lidar.VELODYNE].extrinsics.transform  # type: ignore[union-attr]

    t_lidar_to_cam = np.asarray(t_lidar_to_cam, dtype=np.float64)
    t_lidar_to_ego = np.asarray(t_lidar_to_ego, dtype=np.float64)
    if t_lidar_to_cam.shape != (4, 4) or t_lidar_to_ego.shape != (4, 4):
        raise ValueError("Expected 4x4 transforms for LiDAR->camera and LiDAR->ego.")
    return t_lidar_to_cam, t_lidar_to_ego


def project_center_to_final_image(
    center_lidar: np.ndarray,
    t_lidar_to_cam: np.ndarray,
    k_final: np.ndarray,
    final_width: int = FINAL_WIDTH_DEFAULT,
    final_height: int = FINAL_HEIGHT_DEFAULT,
) -> tuple[bool, float, float, float]:
    """Custom final-FoV test on undistorted+resized pinhole camera."""
    center_cam = transform_points(center_lidar.reshape(1, 3), t_lidar_to_cam)[0]  # type: ignore[misc]
    x_cam, y_cam, z_cam = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])
    if z_cam <= EPS_Z:
        return False, np.nan, np.nan, z_cam

    fx, fy = float(k_final[0, 0]), float(k_final[1, 1])
    cx, cy = float(k_final[0, 2]), float(k_final[1, 2])
    u = fx * (x_cam / z_cam) + cx
    v = fy * (y_cam / z_cam) + cy

    is_inside = (0.0 <= u < float(final_width)) and (0.0 <= v < float(final_height))
    return is_inside, u, v, z_cam


def compute_stats(
    df: pd.DataFrame,
    margin_meters: float,
    x_max_fixed: float,
    grid_resolution: float,
) -> dict[str, Any]:
    """Compute percentiles, suggested bounds, and 0.5m grid sizes."""
    if df.empty:
        return {"stats": {}, "recommended_bounds": {}, "grid_size_estimates": {}}

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    abs_y = np.abs(y)

    stats = {
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "x_p99": float(np.percentile(x, 99.0)),
        "x_p999": float(np.percentile(x, 99.9)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "abs_y_max": float(np.max(abs_y)),
        "abs_y_p99": float(np.percentile(abs_y, 99.0)),
        "abs_y_p999": float(np.percentile(abs_y, 99.9)),
        "abs_y_p9995": float(np.percentile(abs_y, 99.95)),
    }

    y_bound_strict = stats["abs_y_max"] + margin_meters
    y_bound_robust_999 = stats["abs_y_p999"] + margin_meters
    y_bound_robust_9995 = stats["abs_y_p9995"] + margin_meters

    bounds = {
        "x_fixed_min": 0.0,
        "x_fixed_max": float(x_max_fixed),
        "y_bound_strict": float(y_bound_strict),
        "y_bound_robust_999": float(y_bound_robust_999),
        "y_bound_robust_9995": float(y_bound_robust_9995),
        "y_range_strict": [-float(y_bound_strict), float(y_bound_strict)],
        "y_range_robust_999": [-float(y_bound_robust_999), float(y_bound_robust_999)],
        "y_range_robust_9995": [-float(y_bound_robust_9995), float(y_bound_robust_9995)],
    }

    def grid_xy(y_bound: float) -> dict[str, int]:
        x_range = x_max_fixed
        y_range = 2.0 * y_bound
        nx = int(np.ceil(x_range / grid_resolution))
        ny = int(np.ceil(y_range / grid_resolution))
        return {
            "nx": nx,
            "ny": ny,
            "num_cells": nx * ny,
        }

    grid_size_estimates = {
        "grid_resolution_m": float(grid_resolution),
        "strict": grid_xy(y_bound_strict),
        "robust_999": grid_xy(y_bound_robust_999),
        "robust_9995": grid_xy(y_bound_robust_9995),
    }

    return {
        "stats": stats,
        "recommended_bounds": bounds,
        "grid_size_estimates": grid_size_estimates,
    }


def save_plots(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Save x-vs-y and BEV-style scatter plots."""
    xy_path = output_dir / "pedestrian_centers_xy.png"
    bev_path = output_dir / "pedestrian_centers_bev.png"

    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No filtered pedestrian centers", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(xy_path, dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No filtered pedestrian centers", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(bev_path, dpi=180)
        plt.close(fig)
        return xy_path, bev_path

    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x, y, s=5, alpha=0.25, edgecolors="none")
    ax.set_xlabel("x (m, ego forward)")
    ax.set_ylabel("y (m, ego left/right)")
    ax.set_title("Filtered pedestrian centers (x vs y)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(xy_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(-y, x, s=5, alpha=0.25, edgecolors="none")
    ax.set_xlabel("-y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Filtered pedestrian centers (BEV-style)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(bev_path, dpi=200)
    plt.close(fig)
    return xy_path, bev_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BEV bounds after filtering by final camera FoV."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--balance", type=float, default=BALANCE_DEFAULT)
    parser.add_argument("--final-width", type=int, default=FINAL_WIDTH_DEFAULT)
    parser.add_argument("--final-height", type=int, default=FINAL_HEIGHT_DEFAULT)
    parser.add_argument("--margin-meters", type=float, default=MARGIN_METERS_DEFAULT)
    parser.add_argument("--x-max-fixed", type=float, default=X_MAX_FIXED_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if ZOD_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ZOD SDK import failed. Run this script in the same environment used by "
            "the notebooks where ZOD SDK is available."
        ) from ZOD_IMPORT_ERROR

    parquet_path = args.output_dir / "pedestrian_centers_filtered.parquet"
    csv_fallback_path = args.output_dir / "pedestrian_centers_filtered.csv"
    summary_path = args.output_dir / "bev_bounds_summary.json"

    counters = Counters()
    timings = Timings()

    t_start = time.perf_counter()
    frame_dirs = iter_frame_dirs(args.dataset_root, args.max_frames)
    counters.total_frames_discovered = len(frame_dirs)

    print(f"DATASET_ROOT: {args.dataset_root}")
    print(f"Valid frames discovered (unsorted): {len(frame_dirs)}")
    print(f"Output dir: {args.output_dir}")
    print(
        f"Final camera config: balance={args.balance}, "
        f"final_size=({args.final_width}, {args.final_height}), "
        f"grid_resolution={GRID_RESOLUTION_DEFAULT}"
    )

    rows: list[dict[str, Any]] = []
    progress = tqdm(frame_dirs, desc="Scanning frames", unit="frame")

    for i, frame_dir in enumerate(progress, start=1):
        frame_id = frame_dir.name

        frame_start = time.perf_counter()
        try:
            t0 = time.perf_counter()
            zod_frame = make_zodframe_from_raw(frame_dir)
            calib = zod_frame.calibration
            timings.read_calib_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            cam_calib = calib.cameras[Camera.FRONT]  # type: ignore[index]
            k_final = compute_final_camera_intrinsics(
                cam_calib=cam_calib,
                final_width=args.final_width,
                final_height=args.final_height,
                balance=args.balance,
            )
            t_lidar_to_cam, t_lidar_to_ego = get_transforms(calib)
            timings.compute_intrinsics_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            anns = zod_frame.get_annotation(AnnotationProject.OBJECT_DETECTION)  # type: ignore[arg-type]
            timings.load_annotations_sec += time.perf_counter() - t0

            t0 = time.perf_counter()
            for obj in anns:
                if obj is None:
                    continue
                counters.total_objects += 1

                if getattr(obj, "name", None) != "Pedestrian":
                    continue
                counters.pedestrian_objects += 1

                box3d = getattr(obj, "box3d", None)
                if box3d is None:
                    continue
                counters.ped_objects_with_3d += 1

                center_lidar = np.asarray(box3d.center, dtype=np.float64)

                is_inside, _, _, z_cam = project_center_to_final_image(
                    center_lidar=center_lidar,
                    t_lidar_to_cam=t_lidar_to_cam,
                    k_final=k_final,
                    final_width=args.final_width,
                    final_height=args.final_height,
                )
                if z_cam <= EPS_Z:
                    counters.ped_centers_behind_camera += 1
                    continue
                if not is_inside:
                    counters.ped_centers_outside_final_fov += 1
                    continue

                center_ego = transform_points(center_lidar.reshape(1, 3), t_lidar_to_ego)[0]  # type: ignore[misc]
                rows.append(
                    {
                        "frame_id": frame_id,
                        "x": float(center_ego[0]),
                        "y": float(center_ego[1]),
                        "z": float(center_ego[2]),
                    }
                )
                counters.ped_centers_inside_final_fov += 1
            timings.obj_loop_sec += time.perf_counter() - t0

            counters.frames_processed += 1
        except Exception as exc:
            counters.failed_frames += 1
            tqdm.write(f"Failed frame {frame_id}: {type(exc).__name__}: {exc}")
        finally:
            timings.frame_loop_sec += time.perf_counter() - frame_start

        if i % PRINT_EVERY == 0:
            tqdm.write(
                f"[{i}/{len(frame_dirs)}] processed={counters.frames_processed} "
                f"failed={counters.failed_frames} kept={counters.ped_centers_inside_final_fov}"
            )

    timings.total_sec = time.perf_counter() - t_start

    df = pd.DataFrame(rows)
    if df.empty:
        print("Warning: no pedestrian centers kept after final FoV filtering.")

    parquet_saved = True
    try:
        df.to_parquet(parquet_path, index=False)
        centers_path = parquet_path
    except Exception as exc:
        parquet_saved = False
        print(f"Parquet write failed ({type(exc).__name__}); falling back to CSV.")
        df.to_csv(csv_fallback_path, index=False)
        centers_path = csv_fallback_path

    result = compute_stats(
        df=df,
        margin_meters=args.margin_meters,
        x_max_fixed=args.x_max_fixed,
        grid_resolution=GRID_RESOLUTION_DEFAULT,
    )
    xy_path, bev_path = save_plots(df, args.output_dir)

    summary = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "max_frames": args.max_frames,
            "balance": args.balance,
            "final_width": args.final_width,
            "final_height": args.final_height,
            "margin_meters": args.margin_meters,
            "x_max_fixed": args.x_max_fixed,
            "grid_resolution": GRID_RESOLUTION_DEFAULT,
            "output_dir": str(args.output_dir),
        },
        "counters": {
            "total_frames_discovered": counters.total_frames_discovered,
            "frames_processed": counters.frames_processed,
            "failed_frames": counters.failed_frames,
            "total_objects": counters.total_objects,
            "pedestrian_objects": counters.pedestrian_objects,
            "pedestrian_objects_with_3d": counters.ped_objects_with_3d,
            "ped_3d_centers_inside_final_fov": counters.ped_centers_inside_final_fov,
            "ped_3d_centers_outside_final_fov": counters.ped_centers_outside_final_fov,
            "ped_3d_centers_behind_camera": counters.ped_centers_behind_camera,
        },
        "timings_sec": {
            "total": timings.total_sec,
            "frame_loop": timings.frame_loop_sec,
            "read_calibration": timings.read_calib_sec,
            "compute_intrinsics_and_transforms": timings.compute_intrinsics_sec,
            "load_annotations": timings.load_annotations_sec,
            "object_loop": timings.obj_loop_sec,
        },
        "artifacts": {
            "centers_path": str(centers_path),
            "parquet_saved": parquet_saved,
            "summary_path": str(summary_path),
            "xy_plot_path": str(xy_path),
            "bev_plot_path": str(bev_path),
        },
        "result": result,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"frames discovered: {counters.total_frames_discovered}")
    print(f"frames processed:  {counters.frames_processed}")
    print(f"failed frames:     {counters.failed_frames}")
    print(f"total objects:     {counters.total_objects}")
    print(f"ped objects:       {counters.pedestrian_objects}")
    print(f"ped with 3D:       {counters.ped_objects_with_3d}")
    print(f"ped kept:          {counters.ped_centers_inside_final_fov}")
    print(f"ped outside FoV:   {counters.ped_centers_outside_final_fov}")
    print(f"ped behind camera: {counters.ped_centers_behind_camera}")
    print(f"elapsed sec:       {timings.total_sec:.2f}")

    stats = result.get("stats", {})
    bounds = result.get("recommended_bounds", {})
    if stats:
        print("\n--- Distribution stats ---")
        print(
            "x_min={x_min:.3f}, x_max={x_max:.3f}, x_p99={x_p99:.3f}, x_p999={x_p999:.3f}".format(
                **stats
            )
        )
        print("y_min={y_min:.3f}, y_max={y_max:.3f}".format(**stats))
        print(
            "abs_y_max={abs_y_max:.3f}, abs_y_p99={abs_y_p99:.3f}, "
            "abs_y_p999={abs_y_p999:.3f}, abs_y_p9995={abs_y_p9995:.3f}".format(**stats)
        )
    if bounds:
        print("\n--- Suggested BEV bounds ---")
        print(f"x fixed: [0, {args.x_max_fixed}]")
        print(
            f"strict y: [-{bounds['y_bound_strict']:.3f}, {bounds['y_bound_strict']:.3f}]"
        )
        print(
            f"robust y (99.9): [-{bounds['y_bound_robust_999']:.3f}, "
            f"{bounds['y_bound_robust_999']:.3f}]"
        )
        print(
            f"robust y (99.95): [-{bounds['y_bound_robust_9995']:.3f}, "
            f"{bounds['y_bound_robust_9995']:.3f}]"
        )

    print("\nArtifacts:")
    print(f"- centers table: {centers_path}")
    print(f"- summary json:  {summary_path}")
    print(f"- XY plot:       {xy_path}")
    print(f"- BEV plot:      {bev_path}")


if __name__ == "__main__":
    main()
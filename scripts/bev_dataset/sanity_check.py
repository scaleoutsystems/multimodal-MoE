#!/usr/bin/env python3
"""
Sanity-check the NuScenes-style MMDetection3D dataset built by build_infos.py.

This script is **read-only** — it never modifies the dataset.  It loads one
info pickle, inspects the first sample entry, verifies paths and array
shapes, and saves a BEV scatter plot with 3D-box centres overlaid.

Outputs
-------
* Terminal printout of all inspection results.
* A text log mirroring the terminal output saved to:
  /home/edgelab/multimodal-MoE/outputs/analysis/bev_dataset/sanity_check_log.txt
* A BEV plot saved to:
  /home/edgelab/multimodal-MoE/outputs/analysis/bev_dataset/bev_sanity_plot.png
* An image-space plot with projected 3D box wireframes saved to:
  /home/edgelab/multimodal-MoE/outputs/analysis/bev_dataset/image_box_sanity_plot.png

Example
-------
python scripts/bev_dataset/sanity_check.py

python scripts/bev_dataset/sanity_check.py \\
    --dataset-root /mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes \\
    --info-pkl     infos/zod_nuscenes_infos_train.pkl
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ==================================================================
# Tee — duplicate all print() output to a text file
# ==================================================================
class _Tee(io.TextIOBase):
    """Write to both the real stdout and a log file simultaneously."""

    def __init__(self, log_path: Path) -> None:
        super().__init__()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._terminal = sys.stdout
        self._log = open(log_path, "w")

    def write(self, data: str) -> int:
        self._terminal.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()


# ==================================================================
# Defaults
# ==================================================================
DEFAULT_DATASET_ROOT = Path(
    "/mnt/tier2/project/p201392/u103958/zod_moe/zod_nuscenes"
)
DEFAULT_SOURCE_ROOT = Path(
    "/mnt/tier2/project/p201392/u103958/zod_moe"
)
DEFAULT_INFO_PKL = "infos/zod_nuscenes_infos_train.pkl"
DEFAULT_OUT_DIR = Path(
    "/home/users/u103958/projects/multimodal-MoE/outputs/analysis/bev_dataset"
)


# ==================================================================
# CLI
# ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check the ZOD-NuScenes dataset.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root of intermediate artifacts (bev_images/, labels/).",
    )
    p.add_argument(
        "--info-pkl",
        type=str,
        default=DEFAULT_INFO_PKL,
        help="Relative path (inside dataset root) to the info pickle.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the sample to inspect inside data_list.",
    )
    return p.parse_args()


# ==================================================================
# Helpers
# ==================================================================
def check_dataset_structure(dataset_root: Path) -> None:
    """Verify that the expected top-level directories exist."""
    required = [
        "samples",
        "samples/CAM_FRONT",
        "samples/LIDAR_TOP",
        "infos",
        "splits",
        "index",
    ]
    print("--- Directory structure check ---")
    all_ok = True
    for rel in required:
        p = dataset_root / rel
        exists = p.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {rel:30s}  {status}")
        if not exists:
            all_ok = False
    if not all_ok:
        raise FileNotFoundError(
            "One or more required directories are missing under "
            f"{dataset_root}.  Was build_infos.py run successfully?"
        )
    print()


def load_pickle(pkl_path: Path) -> dict:
    """Load and return a pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def inspect_sample(sample: dict, dataset_root: Path) -> None:
    """Print and validate one sample entry from data_list."""

    print("--- Sample entry ---")
    print(f"  sample_idx : {sample['sample_idx']}")
    print(f"  token      : {sample['token']}")
    print(f"  lidar_path : {sample['lidar_points']['lidar_path']}")
    print(f"  num_pts_feats : {sample['lidar_points']['num_pts_feats']}")

    cam = sample["images"]["CAM_FRONT"]
    print(f"  img_path   : {cam['img_path']}")
    print(f"  cam2img    : shape {cam['cam2img'].shape}  dtype {cam['cam2img'].dtype}")
    print(f"  lidar2cam  : shape {cam['lidar2cam'].shape}  dtype {cam['lidar2cam'].dtype}")

    n_inst = len(sample.get("instances", []))
    print(f"  instances  : {n_inst}")
    if n_inst > 0:
        first = sample["instances"][0]
        print(f"    first bbox_label_3d : {first['bbox_label_3d']}")
        print(f"    first bbox_3d       : {first['bbox_3d']}")

    ctx = sample.get("context")
    if ctx:
        print(f"  context    : {ctx}")
    print()

    # --- shape assertions ---
    assert cam["cam2img"].shape == (3, 3), (
        f"cam2img should be (3,3), got {cam['cam2img'].shape}"
    )
    assert cam["lidar2cam"].shape == (4, 4), (
        f"lidar2cam should be (4,4), got {cam['lidar2cam'].shape}"
    )

    # --- path existence ---
    img_abs = dataset_root / cam["img_path"]
    lid_abs = dataset_root / sample["lidar_points"]["lidar_path"]

    for tag, p in [("image", img_abs), ("lidar", lid_abs)]:
        resolved = p.resolve() if p.is_symlink() else p
        if not resolved.exists():
            raise FileNotFoundError(f"{tag} file not found: {p}  (resolves to {resolved})")
        print(f"  {tag} file exists: {p}")
    print()


def load_lidar_points(dataset_root: Path, lidar_rel_path: str) -> np.ndarray:
    """Load an XYZI float32 point cloud from a .bin file."""
    path = dataset_root / lidar_rel_path
    raw = np.fromfile(str(path), dtype=np.float32)
    points = raw.reshape(-1, 4)

    print("--- LiDAR point cloud ---")
    print(f"  path  : {path}")
    print(f"  shape : {points.shape}")
    print(f"  dtype : {points.dtype}")
    print(f"  x range : [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  y range : [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  z range : [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print()
    return points


def plot_bev_points_and_box_centers(
    points: np.ndarray,
    sample: dict,
    out_path: Path,
) -> None:
    """Save a bird's-eye-view scatter of LiDAR points with box centres."""
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=0.05,
        c="steelblue",
        alpha=0.3,
        rasterized=True,
    )

    instances = sample.get("instances", [])
    if instances:
        centres = np.array([inst["bbox_3d"][:2] for inst in instances])
        ax.scatter(
            centres[:, 0],
            centres[:, 1],
            s=40,
            c="red",
            marker="x",
            linewidths=1.5,
            label=f"box centres ({len(instances)})",
        )
        ax.legend(fontsize=10)

    ax.set_xlabel("x  (forward, m)")
    ax.set_ylabel("y  (left, m)")
    ax.set_title(f"BEV sanity — sample {sample['sample_idx']}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BEV plot → {out_path}")


# ------------------------------------------------------------------
# Image-space projected box visualization
# ------------------------------------------------------------------

# 12 edges of a cuboid defined by 8 corner indices.
# Bottom face: 0-1-2-3, top face: 4-5-6-7, verticals: 0-4, 1-5, 2-6, 3-7.
_CUBOID_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def load_label_json(label_path: Path) -> dict:
    """Load a label JSON produced by build_zod_moe_dataset.py."""
    with open(label_path) as f:
        return json.load(f)


def draw_projected_boxes_on_image(
    image_path: Path,
    label_data: dict,
    out_path: Path,
) -> None:
    """Draw pre-computed projected 3D box wireframes on the final image.

    Each instance in *label_data* is expected to carry:
      - ``projected_center_uv``  — [u, v]
      - ``projected_corners_uv`` — list of 8 × [u, v]

    These were already computed by the build script, so no re-projection
    is needed here.
    """
    img = plt.imread(str(image_path))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)

    instances = label_data.get("instances", [])
    n_drawn = 0

    for inst in instances:
        center = inst.get("projected_center_uv")
        corners = inst.get("projected_corners_uv")

        if center is None or corners is None:
            continue
        if len(corners) != 8:
            continue

        cu, cv = float(center[0]), float(center[1])
        if not (np.isfinite(cu) and np.isfinite(cv)):
            continue

        pts = np.array(corners, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            continue

        # Draw cuboid edges
        for i, j in _CUBOID_EDGES:
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color="cyan",
                linewidth=0.8,
                alpha=0.9,
            )

        # Draw corner dots
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="dodgerblue", zorder=3)

        # Draw projected centre
        ax.plot(cu, cv, "ro", markersize=3, zorder=4)

        n_drawn += 1

    frame_id = label_data.get("frame_id", "?")
    ax.set_title(
        f"Projected 3D boxes — frame {frame_id}  "
        f"({n_drawn}/{len(instances)} instances drawn)",
        fontsize=12,
    )
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image-box plot → {out_path}  ({n_drawn} boxes drawn)")


# ==================================================================
# Main
# ==================================================================
def main() -> None:
    args = parse_args()
    root = args.dataset_root

    log_path = args.out_dir / "sanity_check_log.txt"
    tee = _Tee(log_path)
    sys.stdout = tee

    print(f"Dataset root: {root}\n")

    # 1. Structure check
    check_dataset_structure(root)

    # 2. Load pickle
    pkl_path = root / args.info_pkl
    assert pkl_path.exists(), f"Info pickle not found: {pkl_path}"
    info = load_pickle(pkl_path)

    print("--- Info pickle ---")
    print(f"  path       : {pkl_path}")
    print(f"  top keys   : {list(info.keys())}")
    print(f"  metainfo   : {info['metainfo']}")
    print(f"  data_list  : {len(info['data_list'])} samples")
    print()

    # 3. Inspect one sample
    idx = args.sample_index
    assert idx < len(info["data_list"]), (
        f"--sample-index {idx} out of range (data_list has {len(info['data_list'])} entries)"
    )
    sample = info["data_list"][idx]
    inspect_sample(sample, root)

    # 4. Load LiDAR
    points = load_lidar_points(root, sample["lidar_points"]["lidar_path"])

    # 5. BEV plot
    out_plot = args.out_dir / "bev_sanity_plot.png"
    plot_bev_points_and_box_centers(points, sample, out_plot)

    # 6. Image-space projected-box plot
    frame_id = str(sample["sample_idx"])
    src_img_path = args.source_root / "bev_images" / f"{frame_id}.png"
    src_label_path = args.source_root / "labels" / f"{frame_id}.json"

    out_img_plot = args.out_dir / "image_box_sanity_plot.png"

    if src_img_path.exists() and src_label_path.exists():
        label_data = load_label_json(src_label_path)
        draw_projected_boxes_on_image(src_img_path, label_data, out_img_plot)
    else:
        if not src_img_path.exists():
            print(f"  WARNING: source image not found: {src_img_path}")
        if not src_label_path.exists():
            print(f"  WARNING: source label not found: {src_label_path}")
        print("  Skipping image-box sanity plot.")

    # 7. Summary
    print()
    print("=" * 50)
    print("SANITY CHECK PASSED")
    print("=" * 50)
    print(f"  Dataset root   : {root}")
    print(f"  Info pickle    : {pkl_path}")
    print(f"  Sample inspected : {sample['sample_idx']}")
    print(f"  LiDAR points   : {points.shape[0]}")
    print(f"  Instances      : {len(sample.get('instances', []))}")
    print(f"  BEV plot       : {out_plot}")
    print(f"  Image-box plot : {out_img_plot}")
    print(f"  Log file       : {log_path}")
    print("=" * 50)

    sys.stdout = tee._terminal
    tee.close()
    print(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout = sys.__stdout__
        import traceback
        traceback.print_exc()
        sys.exit(1)
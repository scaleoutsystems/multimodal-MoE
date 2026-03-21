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

Example
-------
python scripts/bev_dataset/sanity_check.py

python scripts/bev_dataset/sanity_check.py \\
    --dataset-root /mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/zod_nuscenes \\
    --info-pkl     infos/zod_nuscenes_infos_train.pkl
"""

from __future__ import annotations

import argparse
import io
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
    "/mnt/ZOD_clone_2018_scaleout_zenseact/zod_moe/zod_nuscenes"
)
DEFAULT_INFO_PKL = "infos/zod_nuscenes_infos_train.pkl"
DEFAULT_OUT_DIR = Path(
    "/home/edgelab/multimodal-MoE/outputs/analysis/bev_dataset"
)


# ==================================================================
# CLI
# ==================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check the ZOD-NuScenes dataset.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
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

    # 6. Summary
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
    print(f"  Log file       : {log_path}")
    print("=" * 50)

    sys.stdout = tee._terminal
    tee.close()
    print(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    main()

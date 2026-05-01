#!/usr/bin/env python3
"""Render "Sparse LiDAR depth on image" with large fonts — no GPU needed.

Sparse depth is pure geometry: project LiDAR points through the camera
matrices stored in the dataset sample, exactly as BaseDepthTransform does.
No model forward pass, no CUDA.

Usage (from repo root):
  PY=/home/users/u103958/miniconda3/envs/multimodal-moe/bin/python

  $PY scripts/bev_scripts/plot_sparse_depth.py \\
      mmdetection3d/configs/zod/zod_camera_only_ftlr_d60_splat0.py \\
      --out outputs/sparse_depth_splat0.png \\
      --label "no neighbourhood expansion"

  $PY scripts/bev_scripts/plot_sparse_depth.py \\
      mmdetection3d/configs/zod/zod_camera_only_ftlr_d60.py \\
      --out outputs/sparse_depth_splat1.png \\
      --label "3×3 neighbourhood expansion"
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mmdetection3d"))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

# ── font sizes ──────────────────────────────────────────────────────────
FS_TITLE = 20
FS_LABEL = 24
FS_TICK  = 18
FS_CB    = 24
FS_STATS = 20


# ── stub open3d / visualization so register_all_modules doesn't need libGL ──
def _stub_headless():
    import types
    for name in ("open3d", "open3d.geometry", "open3d.utility", "open3d.io",
                 "open3d.visualization",
                 "mmdet3d.visualization", "mmdet3d.visualization.local_visualizer"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    vis = sys.modules["mmdet3d.visualization"]
    if not hasattr(vis, "Det3DLocalVisualizer"):
        vis.Det3DLocalVisualizer = None


def load_dataset(config_path: str):
    _stub_headless()
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from mmengine.config import Config
    from mmdet3d.registry import DATASETS
    cfg = Config.fromfile(config_path)
    return DATASETS.build(cfg.val_dataloader.dataset)


def get_view_transform_params(config_path: str) -> dict:
    """Extract image_size, splat_radius, dbound from config."""
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)
    # view_transform is nested inside model
    vt = cfg.model.view_transform
    return dict(
        image_size  = tuple(vt.image_size),
        splat_radius= int(getattr(vt, "splat_radius", 0)),
    )


def compute_sparse_depth(sample: dict, image_size: tuple, splat_radius: int) -> np.ndarray:
    """Replicate BaseDepthTransform depth rasterisation exactly."""
    iH, iW = image_size
    depth = np.zeros((iH, iW), dtype=np.float32)

    # ── pull tensors out of the sample ──────────────────────────────────
    pts_raw = sample["inputs"]["points"]
    if hasattr(pts_raw, "tensor"):   # BasePoints wrapper
        pts_raw = pts_raw.tensor
    pts_xyz = pts_raw[:, :3].float()

    ds = sample["data_samples"]
    meta = ds.metainfo if hasattr(ds, "metainfo") else ds
    lidar2img   = torch.tensor(np.asarray(meta["lidar2img"]),  dtype=torch.float32)
    img_aug_mat = torch.tensor(np.asarray(meta.get("img_aug_matrix",  np.eye(4))), dtype=torch.float32)
    lidar_aug   = torch.tensor(np.asarray(meta.get("lidar_aug_matrix", np.eye(4))), dtype=torch.float32)

    # handle (N_cam, 4, 4) or (4, 4)
    if lidar2img.dim() == 3:
        lidar2img = lidar2img[0]
    if img_aug_mat.dim() == 3:
        img_aug_mat = img_aug_mat[0]

    # ── inverse lidar aug (same as depth_lss.py) ────────────────────────
    cur = pts_xyz.clone()
    cur = cur - lidar_aug[:3, 3]
    cur = torch.linalg.inv(lidar_aug[:3, :3]) @ cur.T  # (3, N)

    # ── lidar → image ────────────────────────────────────────────────────
    proj = lidar2img[:3, :3] @ cur + lidar2img[:3, 3:4]
    dist = proj[2].clone()
    proj[2] = proj[2].clamp(min=1e-5)
    proj[:2] /= proj[2:3]

    # ── image aug ────────────────────────────────────────────────────────
    proj = img_aug_mat[:3, :3] @ proj + img_aug_mat[:3, 3:4]
    uv  = proj[:2].T  # (N, 2)  — u=col, v=row

    # ── filter valid ─────────────────────────────────────────────────────
    dist_np = dist.numpy()
    u_np = uv[:, 0].numpy()
    v_np = uv[:, 1].numpy()
    valid = (dist_np > 0) & (v_np >= 0) & (v_np < iH) & (u_np >= 0) & (u_np < iW)

    rows = v_np[valid].astype(np.int32)
    cols = u_np[valid].astype(np.int32)
    dvals = dist_np[valid]

    # ── splat ────────────────────────────────────────────────────────────
    r = splat_radius
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            nr = rows + dr; nc = cols + dc
            ok = (nr >= 0) & (nr < iH) & (nc >= 0) & (nc < iW)
            nr, nc, dv = nr[ok], nc[ok], dvals[ok]
            cur_d = depth[nr, nc]
            depth[nr, nc] = np.where((cur_d == 0) | (dv < cur_d), dv, cur_d)

    return depth


def plot_sparse_depth(sd: np.ndarray, out_path: str, label: str = "", dpi: int = 150):
    iH, iW = sd.shape
    nz = sd[sd > 0]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("black")
    ax.set_xlim(0, iW); ax.set_ylim(iH, 0)

    if len(nz) > 0:
        sd_plot = np.where(sd > 0, sd, np.nan)
        im = ax.imshow(sd_plot, cmap="turbo", origin="upper", aspect="auto",
                       norm=LogNorm(vmin=max(nz.min(), 1.0), vmax=nz.max()))
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("depth (m)", fontsize=FS_CB)
        cbar.ax.tick_params(labelsize=FS_TICK)

    title = f"Sparse LiDAR depth on image  ({iH}×{iW} px)"
    if label:
        title += f"\n{label}"
    ax.set_title(title, fontsize=FS_TITLE, pad=10)
    ax.set_xlabel("u (pixels)", fontsize=FS_LABEL)
    ax.set_ylabel("v (pixels)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)

    stats = f"pixels with depth: {len(nz):,}"
    if len(nz) > 0:
        stats += f"\nrange: [{nz.min():.1f}, {nz.max():.1f}] m"
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=FS_STATS,
            va="top", color="white",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6, ec="gray"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config",  help="MMDet3D config (.py)")
    p.add_argument("--out",   default="outputs/sparse_depth.png")
    p.add_argument("--label", default="")
    p.add_argument("--dpi",   type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Config : {args.config}")
    print(f"Output : {args.out}")

    params = get_view_transform_params(args.config)
    print(f"image_size={params['image_size']}  splat_radius={params['splat_radius']}")

    print("Loading dataset …")
    dataset = load_dataset(args.config)

    print("Computing sparse depth from sample[0] …")
    sample = dataset[0]
    sd = compute_sparse_depth(sample, params["image_size"], params["splat_radius"])

    nz = (sd > 0).sum()
    print(f"Nonzero pixels: {nz:,} / {sd.size:,}  ({100*nz/sd.size:.2f}%)")
    plot_sparse_depth(sd, args.out, label=args.label, dpi=args.dpi)


if __name__ == "__main__":
    main()

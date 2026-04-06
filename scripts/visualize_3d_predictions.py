#!/usr/bin/env python3
"""Visualize LiDAR point cloud + GT boxes + predicted boxes in 3D.

Usage (run from multimodal-MoE root):
--------------------------------------
python scripts/visualize_3d_predictions.py \
    --config  mmdetection3d/configs/zod/zod_lidar_only.py \
    --ckpt    outputs/runs/zod_lidar_only/zod-lidar-only_4452913/best_mAP_0.50_epoch_18.pth \
    --out-dir outputs/runs/zod_lidar_only/zod-lidar-only_4452913/vis3d \
    --num-samples 8 \
    --score-thr 0.25
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ──────────────────────────────────────────────────────────────────────────────
# Box helpers
# ──────────────────────────────────────────────────────────────────────────────

def box_to_corners(box: np.ndarray) -> np.ndarray:
    """[x, y, z_bottom, dx, dy, dz, yaw] → (8, 3) corners."""
    x, y, z_bot, dx, dy, dz, yaw = box[:7]
    hx, hy = dx / 2, dy / 2
    local = np.array([
        [-hx, -hy, 0], [ hx, -hy, 0], [ hx,  hy, 0], [-hx,  hy, 0],
        [-hx, -hy, dz],[ hx, -hy, dz],[ hx,  hy, dz],[-hx,  hy, dz],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ R.T + np.array([x, y, z_bot])


BOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]


def draw_box_3d(ax, corners, color, lw=1.4, alpha=0.9):
    segs = [[corners[i], corners[j]] for i, j in BOX_EDGES]
    ax.add_collection3d(Line3DCollection(segs, colors=color,
                                         linewidths=lw, alpha=alpha))


def draw_box_bev(ax, corners, color, lw=1.4, alpha=0.9):
    bottom = corners[:4, :2]
    poly = plt.Polygon(bottom, fill=False, edgecolor=color,
                       linewidth=lw, alpha=alpha)
    ax.add_patch(poly)


# ──────────────────────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────────────────────

BG = '#12122a'

def plot_sample(pts, gt_boxes, pred_boxes, pred_scores,
                sample_idx, out_path, score_thr):
    keep = pred_scores >= score_thr
    pred_boxes  = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    rng = np.random.default_rng(0)
    if len(pts) > 40_000:
        pts = pts[rng.choice(len(pts), 40_000, replace=False)]

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor(BG)

    h = pts[:, 2]
    h_norm = np.clip((h - h.min()) / (h.ptp() + 1e-6), 0, 1)

    gt_corners   = [box_to_corners(b) for b in gt_boxes]
    pred_corners = [box_to_corners(b) for b in pred_boxes]

    # ── BEV ──────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 1)
    ax2.set_facecolor(BG)
    ax2.set_aspect('equal')
    ax2.scatter(pts[:, 0], pts[:, 1], c=h_norm, cmap='plasma',
                s=0.3, alpha=0.5, rasterized=True)
    for c in gt_corners:
        draw_box_bev(ax2, c, '#00ff88')
    for c, sc in zip(pred_corners, pred_scores):
        draw_box_bev(ax2, c, '#ff4444')
        cx, cy = c[:4, 0].mean(), c[:4, 1].mean()
        ax2.text(cx, cy, f'{sc:.2f}', color='#ff9999', fontsize=5,
                 ha='center', va='center')
    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Y (m)', color='white')
    ax2.tick_params(colors='white')
    ax2.set_title(f'BEV  ·  sample {sample_idx}\n'
                  f'GT: {len(gt_boxes)}   Pred ≥{score_thr}: {len(pred_boxes)}',
                  color='white', pad=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#33335a')

    # ── 3-D ──────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 2, 2, projection='3d')
    ax3.set_facecolor(BG)
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                c=h_norm, cmap='plasma', s=0.3, alpha=0.35, rasterized=True)
    for c in gt_corners:
        draw_box_3d(ax3, c, '#00ff88')
    for c in pred_corners:
        draw_box_3d(ax3, c, '#ff4444')

    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#2a2a4a')
    ax3.tick_params(colors='white', labelsize=7)
    ax3.set_xlabel('X', color='white', labelpad=2)
    ax3.set_ylabel('Y', color='white', labelpad=2)
    ax3.set_zlabel('Z', color='white', labelpad=2)
    ax3.set_title('3-D perspective', color='white', pad=8)
    ax3.view_init(elev=25, azim=-55)

    if len(pts):
        xc, yc = pts[:, 0].mean(), pts[:, 1].mean()
        r = max(pts[:, 0].ptp(), pts[:, 1].ptp()) / 2 + 3
        ax3.set_xlim(xc - r, xc + r)
        ax3.set_ylim(yc - r, yc + r)
        ax3.set_zlim(pts[:, 2].min() - 0.5, pts[:, 2].max() + 2.5)

    handles = [mpatches.Patch(color='#00ff88',
                               label=f'GT ({len(gt_boxes)})'),
               mpatches.Patch(color='#ff4444',
                               label=f'Pred ≥{score_thr} ({len(pred_boxes)})')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               facecolor='#1e1e3a', edgecolor='#44446a',
               labelcolor='white', fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  saved → {out_path}')


# ──────────────────────────────────────────────────────────────────────────────
# mmdet3d setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_mmdet3d(repo_root: Path):
    # Add repo root so mmdet3d finds its own modules.
    # Add BEVFusion project dir so `custom_imports` in the config can find
    # `bevfusion` package. Do NOT manually import it here — Config.fromfile
    # triggers custom_imports itself, and double-registration raises KeyError.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / 'projects' / 'BEVFusion'))
    from mmdet3d.utils import register_all_modules
    register_all_modules()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      required=True)
    p.add_argument('--ckpt',        required=True)
    p.add_argument('--out-dir',     required=True)
    p.add_argument('--num-samples', type=int,   default=8)
    p.add_argument('--score-thr',   type=float, default=0.25)
    p.add_argument('--split',       default='val',
                   choices=['train', 'val', 'test'])
    p.add_argument('--device',      default='cuda:0')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1] / 'mmdetection3d'
    setup_mmdet3d(repo_root)

    from mmengine.config import Config
    from mmengine.registry import DATASETS
    from mmdet3d.apis import init_model
    import torch

    cfg = Config.fromfile(args.config)

    # ── build dataset ─────────────────────────────────────────────────────────
    dl_cfg = getattr(cfg, f'{args.split}_dataloader', cfg.val_dataloader)
    ds_cfg = dl_cfg.dataset
    dataset = DATASETS.build(ds_cfg)
    print(f'Dataset: {len(dataset)} samples ({args.split})')

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading model …')
    model = init_model(args.config, args.ckpt, device=args.device)
    model.eval()

    data_root = Path(dataset.data_root)
    print(f'Scanning for annotated samples (need {args.num_samples}, score_thr={args.score_thr})')

    # Collect indices of samples that have at least one GT instance
    annotated_indices = []
    for i in range(len(dataset)):
        info = dataset.get_data_info(i)
        if len(info.get('instances', [])) > 0:
            annotated_indices.append(i)
        if len(annotated_indices) >= args.num_samples:
            break
    print(f'Found {len(annotated_indices)} annotated samples (scanned up to index {annotated_indices[-1] if annotated_indices else "?"})')

    for i in annotated_indices:
        data_info = dataset.get_data_info(i)
        sample_idx = str(data_info.get('sample_idx', i))

        # ── point cloud ───────────────────────────────────────────────────────
        lidar_rel = data_info.get('lidar_points', {}).get('lidar_path', '')
        lidar_abs = str(data_root / lidar_rel) if lidar_rel else None
        if lidar_abs and Path(lidar_abs).exists():
            pts = np.fromfile(lidar_abs, dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            pts = np.zeros((0, 3), dtype=np.float32)

        # ── GT boxes ──────────────────────────────────────────────────────────
        gt_list = []
        for inst in data_info.get('instances', []):
            b = inst.get('bbox_3d')
            if b is not None:
                gt_list.append(np.array(b[:7], dtype=np.float32))
        gt_boxes = np.stack(gt_list) if gt_list else np.zeros((0, 7))

        # ── inference via dataset pipeline ────────────────────────────────────
        sample = dataset[i]
        # collate into a batch of 1
        from mmengine.structures import InstanceData
        from mmdet3d.structures import Det3DDataSample
        from mmengine.dataset import pseudo_collate

        batch = pseudo_collate([sample])

        # move tensors to device
        def to_device(obj, dev):
            if isinstance(obj, torch.Tensor):
                return obj.to(dev)
            if isinstance(obj, dict):
                return {k: to_device(v, dev) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_device(v, dev) for v in obj]
            return obj

        batch = to_device(batch, args.device)

        with torch.no_grad():
            results = model.test_step(batch)

        pred = results[0].pred_instances_3d
        pred_boxes_np  = pred.bboxes_3d.tensor.cpu().numpy()
        pred_scores_np = pred.scores_3d.cpu().numpy()

        # ── z diagnostic ───────────────────────────────────────────────────────
        from mmdet3d.structures import LiDARInstance3DBoxes
        high_conf = pred_scores_np >= args.score_thr
        if gt_boxes.shape[0] > 0:
            gt_z = gt_boxes[:, 2]
            gt_dz = gt_boxes[:, 5]
            print(f'\n  sample {sample_idx}: GT z_bottom  min={gt_z.min():.3f}  '
                  f'mean={gt_z.mean():.3f}  max={gt_z.max():.3f}  '
                  f'mean_dz={gt_dz.mean():.3f}')
        if high_conf.sum() > 0:
            pz = pred_boxes_np[high_conf, 2]
            pdz = pred_boxes_np[high_conf, 5]
            print(f'  sample {sample_idx}: Pred z       min={pz.min():.3f}  '
                  f'mean={pz.mean():.3f}  max={pz.max():.3f}  '
                  f'mean_dz={pdz.mean():.3f}')
            if gt_boxes.shape[0] > 0:
                shift = pz.mean() - gt_z.mean()
                print(f'  sample {sample_idx}: mean(pred_z) - mean(gt_z) = {shift:.3f}  '
                      f'(unmatched; ~half dz = {gt_dz.mean()/2:.3f})')

        # ── MATCHED z diagnostic (Hungarian-style IoU matching) ────────────────
        if gt_boxes.shape[0] > 0 and high_conf.sum() > 0:
            gt_t  = torch.tensor(gt_boxes, dtype=torch.float32)
            pr_t  = torch.tensor(pred_boxes_np[high_conf], dtype=torch.float32)
            gt_b  = LiDARInstance3DBoxes(gt_t)
            pr_b  = LiDARInstance3DBoxes(pr_t)
            iou_mat = LiDARInstance3DBoxes.overlaps(pr_b, gt_b).numpy()  # [N_pred, N_gt]
            # For each GT, find the best matching prediction
            matched_shifts = []
            matched_ious   = []
            for g in range(iou_mat.shape[1]):
                best_p = int(iou_mat[:, g].argmax())
                best_iou = float(iou_mat[best_p, g])
                if best_iou > 0.1:   # loosely matched
                    dz_shift = float(pred_boxes_np[high_conf][best_p, 2]) - float(gt_boxes[g, 2])
                    matched_shifts.append(dz_shift)
                    matched_ious.append(best_iou)
            if matched_shifts:
                arr = np.array(matched_shifts)
                iou_arr = np.array(matched_ious)
                print(f'  sample {sample_idx}: MATCHED z-shift (pred_z - gt_z) '
                      f'n={len(arr)}  mean={arr.mean():.3f}  '
                      f'min={arr.min():.3f}  max={arr.max():.3f}  '
                      f'mean_iou={iou_arr.mean():.3f}')
            else:
                print(f'  sample {sample_idx}: MATCHED: no predictions with IoU > 0.1')

        # ── plot ──────────────────────────────────────────────────────────────
        out_path = os.path.join(args.out_dir, f'vis3d_{i}_{sample_idx}.png')
        plot_sample(pts, gt_boxes, pred_boxes_np, pred_scores_np,
                    sample_idx, out_path, args.score_thr)

    print(f'\nDone. Saved to {args.out_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Plot BEV predictions vs ground truth from a BEVFusion checkpoint.

Usage:
    srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 \
     --account=p201222 --qos=default --time=01:00:00 --pty \
     /home/users/u103958/miniconda3/envs/multimodal-moe/bin/python \
     scripts/bev_scripts/plot_bev.py \
     outputs/runs/zod_lidar_only/zod-lidar-only_4440636/best_mAP_1.0m_epoch_14.pth \
     --score-thr 0.2

Outputs are saved to:
    outputs/runs/<run_name>/visualizations/bev_pred_vs_gt_epoch_<N>_<keyframe>.png

``run_name`` is inferred from the checkpoint path by taking the directory
name immediately containing the checkpoint file.  For instance:
    outputs/runs/zod_bevfusion/epoch_10.pth  -->  run_name = "zod_bevfusion"
"""

import argparse
import os
import os.path as osp
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

MMDET3D_DIR = osp.join(osp.dirname(__file__), '..', '..', 'mmdetection3d')
sys.path.insert(0, osp.abspath(MMDET3D_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot BEV predictions vs GT from a BEVFusion checkpoint.')
    parser.add_argument('checkpoint', help='Path to the .pth checkpoint file')
    parser.add_argument('--config', default=None,
                        help='Override config path (default: auto-detect from work_dir)')
    parser.add_argument('--score-thr', type=float, default=0.2,
                        help='Minimum prediction score to display')
    parser.add_argument('--max-samples', type=int, default=8,
                        help='Maximum number of val samples to plot')
    return parser.parse_args()


def infer_run_name(ckpt_path):
    """Infer run_name from the checkpoint directory."""
    return osp.basename(osp.dirname(osp.abspath(ckpt_path)))


def infer_epoch(ckpt_path):
    """Extract epoch number from filename like 'epoch_10.pth'."""
    basename = osp.splitext(osp.basename(ckpt_path))[0]
    m = re.search(r'epoch[_-]?(\d+)', basename)
    if m:
        return int(m.group(1))
    return basename


def find_config(ckpt_path, override=None):
    """Locate the config file for this run."""
    if override:
        return override
    run_dir = osp.dirname(osp.abspath(ckpt_path))
    for name in os.listdir(run_dir):
        if name.endswith('.py') and not name.startswith('__'):
            return osp.join(run_dir, name)
    raise FileNotFoundError(
        f'No config .py found in {run_dir}. Use --config to specify one.')


def corners_from_box(cx, cy, dx, dy, yaw):
    cos, sin = np.cos(yaw), np.sin(yaw)
    hdx, hdy = dx / 2, dy / 2
    corners = np.array([
        [-hdx, -hdy],
        [ hdx, -hdy],
        [ hdx,  hdy],
        [-hdx,  hdy],
    ])
    R = np.array([[cos, -sin], [sin, cos]])
    return (R @ corners.T).T + np.array([cx, cy])


def draw_box(ax, cx, cy, dx, dy, yaw, color='green', lw=1.0):
    pts = corners_from_box(cx, cy, dx, dy, yaw)
    poly = patches.Polygon(pts, closed=True, fill=False,
                           edgecolor=color, linewidth=lw)
    ax.add_patch(poly)


def keyframe_id(lidar_path):
    if not lidar_path:
        return 'unknown'
    return osp.splitext(osp.basename(lidar_path))[0]


def load_gt_from_info(dataset, idx):
    """Load GT boxes from dataset info, bypassing the pipeline."""
    info = dataset.get_data_info(idx)
    instances = info.get('instances', [])
    if not instances:
        return np.empty((0, 7))

    label_mapping = getattr(dataset, 'label_mapping', None)
    boxes = []
    for inst in instances:
        if not inst.get('bbox_3d_isvalid', True):
            continue
        if label_mapping is not None:
            if label_mapping.get(inst['bbox_label_3d'], -1) == -1:
                continue
        boxes.append(inst['bbox_3d'])
    if not boxes:
        return np.empty((0, 7))
    return np.array(boxes)


def main():
    args = parse_args()

    # Sparse convolutions (mmcv / spconv) are CUDA-only.  Fail early with a
    # clear message rather than crashing inside the sparse encoder.
    if not torch.cuda.is_available():
        print(
            'ERROR: No CUDA GPU detected.  plot_bev.py requires a GPU because '
            'the LiDAR sparse encoder uses mmcv sparse convolutions that are '
            'not implemented on CPU.\n'
            '\n'
            'Run on a GPU node instead, e.g.:\n'
            '  srun --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:1 \\\n'
            '       --account=p201222 --qos=default --pty \\\n'
            '       /home/users/u103958/miniconda3/envs/multimodal-moe/bin/python \\\n'
            '       scripts/bev_scripts/plot_bev.py <checkpoint> [--score-thr 0.2]\n'
            '\n'
            'Or submit as a short Slurm batch job.',
            file=sys.stderr)
        sys.exit(1)

    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    config_path = find_config(args.checkpoint, args.config)
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    if hasattr(cfg, 'custom_imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)

    from mmdet3d.registry import MODELS, DATASETS

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.eval()
    model = model.cuda()

    val_cfg = cfg.val_dataloader.dataset
    val_dataset = DATASETS.build(val_cfg)

    run_name = infer_run_name(args.checkpoint)
    epoch = infer_epoch(args.checkpoint)
    # Use an absolute path anchored to the repo root so the output location is
    # the same regardless of cwd (the compute node may resolve relative paths
    # differently from the login node).
    #repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    #out_dir = osp.join(repo_root, 'outputs', 'runs', run_name, 'visualizations')
    #os.makedirs(out_dir, exist_ok=True)
    # adjust to the project root and run name
    out_dir = "~/projects/multimodal-MoE/outputs/runs/zod_lidar_only/zod-lidar-only_4440636/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')
    n_samples = min(args.max_samples, len(val_dataset))
    print(f'Plotting {n_samples} samples from val set  '
          f'(run={run_name}, epoch={epoch}, score_thr={args.score_thr})')

    for idx in range(n_samples):
        batch = pseudo_collate([val_dataset[idx]])
        batch = _to_cuda(batch)

        with torch.no_grad():
            data = model.data_preprocessor(batch, training=False)
            results = model(**data, mode='predict')

        result = results[0]
        pred_bboxes = result.pred_instances_3d.bboxes_3d
        pred_scores = result.pred_instances_3d.scores_3d

        gt_np = load_gt_from_info(val_dataset, idx)
        points = batch['inputs']['points'][0].cpu().numpy()

        lidar_path = ''
        try:
            ds = batch['data_samples'][0]
            lidar_path = ds.metainfo.get('lidar_path', '')
        except (KeyError, IndexError, AttributeError):
            pass
        kf_id = keyframe_id(lidar_path)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.scatter(points[:, 0], points[:, 1], s=0.05, c='gray', alpha=0.3)

        n_gt = len(gt_np)
        for b in gt_np:
            draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                     color='green', lw=1.2)

        all_scores = pred_scores.cpu().numpy()
        mask = all_scores >= args.score_thr
        pred_np = (pred_bboxes.tensor.cpu().numpy()
                   if hasattr(pred_bboxes, 'tensor')
                   else pred_bboxes.cpu().numpy())
        n_pred = int(mask.sum())
        for i, b in enumerate(pred_np[mask]):
            score = all_scores[mask][i]
            draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                     color='red', lw=1.0)
            ax.text(b[0], b[1], f'{score:.2f}', fontsize=5, color='red',
                    ha='center', va='bottom')

        ax.set_aspect('equal')
        ax.set_title(
            f'BEV val pred(red) vs GT(green)  epoch {epoch}  '
            f'[keyframe {kf_id}]\n'
            f'GT: {n_gt}  |  pred (score\u2265{args.score_thr}): {n_pred}  |  '
            f'max score: {all_scores.max():.3f}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        out_path = osp.join(out_dir,
                            f'bev_pred_vs_gt_epoch_{epoch}_{kf_id}.png')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'  [{idx+1}/{n_samples}] Saved {out_path}  '
              f'(GT={n_gt}, pred={n_pred})')

    print(f'Done. Visualizations saved to {out_dir}/')


def _to_cuda(obj):
    """Recursively move tensors in nested dicts/lists to CUDA."""
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    if isinstance(obj, dict):
        return {k: _to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cuda(v) for v in obj)
    return obj


if __name__ == '__main__':
    main()

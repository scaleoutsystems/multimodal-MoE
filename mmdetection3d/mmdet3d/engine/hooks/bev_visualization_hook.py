"""Lightweight BEV visualization hooks for BEVFusion training diagnostics.

BEVFeatureVisualizationHook  – L2-norm heatmap of FPN BEV features
BEVPredictionVisualizationHook – predicted vs GT boxes in BEV over LiDAR points
"""

import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


# ── shared helpers ──────────────────────────────────────────────────────

def _unwrap_model(runner):
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    return model


def _should_visualize(runner):
    """True at epochs 1, 10, 20, 30, every 50th, and the final epoch."""
    epoch = runner.epoch + 1
    max_epochs = runner.max_epochs
    if epoch in {1, 10, 20, 30, max_epochs}:
        return True
    if epoch % 50 == 0:
        return True
    return False


def _first_sample_batch(runner):
    """Build a single-sample batch from val or train dataset[0]."""
    from mmengine.dataset import pseudo_collate
    try:
        dataset = runner.val_dataloader.dataset
    except (AttributeError, TypeError):
        dataset = runner.train_dataloader.dataset
    if len(dataset) == 0:
        return None
    return pseudo_collate([dataset[0]])


def _first_train_sample_batch(runner):
    """Build a single-sample batch from train dataset[0]."""
    from mmengine.dataset import pseudo_collate
    dataset = runner.train_dataloader.dataset
    if len(dataset) == 0:
        return None
    return pseudo_collate([dataset[0]])


def _ensure_dir(path):
    os.makedirs(osp.dirname(path), exist_ok=True)


def _preprocess_and_forward(model, batch, mode='predict'):
    data = model.data_preprocessor(batch, training=False)
    return model(**data, mode=mode), data


def _load_gt_from_info(dataset, idx):
    """Load GT boxes directly from dataset info, bypassing the pipeline.

    Uses ``dataset.get_data_info(idx)`` (works even with lazy-loaded datasets
    where ``data_list`` may be empty) and ``dataset.label_mapping`` to keep
    only instances whose raw label maps to a valid class (mapped != -1).

    Returns (N, 7) numpy array [x, y, z, dx, dy, dz, yaw] in LiDAR frame.
    """
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


def _keyframe_id_from_path(lidar_path):
    """Extract keyframe ID (e.g. '000011') from a lidar_path like
    '.../LIDAR_TOP/000011.bin'."""
    if not lidar_path:
        return 'unknown'
    return osp.splitext(osp.basename(lidar_path))[0]


# ── Hook 1: FPN feature-map heatmap ────────────────────────────────────

@HOOKS.register_module()
class BEVFeatureVisualizationHook(Hook):
    """Save L2-norm BEV heatmaps of sparse encoder and FPN outputs."""

    priority = 'LOW'

    def after_train_epoch(self, runner):
        if not _should_visualize(runner):
            return
        epoch = runner.epoch + 1
        model = _unwrap_model(runner)
        batch = _first_sample_batch(runner)
        if batch is None:
            return

        captured = {}

        def _make_hook(name):
            def _hook(module, inp, out):
                captured[name] = out
            return _hook

        h1 = model.pts_middle_encoder.register_forward_hook(
            _make_hook('sparse'))
        h2 = model.pts_neck.register_forward_hook(
            _make_hook('fpn'))

        prev_training = model.training
        model.eval()
        with torch.no_grad():
            try:
                _preprocess_and_forward(model, batch, mode='predict')
            except Exception as e:
                runner.logger.warning(
                    f'BEVFeatureVisualizationHook forward failed: {e}')
        model.train(prev_training)
        h1.remove()
        h2.remove()

        vis_dir = osp.join(runner.work_dir, 'visualizations')

        stages = [
            ('sparse', 'bev_before_backbone', 'BEV features (before backbone)'),
            ('fpn',    'bev_after_fpn',        'BEV features (after FPN)'),
        ]
        for key, filename_tag, title_prefix in stages:
            feat = captured.get(key)
            if feat is None:
                continue
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feat = feat[0]  # drop batch dim  → (C, H, W)
            heatmap = feat.norm(dim=0).cpu().numpy()

            vmax = np.percentile(heatmap[heatmap > 0], 95) \
                if (heatmap > 0).any() else 1.0

            out_path = osp.join(vis_dir,
                                f'{filename_tag}_epoch_{epoch}.png')
            _ensure_dir(out_path)

            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            im = ax.imshow(heatmap, cmap='viridis', origin='lower',
                            aspect='equal', vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel('W  (X grid)')
            ax.set_ylabel('H  (Y grid)')
            ax.set_title(f'{title_prefix} \u2013 L2 norm  (epoch {epoch})')
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            runner.logger.info(f'Saved {out_path}')


# ── Hook 2: prediction vs GT in BEV ────────────────────────────────────

def _corners_from_box(cx, cy, dx, dy, yaw):
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


def _draw_box(ax, cx, cy, dx, dy, yaw, color='green', lw=1.0):
    pts = _corners_from_box(cx, cy, dx, dy, yaw)
    poly = patches.Polygon(pts, closed=True, fill=False,
                           edgecolor=color, linewidth=lw)
    ax.add_patch(poly)


@HOOKS.register_module()
class BEVPredictionVisualizationHook(Hook):
    """Overlay predicted and GT boxes in BEV for a fixed sample."""

    priority = 'LOW'

    def __init__(self, score_thr=0.3):
        self.score_thr = score_thr

    def after_train_epoch(self, runner):
        if not _should_visualize(runner):
            return
        epoch = runner.epoch + 1
        model = _unwrap_model(runner)
        batch = _first_train_sample_batch(runner)
        if batch is None:
            return

        prev_training = model.training
        model.eval()
        with torch.no_grad():
            data = model.data_preprocessor(batch, training=False)
            results = model(**data, mode='predict')
        model.train(prev_training)

        result = results[0]
        pred_bboxes = result.pred_instances_3d.bboxes_3d
        pred_scores = result.pred_instances_3d.scores_3d

        gt_bboxes = None
        if hasattr(result, 'gt_instances_3d') and \
                hasattr(result.gt_instances_3d, 'bboxes_3d'):
            gt_bboxes = result.gt_instances_3d.bboxes_3d
        elif 'data_samples' in batch:
            ds = batch['data_samples'][0]
            if hasattr(ds, 'gt_instances_3d') and \
                    hasattr(ds.gt_instances_3d, 'bboxes_3d'):
                gt_bboxes = ds.gt_instances_3d.bboxes_3d

        points = batch['inputs']['points'][0].cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(points[:, 0], points[:, 1], s=0.05, c='gray', alpha=0.3)

        n_gt = 0
        if gt_bboxes is not None:
            gt_np = gt_bboxes.tensor.cpu().numpy() \
                if hasattr(gt_bboxes, 'tensor') else gt_bboxes.cpu().numpy()
            n_gt = len(gt_np)
            for b in gt_np:
                _draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                          color='green', lw=1.2)

        all_scores = pred_scores.cpu().numpy()
        mask = all_scores >= self.score_thr
        pred_np = pred_bboxes.tensor.cpu().numpy() \
            if hasattr(pred_bboxes, 'tensor') else pred_bboxes.cpu().numpy()
        n_pred = int(mask.sum())
        for i, b in enumerate(pred_np[mask]):
            score = all_scores[mask][i]
            _draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                      color='red', lw=1.0)
            ax.text(b[0], b[1], f'{score:.2f}', fontsize=5, color='red',
                    ha='center', va='bottom')

        lidar_path = ''
        try:
            ds = batch['data_samples'][0]
            lidar_path = ds.metainfo.get('lidar_path', '')
        except (KeyError, IndexError, AttributeError):
            pass
        keyframe = _keyframe_id_from_path(lidar_path)

        ax.set_aspect('equal')
        ax.set_title(
            f'BEV pred(red) vs GT(green)  epoch {epoch}  '
            f'[train | keyframe {keyframe}]\n'
            f'GT: {n_gt}  |  pred (score\u2265{self.score_thr}): {n_pred}  |  '
            f'max score: {all_scores.max():.3f}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        out_path = osp.join(runner.work_dir, 'visualizations',
                            f'bev_pred_vs_gt_epoch_{epoch}.png')
        _ensure_dir(out_path)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        runner.logger.info(
            f'Saved {out_path}  (GT={n_gt}, pred={n_pred}, '
            f'max_score={all_scores.max():.3f})')


# ── Hook 3: validation prediction vs GT in BEV ─────────────────────────

@HOOKS.register_module()
class BEVValPredictionVisualizationHook(Hook):
    """Overlay predicted and GT boxes in BEV for a fixed *validation* sample.

    GT annotations are loaded directly from the dataset info dict (bypassing
    the test pipeline which omits ``LoadAnnotations3D``).  The model runs in
    ``predict`` mode on the unmodified val pipeline output so predictions
    reflect true inference behaviour.
    """

    priority = 'LOW'

    def __init__(self, score_thr=0.3, sample_idx=0):
        self.score_thr = score_thr
        self.sample_idx = sample_idx

    def after_train_epoch(self, runner):
        if not _should_visualize(runner):
            return
        epoch = runner.epoch + 1
        model = _unwrap_model(runner)

        try:
            val_dataset = runner.val_dataloader.dataset
        except (AttributeError, TypeError):
            runner.logger.warning(
                'BEVValPredictionVisualizationHook: no val_dataloader')
            return

        if len(val_dataset) == 0:
            return

        idx = min(self.sample_idx, len(val_dataset) - 1)

        from mmengine.dataset import pseudo_collate
        batch = pseudo_collate([val_dataset[idx]])
        if batch is None:
            return

        prev_training = model.training
        model.eval()
        with torch.no_grad():
            data = model.data_preprocessor(batch, training=False)
            results = model(**data, mode='predict')
        model.train(prev_training)

        result = results[0]
        pred_bboxes = result.pred_instances_3d.bboxes_3d
        pred_scores = result.pred_instances_3d.scores_3d

        gt_np = _load_gt_from_info(val_dataset, idx)

        points = batch['inputs']['points'][0].cpu().numpy()

        lidar_path = ''
        try:
            ds = batch['data_samples'][0]
            lidar_path = ds.metainfo.get('lidar_path', '')
        except (KeyError, IndexError, AttributeError):
            pass
        keyframe = _keyframe_id_from_path(lidar_path)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.scatter(points[:, 0], points[:, 1], s=0.05, c='gray', alpha=0.3)

        n_gt = len(gt_np)
        for b in gt_np:
            _draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                      color='green', lw=1.2)

        all_scores = pred_scores.cpu().numpy()
        mask = all_scores >= self.score_thr
        pred_np = pred_bboxes.tensor.cpu().numpy() \
            if hasattr(pred_bboxes, 'tensor') else pred_bboxes.cpu().numpy()
        n_pred = int(mask.sum())
        for i, b in enumerate(pred_np[mask]):
            score = all_scores[mask][i]
            _draw_box(ax, b[0], b[1], b[3], b[4], b[6],
                      color='red', lw=1.0)
            ax.text(b[0], b[1], f'{score:.2f}', fontsize=5, color='red',
                    ha='center', va='bottom')

        ax.set_aspect('equal')
        ax.set_title(
            f'BEV pred(red) vs GT(green)  epoch {epoch}  '
            f'[val | keyframe {keyframe}]\n'
            f'GT: {n_gt}  |  pred (score\u2265{self.score_thr}): {n_pred}  |  '
            f'max score: {all_scores.max():.3f}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        out_path = osp.join(runner.work_dir, 'visualizations',
                            f'bev_val_pred_vs_gt_epoch_{epoch}.png')
        _ensure_dir(out_path)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        runner.logger.info(
            f'Saved {out_path}  (GT={n_gt}, pred={n_pred}, '
            f'max_score={all_scores.max():.3f})')

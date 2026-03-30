"""Camera-BEV and fusion visualization hooks for BEVFusion training diagnostics.

BEVCameraFeatureVisualizationHook
    L2-norm heatmaps of camera BEV features (after DepthLSSTransform)
    and fused BEV features (after ConvFuser).

DepthTransformDiagnosticHook
    Multi-panel diagnostic for DepthLSSTransform: sparse depth input,
    predicted depth map, depth-distribution entropy, and processed
    depth-feature activation.  Designed to surface common camera→BEV
    projection failures (incorrect geometry, collapsed depth, noisy lifting).
"""

import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

from .bev_visualization_hook import (
    _ensure_dir,
    _first_sample_batch,
    _preprocess_and_forward,
    _should_visualize,
)


def _unwrap(runner):
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    return model


# ── Hook 1: camera BEV + fused BEV heatmaps ────────────────────────────

@HOOKS.register_module()
class BEVCameraFeatureVisualizationHook(Hook):
    """L2-norm heatmaps of camera BEV and fused BEV features.

    Captures the output of ``view_transform`` (camera BEV, e.g. 80-ch)
    and ``fusion_layer`` (fused BEV, e.g. 256-ch) via forward hooks.
    Runs on the same epoch schedule as ``BEVFeatureVisualizationHook``.
    """

    priority = 'LOW'

    def after_train_epoch(self, runner):
        if not _should_visualize(runner):
            return
        epoch = runner.epoch + 1
        model = _unwrap(runner)

        if getattr(model, 'view_transform', None) is None:
            return

        batch = _first_sample_batch(runner)
        if batch is None:
            return

        captured = {}

        def _make_hook(name):
            def _hook(_module, _inp, out):
                captured[name] = out
            return _hook

        handles = []
        handles.append(
            model.view_transform.register_forward_hook(
                _make_hook('camera_bev')))
        if getattr(model, 'fusion_layer', None) is not None:
            handles.append(
                model.fusion_layer.register_forward_hook(
                    _make_hook('fused_bev')))

        prev_training = model.training
        model.eval()
        with torch.no_grad():
            try:
                _preprocess_and_forward(model, batch, mode='predict')
            except Exception as e:
                runner.logger.warning(
                    f'BEVCameraFeatureVisualizationHook forward failed: {e}')
        model.train(prev_training)
        for h in handles:
            h.remove()

        vis_dir = osp.join(runner.work_dir, 'visualizations')
        stages = [
            ('camera_bev', 'camera_bev_features',
             'Camera BEV (after DepthLSSTransform)'),
            ('fused_bev', 'fused_bev_features',
             'Fused BEV (after ConvFuser)'),
        ]

        for key, filename_tag, title_prefix in stages:
            feat = captured.get(key)
            if feat is None:
                continue
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            feat = feat[0]  # drop batch dim → (C, H, W)
            heatmap = feat.float().norm(dim=0).cpu().numpy()

            vmax = (np.percentile(heatmap[heatmap > 0], 95)
                    if (heatmap > 0).any() else 1.0)

            out_path = osp.join(
                vis_dir, f'{filename_tag}_epoch_{epoch}.png')
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


# ── Hook 2: DepthLSSTransform diagnostic ────────────────────────────────

@HOOKS.register_module()
class DepthTransformDiagnosticHook(Hook):
    """Multi-panel diagnostic for camera → BEV projection quality.

    Panels (2×2 figure):
      1. Sparse LiDAR depth projected onto the image (dtransform input).
      2. Predicted depth map (argmax of depthnet softmax).
      3. Depth-distribution entropy (per feature-pixel confidence).
      4. L2 norm of processed depth features (dtransform output).

    Requires that ``model.view_transform`` is a ``DepthLSSTransform``
    (or compatible) with ``dtransform`` and ``depthnet`` sub-modules.
    """

    priority = 'LOW'

    def after_train_epoch(self, runner):
        if not _should_visualize(runner):
            return
        epoch = runner.epoch + 1
        model = _unwrap(runner)

        vt = getattr(model, 'view_transform', None)
        if vt is None:
            return
        if not (hasattr(vt, 'dtransform') and hasattr(vt, 'depthnet')):
            return

        batch = _first_sample_batch(runner)
        if batch is None:
            return

        captured = {}

        def _dt_hook(_module, inp, out):
            captured['sparse_depth'] = inp[0].detach()
            captured['dt_out'] = out.detach()

        def _dn_hook(_module, _inp, out):
            captured['dn_out'] = out.detach()

        h1 = vt.dtransform.register_forward_hook(_dt_hook)
        h2 = vt.depthnet.register_forward_hook(_dn_hook)

        prev_training = model.training
        model.eval()
        with torch.no_grad():
            try:
                _preprocess_and_forward(model, batch, mode='predict')
            except Exception as e:
                runner.logger.warning(
                    f'DepthTransformDiagnosticHook forward failed: {e}')
        model.train(prev_training)
        h1.remove()
        h2.remove()

        D = vt.D
        dbound = vt.dbound
        vis_dir = osp.join(runner.work_dir, 'visualizations')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # --- Panel 1: sparse LiDAR depth on image ---
        sd = captured.get('sparse_depth')
        if sd is not None:
            sd_np = sd[0, 0].cpu().float().numpy()  # cam 0, channel 0
            ax = axes[0, 0]
            im = ax.imshow(sd_np, cmap='magma', origin='upper', aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8, label='depth (m)')
            ax.set_title('Sparse LiDAR depth on image')
            nz = sd_np[sd_np > 0]
            stats = f'pixels with depth: {len(nz)}'
            if len(nz) > 0:
                stats += f'\nrange: [{nz.min():.1f}, {nz.max():.1f}] m'
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=8,
                    va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        else:
            axes[0, 0].set_visible(False)

        # --- Panel 2: predicted depth (argmax) ---
        dn = captured.get('dn_out')
        if dn is not None:
            dn0 = dn[0].float()  # cam 0: (D+C, fH, fW)
            depth_logits = dn0[:D]
            depth_dist = torch.softmax(depth_logits, dim=0)
            depth_bins = torch.arange(
                dbound[0], dbound[1], dbound[2], device='cpu')
            if depth_bins.shape[0] > D:
                depth_bins = depth_bins[:D]
            depth_map = depth_bins[
                depth_dist.argmax(dim=0).cpu()].numpy()

            ax = axes[0, 1]
            im = ax.imshow(depth_map, cmap='turbo', origin='upper',
                           aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8, label='depth (m)')
            ax.set_title('Predicted depth (argmax of softmax)')

            # --- Panel 3: entropy of depth distribution ---
            eps = 1e-8
            entropy = -(depth_dist * (depth_dist + eps).log()
                        ).sum(dim=0).cpu().numpy()

            ax = axes[1, 0]
            im = ax.imshow(entropy, cmap='hot', origin='upper',
                           aspect='auto')
            fig.colorbar(im, ax=ax, shrink=0.8, label='entropy (nats)')
            ax.set_title('Depth distribution entropy')
            ax.text(0.02, 0.98,
                    f'mean: {entropy.mean():.2f}\nmax: {entropy.max():.2f}',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        else:
            axes[0, 1].set_visible(False)
            axes[1, 0].set_visible(False)

        # --- Panel 4: dtransform output (processed depth features) ---
        dt = captured.get('dt_out')
        if dt is not None:
            dt0 = dt[0].float()  # cam 0: (64, fH, fW)
            dt_norm = dt0.norm(dim=0).cpu().numpy()
            vmax = (np.percentile(dt_norm[dt_norm > 0], 95)
                    if (dt_norm > 0).any() else 1.0)
            ax = axes[1, 1]
            im = ax.imshow(dt_norm, cmap='viridis', origin='upper',
                           aspect='auto', vmin=0, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title('Processed depth features (L2 norm)')
        else:
            axes[1, 1].set_visible(False)

        out_path = osp.join(
            vis_dir, f'depth_transform_diagnostic_epoch_{epoch}.png')
        _ensure_dir(out_path)
        fig.suptitle(
            f'DepthLSSTransform diagnostic \u2013 epoch {epoch}', fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        runner.logger.info(f'Saved {out_path}')

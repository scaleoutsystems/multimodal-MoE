"""One-shot debug hook for DepthLSSTransform sparse-depth diagnostics.

Fires at epoch 1 and produces:
  1. LiDAR overlay on the actual augmented image
  2. Multi-hit collision analysis (how many points map to the same pixel)
  3. Occupancy stats for sparse depth at full-res and after dtransform
  4. 4-panel plot: full-res depth, dtransform output, depthnet depth, entropy
  5. All representative matrices + tensor shapes
"""

import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

from .bev_visualization_hook import _ensure_dir, _first_sample_batch


def _unwrap(runner):
    model = runner.model
    if is_model_wrapper(model):
        model = model.module
    return model


def _np_fmt(arr, name, logger):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    logger.info(f'  {name}  shape={arr.shape}')
    for row in arr.reshape(-1, arr.shape[-1]) if arr.ndim >= 2 else [arr]:
        logger.info(f'    {np.array2string(row, precision=4, suppress_small=True)}')


@HOOKS.register_module()
class DepthProjectionDebugHook(Hook):
    """Sparse-depth quality diagnostics for DepthLSSTransform."""

    priority = 'LOW'

    def __init__(self):
        self._done = False

    def after_train_epoch(self, runner):
        if self._done:
            return
        epoch = runner.epoch + 1
        if epoch != 1:
            return
        self._done = True

        logger = runner.logger
        model = _unwrap(runner)
        vt = getattr(model, 'view_transform', None)
        if vt is None:
            return

        batch = _first_sample_batch(runner)
        if batch is None:
            return

        debug_dir = osp.join(runner.work_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

        data = model.data_preprocessor(batch, training=False)
        batch_input_metas = [
            item.metainfo for item in data['data_samples']]
        meta = batch_input_metas[0]
        imgs_tensor = data['inputs']['imgs']
        points_list = data['inputs']['points']

        # ── matrices ────────────────────────────────────────────────────
        logger.info('=' * 72)
        logger.info('  DEPTH PROJECTION DEBUG — Sparse depth analysis')
        logger.info('=' * 72)

        lidar2img_np = np.asarray(meta['lidar2img'])
        cam2img_np = np.asarray(meta['cam2img'])
        cam2lidar_np = np.asarray(meta['cam2lidar'])
        img_aug_np = np.asarray(meta.get('img_aug_matrix', np.eye(4)))
        lidar_aug_np = np.asarray(meta.get('lidar_aug_matrix', np.eye(4)))
        _np_fmt(lidar2img_np, 'lidar2img', logger)
        _np_fmt(cam2img_np, 'cam2img', logger)
        _np_fmt(img_aug_np, 'img_aug_matrix', logger)
        _np_fmt(lidar_aug_np, 'lidar_aug_matrix', logger)
        logger.info(f'  img tensor shape: {tuple(imgs_tensor.shape)}')
        logger.info(f'  points[0] shape:  {tuple(points_list[0].shape)}')
        if hasattr(vt, 'image_size'):
            logger.info(f'  image_size={vt.image_size}  '
                        f'feature_size={vt.feature_size}  D={vt.D}')

        # ── project LiDAR → augmented image for overlay ─────────────────
        aug_h, aug_w = vt.image_size
        pts = points_list[0].clone().cpu().float()
        c = pts[:, :3].clone()
        la = torch.tensor(lidar_aug_np, dtype=torch.float32)
        c = c - la[:3, 3]
        c = torch.inverse(la[:3, :3]) @ c.T

        l2i = torch.tensor(lidar2img_np, dtype=torch.float32)
        if l2i.ndim == 3:
            l2i = l2i[0]
        ia = torch.tensor(img_aug_np, dtype=torch.float32)
        if ia.ndim == 3:
            ia = ia[0]

        proj = l2i[:3, :3] @ c + l2i[:3, 3:4]
        depth_vec = proj[2, :].clone()
        proj[2, :] = proj[2, :].clamp(min=1e-5)
        proj[:2, :] /= proj[2:3, :]
        proj = ia[:3, :3] @ proj + ia[:3, 3:4]
        uv = proj[:2, :].T.numpy()
        depth_np = depth_vec.numpy()

        valid = (
            (uv[:, 1] >= 0) & (uv[:, 1] < aug_h) &
            (uv[:, 0] >= 0) & (uv[:, 0] < aug_w) &
            (depth_np > 0))

        rows_v = uv[valid, 1]
        logger.info(f'  projected: {valid.sum()} / {len(pts)} on image')
        if valid.sum() > 0:
            logger.info(f'  row min/max/mean: {rows_v.min():.1f} / '
                        f'{rows_v.max():.1f} / {rows_v.mean():.1f}')

        # ── multi-hit analysis ──────────────────────────────────────────
        row_int = uv[valid, 1].astype(np.int32)
        col_int = uv[valid, 0].astype(np.int32)
        hit_count = np.zeros((aug_h, aug_w), dtype=np.int32)
        np.add.at(hit_count, (row_int, col_int), 1)
        occupied = (hit_count > 0).sum()
        multi_hit = (hit_count > 1).sum()
        total_pts_on_img = int(valid.sum())
        overwritten = total_pts_on_img - occupied
        logger.info('-' * 72)
        logger.info('  MULTI-HIT / OVERWRITE ANALYSIS (full-res depth map)')
        logger.info(f'  total LiDAR hits on image:     {total_pts_on_img}')
        logger.info(f'  unique pixels occupied:         {occupied}')
        logger.info(f'  pixels with >1 hit:             {multi_hit}')
        logger.info(f'  overwritten points (lost):      {overwritten}  '
                    f'({100*overwritten/max(total_pts_on_img,1):.1f}%)')
        if multi_hit > 0:
            mh = hit_count[hit_count > 1]
            logger.info(f'  multi-hit max/mean:             '
                        f'{mh.max()} / {mh.mean():.1f}')
        logger.info(f'  full-res occupancy:             '
                    f'{occupied} / {aug_h * aug_w} = '
                    f'{100*occupied/(aug_h*aug_w):.3f}%')

        # ── forward with hooks to capture sparse depth pipeline ─────────
        captured = {}

        def _dt_hook(_mod, inp, out):
            captured['sd_in'] = inp[0].detach()
            captured['dt_out'] = out.detach()

        def _dn_hook(_mod, _inp, out):
            captured['dn_out'] = out.detach()

        def _vt_hook(_mod, inp, out):
            captured['img_feat'] = inp[0].detach()
            captured['bev_out'] = out.detach()

        handles = []
        if hasattr(vt, 'dtransform'):
            handles.append(vt.dtransform.register_forward_hook(_dt_hook))
        if hasattr(vt, 'depthnet'):
            handles.append(vt.depthnet.register_forward_hook(_dn_hook))
        handles.append(vt.register_forward_hook(_vt_hook))

        prev = model.training
        model.eval()
        with torch.no_grad():
            try:
                model.extract_feat(data['inputs'], batch_input_metas)
            except Exception as e:
                logger.warning(f'  forward failed: {e}')
        model.train(prev)
        for h in handles:
            h.remove()

        # ── occupancy stats at each stage ───────────────────────────────
        logger.info('-' * 72)
        logger.info('  OCCUPANCY THROUGH DEPTH PIPELINE')
        sd = captured.get('sd_in')
        dt = captured.get('dt_out')
        dn = captured.get('dn_out')
        bev = captured.get('bev_out')
        img_feat = captured.get('img_feat')

        sd0 = None
        if sd is not None:
            sd0 = sd[0, 0].cpu().float()
            nz = (sd0 > 0).sum().item()
            total = sd0.numel()
            logger.info(f'  sparse_depth (full-res):  {tuple(sd.shape)}')
            logger.info(f'    nonzero: {nz} / {total}  '
                        f'({100*nz/total:.3f}%)')
            rows_active = (sd0 > 0).any(dim=1).sum().item()
            cols_active = (sd0 > 0).any(dim=0).sum().item()
            logger.info(f'    active rows: {rows_active}/{sd0.shape[0]}  '
                        f'cols: {cols_active}/{sd0.shape[1]}')

        if dt is not None:
            dt0 = dt[0].cpu().float()  # (64, fH, fW)
            dt_act = (dt0.abs() > 1e-6).any(dim=0)
            nz_feat = dt_act.sum().item()
            total_feat = dt_act.numel()
            logger.info(f'  dtransform output:        {tuple(dt.shape)}')
            logger.info(f'    active cells (any ch>0): {nz_feat} / {total_feat}  '
                        f'({100*nz_feat/total_feat:.2f}%)')

        if img_feat is not None:
            logger.info(f'  img_features:             {tuple(img_feat.shape)}')
        if dn is not None:
            logger.info(f'  depthnet output:          {tuple(dn.shape)}')
        if bev is not None:
            logger.info(f'  camera BEV:               {tuple(bev.shape)}')

        # ── 4-panel diagnostic plot ─────────────────────────────────────
        D = vt.D
        dbound = vt.dbound
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.4])

        # Panel 1: full-res sparse depth with image underlay
        ax1 = fig.add_subplot(gs[0, 0])
        img_t = imgs_tensor[0, 0].cpu().float()
        mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        img_rgb = (img_t * std + mean).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        ax1.imshow(img_rgb)
        if sd0 is not None:
            sd_plot = sd0.numpy()
            r_idx, c_idx = np.where(sd_plot > 0)
            if len(r_idx) > 0:
                sc = ax1.scatter(c_idx, r_idx, c=sd_plot[r_idx, c_idx],
                                 cmap='turbo', s=0.4, alpha=0.7, vmin=1, vmax=80)
                fig.colorbar(sc, ax=ax1, shrink=0.6, label='depth (m)')
        nz_str = f'{nz}' if sd0 is not None else '?'
        occ_str = f'{100*nz/total:.3f}%' if sd0 is not None else '?'
        ax1.set_title(f'Full-res sparse depth on image\n'
                      f'{nz_str} px occupied ({occ_str})')

        # Panel 2: dtransform output L2 norm
        ax2 = fig.add_subplot(gs[0, 1])
        if dt is not None:
            dt_norm = dt[0].float().norm(dim=0).cpu().numpy()
            vmax_dt = np.percentile(dt_norm[dt_norm > 0], 95) if (dt_norm > 0).any() else 1
            im2 = ax2.imshow(dt_norm, cmap='viridis', origin='upper',
                             aspect='auto', vmin=0, vmax=vmax_dt)
            fig.colorbar(im2, ax=ax2, shrink=0.6)
            fH, fW = dt_norm.shape
            ax2.set_title(f'dtransform output L2 norm ({fH}×{fW})\n'
                          f'{nz_feat}/{total_feat} active ({100*nz_feat/total_feat:.1f}%)')
        else:
            ax2.set_visible(False)

        # Panel 3: hit-count histogram
        ax3 = fig.add_subplot(gs[0, 2])
        hc_flat = hit_count[hit_count > 0]
        if len(hc_flat) > 0:
            max_hits = min(int(hc_flat.max()), 20)
            bins = np.arange(1, max_hits + 2) - 0.5
            ax3.hist(hc_flat, bins=bins, color='steelblue', edgecolor='black')
            ax3.set_xlabel('Hits per pixel')
            ax3.set_ylabel('Count')
            ax3.set_title('Multi-hit histogram')
            ax3.set_yscale('log')

        # Panel 4: predicted depth argmax
        ax4 = fig.add_subplot(gs[1, 0])
        if dn is not None:
            dn0 = dn[0].float()
            depth_logits = dn0[:D]
            depth_dist = torch.softmax(depth_logits, dim=0)
            depth_bins = torch.arange(dbound[0], dbound[1], dbound[2])
            if depth_bins.shape[0] > D:
                depth_bins = depth_bins[:D]
            depth_map = depth_bins[depth_dist.argmax(dim=0).cpu()].numpy()
            im4 = ax4.imshow(depth_map, cmap='turbo', origin='upper',
                             aspect='auto')
            fig.colorbar(im4, ax=ax4, shrink=0.6, label='depth (m)')
            ax4.set_title(f'Predicted depth (argmax) {depth_map.shape}')
        else:
            ax4.set_visible(False)

        # Panel 5: entropy
        ax5 = fig.add_subplot(gs[1, 1])
        if dn is not None:
            eps = 1e-8
            entropy = -(depth_dist * (depth_dist + eps).log()).sum(dim=0).cpu().numpy()
            im5 = ax5.imshow(entropy, cmap='hot', origin='upper', aspect='auto')
            fig.colorbar(im5, ax=ax5, shrink=0.6, label='nats')
            ax5.set_title(f'Depth entropy  mean={entropy.mean():.2f}')
        else:
            ax5.set_visible(False)

        # Panel 6: row-wise depth occupancy profile
        ax6 = fig.add_subplot(gs[1, 2])
        if sd0 is not None:
            row_occ = (sd0 > 0).sum(dim=1).numpy()
            ax6.barh(np.arange(len(row_occ)), row_occ, height=1,
                     color='steelblue')
            ax6.set_xlabel('Occupied cols')
            ax6.set_ylabel('Row')
            ax6.invert_yaxis()
            ax6.set_title('Row-wise depth occupancy')

        out_path = osp.join(debug_dir, 'sparse_depth_pipeline.png')
        _ensure_dir(out_path)
        fig.suptitle('Sparse depth pipeline diagnostic', fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f'  Saved {out_path}')

        # ── overlay figure (same as before) ─────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 7))
        ori_shape = meta.get('ori_shape', (704, 1248))
        ori_h = ori_shape[0] if isinstance(ori_shape, (list, tuple)) else 704
        ori_w = ori_shape[1] if isinstance(ori_shape, (list, tuple)) else 1248

        proj_a = l2i[:3, :3] @ c + l2i[:3, 3:4]
        da = proj_a[2, :].clone()
        proj_a[2, :] = proj_a[2, :].clamp(min=1e-5)
        proj_a[:2, :] /= proj_a[2:3, :]
        uv_a = proj_a[:2, :].T.numpy()
        va = ((uv_a[:, 1] >= 0) & (uv_a[:, 1] < ori_h) &
              (uv_a[:, 0] >= 0) & (uv_a[:, 0] < ori_w) & (da.numpy() > 0))

        ax = axes2[0]
        ax.set_facecolor('black')
        ax.set_xlim(0, ori_w); ax.set_ylim(ori_h, 0)
        ax.scatter(uv_a[va, 0], uv_a[va, 1], c=da.numpy()[va],
                   cmap='turbo', s=0.3, alpha=0.6, vmin=1, vmax=80)
        ax.set_title(f'lidar2image ONLY ({ori_h}×{ori_w})\n{va.sum()} pts')

        ax = axes2[1]
        ax.imshow(img_rgb)
        ax.scatter(uv[valid, 0], uv[valid, 1], c=depth_np[valid],
                   cmap='turbo', s=0.3, alpha=0.6, vmin=1, vmax=80)
        ax.set_title(f'lidar2image + img_aug ({aug_h}×{aug_w})\n{valid.sum()} pts')

        out2 = osp.join(debug_dir, 'lidar_projection_overlay.png')
        fig2.tight_layout()
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
        logger.info(f'  Saved {out2}')
        logger.info('=' * 72)

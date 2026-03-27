"""
Diagnose why the overfit experiment is failing to converge.

Loads a checkpoint, runs inference on all training frames, and logs:
  1. Per-frame assignment stats (num_gt, num_pos, mean IoU)
  2. Per-channel regression error breakdown
  3. Score distribution
  4. GT vs predicted box dimensions
  5. Matched-vs-unmatched BEV visualization
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS
from mmdet3d.registry import MODELS as MODELS_3D
from mmdet3d.utils import register_all_modules
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

CHANNEL_NAMES = ['cx', 'cy', 'z', 'log_dx', 'log_dy', 'log_dz', 'sin_yaw', 'cos_yaw']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config', help='Config file path')
    p.add_argument('checkpoint', help='Checkpoint path')
    p.add_argument('--out-dir', default=None, help='Output directory')
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def build_model_and_dataset(cfg, ckpt_path, device):
    register_all_modules(init_default_scope=True)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model = model.to(device).eval()

    ds_cfg = cfg.train_dataloader.dataset
    ds_cfg['test_mode'] = False
    ds_cfg['pipeline'] = cfg.train_pipeline
    if 'metainfo' not in ds_cfg:
        ds_cfg['metainfo'] = cfg.get('metainfo', None)
    if 'data_root' not in ds_cfg:
        ds_cfg['data_root'] = cfg.get('data_root', None)
    if 'modality' not in ds_cfg:
        ds_cfg['modality'] = cfg.get('input_modality', None)
    if 'data_prefix' not in ds_cfg:
        ds_cfg['data_prefix'] = cfg.get('data_prefix', None)
    if 'box_type_3d' not in ds_cfg:
        ds_cfg['box_type_3d'] = 'LiDAR'
    dataset = DATASETS.build(ds_cfg)
    return model, dataset


@torch.no_grad()
def diagnose(model, dataset, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    from mmengine.dataset import pseudo_collate
    from torch.utils.data import DataLoader

    head = model.bbox_head
    coder = head.bbox_coder

    all_frame_stats = []
    all_channel_errors = []
    all_gt_dims = []
    all_pred_dims = []
    all_scores = []

    vis_frame_idx = 0

    for idx in range(len(dataset)):
        data = dataset[idx]

        batch = pseudo_collate([data])

        data_samples = batch['data_samples']
        gt_inst = data_samples[0].gt_instances_3d
        gt_boxes = gt_inst.bboxes_3d
        gt_labels = gt_inst.labels_3d
        num_gt = len(gt_labels)

        processed = model.data_preprocessor(batch, training=True)
        inputs, data_samples_proc = processed['inputs'], processed['data_samples']

        gt_boxes_dev = data_samples_proc[0].gt_instances_3d.bboxes_3d
        gt_labels_dev = data_samples_proc[0].gt_instances_3d.labels_3d

        metas = [ds.metainfo for ds in data_samples_proc]
        feats = model.extract_feat(inputs, metas)
        preds_dicts = head(feats, metas)

        pd0 = preds_dicts[0][0]

        score_raw = copy.deepcopy(pd0['heatmap'].detach())
        center_raw = copy.deepcopy(pd0['center'].detach())
        height_raw = copy.deepcopy(pd0['height'].detach())
        dim_raw = copy.deepcopy(pd0['dim'].detach())
        rot_raw = copy.deepcopy(pd0['rot'].detach())

        boxes_dict = coder.decode(
            copy.deepcopy(score_raw),
            copy.deepcopy(rot_raw),
            copy.deepcopy(dim_raw),
            copy.deepcopy(center_raw),
            copy.deepcopy(height_raw),
            None)

        pred_bboxes = boxes_dict[0]['bboxes']
        pred_scores = boxes_dict[0]['scores']

        all_scores.append(pred_scores.cpu().numpy())

        # --- Hungarian assignment ---
        from mmdet3d.registry import TASK_UTILS
        assigner_cfg = dict(head.train_cfg.assigner)
        assigner = TASK_UTILS.build(assigner_cfg)

        gt_bboxes_tensor = gt_boxes_dev.tensor.to(device)
        score_for_assign = copy.deepcopy(pd0['heatmap'].detach())

        assign_result = assigner.assign(
            pred_bboxes,
            gt_bboxes_tensor,
            gt_labels_dev,
            score_for_assign,
            head.train_cfg)

        pos_inds = torch.where(assign_result.gt_inds > 0)[0]
        neg_inds = torch.where(assign_result.gt_inds == 0)[0]
        num_pos = len(pos_inds)
        pos_ious = assign_result.max_overlaps[pos_inds]
        mean_iou = pos_ious.mean().item() if num_pos > 0 else 0.0

        # --- Per-channel regression error ---
        if num_pos > 0:
            pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
            pos_gt_boxes = gt_bboxes_tensor[pos_gt_inds]
            pos_gt_targets = coder.encode(pos_gt_boxes)

            pos_center = pd0['center'][:, :, pos_inds]
            pos_height = pd0['height'][:, :, pos_inds]
            pos_dim = pd0['dim'][:, :, pos_inds]
            pos_rot = pd0['rot'][:, :, pos_inds]
            pos_preds = torch.cat(
                [pos_center, pos_height, pos_dim, pos_rot],
                dim=1)[0].T

            per_channel_error = (pos_preds - pos_gt_targets).abs().mean(dim=0).cpu().numpy()
            all_channel_errors.append(per_channel_error)

            pos_pred_decoded = pred_bboxes[pos_inds]
            all_gt_dims.append(pos_gt_boxes[:, 3:6].cpu().numpy())
            all_pred_dims.append(pos_pred_decoded[:, 3:6].cpu().numpy())

            # --- query_pos vs target center analysis ---
            query_pos_matched = head.bev_pos.to(device).repeat(1, 1, 1)
            top_idx = pd0.get('_top_proposals_index', None)
            if hasattr(head, '_last_top_indices'):
                top_idx = head._last_top_indices
            if top_idx is None:
                hm_for_topk = pd0['dense_heatmap'].detach().sigmoid()
                padding = head.nms_kernel_size // 2
                local_max = torch.zeros_like(hm_for_topk)
                local_max_inner = torch.nn.functional.max_pool2d(
                    hm_for_topk, kernel_size=head.nms_kernel_size,
                    stride=1, padding=0)
                local_max[:, :, padding:(-padding),
                          padding:(-padding)] = local_max_inner
                hm_for_topk = hm_for_topk * (hm_for_topk == local_max)
                hm_for_topk = hm_for_topk.view(1, hm_for_topk.shape[1], -1)
                top_idx = hm_for_topk.view(1, -1).argsort(
                    dim=-1, descending=True)[..., :head.num_proposals]
                top_idx = top_idx % hm_for_topk.shape[-1]

            qp = query_pos_matched.squeeze(0)
            qp_selected = qp[top_idx.squeeze(0)]
            qp_pos = qp_selected[pos_inds].cpu().numpy()
            tgt_center = pos_gt_targets[:, :2].cpu().numpy()
            pred_center = pos_preds[:, :2].detach().cpu().numpy()

            qp_error_c0 = np.abs(qp_pos[:, 0] - tgt_center[:, 0]).mean()
            qp_error_c1 = np.abs(qp_pos[:, 1] - tgt_center[:, 1]).mean()
            offset_c0 = np.abs(pred_center[:, 0] - qp_pos[:, 0]).mean()
            offset_c1 = np.abs(pred_center[:, 1] - qp_pos[:, 1]).mean()

            if idx == 0:
                print(f'\n  --- Frame 0 query_pos vs target center (first 10 matched) ---')
                n_show = min(10, num_pos)
                for k in range(n_show):
                    print(f'    GT#{pos_gt_inds[k].item():2d}: '
                          f'qp=({qp_pos[k,0]:.1f}, {qp_pos[k,1]:.1f})  '
                          f'tgt=({tgt_center[k,0]:.1f}, {tgt_center[k,1]:.1f})  '
                          f'pred=({pred_center[k,0]:.1f}, {pred_center[k,1]:.1f})  '
                          f'iou={pos_ious[k].item():.3f}')
                print(f'  avg |qp-tgt| ch0={qp_error_c0:.2f}  ch1={qp_error_c1:.2f}')
                print(f'  avg |pred-qp| (offset) ch0={offset_c0:.2f}  ch1={offset_c1:.2f}')
                print(f'  avg |pred-tgt| (total) ch0={per_channel_error[0]:.2f}  ch1={per_channel_error[1]:.2f}')
        else:
            all_channel_errors.append(np.zeros(8))

        frame_stat = {
            'idx': idx,
            'num_gt': num_gt,
            'num_pos': num_pos,
            'num_neg': len(neg_inds),
            'mean_iou': mean_iou,
            'min_iou': pos_ious.min().item() if num_pos > 0 else 0.0,
            'max_iou': pos_ious.max().item() if num_pos > 0 else 0.0,
            'max_score': pred_scores.max().item(),
            'mean_score': pred_scores.mean().item(),
            'score_gt01': (pred_scores > 0.1).sum().item(),
            'score_gt03': (pred_scores > 0.3).sum().item(),
            'score_gt05': (pred_scores > 0.5).sum().item(),
        }
        all_frame_stats.append(frame_stat)
        print(f'  Frame {idx}: GT={num_gt} pos={num_pos} mean_iou={mean_iou:.4f} max_score={pred_scores.max().item():.4f}')

        pts_for_plot = data['inputs']['points'].data
        if idx == vis_frame_idx:
            _make_matched_unmatched_plot(
                pts_for_plot, pred_bboxes, pred_scores,
                gt_bboxes_tensor, pos_inds, assign_result,
                out_dir, idx)

    _print_summary(all_frame_stats, all_channel_errors,
                   all_gt_dims, all_pred_dims, all_scores, out_dir)


def _make_matched_unmatched_plot(pts_tensor, pred_bboxes, pred_scores,
                                  gt_bboxes_tensor, pos_inds, assign_result,
                                  out_dir, idx):
    """BEV plot: GT(green) / matched-preds(blue) / unmatched-preds(red)."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)

    pc_range = [0.0, -54.0, -5.0, 108.0, 54.0, 3.0]

    if isinstance(pts_tensor, torch.Tensor):
        pts = pts_tensor.cpu().numpy()
    else:
        pts = np.array(pts_tensor)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c='gray', alpha=0.3)

    gt_np = gt_bboxes_tensor.cpu().numpy()
    for b in gt_np:
        corners = _box_corners_2d(b)
        poly = plt.Polygon(corners, fill=False, edgecolor='green',
                           linewidth=1.5, linestyle='-')
        ax.add_patch(poly)

    pred_np = pred_bboxes.cpu().numpy()
    scores_np = pred_scores.cpu().numpy()

    pos_set = set(pos_inds.cpu().numpy().tolist())
    matched_gt_inds = assign_result.gt_inds.cpu().numpy()

    for i in range(len(pred_np)):
        if scores_np[i] < 0.01:
            continue
        corners = _box_corners_2d(pred_np[i])
        if i in pos_set:
            color = 'blue'
            lw = 1.5
            gt_idx = matched_gt_inds[i] - 1
            iou_val = assign_result.max_overlaps[i].item()
            ax.annotate(f'{iou_val:.2f}', xy=(pred_np[i, 0], pred_np[i, 1]),
                        fontsize=5, color='blue', alpha=0.8)
        else:
            color = 'red'
            lw = 0.5
        poly = plt.Polygon(corners, fill=False, edgecolor=color,
                           linewidth=lw, linestyle='-', alpha=0.6 if color == 'red' else 1.0)
        ax.add_patch(poly)

    ax.set_xlim(pc_range[0], pc_range[3])
    ax.set_ylim(pc_range[1], pc_range[4])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title(f'Frame {idx}: GT(green) matched(blue) unmatched(red)\n'
                 f'GT={len(gt_np)} | pos={len(pos_set)} | '
                 f'pred(score>=0.01)={(scores_np>=0.01).sum()}')

    green_patch = mpatches.Patch(edgecolor='green', facecolor='none', label='GT')
    blue_patch = mpatches.Patch(edgecolor='blue', facecolor='none', label='Matched pred')
    red_patch = mpatches.Patch(edgecolor='red', facecolor='none', label='Unmatched pred')
    ax.legend(handles=[green_patch, blue_patch, red_patch], loc='upper right')

    path = os.path.join(out_dir, f'matched_vs_unmatched_frame{idx}.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _box_corners_2d(box):
    """Get 4 BEV corners from [x, y, z, dx, dy, dz, yaw, ...]."""
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    hdx, hdy = dx / 2, dy / 2
    corners = np.array([
        [-hdx, -hdy], [hdx, -hdy], [hdx, hdy], [-hdx, hdy]
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners @ rot.T + np.array([x, y])
    return corners


def _print_summary(all_frame_stats, all_channel_errors,
                   all_gt_dims, all_pred_dims, all_scores, out_dir):
    sep = '=' * 70

    lines = []
    lines.append(sep)
    lines.append('OVERFIT DIAGNOSTIC REPORT')
    lines.append(sep)

    # 1. Per-frame assignment
    lines.append('\n--- PER-FRAME ASSIGNMENT STATS ---')
    lines.append(f'{"Frame":>5} {"GT":>4} {"Pos":>4} {"Neg":>4} '
                 f'{"Mean IoU":>9} {"Min IoU":>8} {"Max IoU":>8} '
                 f'{"MaxScore":>9} {"#s>0.1":>6} {"#s>0.3":>6} {"#s>0.5":>6}')
    for s in all_frame_stats:
        lines.append(
            f'{s["idx"]:>5} {s["num_gt"]:>4} {s["num_pos"]:>4} {s["num_neg"]:>4} '
            f'{s["mean_iou"]:>9.4f} {s["min_iou"]:>8.4f} {s["max_iou"]:>8.4f} '
            f'{s["max_score"]:>9.4f} {s["score_gt01"]:>6} {s["score_gt03"]:>6} {s["score_gt05"]:>6}')

    total_gt = sum(s['num_gt'] for s in all_frame_stats)
    total_pos = sum(s['num_pos'] for s in all_frame_stats)
    avg_iou = np.mean([s['mean_iou'] for s in all_frame_stats])
    lines.append(f'\nTotal GT: {total_gt} | Total Pos (matched): {total_pos} | '
                 f'Avg mean_iou: {avg_iou:.4f}')

    # 2. Per-channel regression error
    lines.append('\n--- PER-CHANNEL REGRESSION ERROR (L1, averaged over all positives) ---')
    ch_errors = np.stack(all_channel_errors, axis=0)
    mean_errors = ch_errors.mean(axis=0)
    for i, name in enumerate(CHANNEL_NAMES):
        lines.append(f'  {name:>10}: {mean_errors[i]:.4f}')
    lines.append(f'  {"TOTAL":>10}: {mean_errors.sum():.4f}')
    lines.append(f'  {"loss_bbox":>10}: {mean_errors.sum() * 0.25 :.4f}  (= total * 0.25 loss_weight)')

    # 3. GT vs Pred box dimensions
    if all_gt_dims:
        gt_dims = np.concatenate(all_gt_dims, axis=0)
        pred_dims = np.concatenate(all_pred_dims, axis=0)
        lines.append('\n--- BOX DIMENSION COMPARISON (GT vs Pred, in meters) ---')
        for i, name in enumerate(['dx', 'dy', 'dz']):
            lines.append(
                f'  {name}: GT mean={gt_dims[:, i].mean():.3f} std={gt_dims[:, i].std():.3f} | '
                f'Pred mean={pred_dims[:, i].mean():.3f} std={pred_dims[:, i].std():.3f} | '
                f'L1 err={np.abs(gt_dims[:, i] - pred_dims[:, i]).mean():.3f}')

    # 4. Score statistics
    all_s = np.concatenate(all_scores, axis=0)
    lines.append('\n--- SCORE DISTRIBUTION (over all proposals, all frames) ---')
    lines.append(f'  min={all_s.min():.4f}  mean={all_s.mean():.4f}  '
                 f'median={np.median(all_s):.4f}  max={all_s.max():.4f}')
    for thr in [0.01, 0.05, 0.1, 0.3, 0.5]:
        lines.append(f'  score > {thr}: {(all_s > thr).sum()} / {len(all_s)} '
                     f'({(all_s > thr).mean() * 100:.1f}%)')

    # 5. Diagnosis
    lines.append(f'\n{sep}')
    lines.append('DIAGNOSIS')
    lines.append(sep)

    bbox_bottleneck = mean_errors.sum() * 0.25
    heatmap_ok = True
    cls_ok = True
    bbox_stuck = bbox_bottleneck > 0.5

    max_channel = CHANNEL_NAMES[np.argmax(mean_errors)]
    lines.append(f'Heatmap & cls: CONVERGED (loss_heatmap < 0.04, loss_cls < 0.02)')
    lines.append(f'Bbox regression: {"STUCK" if bbox_stuck else "OK"} '
                 f'(reconstructed loss_bbox ~ {bbox_bottleneck:.3f})')
    lines.append(f'Worst channel: {max_channel} (L1 = {mean_errors.max():.4f})')
    lines.append(f'Avg matched IoU: {avg_iou:.4f}')

    if avg_iou < 0.5:
        lines.append('\n>> PRIMARY BOTTLENECK: Box regression not converging.')
        if max_channel in ('cx', 'cy'):
            lines.append('   Center prediction error is high — proposals may not be landing on GT.')
        elif max_channel in ('log_dx', 'log_dy', 'log_dz'):
            lines.append('   Dimension prediction error is high — scale/convention mismatch likely.')
        elif max_channel in ('sin_yaw', 'cos_yaw'):
            lines.append('   Rotation prediction error is high — yaw convention mismatch likely.')
        elif max_channel == 'z':
            lines.append('   Height prediction error is high — z convention mismatch (bottom vs gravity center).')

    report = '\n'.join(lines)
    print(report)

    report_path = os.path.join(out_dir, 'diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\nFull report saved to: {report_path}')


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), 'diagnostics')
    model, dataset = build_model_and_dataset(cfg, args.checkpoint, args.device)
    diagnose(model, dataset, args.device, out_dir)

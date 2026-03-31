# Copyright (c) OpenMMLab. All rights reserved.
"""Center-distance-based 3D detection metric (nuScenes-style matching).

Matches predictions to ground truths using BEV center distance instead of
3D IoU.  This is the standard approach for outdoor driving benchmarks
(nuScenes, Waymo) and is more forgiving for small objects like pedestrians
where tiny localization errors can tank volumetric IoU.

Default distance thresholds follow the nuScenes convention:
    pedestrian: [0.5, 1.0, 2.0, 4.0] metres

Usage in config:
    val_evaluator = dict(
        type='CenterDistanceMetric',
        dist_thr=[0.5, 1.0, 2.0, 4.0])
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet3d.registry import METRICS


def _center_distance_eval(gt_annos, dt_annos, dist_thresholds, classes,
                          logger=None):
    """Evaluate using BEV center distance matching.

    For each distance threshold:
      - Sort detections by confidence (descending).
      - Greedily match to the closest unmatched GT within the threshold.
      - Compute per-class AP (area under precision-recall curve) and recall.

    Returns dict of {metric_name: value}.
    """
    n_classes = len(classes)
    assert len(gt_annos) == len(dt_annos)

    # Organize per-class predictions and GTs across all samples.
    # pred[cls][img_id] = list of (center_xy, score)
    # gt_map[cls][img_id] = list of center_xy
    pred = {c: {} for c in range(n_classes)}
    gt_map = {c: {} for c in range(n_classes)}

    for img_id in range(len(dt_annos)):
        dt = dt_annos[img_id]
        gt = gt_annos[img_id]

        # detections
        if len(dt['labels_3d']) > 0:
            dt_boxes = dt['bboxes_3d']
            if hasattr(dt_boxes, 'tensor'):
                dt_centers = dt_boxes.tensor[:, :2].numpy()
            elif hasattr(dt_boxes, 'numpy'):
                dt_centers = dt_boxes[:, :2].numpy()
            else:
                dt_centers = np.array(dt_boxes)[:, :2]
            dt_scores = dt['scores_3d']
            if hasattr(dt_scores, 'numpy'):
                dt_scores = dt_scores.numpy()
            dt_labels = dt['labels_3d']
            if hasattr(dt_labels, 'numpy'):
                dt_labels = dt_labels.numpy()

            for i in range(len(dt_labels)):
                c = int(dt_labels[i])
                if c not in pred:
                    pred[c] = {}
                if img_id not in pred[c]:
                    pred[c][img_id] = []
                pred[c][img_id].append((dt_centers[i], float(dt_scores[i])))

        # ground truths
        gt_boxes_3d = gt['gt_bboxes_3d']
        gt_labels = gt['gt_labels_3d']
        if hasattr(gt_boxes_3d, 'tensor'):
            gt_centers = gt_boxes_3d.tensor[:, :2].numpy()
        elif hasattr(gt_boxes_3d, 'numpy'):
            gt_centers = gt_boxes_3d[:, :2].numpy()
        else:
            gt_centers = np.array(gt_boxes_3d)[:, :2] if len(
                gt_boxes_3d) > 0 else np.zeros((0, 2))

        for i in range(len(gt_labels)):
            c = int(gt_labels[i])
            if c not in gt_map:
                gt_map[c] = {}
            if img_id not in gt_map[c]:
                gt_map[c][img_id] = []
            gt_map[c][img_id].append(gt_centers[i])

    ret_dict = OrderedDict()
    header = ['classes']
    table_columns = []
    class_col = []

    for dist_thr in dist_thresholds:
        header.append(f'AP_{dist_thr:.1f}m')
        header.append(f'AR_{dist_thr:.1f}m')

    ap_per_thr = {d: [] for d in dist_thresholds}
    ar_per_thr = {d: [] for d in dist_thresholds}

    for cls_id in range(n_classes):
        cls_name = classes[cls_id]
        class_col.append(cls_name)

        # Collect all image ids for this class
        all_img_ids = set()
        if cls_id in gt_map:
            all_img_ids.update(gt_map[cls_id].keys())
        if cls_id in pred:
            all_img_ids.update(pred[cls_id].keys())

        for dist_thr in dist_thresholds:
            # Build GT records per image
            npos = 0
            gt_recs = {}
            for img_id in all_img_ids:
                gts = gt_map.get(cls_id, {}).get(img_id, [])
                npos += len(gts)
                gt_recs[img_id] = {
                    'centers': np.array(gts) if len(gts) > 0 else np.zeros(
                        (0, 2)),
                    'matched': [False] * len(gts)
                }

            # Collect all predictions, sort by score
            all_preds = []
            for img_id in all_img_ids:
                for center, score in pred.get(cls_id, {}).get(img_id, []):
                    all_preds.append((img_id, center, score))
            all_preds.sort(key=lambda x: -x[2])

            tp = np.zeros(len(all_preds))
            fp = np.zeros(len(all_preds))

            for idx, (img_id, p_center, _score) in enumerate(all_preds):
                rec = gt_recs.get(img_id)
                if rec is None or len(rec['centers']) == 0:
                    fp[idx] = 1
                    continue

                dists = np.linalg.norm(
                    rec['centers'] - p_center.reshape(1, 2), axis=1)
                min_idx = int(np.argmin(dists))
                min_dist = dists[min_idx]

                if min_dist <= dist_thr and not rec['matched'][min_idx]:
                    tp[idx] = 1
                    rec['matched'][min_idx] = True
                else:
                    fp[idx] = 1

            if npos == 0:
                ap_per_thr[dist_thr].append(0.0)
                ar_per_thr[dist_thr].append(0.0)
                ret_dict[f'{cls_name}_AP_{dist_thr:.1f}m'] = 0.0
                ret_dict[f'{cls_name}_rec_{dist_thr:.1f}m'] = 0.0
                continue

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recall = cum_tp / npos
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

            # AP via area under monotonically decreasing precision-recall curve
            mrec = np.concatenate([[0.0], recall, [1.0]])
            mpre = np.concatenate([[0.0], precision, [0.0]])
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[indices + 1] - mrec[indices]) *
                              mpre[indices + 1]))

            final_recall = float(recall[-1]) if len(recall) > 0 else 0.0

            ap_per_thr[dist_thr].append(ap)
            ar_per_thr[dist_thr].append(final_recall)
            ret_dict[f'{cls_name}_AP_{dist_thr:.1f}m'] = ap
            ret_dict[f'{cls_name}_rec_{dist_thr:.1f}m'] = final_recall

    # Mean AP / AR across classes
    for dist_thr in dist_thresholds:
        mAP = float(np.mean(ap_per_thr[dist_thr])) if ap_per_thr[
            dist_thr] else 0.0
        mAR = float(np.mean(ar_per_thr[dist_thr])) if ar_per_thr[
            dist_thr] else 0.0
        ret_dict[f'mAP_{dist_thr:.1f}m'] = mAP
        ret_dict[f'mAR_{dist_thr:.1f}m'] = mAR

    # Build table
    table_columns = [class_col + ['Overall']]
    for dist_thr in dist_thresholds:
        ap_col = [
            ret_dict.get(f'{c}_AP_{dist_thr:.1f}m', 0.0)
            for c in classes
        ] + [ret_dict[f'mAP_{dist_thr:.1f}m']]
        ar_col = [
            ret_dict.get(f'{c}_rec_{dist_thr:.1f}m', 0.0)
            for c in classes
        ] + [ret_dict[f'mAR_{dist_thr:.1f}m']]
        table_columns.append([f'{x:.4f}' for x in ap_col])
        table_columns.append([f'{x:.4f}' for x in ar_col])

    table_data = [header]
    table_data += list(zip(*table_columns))
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    if logger:
        logger.info('\n' + table.table)

    return ret_dict


@METRICS.register_module()
class CenterDistanceMetric(BaseMetric):
    """BEV center-distance-based 3D detection metric.

    Matches predictions to ground truths by BEV center distance (L2 in the
    XY plane) instead of 3D IoU.  This follows the nuScenes evaluation
    protocol and is more appropriate for outdoor driving scenes.

    Args:
        dist_thr (list[float]): Distance thresholds in metres.
            Defaults to [0.5, 1.0, 2.0, 4.0] (nuScenes pedestrian convention).
        collect_device (str): Device for distributed gathering.
        prefix (str, optional): Metric name prefix.
    """

    def __init__(self,
                 dist_thr: Union[float, List[float]] = [0.5, 1.0, 2.0, 4.0],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.dist_thr = [dist_thr] if isinstance(dist_thr, float) else dist_thr

    def process(self, data_batch: dict,
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = {}
            for k, v in pred_3d.items():
                cpu_pred_3d[k] = v.to('cpu') if hasattr(v, 'to') else v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        ann_infos, pred_results = [], []
        for eval_ann, pred in results:
            ann_infos.append(eval_ann)
            pred_results.append(pred)

        ret = _center_distance_eval(
            ann_infos, pred_results, self.dist_thr,
            self.dataset_meta['classes'], logger=logger)
        return ret

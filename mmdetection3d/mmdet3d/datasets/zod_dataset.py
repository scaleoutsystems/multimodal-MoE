from typing import Callable, List, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class ZODDataset(NuScenesDataset):
    """ZOD dataset wrapper.

    ZOD ``bbox_3d`` stores z as **bottom-center** (origin 0.5, 0.5, 0),
    whereas vanilla nuScenes uses **geometric center** (origin 0.5, 0.5, 0.5).
    This subclass overrides ``parse_ann_info`` so the LiDARInstance3DBoxes
    constructor receives the correct origin and does not subtract an extra
    half-height from z.
    """

    METAINFO = {
        'classes': ('pedestrian',),
        'version': 'zod-v1',
        'palette': [(0, 0, 230)],
    }

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super(NuScenesDataset, self).parse_ann_info(info)

        if ann_info is not None:
            ann_info = self._filter_with_mask(ann_info)

            if self.with_velocity:
                gt_bboxes_3d = ann_info['gt_bboxes_3d']
                gt_velocities = ann_info['velocities']
                nan_mask = np.isnan(gt_velocities[:, 0])
                gt_velocities[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate(
                    [gt_bboxes_3d, gt_velocities], axis=-1)
                ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        else:
            ann_info = dict()
            if self.with_velocity:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 9), dtype=np.float32)
            else:
                ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes
from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class ZODDataset(NuScenesDataset):
    """ZOD dataset wrapper.

    ZOD ``bbox_3d`` stores z as **bottom-center** (origin 0.5, 0.5, 0),
    whereas vanilla nuScenes uses **geometric center** (origin 0.5, 0.5, 0.5).
    This subclass overrides ``parse_ann_info`` to declare the correct origin
    so that ``LiDARInstance3DBoxes`` applies no z-shift.

    METAINFO is intentionally inherited from NuScenesDataset (full 10-class
    list). The ZOD pickle stores ``bbox_label_3d = 7`` (the nuScenes index
    for "pedestrian") so that ``Det3DDataset.label_mapping`` — built from
    METAINFO against the config's ``metainfo['classes']`` — correctly maps
    raw label 7 → training class 0. Overriding METAINFO to a 1-class list
    would remove key 7 from label_mapping and cause a KeyError at init.
    """

    def parse_ann_info(self, info: dict) -> dict:
        # Skip NuScenesDataset.parse_ann_info entirely and call
        # Det3DDataset.parse_ann_info directly for label mapping + filtering.
        # We then replicate NuScenesDataset's filtering/velocity logic and
        # wrap boxes with origin=(0.5, 0.5, 0) (bottom-center) instead of
        # origin=(0.5, 0.5, 0.5) (geometric center).
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

        # ZOD boxes are bottom-center: declare origin=(0.5, 0.5, 0) so no
        # z-shift is applied (NuScenesDataset would use (0.5, 0.5, 0.5) and
        # incorrectly subtract dz/2 from an already-bottom-center z).
        gt_bboxes_3d = LiDARInstance3DBoxes(
            ann_info['gt_bboxes_3d'],
            box_dim=ann_info['gt_bboxes_3d'].shape[-1],
            origin=(0.5, 0.5, 0)).convert_to(self.box_mode_3d)

        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info

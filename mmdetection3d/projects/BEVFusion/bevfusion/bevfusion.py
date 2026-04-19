from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):
    """BEVFusion multi-modal 3D detector with optional MoE routing.

    The detector builds BEV features from camera and/or LiDAR inputs,
    optionally fuses them, then passes the result through a backbone + neck
    to produce the feature map consumed by the detection head.

    **MoE variants** — exactly one fusion/MoE path is active per run.
    Omit all MoE configs for baseline (ConvFuser) behavior.

    Variant A ``joint_modality_moe_cfg``
        JointModalityMoEBlock — each expert receives both cam_bev and
        lidar_bev and produces a single fused output.  No ConvFuser.

    Variant B ``modality_specific_moe_cfg``
        ModalitySpecificMoEBlock — separate cam/lidar expert pools with
        a joint gate, followed by concat + 1×1 conv fusion inside the
        block.  No ConvFuser.

    Variant C ``fusion_layer`` + ``bev_moe_cfg``
        ConvFuser first, then BEVMoEBlock on the fused BEV.
        This is the ONLY variant that uses ConvFuser.

    Variant D ``bev_moe_cfg`` (no camera branch)
        LiDAR-only — BEVMoEBlock directly on the LiDAR BEV.
        No fusion of any kind.
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        # ── MoE configs (all optional — omit for baseline behavior) ───────
        # Variant A – joint-modality experts (replaces ConvFuser):
        joint_modality_moe_cfg: Optional[dict] = None,
        # Variant B – modality-specific experts (replaces ConvFuser):
        modality_specific_moe_cfg: Optional[dict] = None,
        # Variant C/D – post-fusion / LiDAR-only MoE:
        bev_moe_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        # fusion_layer is ConvFuser — used ONLY in Variant C (and baseline).
        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        # ── MoE blocks (only built when config is provided) ──────────────
        # Variant A – JointModalityMoEBlock: each expert receives both
        # cam_bev and lidar_bev and produces a single fused output.
        self.joint_modality_moe = (
            MODELS.build(joint_modality_moe_cfg)
            if joint_modality_moe_cfg else None
        )

        # Variant B – ModalitySpecificMoEBlock: separate cam/lidar expert
        # pools with a joint gate, fused via concat + 1×1 conv inside.
        self.modality_specific_moe = (
            MODELS.build(modality_specific_moe_cfg)
            if modality_specific_moe_cfg else None
        )

        # Variant C/D – post-fusion or LiDAR-only BEVMoEBlock.
        self.bev_moe = MODELS.build(bev_moe_cfg) if bev_moe_cfg else None

        # Accumulator for MoE auxiliary losses; populated in extract_feat(),
        # consumed in loss(), then reset on the next forward.
        self._moe_aux_loss: Optional[Tensor] = None

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
            # mmcv voxelizer outputs coords as (batch, Z, Y, X)
            # BEVFusionSparseEncoder expects (batch, Y, X, Z)
            coords = coords[:, [0, 2, 3, 1]]
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            if res.shape[0] == 0:
                res = res.new_zeros(1, res.shape[1])
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        """Build the BEV representation and (optionally) apply MoE routing.

        Exactly one fusion / MoE path is active per run:

        Variant A  ``self.joint_modality_moe``
            (cam_bev, lidar_bev) → JointModalityMoEBlock → fused_bev.
            No ConvFuser.

        Variant B  ``self.modality_specific_moe``
            cam_bev → cam experts, lidar_bev → lidar experts,
            then concat + 1×1 conv inside the block → fused_bev.
            No ConvFuser.

        Variant C  ``self.fusion_layer`` + ``self.bev_moe``
            (cam_bev, lidar_bev) → ConvFuser → BEVMoEBlock → fused_bev.
            This is the ONLY variant that uses ConvFuser.

        Variant D  ``self.bev_moe`` (no camera branch)
            lidar_bev → BEVMoEBlock → fused_bev.
            No fusion of any kind.

        Baseline  ``self.fusion_layer`` only (no MoE)
            (cam_bev, lidar_bev) → ConvFuser → fused_bev.
        """
        self._moe_aux_loss = None
        moe_aux_parts: List[Tensor] = []

        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []

        # ── 1. Camera branch ─────────────────────────────────────────
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            # bev_pool produces camera BEV in (B, C, X, Y) order, but the
            # LiDAR BEV is in (B, C, Y, X) after the mmcv-voxelizer coord
            # reorder in extract_pts_feat.  Transpose so both branches use
            # the same spatial convention before fusion.
            img_feature = img_feature.transpose(-1, -2)
            features.append(img_feature)

        # ── 2. LiDAR branch ──────────────────────────────────────────
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        # ── 3. Fusion / MoE dispatch ─────────────────────────────────
        # Exactly one of these paths fires per run.
        if self.joint_modality_moe is not None:
            # Variant A: both BEVs → JointModalityMoEBlock → fused_bev
            assert len(features) == 2, (
                'Variant A (joint_modality_moe) requires both camera '
                'and LiDAR inputs')
            cam_bev, lidar_bev = features[0], features[1]
            x, jm_info = self.joint_modality_moe(
                cam_bev, lidar_bev, batch_input_metas)
            moe_aux_parts.append(jm_info['aux_loss'])

        elif self.modality_specific_moe is not None:
            # Variant B: separate expert pools → concat + 1×1 → fused_bev
            assert len(features) == 2, (
                'Variant B (modality_specific_moe) requires both camera '
                'and LiDAR inputs')
            cam_bev, lidar_bev = features[0], features[1]
            x, ms_info = self.modality_specific_moe(
                cam_bev, lidar_bev, batch_input_metas)
            moe_aux_parts.append(ms_info['aux_loss'])

        elif self.fusion_layer is not None:
            # Variant C (with bev_moe) or baseline (without bev_moe):
            # ConvFuser fuses the feature list.
            x = self.fusion_layer(features)

        else:
            # Variant D / LiDAR-only: no fusion at all.
            assert len(features) == 1, (
                'No fusion layer configured but got multiple feature '
                f'maps ({len(features)})')
            x = features[0]

        # ── 4. Post-fusion MoE (Variant C / D only) ──────────────────
        if self.bev_moe is not None:
            x, bev_info = self.bev_moe(x, batch_input_metas)
            moe_aux_parts.append(bev_info['aux_loss'])

        # ── 5. Backbone + Neck ────────────────────────────────────────
        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        if moe_aux_parts:
            self._moe_aux_loss = sum(moe_aux_parts)

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        # extract_feat runs the full pipeline (including any MoE blocks)
        # and stores auxiliary losses on self._moe_aux_loss for us to
        # collect here.
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        # Depth supervision auxiliary loss (from DepthLSSTransform).
        vt = getattr(self, 'view_transform', None)
        aux = getattr(vt, '_aux_depth_loss', None) if vt else None
        if aux is not None and aux.numel() > 0 and aux.item() > 0:
            losses['aux_depth_loss'] = aux

        # MoE auxiliary losses.  Log switch_balance_loss and load_loss separately
        # so they appear as distinct entries in the training log.  The combined
        # aux_loss tensor already carries the correct gradient via extract_feat;
        # we read per-block _moe_info to get the split.
        if self._moe_aux_loss is not None:
            _bal_parts: list = []
            _ld_parts:  list = []
            for _block_name in ('bev_moe', 'joint_modality_moe',
                                'modality_specific_moe'):
                _block = getattr(self, _block_name, None)
                _info  = getattr(_block, '_moe_info', None) if _block else None
                if _info is not None:
                    _bal = _info.get('balance_loss')
                    _ld  = _info.get('load_loss')
                    if _bal is not None:
                        _bal_parts.append(_bal)
                    if _ld is not None:
                        _ld_parts.append(_ld)
            if _bal_parts:
                losses['moe_balance_loss'] = sum(_bal_parts)
                losses['moe_load_loss']    = sum(_ld_parts) if _ld_parts else \
                    self._moe_aux_loss.detach() * 0
            else:
                losses['moe_aux_loss'] = self._moe_aux_loss

        return losses

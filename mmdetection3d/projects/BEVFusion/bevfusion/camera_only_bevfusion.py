"""Camera-only BEVFusion detector with LiDAR-assisted depth (DepthLSS).
CUSTOM MODEL CLASS FOR ZOD MOE THESIS
registered as a new MMDetection3D model.

Design choices: 
- voxelize_cfg is popped from data_preprocessor before building Det3DDataPreprocessor — same pattern as BEVFusion.__init__, keeps the config dict format compatible if you ever copy/paste.
- self.fusion_layer = None is set explicitly so BEVCameraFeatureVisualizationHook (which tests getattr(model, 'fusion_layer', None)) degrades gracefully.
- LiDAR points are still forwarded to DepthLSSTransform via batch_inputs_dict['points'] — identical to the full BEVFusion path. No voxelizer is created or called.
- x.transpose(-1, -2) after view_transform — same as full BEVFusion, swaps bev_pool's (B, C, X, Y) to (B, C, Y, X) for SECOND.
- _aux_depth_loss is picked up from view_transform in loss() — same as full BEVFusion.

LiDAR points are still loaded by the data pipeline and forwarded to
``DepthLSSTransform`` for:
  - sparse-depth projection into the image plane (dtransform input), and
  - auxiliary cross-entropy depth supervision loss.

No LiDAR BEV feature tensor is produced.  The camera BEV output of
``view_transform`` feeds directly into ``pts_backbone`` (SECOND) and
``pts_neck`` (SECONDFPN), then into the detection head (TransFusionHead).

Architecture::

    img  ──►  img_backbone  ──►  img_neck
                                    │
                            view_transform  ◄── LiDAR pts (depth only)
                                    │
                            camera BEV  (B, cam_ch, H, W)
                                    │   [transpose X↔Y to match LiDAR convention]
                            pts_backbone  (SECOND, in_channels = cam_ch)
                                    │
                            pts_neck  (SECONDFPN)
                                    │
                            bbox_head  (TransFusionHead)
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList


@MODELS.register_module()
class CameraOnlyBEVFusion(Base3DDetector):
    """Camera-only variant of BEVFusion with LiDAR-assisted DepthLSS.

    Key differences from the full ``BEVFusion`` class:

    * No voxel encoder, sparse encoder, SECOND-LiDAR branch, or ConvFuser.
    * LiDAR points are still loaded and passed to ``DepthLSSTransform`` so
      that (a) projected LiDAR depth guides the depth-net, and (b) the
      auxiliary depth loss is available during training.
    * ``pts_backbone`` and ``pts_neck`` play the same role as in the full
      model — they process the BEV map after the view transform — but now
      their input is the camera BEV (``out_channels`` of ``view_transform``)
      rather than the 256-ch fused BEV.  Set ``pts_backbone.in_channels``
      to match ``view_transform.out_channels`` (typically 80).
    * ``fusion_layer`` is always ``None``; it is exposed as an attribute so
      that hooks that test ``getattr(model, 'fusion_layer', None)`` degrade
      gracefully.
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        img_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ) -> None:
        # ``voxelize_cfg`` appears in BEVFusion-style configs for structural
        # compatibility; pop it before handing ``data_preprocessor`` to the
        # parent (Det3DDataPreprocessor does not need it here).
        if isinstance(data_preprocessor, dict):
            data_preprocessor.pop('voxelize_cfg', None)

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.img_backbone = (
            MODELS.build(img_backbone) if img_backbone is not None else None)
        self.img_neck = (
            MODELS.build(img_neck) if img_neck is not None else None)
        self.view_transform = (
            MODELS.build(view_transform) if view_transform is not None else None)

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.bbox_head = MODELS.build(bbox_head)

        # Explicitly None so hooks that test `getattr(model, 'fusion_layer')`
        # skip the fused-BEV visualisation branch without errors.
        self.fusion_layer = None

        self.init_weights()

    # ------------------------------------------------------------------
    # Required abstract method
    # ------------------------------------------------------------------

    def _forward(self, batch_inputs, batch_data_samples=None):
        pass

    # ------------------------------------------------------------------
    # Loss / gradient book-keeping (identical to BEVFusion)
    # ------------------------------------------------------------------

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def with_bbox_head(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        """Build the camera BEV representation and return head features.

        Steps:
          1. Run Swin-T backbone + GeneralizedLSSFPN neck on the image(s).
          2. Pass image features + raw LiDAR points through DepthLSSTransform
             (bev_pool).  The LiDAR points are used *only* to compute the
             sparse depth map fed to ``dtransform``; no LiDAR BEV tensor is
             produced.
          3. Transpose the BEV from (B, C, X, Y) → (B, C, Y, X) to match the
             spatial convention used by SECOND / SECONDFPN.
          4. Run SECOND backbone and SECONDFPN neck.
        """
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)

        if imgs is None:
            raise ValueError('CameraOnlyBEVFusion requires image input.')

        imgs = imgs.contiguous()

        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for meta in batch_input_metas:
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

        # ── 1. Image backbone + neck ─────────────────────────────────
        B, N, C, H, W = imgs.size()
        x = imgs.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # ── 2. View transform (DepthLSSTransform) ───────────────────
        # Run in fp32 to avoid numerical issues in bev_pool / depth projection.
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
            )

        # ── 3. Transpose X↔Y ────────────────────────────────────────
        # bev_pool outputs (B, C, X, Y); SECOND expects (B, C, Y, X).
        x = x.transpose(-1, -2)

        # ── 4. BEV backbone + neck ───────────────────────────────────
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    # ------------------------------------------------------------------
    # Loss / predict entry points
    # ------------------------------------------------------------------

    def loss(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> Dict[str, Tensor]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
        losses.update(bbox_loss)

        # Auxiliary depth supervision loss stored by DepthLSSTransform.
        vt = getattr(self, 'view_transform', None)
        aux = getattr(vt, '_aux_depth_loss', None) if vt else None
        if aux is not None and aux.numel() > 0 and aux.item() > 0:
            losses['aux_depth_loss'] = aux

        return losses

    def predict(
        self,
        batch_inputs_dict: Dict[str, Optional[Tensor]],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        return self.add_pred_to_datasample(batch_data_samples, outputs)

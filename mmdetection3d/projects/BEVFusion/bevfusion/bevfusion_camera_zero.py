"""FOR ZOD_MOE Thesis:
BEVFusionCameraZero — camera-zeroed ablation variant of BEVFusion.

Replaces the camera BEV tensor with torch.zeros_like() immediately after
the view-transform step, right before the features are concatenated and
passed to the fusion layer (ConvFuser).  All other network components —
camera backbone, neck, view transform, LiDAR branch, ConvFuser, backbone,
neck, and detection head — use their learned weights unchanged.

Why zero here?
  - Zeroing *after* view-transform (inside extract_img_feat) is the most
    principled ablation point because the fusion layer's convolutional
    filters are still applied (to all-zero camera channels), so the fused
    BEV representation contains exactly the information that the LiDAR
    branch would contribute when camera input is absent.
  - An alternative would be to skip the camera branch entirely and route
    only pts_feature through the backbone/neck, but this changes the
    architecture (bypasses ConvFuser entirely), making the comparison less
    apples-to-apples.

The zero happens after the view-transform (so the tensor shape is right) but before the ConvFuser receives it. The ConvFuser's camera-channel filters are applied to zeros — net contribution = 0, while lidar channels pass through normally.

Usage:
  Use config ``zod_bevfusion_finetune_camzero.py`` which simply overrides
  model.type to 'BEVFusionCameraZero' while keeping all weights the same.
  Run this config against the full-model checkpoint for an ablation that
  isolates the LiDAR-only contribution inside the trained BEVFusion model.
"""
import torch
from mmdet3d.registry import MODELS

from .bevfusion import BEVFusion


@MODELS.register_module()
class BEVFusionCameraZero(BEVFusion):
    """BEVFusion with camera BEV features zeroed before fusion.

    Identical to :class:`BEVFusion` in every way except that
    ``extract_img_feat`` returns a zero tensor of the same shape as the
    real camera BEV.  This lets the ConvFuser see lidar BEV unchanged while
    receiving no signal from the camera branch.
    """

    def extract_img_feat(self, *args, **kwargs) -> torch.Tensor:
        cam_bev = super().extract_img_feat(*args, **kwargs)
        return torch.zeros_like(cam_bev)

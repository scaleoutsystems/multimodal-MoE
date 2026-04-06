"""Joint-modality (camera + LiDAR) Mixture-of-Experts fusion block.

FusionMoEBlock replaces ConvFuser for Variant B.  Each FusionExpert
receives both camera and LiDAR BEV maps and learns its own fusion
strategy (concat -> conv).  A TopkGate routes samples to experts based
on pooled features from both modalities plus optional context.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .losses import importance_loss, load_loss
from .routing import ContextEncoder, TopkGate


@MODELS.register_module()
class FusionExpert(nn.Module):
    """Single fusion expert: concatenate two BEV maps and convolve.

    Each expert has independent weights so it can learn a distinct
    fusion strategy (e.g., camera-dominant vs. LiDAR-dominant).

    Args:
        cam_channels: Camera BEV channels (e.g. 80).
        lidar_channels: LiDAR BEV channels (e.g. 256).
        out_channels: Output fused BEV channels (e.g. 256).
    """

    def __init__(self, cam_channels: int, lidar_channels: int,
                 out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(cam_channels + lidar_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cam_bev: Tensor, lidar_bev: Tensor) -> Tensor:
        return self.fuse(torch.cat([cam_bev, lidar_bev], dim=1))


@MODELS.register_module()
class FusionMoEBlock(nn.Module):
    """Joint-modality MoE block that replaces ConvFuser.

    Routes each sample to k fusion experts based on pooled features
    from both BEV maps plus optional context metadata.

    Interface is compatible with ConvFuser: accepts a list of tensors
    [cam_bev, lidar_bev] and returns a single fused tensor.

    Args:
        cam_channels: Camera BEV channels.
        lidar_channels: LiDAR BEV channels.
        out_channels: Output fused BEV channels.
        num_experts: Number of fusion experts.
        k: Top-k experts per sample.
        importance_coef: Weight for importance balancing loss.
        load_coef: Weight for load balancing loss.
        context_cfg: If provided, build a ContextEncoder with these kwargs.
    """

    def __init__(self,
                 cam_channels: int = 80,
                 lidar_channels: int = 256,
                 out_channels: int = 256,
                 num_experts: int = 4,
                 k: int = 1,
                 importance_coef: float = 0.01,
                 load_coef: float = 0.01,
                 context_cfg: Optional[dict] = None):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef

        self.experts = nn.ModuleList([
            FusionExpert(cam_channels, lidar_channels, out_channels)
            for _ in range(num_experts)
        ])

        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        self.gate = TopkGate(
            feat_dim=cam_channels + lidar_channels,
            num_experts=num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        self.cam_pool = nn.AdaptiveAvgPool2d(1)
        self.lidar_pool = nn.AdaptiveAvgPool2d(1)

        # Store aux info for loss collection (same pattern as depth_lss.py)
        self._moe_aux_loss: Optional[Tensor] = None
        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(self,
                inputs: List[Tensor],
                batch_input_metas: Optional[List[dict]] = None,
                ) -> Tensor:
        """Fuse camera and LiDAR BEV maps via routed experts.

        Args:
            inputs: [cam_bev (B,Cc,H,W), lidar_bev (B,Cl,H,W)].
                Compatible with ConvFuser's interface.
            batch_input_metas: Per-sample metadata (for context routing).

        Returns:
            Fused BEV tensor (B, out_channels, H, W).
        """
        cam_bev, lidar_bev = inputs[0], inputs[1]
        B = cam_bev.shape[0]

        cam_feat = self.cam_pool(cam_bev).flatten(1)       # (B, Cc)
        lidar_feat = self.lidar_pool(lidar_bev).flatten(1)  # (B, Cl)
        feat = torch.cat([cam_feat, lidar_feat], dim=1)     # (B, Cc+Cl)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)

        gate_out = self.gate(feat, ctx)

        out = cam_bev.new_zeros(B, self.out_channels,
                                cam_bev.shape[2], cam_bev.shape[3])
        expert_counts = torch.zeros(self.num_experts, device=cam_bev.device)

        for b in range(B):
            sample_out = torch.zeros_like(out[b:b + 1])
            for j in range(self.k):
                eidx = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](cam_bev[b:b + 1],
                                                lidar_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
                expert_counts[eidx] += 1
            out[b] = sample_out[0]

        aux = importance_loss(gate_out.probs, self.importance_coef)
        aux = aux + load_loss(expert_counts, self.load_coef)

        self._moe_aux_loss = aux
        self._moe_info = {
            'probs': gate_out.probs.detach(),
            'topk_idx': gate_out.topk_idx.detach(),
            'aux_loss': aux.detach(),
        }
        return out

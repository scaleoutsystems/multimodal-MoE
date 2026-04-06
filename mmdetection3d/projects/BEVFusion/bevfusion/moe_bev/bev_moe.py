"""Single-input BEV Mixture-of-Experts block.

BEVMoEBlock is the workhorse module used for:
  - Variant A: modality-specific experts (camera channels=80, LiDAR channels=256)
  - Variant C: post-fusion MoE (channels=256)
  - Variant D: LiDAR-only MoE (channels=256)

It global-average-pools the BEV map, optionally appends a context vector,
routes through TopkGate, dispatches to BEVResidualExpert modules, and
returns a weighted combination of expert outputs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .losses import importance_loss, load_loss
from .routing import ContextEncoder, TopkGate


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Context-aware Mixture-of-Experts block for BEV feature maps.

    Args:
        channels: Number of BEV feature channels (input == output).
        num_experts: Number of expert modules.
        k: Number of experts selected per sample (top-k routing).
        num_convs: Number of conv layers inside each expert.
        importance_coef: Weight for the importance balancing loss.
        load_coef: Weight for the load balancing loss.
        context_cfg: If provided, build a ContextEncoder with these kwargs.
            Set to ``None`` to disable context-infused routing.
    """

    def __init__(self,
                 channels: int,
                 num_experts: int = 4,
                 k: int = 1,
                 num_convs: int = 2,
                 importance_coef: float = 0.01,
                 load_coef: float = 0.01,
                 context_cfg: Optional[dict] = None):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef

        self.experts = make_bev_experts(num_experts, channels, num_convs)

        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        self.gate = TopkGate(
            feat_dim=channels,
            num_experts=num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        x_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x_bev: BEV feature map (B, C, H, W).
            batch_input_metas: Per-sample metadata dicts. Required only
                when context_encoder is configured (contains 'context' key).

        Returns:
            x_out: BEV feature map (B, C, H, W) after expert processing.
            moe_info: Dict with 'probs', 'topk_idx', 'aux_loss' for
                logging and loss collection.
        """
        B = x_bev.shape[0]

        # Pool BEV to per-sample feature vector
        feat = self.pool(x_bev).flatten(1)  # (B, C)

        # Optional context encoding
        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)  # (B, ctx_dim)

        # Gating
        gate_out = self.gate(feat, ctx)

        # Dispatch: run selected experts per sample
        x_out = torch.zeros_like(x_bev)
        expert_counts = torch.zeros(self.num_experts, device=x_bev.device)

        for b in range(B):
            sample_out = torch.zeros_like(x_bev[b:b + 1])  # (1, C, H, W)
            for j in range(self.k):
                eidx = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](x_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
                expert_counts[eidx] += 1
            x_out[b] = sample_out[0]

        # Auxiliary losses
        aux = importance_loss(gate_out.probs, self.importance_coef)
        aux = aux + load_loss(expert_counts, self.load_coef)

        moe_info = {
            'probs': gate_out.probs.detach(),
            'topk_idx': gate_out.topk_idx.detach(),
            'aux_loss': aux,
        }
        return x_out, moe_info

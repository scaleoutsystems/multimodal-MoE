"""Joint-modality (camera + LiDAR) Mixture-of-Experts fusion block.

FusionMoEBlock replaces ConvFuser for Variant B (joint-modality experts).
Each FusionExpert receives both camera and LiDAR BEV maps and learns its own
fusion strategy (concat → conv).  A TopkGate routes samples to experts based
on summary descriptors from both modalities plus optional context.

Router input
------------
Replaced separate plain GAP pools with BEVSummaryHead instances — one for
camera BEV, one for LiDAR BEV.  Their outputs (each router_out_dim=64) are
concatenated before the gate, giving the router a 128-d spatial-aware signal
from both modalities.
    cam_bev   → BEVSummaryHead(cam_channels)   → (B, 64)
    lidar_bev → BEVSummaryHead(lidar_channels)  → (B, 64)
    cat                                          → (B, 128)  [+ ctx]
    TopkGate                                     → (B, E) logits

moe_info contract
-----------------
self._moe_info is written after every forward() with:
    probs        (B, E)  — full softmax distribution
    topk_idx     (B, k)  — selected expert indices per sample
    topk_weights (B, k)  — re-normalised softmax weights
    aux_loss     scalar  — importance + load auxiliary loss
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .losses import importance_loss, load_loss
from .routing import BEVSummaryHead, ContextEncoder, TopkGate


@MODELS.register_module()
class FusionExpert(nn.Module):
    """Single fusion expert: concatenate two BEV maps and convolve.

    Each expert has independent weights so it can learn a distinct
    fusion strategy (e.g., camera-dominant vs. LiDAR-dominant).

    Args:
        cam_channels:   Camera BEV channels (e.g. 80).
        lidar_channels: LiDAR BEV channels (e.g. 256).
        out_channels:   Output fused BEV channels (e.g. 256).
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
        # (B, Cc, H, W) + (B, Cl, H, W) → cat → (B, Cc+Cl, H, W) → conv
        return self.fuse(torch.cat([cam_bev, lidar_bev], dim=1))


@MODELS.register_module()
class FusionMoEBlock(nn.Module):
    """Joint-modality MoE block that replaces ConvFuser (Variant B).

    Routes each sample to k fusion experts based on spatial-aware routing
    descriptors from both BEV maps, plus optional context metadata.

    Interface is compatible with ConvFuser: accepts a list of tensors
    [cam_bev, lidar_bev] and returns a single fused tensor.

    Args:
        cam_channels:     Camera BEV channels.
        lidar_channels:   LiDAR BEV channels.
        out_channels:     Output fused BEV channels.
        num_experts:      Number of fusion experts.
        k:                Top-k experts per sample.
        importance_coef:  Weight for importance balancing loss
                          (config name: moe_importance_loss_weight). Default 1e-2.
        load_coef:        Weight for load balancing loss. Default 1e-2.
        router_pool_size: Spatial size for BEVSummaryHead pooling grid. Default 2.
        router_hidden_dim: Hidden dim of the MLP inside BEVSummaryHead. Default 128.
        router_out_dim:   Output dim per modality BEVSummaryHead. Default 64.
                          Gate sees 2 × router_out_dim (+ ctx_dim) as input.
        context_cfg:      If provided, build a ContextEncoder with these kwargs.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_experts: int = 4,
        k: int = 1,
        importance_coef: float = 0.01,    # moe_importance_loss_weight
        load_coef: float = 0.01,
        router_pool_size: int = 2,
        router_hidden_dim: int = 128,
        router_out_dim: int = 64,
        context_cfg: Optional[dict] = None,
    ):
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

        # Separate BEVSummaryHeads for each modality — they may have different
        # channel counts (cam=80, lidar=256) but both project to router_out_dim.
        # Using BEVSummaryHead (avg+max 2×2 → MLP) instead of plain GAP to
        # preserve coarse spatial context in the routing signal.
        self.cam_summary   = BEVSummaryHead(cam_channels,   router_pool_size,
                                            router_hidden_dim, router_out_dim)
        self.lidar_summary = BEVSummaryHead(lidar_channels, router_pool_size,
                                            router_hidden_dim, router_out_dim)

        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        # Gate input = [cam_summary (router_out_dim), lidar_summary (router_out_dim)]
        # concat → 2 * router_out_dim  (+ optional context_dim).
        # Switching to NoisyTopkGate later only requires changing this line.
        self.gate = TopkGate(
            feat_dim=2 * router_out_dim,
            num_experts=num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        # Written by forward(); read by BEVFusion.extract_feat() for loss
        # collection and by MoERoutingHook for Hook A.
        self._moe_aux_loss: Optional[Tensor] = None
        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        inputs: List[Tensor],
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tensor:
        """Fuse camera and LiDAR BEV maps via routed experts.

        Args:
            inputs:             [cam_bev (B,Cc,H,W), lidar_bev (B,Cl,H,W)].
                                Compatible with ConvFuser's interface.
            batch_input_metas:  Per-sample metadata (for context routing).

        Returns:
            Fused BEV tensor (B, out_channels, H, W).
        """
        cam_bev, lidar_bev = inputs[0], inputs[1]
        B = cam_bev.shape[0]

        # ── Step 1: Build joint routing descriptor ────────────────────
        # Each BEVSummaryHead: (B, C, H, W) → avg+max 2×2 → MLP → (B, 64)
        cam_feat   = self.cam_summary(cam_bev)    # (B, router_out_dim)
        lidar_feat = self.lidar_summary(lidar_bev) # (B, router_out_dim)
        feat = torch.cat([cam_feat, lidar_feat], dim=1)  # (B, 2·router_out_dim)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)  # (B, ctx_dim)

        # ── Step 2: Gate → top-k expert selection ─────────────────────
        gate_out = self.gate(feat, ctx)

        # ── Step 3: Dispatch — each selected expert fuses both BEVs ──
        out = cam_bev.new_zeros(B, self.out_channels,
                                cam_bev.shape[2], cam_bev.shape[3])
        expert_counts = torch.zeros(self.num_experts, device=cam_bev.device)

        for b in range(B):
            sample_out = torch.zeros_like(out[b:b + 1])
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](cam_bev[b:b + 1],
                                                lidar_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
                expert_counts[eidx] += 1
            out[b] = sample_out[0]

        # ── Step 4: Auxiliary losses ───────────────────────────────────
        aux = importance_loss(gate_out.probs, self.importance_coef)
        aux = aux + load_loss(expert_counts, self.load_coef)

        self._moe_aux_loss = aux
        self._moe_info = {
            'probs':        gate_out.probs.detach(),
            'topk_idx':     gate_out.topk_idx.detach(),
            'topk_weights': gate_out.topk_weights.detach(),  # for Hook A
            'aux_loss':     aux.detach(),
        }
        return out

"""Single-input BEV Mixture-of-Experts block.

BEVMoEBlock is the workhorse module used for:
  - Variant A: modality-specific experts (camera channels=80, LiDAR channels=256)
    applied as *separate* independent blocks, each seeing only their own modality.
    For joint-gate modality-specific routing, use ModalitySpecificMoEBlock instead.
  - Variant C: post-fusion MoE on fused BEV (channels=256)
  - Variant D: LiDAR-only MoE (channels=256)

Router input
------------
Replaced plain global average pooling with BEVSummaryHead (avg+max pool to
2×2 grid, flatten, MLP) to preserve coarse spatial structure.  See BEVSummaryHead
docstring for the full shape trace and design rationale.

moe_info contract
-----------------
After every forward() call, self._moe_info is written with:
    probs        (B, E)  — full softmax distribution over all experts
    topk_idx     (B, k)  — indices of the k selected experts per sample
    topk_weights (B, k)  — re-normalised softmax weights for selected experts
    aux_loss     scalar  — combined importance + load auxiliary loss

Hook A (routing mass accumulation) reads topk_idx and topk_weights from
_moe_info after each training/val iteration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .losses import importance_loss, load_loss
from .routing import BEVSummaryHead, ContextEncoder, TopkGate


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Context-aware Mixture-of-Experts block for a single-modality BEV map.

    Args:
        channels:         Number of BEV feature channels (input == output).
        num_experts:      Number of expert modules.
        k:                Top-k experts selected per sample.
        num_convs:        Conv layers inside each BEVResidualExpert.
        importance_coef:  Weight for the importance balancing loss
                          (config name: moe_importance_loss_weight). Default 1e-2.
        load_coef:        Weight for the load balancing loss. Default 1e-2.
        router_pool_size: Spatial size for BEVSummaryHead pooling grid. Default 2.
        router_hidden_dim: Hidden dim of the MLP inside BEVSummaryHead. Default 128.
        router_out_dim:   Output dim of BEVSummaryHead (gate input dim). Default 64.
        context_cfg:      If provided, build a ContextEncoder with these kwargs.
                          Set to None to disable context-infused routing.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        k: int = 1,
        num_convs: int = 2,
        importance_coef: float = 0.01,    # moe_importance_loss_weight
        load_coef: float = 0.01,
        router_pool_size: int = 2,
        router_hidden_dim: int = 128,
        router_out_dim: int = 64,
        context_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef

        # Independent residual-conv experts; with top-1 routing and B=2-4,
        # at most one expert runs per sample — cost ≈ one ConvFuser forward.
        self.experts = make_bev_experts(num_experts, channels, num_convs)

        # BEVSummaryHead replaces plain AdaptiveAvgPool2d(1).
        # It pools to a (pool_size × pool_size) grid with both avg and max,
        # then projects through a small MLP to router_out_dim features.
        # Shape: (B, C, H, W) → (B, router_out_dim)
        self.summary = BEVSummaryHead(
            channels=channels,
            pool_size=router_pool_size,
            hidden_dim=router_hidden_dim,
            out_dim=router_out_dim,
        )

        # Optional context encoder: encodes per-sample metadata (weather,
        # road type, …) into a vector appended to the gate input.
        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        # Gate: router_out_dim (+ optional context_dim) → expert logits.
        # Switching from deterministic to noisy routing later only requires
        # replacing TopkGate with NoisyTopkGate here — no other changes needed.
        self.gate = TopkGate(
            feat_dim=router_out_dim,
            num_experts=num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        # Populated after every forward(); read by MoERoutingHook.
        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        x_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x_bev:              BEV feature map (B, C, H, W).
            batch_input_metas:  Per-sample metadata dicts. Required only when
                                context_encoder is configured.

        Returns:
            x_out:    BEV feature map (B, C, H, W) after expert processing.
            moe_info: Dict with 'probs', 'topk_idx', 'topk_weights',
                      'aux_loss'. Used for loss collection and Hook A.
        """
        B = x_bev.shape[0]

        # ── Step 1: Build routing descriptor ──────────────────────────
        # BEVSummaryHead: (B, C, H, W) → avg+max pool 2×2 → flatten → MLP
        # → (B, router_out_dim). Preserves coarse spatial structure vs GAP.
        feat = self.summary(x_bev)  # (B, router_out_dim)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)  # (B, ctx_dim)

        # ── Step 2: Gate → top-k expert selection ─────────────────────
        gate_out = self.gate(feat, ctx)
        # gate_out.probs        : (B, E)  full softmax distribution
        # gate_out.topk_idx     : (B, k)  selected expert indices
        # gate_out.topk_weights : (B, k)  re-normalised weights

        # ── Step 3: Dispatch to selected experts ──────────────────────
        # Start from zero; residual experts output (x + delta), so the
        # weighted sum reconstructs x + (weighted mean of expert deltas).
        x_out = torch.zeros_like(x_bev)
        expert_counts = torch.zeros(self.num_experts, device=x_bev.device)

        for b in range(B):
            sample_out = torch.zeros_like(x_bev[b:b + 1])  # (1, C, H, W)
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](x_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
                expert_counts[eidx] += 1
            x_out[b] = sample_out[0]

        # ── Step 4: Auxiliary losses ───────────────────────────────────
        # importance_loss is differentiable (trains the gate toward balance).
        # load_loss is detached (monitoring signal only, no gradient).
        aux = importance_loss(gate_out.probs, self.importance_coef)
        aux = aux + load_loss(expert_counts, self.load_coef)

        moe_info = {
            'probs':        gate_out.probs.detach(),
            'topk_idx':     gate_out.topk_idx.detach(),
            'topk_weights': gate_out.topk_weights.detach(),  # for Hook A (routing mass)
            'aux_loss':     aux,
        }
        # Cache on self so MoERoutingHook can read it after each iter
        # without requiring the caller to pass it up the call stack.
        self._moe_info = moe_info
        return x_out, moe_info

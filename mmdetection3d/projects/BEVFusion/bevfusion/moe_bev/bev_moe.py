"""Single-input BEV Mixture-of-Experts block.

BEVMoEBlock is a single-input MoE block used for:
  - Variant C: fusion-then-MoE — applied to the fused BEV after ConvFuser
    and before pts_backbone.
  - Variant D: LiDAR-only MoE — applied to the LiDAR BEV at the same
    insertion point before pts_backbone (no convfuser though).

  For modality-specific experts (joint gate over cam + lidar expert pools),
  use ModalitySpecificMoEBlock instead (modality_specific_moe_cfg in config).

Router input
------------
Replaced plain global average pooling with BEVSummaryHead (avg+max pool to
2×2 grid, flatten, MLP) to preserve coarse spatial structure.  See BEVSummaryHead
docstring for the full shape trace and design rationale.

moe_info contract
-----------------
After every forward() call, self._moe_info is written with:
    full_softmax_probs   (B, E)  — pre-top-k softmax (router belief over all experts)
    sparse_softmax_probs (B, E)  — post-top-k masked softmax (zero on non-selected)
    topk_idx             (B, k)  — indices of the k selected experts per sample
    topk_weights         (B, k)  — re-normalised dispatch weights for selected experts
    aux_loss             scalar  — combined importance + load loss (for extract_feat)
    importance_loss      scalar  — importance balancing loss component (logged separately)
    load_loss            scalar  — load balancing loss component (logged separately)

MoERoutingHook reads full_softmax_probs, topk_idx, and topk_weights from
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
        num_experts: int = 6,
        k: int = 1,
        num_convs: int = 1,
        importance_coef: float = 0.02,    # moe_importance_loss_weight
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

        # Independent residual-conv experts; with top-1 routing
        # at most one expert runs per sample — cost ≈ one ConvFuser forward
        self.experts = make_bev_experts(num_experts, channels, num_convs)

        # BEVSummaryHead pools to a (pool_size × pool_size) grid with both avg and max,
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
            moe_info: Dict with 'full_softmax_probs', 'sparse_softmax_probs',
                      'topk_idx', 'topk_weights', 'aux_loss',
                      'importance_loss', 'load_loss'.
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
        # gate_out.full_softmax_probs   : (B, E)  pre-top-k router belief
        # gate_out.sparse_softmax_probs : (B, E)  post-top-k masked softmax
        # gate_out.topk_idx             : (B, k)  selected expert indices
        # gate_out.topk_weights         : (B, k)  re-normalised dispatch weights

        # ── Step 3: Dispatch to selected experts ──────────────────────
        # Start from zero; residual experts output (x + delta), so the
        # weighted sum reconstructs x + (weighted mean of expert deltas).
        # For each sample in the batch, run the selected experts on that sample,
        # weight their outputs, and add them together.
        x_out = torch.zeros_like(x_bev)
        expert_counts = torch.zeros(self.num_experts, device=x_bev.device)

        for b in range(B):
            sample_out = torch.zeros_like(x_bev[b:b + 1])  # (1, C, H, W)
            for j in range(self.k): # if k=2, j= 0, 1
                eidx   = gate_out.topk_idx[b, j].item() # expert index for this sample
                weight = gate_out.topk_weights[b, j] # weight for this expert
                expert_out = self.experts[eidx](x_bev[b:b + 1]) # run the expert on the sample
                sample_out = sample_out + weight * expert_out # weighted sum of expert outputs
                expert_counts[eidx] += 1 # count how many times each expert was used
            x_out[b] = sample_out[0]  # x_out[b].shape = (C, H, W), sample_out[0].shape = (C, H, W)

        # ── Step 4: Auxiliary losses ───────────────────────────────────
        # importance_loss uses full_softmax_probs (pre-top-k) so all experts
        # receive gradient signal, not just the selected ones.
        # load_loss is detached — monitoring signal only, no gradient.
        imp_loss = importance_loss(gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(expert_counts, self.load_coef)
        aux      = imp_loss + ld_loss

        moe_info = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,   # WITH grad — added to losses by bevfusion
            'load_loss':            ld_loss,    # already detached — added for log visibility
        }
        # Cache on self so MoERoutingHook can read it after each iter
        # without requiring the caller to pass it up the call stack.
        self._moe_info = moe_info
        return x_out, moe_info

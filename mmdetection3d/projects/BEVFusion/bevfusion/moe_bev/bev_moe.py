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
BEVSummaryHead (avg+max pool to a 2×2 grid, flatten, MLP with final LayerNorm)
is used instead of plain global average pooling to preserve coarse spatial
structure.  The final LayerNorm (replacing a previous ReLU) ensures the routing
descriptor is signed and unit-variance so gate logits can grow and carry
meaningful preference signals.  See BEVSummaryHead docstring for the full shape
trace and design rationale.

Dispatch strategy: residual-delta with bootstrap gain
------------------------------------------------------
BEVResidualExperts output  x + delta  (the input feature plus a residual).
Dispatch is therefore implemented as:

    x_out = x_bev + g · Σ_j  w_j · (expert_j(x_bev) − x_bev)

where w_j are Switch-style dispatch weights (see routing.py for definition)
and g = ``residual_gain`` is a constant scalar multiplier on the expert
contribution.

Why the gain exists (bootstrap problem)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Gate weights from TopkGate are Switch-style: they are gathered from the
pre-top-k softmax and NOT renormalised to sum to 1.  At initialisation each
weight is approximately 1/E (≈ 0.17 for 6 experts).  With residual-delta
dispatch and g=1, the expert's effective contribution is then w·Δ ≈ Δ/E,
which is essentially zero because Δ itself is small at init.

Consequence without a gain (g=1):
  - Experts barely influence x_out → detection loss doesn't prefer any
    particular expert → no rich-get-richer signal → gate's softmax stays
    near-uniform → w stays ≈ 1/E → experts never learn to differentiate.
    (Empirically: moe_balance_loss stays pinned at α regardless of hard
     selection imbalance.)

Setting g = num_experts (≈ E) makes g·w ≈ 1 at init, so the expert
contributes at full scale from step 1.  As experts learn distinctive Δs,
the detection gradient w.r.t. w becomes strong (∝ g·Δ), which lets the
gate actually peak → specialisation emerges.

Why the naive dispatch (no residual, no gain) is wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If instead the naive dispatch   x_out = Σ_j w_j · expert_j(x_bev)   were used
with w_j ≈ 1/E, the output BEV feature map would be scaled down E× globally
(because residual experts output x+Δ, so the whole x is scaled too).  That
scaling propagates through pts_backbone and collapses detection performance
(empirically: loss_heatmap rises 2×, matched_ious drops 10×, moe_balance_loss
stays near zero because the gate receives a corrupt gradient signal).

Scaling behaviour with g = num_experts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Gate state                    g·w   Effective expert contribution
    ─────────────────────────────────────────────────────────────────
    Uniform softmax (init)        ≈ 1   x_out ≈ x_bev + Δ   (full scale)
    Mild peaking (w ≈ 0.3)        ≈ 1.8 moderate amplification
    Strong peaking (w ≈ 0.5)      ≈ 3.0 possible instability — watch grad_norm
    Collapsed (w ≈ 1.0)           ≈ E   risk of runaway — unlikely under
                                        switch_balance_loss back-pressure

Sentinel: if grad_norm drifts above ~50 sustained (typical is 5–10), reduce
residual_gain or increase importance_coef.

moe_info contract
-----------------
After every forward() call, self._moe_info is written with:
    full_softmax_probs   (B, E)  — pre-top-k softmax (router belief over all experts).
                                   Used by switch_balance_loss (P_e term) and
                                   dense_mean_prob_per_expert diagnostics.
    sparse_softmax_probs (B, E)  — post-top-k masked softmax (zero on non-selected).
                                   Returned for analysis; NOT used for dispatch.
    topk_idx             (B, k)  — indices of the k selected experts per sample.
                                   Used by selection-frequency diagnostics and
                                   switch_balance_loss (f_e term).
    topk_weights         (B, k)  — Switch-style dispatch weights (NOT renormalised).
                                   Used in the residual-delta dispatch above and
                                   for dispatch_mass_per_expert diagnostics.
    aux_loss             scalar  — combined balance + load loss (returned by
                                   extract_feat for the backbone loss sum).
    balance_loss         scalar  — Switch balance loss only (logged as moe_balance_loss).
    load_loss            scalar  — CV²-of-counts load loss only (logged as moe_load_loss;
                                   detached — monitoring signal, no gradient).

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
from .losses import load_loss, switch_balance_loss
from .routing import BEVSummaryHead, ContextEncoder, TopkGate


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Context-aware Mixture-of-Experts block for a single-modality BEV map.

    Args:
        channels:         Number of BEV feature channels (input == output).
        num_experts:      Number of expert modules.
        k:                Top-k experts selected per sample.
        num_convs:        Conv layers inside each BEVResidualExpert.
        importance_coef:  Weight for the Switch balance loss α
                          (config name: moe_importance_loss_weight). Default 1e-2.
        load_coef:        Weight for the load balancing loss. Default 1e-2.
        residual_gain:    Scalar multiplier applied to the routed expert delta
                          in the residual-delta dispatch:
                             x_out = x_bev + residual_gain · Σ_j w_j · Δ_j
                          Set to num_experts to compensate for the ≈ 1/E
                          magnitude of Switch-style weights at init (bootstrap).
                          Default 1.0 (no amplification — preserves legacy
                          behaviour when the caller doesn't override).
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
        residual_gain: float = 1.0,
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
        self.residual_gain = float(residual_gain)

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
                      'balance_loss', 'load_loss'.
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
        # gate_out.topk_weights         : (B, k)  Switch-style router confidence
        #                                        (NOT renormalised to sum to 1)

        # ── Step 3: Dispatch to selected experts ──────────────────────
        # Residual-delta dispatch with bootstrap gain:
        #   x_out = x_bev + g · Σ_j w_j · (expert_j(x_bev) − x_bev)
        # where g = self.residual_gain.
        #
        # Experts are residual (output x + delta), so (expert_out − x_bev)
        # extracts the delta.  Adding g·w_j-weighted deltas back to x_bev
        # preserves the input scale regardless of w_j magnitude:
        #   k=1, g·w ≈ 1  → x_out ≈ expert_out           (full contribution)
        #   k=1, g·w << 1 → x_out ≈ x_bev + small delta  (soft gating)
        # g = num_experts compensates for Switch-style weights which at init
        # are ≈ 1/E; without this, expert contribution is effectively zero
        # and the gate gets no gradient signal (see class docstring).
        x_out = x_bev.clone()
        expert_counts = torch.zeros(self.num_experts, device=x_bev.device)

        for b in range(B):
            xb = x_bev[b:b + 1]                           # (1, C, H, W)
            delta_sum = torch.zeros_like(xb)
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](xb)
                delta_sum = delta_sum + weight * (expert_out - xb)
                expert_counts[eidx] += 1
            x_out[b] = (xb + self.residual_gain * delta_sum)[0]

        # ── Step 4: Auxiliary losses ───────────────────────────────────
        # switch_balance_loss: multiplies detached dispatch frequency (f_e)
        # by differentiable mean router probability (P_e).  When an expert is
        # overloaded, gradient through P_e pushes the router away from it.
        # load_loss is detached — monitoring signal only, no gradient.
        bal_loss = switch_balance_loss(
            gate_out.full_softmax_probs,
            gate_out.topk_idx,
            self.num_experts,
            self.importance_coef,
        )
        ld_loss  = load_loss(expert_counts, self.load_coef)
        aux      = bal_loss + ld_loss

        moe_info = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'aux_loss':             aux,
            'balance_loss':         bal_loss,   # WITH grad — added to losses by bevfusion
            'load_loss':            ld_loss,    # already detached — added for log visibility
        }
        # Cache on self so MoERoutingHook can read it after each iter
        # without requiring the caller to pass it up the call stack.
        self._moe_info = moe_info
        return x_out, moe_info

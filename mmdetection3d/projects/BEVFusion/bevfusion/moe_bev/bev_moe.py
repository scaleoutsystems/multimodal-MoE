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

Dispatch strategy: residual-delta with Shazeer top-k mixture
------------------------------------------------------------
BEVResidualExperts output  x + delta  (the input feature plus a residual).
Dispatch is therefore implemented as:

    x_out = x_bev + g · Σ_j  w_j · (expert_j(x_bev) − x_bev)

where ``w_j`` are the standard Shazeer top-k mixture weights produced by the
gate — renormalised over the top-k selections so ``Σ_j w_j = 1`` per sample
(see routing.py).  ``g = residual_gain`` is a plain scalar multiplier on the
total expert contribution; default is 1.0.

Why Σ w = 1 matters here
~~~~~~~~~~~~~~~~~~~~~~~~
Because the mixture sum is constant, the magnitude of the residual update is
controlled solely by the delta magnitude and ``g``, not by router peakiness:

    ‖x_out − x_bev‖ = g · ‖Σ_j w_j · Δ_j‖ ≤ g · max_j ‖Δ_j‖

A uniform router and a fully-collapsed router produce the same *scale* of
update — they differ only in *which* expert's delta dominates.  This keeps
the gradient scale to the experts stable across training and removes the
earlier need to couple ``residual_gain`` to ``num_experts``.

Why g = 1 by default
~~~~~~~~~~~~~~~~~~~~
At init each expert's Δ is small (experts haven't learned to differentiate).
With Σ w = 1 and g = 1, the update ``Σ w · Δ`` has the same *scale* as any
individual expert output, so experts contribute at full effective scale from
step 1.  Specialisation is driven by:
    • importance_loss + load_loss (direct gradient on gate logits + noise_std)
    • task-loss gradient through ``w_j`` (requires k ≥ 2; see routing.py)

Tuning g (when needed)
~~~~~~~~~~~~~~~~~~~~~~
    • g < 1   → dampen the expert residual (conservative; useful when the
                  pre-MoE features are already strong and you want experts to
                  only refine them).
    • g = 1   → default, expert outputs replace the residual pathway fully.
    • g > 1   → amplify; rarely needed now that Σ w = 1 and can cause
                  grad_norm spikes.  If you feel tempted, increase ``num_convs``
                  in the experts instead.

Sentinel: if grad_norm drifts above ~50 sustained (typical is 5–10), reduce
residual_gain or increase importance_coef / load_coef.

Balancing losses
----------------
The block emits Shazeer et al. (2017) importance + load losses by default:

    importance_loss — CV² of per-expert total soft probability mass
                      (gradient through gate_probs).  Catches dead experts
                      whose P_e collapses toward 0.
    load_loss       — CV² of the Gaussian-CDF expected dispatch count, using
                      the noisy gate's clean logits + noise std.  Catches
                      collapsed hard dispatch even when P_e looks uniform.

See ``losses.py`` for the derivations, and ``routing.py`` for why both
signals are only fully differentiable under ``NoisyTopkGate``.

moe_info contract
-----------------
After every forward() call, self._moe_info is written with:
    full_softmax_probs   (B, E)  — pre-top-k softmax (router belief over all experts).
                                   Used by importance_loss and the
                                   dense_mean_prob_per_expert diagnostics.
    sparse_softmax_probs (B, E)  — post-top-k masked softmax (zero on non-selected).
                                   Returned for analysis; NOT used for dispatch.
    topk_idx             (B, k)  — indices of the k selected experts per sample.
                                   Used by selection-frequency diagnostics.
    topk_weights         (B, k)  — Shazeer top-k mixture weights (Σ_j = 1).
                                   Used in the residual-delta dispatch above
                                   and for dispatch_mass_per_expert diagnostics.
    aux_loss             scalar  — importance_loss + load_loss (returned by
                                   extract_feat for the backbone loss sum).
    importance_loss      scalar  — Shazeer importance term (logged as
                                   moe_importance_loss).  Differentiable.
    load_loss            scalar  — Shazeer load term — CV² of the Gaussian-CDF
                                   dispatch estimator (logged as moe_load_loss).
                                   Differentiable whenever the gate supplies
                                   a noise_std; a detached zero otherwise.

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
from .routing import BEVSummaryHead, ContextEncoder, NoisyTopkGate, TopkGate


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Context-aware Mixture-of-Experts block for a single-modality BEV map.

    Args:
        channels:         Number of BEV feature channels (input == output).
        num_experts:      Number of expert modules.
        k:                Top-k experts selected per sample.
        num_convs:        Conv layers inside each BEVResidualExpert.
        importance_coef:  Weight α for the Shazeer importance loss
                          (logged as moe_importance_loss).  Penalises uneven
                          per-expert mean soft probability mass.  Default 1e-3.
        load_coef:        Weight α for the Shazeer load loss
                          (logged as moe_load_loss).  Penalises uneven
                          Gaussian-CDF dispatch estimates; requires a noisy
                          gate.  Default 1e-3.
        residual_gain:    Scalar multiplier applied to the routed expert delta
                          in the residual-delta dispatch:
                             x_out = x_bev + residual_gain · Σ_j w_j · Δ_j
                          With Shazeer top-k mixture (Σ_j w_j = 1) the default
                          of 1.0 applies the expert delta at full scale with
                          no dependence on num_experts.  See module docstring
                          for tuning guidance.
        router_pool_size: Spatial size for BEVSummaryHead pooling grid. Default 2.
        router_hidden_dim: Hidden dim of the MLP inside BEVSummaryHead. Default 128.
        router_out_dim:   Output dim of BEVSummaryHead (gate input dim). Default 64.
        context_cfg:      If provided, build a ContextEncoder with these kwargs.
                          Set to None to disable context-infused routing.
        gate_type:        Which gate to use: ``'topk'`` (deterministic, default)
                          or ``'noisy_topk'`` (Shazeer et al. noisy gate that
                          adds learned Gaussian noise during training to
                          encourage load balance).  load_loss only produces a
                          non-zero gradient signal with ``'noisy_topk'``.
        gate_cfg:         Extra kwargs forwarded to the gate constructor.
                          Only used when ``gate_type='noisy_topk'``.
                          Supported keys: ``temperature``, ``noise_floor``,
                          ``input_dropout``, ``logit_dropout``.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        k: int = 1,
        num_convs: int = 1,
        importance_coef: float = 0.001,   # logged as moe_importance_loss
        load_coef: float = 0.001,         # logged as moe_load_loss
        residual_gain: float = 1.0,
        router_pool_size: int = 2,
        router_hidden_dim: int = 128,
        router_out_dim: int = 64,
        context_cfg: Optional[dict] = None,
        gate_type: str = 'topk',
        gate_cfg: Optional[dict] = None,
        # Backward-compat: old configs may still pass ``switch_coef``.  If so
        # use it as the importance_coef (same numeric meaning after the
        # Shazeer refactor: a scalar weight on the primary balancing loss)
        # and warn.  Remove this once all configs have been migrated.
        switch_coef: Optional[float] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.k = k
        if switch_coef is not None:
            import warnings
            warnings.warn(
                'BEVMoEBlock(switch_coef=...) is deprecated; use '
                'importance_coef instead.  The Shazeer importance + load '
                'losses replace the Switch balance loss as the default '
                'regulariser under a noisy gate.  Interpreting the passed '
                'switch_coef as importance_coef for this run.',
                DeprecationWarning, stacklevel=2)
            importance_coef = switch_coef
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
        # gate_type='topk'       → deterministic TopkGate (default).
        # gate_type='noisy_topk' → Shazeer noisy gate (NoisyTopkGate);
        #   adds learned input-dependent Gaussian noise during training to
        #   encourage load balance.  Extra kwargs (temperature, noise_floor,
        #   input_dropout, logit_dropout) can be passed via gate_cfg.
        extra_gate_kwargs = gate_cfg or {}
        if gate_type == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=router_out_dim,
                num_experts=num_experts,
                k=k,
                context_dim=ctx_dim,
                **extra_gate_kwargs,
            )
        else:
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
        # gate_out.sparse_softmax_probs : (B, E)  top-k mixture laid into E
        # gate_out.topk_idx             : (B, k)  selected expert indices
        # gate_out.topk_weights         : (B, k)  Shazeer top-k mixture,
        #                                        Σ_j topk_weights = 1 per sample

        # ── Step 3: Dispatch to selected experts ──────────────────────
        # Residual-delta dispatch:
        #   x_out = x_bev + g · Σ_j w_j · (expert_j(x_bev) − x_bev)
        # where g = self.residual_gain and Σ_j w_j = 1 by construction.
        #
        # Experts are residual (output x + delta), so (expert_out − x_bev)
        # extracts the delta.  Because Σ w = 1 the scale of the residual
        # update is governed by g and the delta magnitudes alone — router
        # peakiness does not change the update magnitude, only *which*
        # expert's delta dominates.  See module docstring for rationale.
        x_out = x_bev.clone()

        for b in range(B):
            xb = x_bev[b:b + 1]                           # (1, C, H, W)
            delta_sum = torch.zeros_like(xb)
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](xb)
                delta_sum = delta_sum + weight * (expert_out - xb)
            x_out[b] = (xb + self.residual_gain * delta_sum)[0]

        # ── Step 4: Shazeer importance + load losses ──────────────────
        # importance_loss — CV² of Σ_b gate_probs[b, e].  Penalises uneven
        #   soft probability mass; gradient flows through the softmax back
        #   to every gate logit.  Catches dead experts.
        # load_loss — CV² of the Gaussian-CDF dispatch estimator using the
        #   noisy gate's clean logits + noise std.  Penalises uneven hard
        #   dispatch; differentiable through clean_logits and noise_std.
        #   Falls back to a detached zero when noise_std is None (e.g. a
        #   deterministic TopkGate or NoisyTopkGate during eval).
        imp_loss = importance_loss(
            gate_out.full_softmax_probs,
            self.importance_coef,
        )
        ld_loss  = load_loss(
            gate_out.clean_logits,
            gate_out.noisy_logits,
            gate_out.noise_std,
            self.k,
            self.load_coef,
        )
        aux = imp_loss + ld_loss

        moe_info = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,   # WITH grad — Shazeer importance
            'load_loss':            ld_loss,    # WITH grad when noisy gate
        }
        # Cache on self so MoERoutingHook can read it after each iter
        # without requiring the caller to pass it up the call stack.
        self._moe_info = moe_info
        return x_out, moe_info

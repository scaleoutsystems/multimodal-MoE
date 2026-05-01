"""Single-input BEV Mixture-of-Experts block (context-supervised routing).

BEVMoEBlock is a single-input MoE block used for:
  - Variant C: fusion-then-MoE — applied to the fused BEV after ConvFuser
    and before pts_backbone.
  - Variant D: LiDAR-only MoE — applied to the LiDAR BEV at the same
    insertion point before pts_backbone (no ConvFuser).

  For modality-specific experts (joint gate over cam + lidar expert pools),
  use ModalitySpecificMoEBlock instead.

Router input
------------
``BEVSummaryHead`` (avg + max pool to a P×P grid, 1×1 + 3×3 spatial mixer,
flatten, two-layer MLP with final LayerNorm) produces the routing
descriptor ``z``.  See ``routing.py`` for the rationale.

Context-supervised routing
--------------------------
Context labels are NOT concatenated into the gate input.  The block uses a
separate ``context_head: Linear(out_dim → num_context_classes)`` whose
output is supervised by ``F.cross_entropy(ctx_logits, ctx_label)``.  The
gradient flows back through ``z`` and shapes the BEV summary descriptor —
expert dispatch remains task-driven (top-k over the router logits) but the
descriptor is biased toward context-relevant directions.

For the full pattern (every variant)::

    z          = self.summary(x_bev)
    gate_out   = self.gate(z)
    ctx_logits = self.context_head(z)
    ctx_loss   = F.cross_entropy(ctx_logits, ctx_label,
                                 label_smoothing=ctx_label_smoothing)
    z_loss     = router_z_loss(gate_out.clean_logits, z_loss_coef)
    aux        = importance_loss + load_loss + ctx_loss_coef · ctx_loss + z_loss

Dispatch strategy: residual-delta with Shazeer top-k mixture
------------------------------------------------------------
Experts output ``x + delta``.  Dispatch is implemented as

    x_out = x_bev + g · Σ_j  w_j · (expert_j(x_bev) − x_bev)

with ``Σ_j w_j = 1`` (Shazeer top-k) and ``g = residual_gain`` a plain
scalar (default 1.0).  See module history below for why the dependency on
``num_experts`` was removed.

moe_info contract
-----------------
After every forward() call, ``self._moe_info`` is written with:

    full_softmax_probs   (B, E)  — softmax over clean_logits (router belief)
    sparse_softmax_probs (B, E)  — top-k mixture laid back into (B, E)
    topk_idx             (B, k)  — selected expert indices
    topk_weights         (B, k)  — Shazeer top-k mixture weights (Σ_j = 1)
    clean_logits         (B, E)  — for downstream router-scale diagnostics
    noisy_logits         (B, E)  — same; equal to clean for TopkGate / eval
    noise_std            (B, E) | None
                                 — per-sample per-expert noise std (training
                                   only, NoisyTopkGate only)
    aux_loss             scalar  — total auxiliary loss (with grad)
    importance_loss      scalar  — Shazeer importance term
    load_loss            scalar  — Shazeer Gaussian-CDF load term
    router_z_loss        scalar  — clean-logit z-regulariser
    ctx_aux_loss         scalar  — unweighted F.cross_entropy(ctx_logits, y)
    ctx_aux_loss_weighted scalar — coef · ctx_aux_loss (what enters aux_loss)
    ctx_aux_acc          scalar  — context classification accuracy
    ctx_target_field     str     — name of the supervised context field
    ctx_pred_hist        list[int] of length num_context_classes
    ctx_label_hist       list[int] of length num_context_classes
    ctx_logits_mean_abs  scalar
    clean_logits_*       router-scale diagnostics (mean/std/abs_mean/min/
                                                     max/lse_mean)
    noisy_logits_*       same, when noisy_logits != clean_logits
    noise_std_*          mean/min/max, when noise_std is not None
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .losses import importance_loss, load_loss, router_z_loss
from .routing import (BEVSummaryHead, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


def _logit_diagnostics(prefix: str, logits: Tensor) -> Dict[str, float]:
    """Compact set of scale diagnostics for a (B, E) logit tensor."""
    with torch.no_grad():
        l = logits.detach()
        return {
            f'{prefix}_mean':     float(l.mean().item()),
            f'{prefix}_std':      float(l.std(unbiased=False).item()),
            f'{prefix}_abs_mean': float(l.abs().mean().item()),
            f'{prefix}_min':      float(l.min().item()),
            f'{prefix}_max':      float(l.max().item()),
            f'{prefix}_lse_mean': float(torch.logsumexp(l, dim=-1).mean().item()),
        }


def _noise_diagnostics(noise_std: Tensor) -> Dict[str, float]:
    with torch.no_grad():
        n = noise_std.detach()
        return {
            'noise_std_mean': float(n.mean().item()),
            'noise_std_min':  float(n.min().item()),
            'noise_std_max':  float(n.max().item()),
        }


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Single-modality BEV Mixture-of-Experts block with context-supervised
    routing.

    Args:
        channels:           Number of BEV feature channels (input == output).
        num_experts:        Number of expert modules.
        k:                  Top-k experts selected per sample (default 2).
        num_convs:          Conv layers inside each BEVResidualExpert.
        importance_coef:    Weight α for the Shazeer importance loss.
        load_coef:          Weight α for the Shazeer load loss.
        z_loss_coef:        Weight λ_z for ``router_z_loss(clean_logits)``.
                            Pass 0 to disable.  Default 1e-4.
        residual_gain:      Scalar multiplier on the routed expert delta in
                            the residual-delta dispatch.  Default 1.0.
        router_pool_size:   Spatial size of the BEVSummaryHead pooled grid.
                            Default 4.
        router_spatial_dim: Channels in the BEVSummaryHead 1×1 + 3×3
                            spatial mixer convs.  Default 128.
        router_hidden_dim:  Hidden width of the BEVSummaryHead MLP.
                            Default 256.
        router_out_dim:     Output dim of BEVSummaryHead (gate input dim).
                            Default 128.
        context_aux_cfg:    Dict configuring the context-supervised
                            auxiliary loss.  Required keys / defaults::

                                target_field      str   (no default; required)
                                loss_coef         float = 0.05
                                label_smoothing   float = 0.0

                            ``target_field`` must be a key of
                            :data:`ZOD_FIELD_REGISTRY`.  Pass ``None`` to
                            disable context supervision entirely (the
                            block then trains with only importance + load
                            + z-loss).
        gate_type:          ``'topk'`` (deterministic) or ``'noisy_topk'``
                            (Shazeer noisy gate).  Default ``'topk'``.
        gate_cfg:           Extra kwargs forwarded to NoisyTopkGate
                            (``temperature``, ``noise_epsilon``).
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        k: int = 2,
        num_convs: int = 1,
        importance_coef: float = 0.02,
        load_coef: float = 0.002,
        z_loss_coef: float = 1e-4,
        residual_gain: float = 1.0,
        router_pool_size: int = 4,
        router_spatial_dim: int = 128,
        router_hidden_dim: int = 256,
        router_out_dim: int = 128,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'topk',
        gate_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef
        self.z_loss_coef = float(z_loss_coef)
        self.residual_gain = float(residual_gain)

        self.experts = make_bev_experts(num_experts, channels, num_convs)

        self.summary = BEVSummaryHead(
            channels=channels,
            pool_size=router_pool_size,
            spatial_dim=router_spatial_dim,
            hidden_dim=router_hidden_dim,
            out_dim=router_out_dim,
        )

        # ── Context auxiliary classification head ─────────────────────
        # Configured via context_aux_cfg = dict(target_field=..., loss_coef=..., label_smoothing=...)
        # The head is a plain Linear from the BEV summary descriptor to
        # the context vocabulary.  Its output is consumed only by
        # F.cross_entropy — never by the gate.
        self.context_aux_cfg: Optional[dict] = None
        self.context_head: Optional[nn.Linear] = None
        self._ctx_vocab_map: Optional[Dict[str, int]] = None
        self._ctx_target_field: Optional[str] = None
        self._ctx_loss_coef: float = 0.0
        self._ctx_label_smoothing: float = 0.0
        if context_aux_cfg is not None:
            self._build_context_head(context_aux_cfg)

        extra_gate_kwargs = gate_cfg or {}
        if gate_type == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=router_out_dim,
                num_experts=num_experts,
                k=k,
                **extra_gate_kwargs,
            )
        else:
            self.gate = TopkGate(
                feat_dim=router_out_dim,
                num_experts=num_experts,
                k=k,
            )

        # Sanity check: the gate must consume z directly (no context concat).
        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        assert gate_in == router_out_dim, (
            f'BEVMoEBlock: gate input dim ({gate_in}) must equal '
            f'router_out_dim ({router_out_dim}) — context vector must NOT '
            f'be concatenated into the router input.')

        self._moe_info: Optional[Dict[str, Any]] = None

    # ── Construction helpers ───────────────────────────────────────────

    def _build_context_head(self, cfg: dict) -> None:
        """Configure and instantiate the context auxiliary classifier."""
        cfg = dict(cfg)  # don't mutate caller's dict
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg must include a 'target_field' "
                "(e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"BEVMoEBlock.context_aux_cfg got unexpected keys: {list(cfg)}")

        vocab = get_context_vocab(target_field)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        num_classes = len(vocab)

        self.context_head = nn.Linear(self.summary.out_dim, num_classes)
        self._ctx_vocab_map = vocab_map
        self._ctx_target_field = target_field
        self._ctx_loss_coef = loss_coef
        self._ctx_label_smoothing = label_smoothing
        self.context_aux_cfg = dict(
            target_field=target_field,
            loss_coef=loss_coef,
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        x_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x_bev:              BEV feature map (B, C, H, W).
            batch_input_metas:  Per-sample metadata dicts.  Required when
                                ``context_aux_cfg`` is configured (used to
                                extract integer context labels).

        Returns:
            x_out:    BEV feature map (B, C, H, W) after expert processing.
            moe_info: Diagnostics dict (see module docstring).
        """
        B = x_bev.shape[0]

        # ── Step 1: Build routing descriptor ──────────────────────────
        z = self.summary(x_bev)  # (B, router_out_dim)

        # ── Step 2: Gate → top-k expert selection ─────────────────────
        # gate.forward consumes z DIRECTLY — context is never concatenated.
        gate_out = self.gate(z)

        # ── Step 3: Dispatch to selected experts ──────────────────────
        x_out = x_bev.clone()
        for b in range(B):
            xb = x_bev[b:b + 1]
            delta_sum = torch.zeros_like(xb)
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](xb)
                delta_sum = delta_sum + weight * (expert_out - xb)
            x_out[b] = (xb + self.residual_gain * delta_sum)[0]

        # ── Step 4: Auxiliary losses ──────────────────────────────────
        imp_loss = importance_loss(
            gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(
            gate_out.clean_logits, gate_out.noisy_logits,
            gate_out.noise_std, self.k, self.load_coef)
        z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)

        # Context auxiliary classification ----------------------------------
        ctx_loss_raw = z.new_zeros(())
        ctx_loss_weighted = z.new_zeros(())
        ctx_acc = z.new_zeros(())
        ctx_pred_hist: List[int] = []
        ctx_label_hist: List[int] = []
        ctx_logits_mean_abs = 0.0

        if self.context_head is not None:
            if batch_input_metas is None:
                raise RuntimeError(
                    'BEVMoEBlock: context_aux_cfg is configured but '
                    'batch_input_metas was not passed to forward().')
            ctx_logits = self.context_head(z)                      # (B, K)
            assert ctx_logits.dim() == 2 and \
                ctx_logits.shape[0] == B and \
                ctx_logits.shape[1] == self.context_aux_cfg['num_classes'], (
                f'context_head produced unexpected shape '
                f'{tuple(ctx_logits.shape)}; expected '
                f'(B, num_context_classes)')
            ctx_labels = extract_context_labels(
                batch_input_metas,
                self._ctx_target_field,
                self._ctx_vocab_map,
                z.device,
            )
            assert ctx_labels.dtype == torch.long and ctx_labels.shape == (B,)
            ctx_loss_raw = F.cross_entropy(
                ctx_logits, ctx_labels,
                label_smoothing=self._ctx_label_smoothing)
            ctx_loss_weighted = self._ctx_loss_coef * ctx_loss_raw
            with torch.no_grad():
                pred = ctx_logits.argmax(dim=-1)
                ctx_acc = (pred == ctx_labels).float().mean()
                num_classes = self.context_aux_cfg['num_classes']
                ctx_pred_hist = torch.bincount(
                    pred, minlength=num_classes).cpu().tolist()
                ctx_label_hist = torch.bincount(
                    ctx_labels, minlength=num_classes).cpu().tolist()
                ctx_logits_mean_abs = float(ctx_logits.abs().mean().item())

        aux = imp_loss + ld_loss + z_loss + ctx_loss_weighted

        # ── Step 5: Build moe_info ────────────────────────────────────
        moe_info: Dict[str, Any] = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'clean_logits':         gate_out.clean_logits.detach(),
            'noisy_logits':         gate_out.noisy_logits.detach(),
            'noise_std':            (gate_out.noise_std.detach()
                                     if gate_out.noise_std is not None else None),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
            'router_z_loss':        z_loss,
            # ctx_aux_loss is the unweighted F.cross_entropy value, kept
            # detached for inspection only (does NOT enter gradient flow
            # via this dict).
            'ctx_aux_loss':         (ctx_loss_raw.detach()
                                     if isinstance(ctx_loss_raw, Tensor)
                                     else ctx_loss_raw),
            # ctx_aux_loss_weighted is the gradient-bearing tensor that
            # entered aux_loss above.  Keep it WITH grad so BEVFusion.loss
            # can split aux_loss into per-component log entries without
            # losing gradient signal.
            'ctx_aux_loss_weighted': ctx_loss_weighted,
            'ctx_aux_acc':          ctx_acc.detach()
                                    if isinstance(ctx_acc, Tensor) else ctx_acc,
            'ctx_target_field':     self._ctx_target_field,
            'ctx_pred_hist':        ctx_pred_hist,
            'ctx_label_hist':       ctx_label_hist,
            'ctx_logits_mean_abs':  ctx_logits_mean_abs,
        }
        moe_info.update(_logit_diagnostics('clean_logits', gate_out.clean_logits))
        if gate_out.noisy_logits is not gate_out.clean_logits and \
                not torch.equal(gate_out.noisy_logits, gate_out.clean_logits):
            moe_info.update(
                _logit_diagnostics('noisy_logits', gate_out.noisy_logits))
        if gate_out.noise_std is not None:
            moe_info.update(_noise_diagnostics(gate_out.noise_std))

        self._moe_info = moe_info
        return x_out, moe_info

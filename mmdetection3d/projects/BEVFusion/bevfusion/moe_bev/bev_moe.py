"""Single-input BEV Mixture-of-Experts block (context-supervised routing).

BEVMoEBlock is a single-input MoE block used for:
  - Variant C: fusion-then-MoE — applied to the fused BEV after ConvFuser
    and before pts_backbone.
  - Variant D: LiDAR-only MoE — applied after SECONDFPN (post-neck,
    pre-bbox_head), operating on the semantically rich 512-ch FPN output.

  For modality-specific experts (joint gate over cam + lidar expert pools),
  use ModalitySpecificMoEBlock instead.

Single-summary design (z_ctx-driven gate)
------------------------------------------
A single ``BEVResSummaryEncoder`` branch (stem + 3 residual conv blocks +
avg + max grid pooling at a small ``P × P``, default ``P = 2``) produces
the descriptor consumed by the gate:

    z_ctx     — context branch; optimised by the context auxiliary CE.

The gate consumes ``z_ctx`` with stop-gradient so the context descriptor's
shape is fixed by context CE alone, while the gate's own ``Linear`` weights
remain task-driven via the dispatch path:

    z_ctx      = self.context_summary(x_bev)             # (B, 256)

    z_gate     = z_ctx.detach()                         # (B, 256), stop-gradient
    gate_out   = self.gate(z_gate)                       # task-driven Linear → top-k
    ctx_logits = self.context_head(z_ctx)                # MLP head on z_ctx (full grad)

The encoder learns: spatial structure first (residual conv blocks at full
BEV resolution) → coarse spatial summary (avg + max grid pooling to a
``(P, P)`` map, concatenated along the channel axis) → projected
descriptor.  Replacing the previous ``AdaptiveAvgPool2d(1)`` step with
avg + max grid pooling preserves coarse BEV layout (front/back occupancy,
sparse-vs-dense scenes, highway vs urban geometry) which is intended to
improve routing/context separability across driving regimes without any
other architectural change.

Why a single descriptor (history)
---------------------------------
An earlier dual-summary design carried a separate ``router_summary``
branch (z_router) so the gate received ``cat([z_router, z_ctx.detach()])``.
In practice z_router was overwhelmingly shaped by the auxiliary balance
losses (``importance_loss``, ``load_loss``, ``switch_balance_loss``,
``router_z_loss``) — its detection-task gradient travelled through the
discrete top-k softmax and was much weaker than the direct gate-Linear
update path.  z_router therefore acted mostly as a freshly-initialised
noise channel into the gate input.  Removing it halves the descriptor
parameters, eliminates that noise channel, and forces routing to be a
function of the context-discriminative descriptor.  Detection-task
gradient still drives expert specialisation through ``self.gate`` (its
own ``Linear`` layer is updated via the softmax → top-k weights → expert
outputs → detection loss path), so the router remains task-driven; what
changes is its *input space*, which is now anchored to ``z_ctx``.

This design lets:

- ``z_ctx`` learn road_type cleanly under weighted CE without detection
  gradients competing inside the same descriptor.
- The gate read context information through ``z_ctx.detach()`` — it
  *sees* context but cannot differentiate through it.
- The gate's ``Linear`` weights still receive task gradient via the
  dispatch path, so expert specialisation remains task-driven.

Context-supervised routing
--------------------------
Context labels are NOT fed to the gate.  The block's ``context_head``
(a small MLP — ``Linear → ReLU → LayerNorm → Dropout → Linear``) consumes
``z_ctx`` exclusively and is supervised by one of:

    'ce'          — plain ``F.cross_entropy`` with optional label smoothing.
    'weighted_ce' — ``F.cross_entropy`` with per-class weights; the
                    default for imbalanced context fields
                    (e.g. ``road_type`` on ZOD where 'city' dominates).
                    ``class_weights`` accepts either an explicit list of
                    floats (one per class, in the order of the
                    ``ZOD_FIELD_REGISTRY`` vocab) or the string
                    ``'inverse_frequency'`` which uses the
                    ZOD-training-distribution-derived fallback weights.
    'focal'       — focal CE with modulating factor ``(1 - p_t)^gamma``.
                    Kept as an option; in practice 'focal' still
                    collapsed to the majority-class under heavy class
                    imbalance on ZOD, so 'weighted_ce' is preferred.

For the full pattern (LiDAR-only NoisyTopkGate setup, post-SECONDFPN)::

    # x_bev is e.g. (B, 256, H, W) from pts_middle_encoder (pre_backbone)
    z_ctx      = self.context_summary(x_bev)             # BEVResSummaryEncoder
    z_gate     = z_ctx.detach()                         # stop-gradient
    gate_out   = self.gate(z_gate)                       # TopkGate — dispatch
    ctx_logits = self.context_head(z_ctx)     # MLP head on z_ctx
    # loss_type='weighted_ce' (default in this file):
    ctx_loss   = F.cross_entropy(ctx_logits, ctx_label,
                                 weight=class_weights,
                                 label_smoothing=label_smoothing)
    z_loss     = router_z_loss(gate_out.clean_logits, z_loss_coef)
    switch     = switch_balance_loss(
                     gate_out.full_softmax_probs,
                     gate_out.clean_topk_idx,   # ← clean, not noisy
                     num_experts,
                     switch_balance_coef)
    aux = (importance_loss + load_loss + z_loss + switch
           + ctx_loss_coef · ctx_loss)

``switch_balance_loss`` is fed ``clean_topk_idx`` rather than the noisy
``topk_idx`` so it disciplines the *clean* router that is used at
validation time; see ``moe_bev/losses.py::switch_balance_loss``.

Dispatch strategy: residual-delta with Shazeer top-k mixture
------------------------------------------------------------
Experts output ``x + delta``.  Dispatch is implemented as

    x_out = x_bev + g · Σ_j  w_j · (expert_j(x_bev) − x_bev)

with ``Σ_j w_j = 1`` (Shazeer top-k) and ``g = residual_gain`` a plain
scalar (default 1.0).  See module history below for why the dependency on
``num_experts`` was removed.

Dense (``gate_type='dense'``) dispatch
--------------------------------------
When ``gate_type='dense'`` the block bypasses the top-k cliff entirely
and dispatches every expert on every sample, mixing them by the full
pre-top-k softmax probabilities::

    x_out = x_bev + g · Σ_{e=1..E}  p_e · (expert_e(x_bev) − x_bev)

with ``p_e = full_softmax_probs[:, e]`` (Σ_e p_e = 1 by construction).
This removes the three top-k pathologies that drove the LiDAR-only AP gap
in runs 4552697 / 4554362 (see canvas
``canvases/lidar-moe-ap-gap-diagnosis.canvas.tsx``):

  1. Gradient starvation — every expert receives a non-zero detection
     gradient on every sample (weight = ``p_e``, never zero).
  2. Discontinuous expert switching — small logit changes produce small
     mixing-weight changes; no top-k membership flips.
  3. Train/val routing mismatch — the same softmax weights drive
     dispatch in train and eval (no noisy-vs-clean discrepancy).

Under dense, ``switch_balance_loss`` becomes a constant (``α·E·1`` with
all experts always selected) and should be disabled by setting
``switch_balance_coef=0``.  ``importance_loss`` and ``router_z_loss``
still bite — they regularise the soft balance and the logit scale and
remain useful.  ``ctx_gate_warmup`` is also pointless (the temperature
schedule was a fix for cliff brittleness) so leave it disabled.

For diagnostic compatibility the dense path still populates the usual
``topk_idx`` / ``topk_weights`` fields on ``moe_info`` with the experts
sorted descending by their dense probability, so ``MoERoutingHook``,
``ExpertRespawnHook`` and the context-routing hooks see meaningful
``top1_selection_freq``-style metrics (now equal to the dense argmax
frequency).  ``topk_selection_freq_per_expert`` becomes 1.0 for every
expert (every expert is "in top-k" when k = E) — read it together with
``dense_mean_prob_per_expert`` for the real signal.

moe_info contract
-----------------
After every forward() call, ``self._moe_info`` is written with:

    full_softmax_probs   (B, E)  — softmax over clean_logits (router belief)
    sparse_softmax_probs (B, E)  — top-k mixture laid back into (B, E)
    topk_idx             (B, k)  — dispatch top-k (noisy for NoisyTopkGate
                                   during training)
    topk_weights         (B, k)  — Shazeer top-k mixture weights (Σ_j = 1)
    clean_topk_idx       (B, k)  — deterministic clean top-k (equal to
                                   topk_idx for TopkGate and for
                                   NoisyTopkGate in eval mode)
    clean_logits         (B, E)  — for downstream router-scale diagnostics
    noisy_logits         (B, E)  — same; equal to clean for TopkGate / eval
    noise_std            (B, E) | None
                                 — per-sample per-expert noise std (training
                                   only, NoisyTopkGate only)
    aux_loss             scalar  — total auxiliary loss (with grad)
    importance_loss      scalar  — Shazeer importance term
    load_loss            scalar  — Shazeer Gaussian-CDF load term
    switch_balance_loss  scalar  — Fedus Switch balance term, computed
                                   from clean_topk_idx (disabled when
                                   switch_balance_coef == 0)
    router_z_loss        scalar  — clean-logit z-regulariser
    ctx_aux_loss         scalar  — unweighted context loss (weighted_ce / ce / focal)
    ctx_aux_loss_weighted scalar — coef · ctx_aux_loss (what enters aux_loss)
    ctx_aux_acc          scalar  — context classification accuracy
    ctx_target_field     str     — name of the supervised context field
    ctx_pred_hist        list[int] of length num_context_classes
    ctx_label_hist       list[int] of length num_context_classes
    ctx_logits_mean_abs  scalar
    ctx_loss_type        str     — 'weighted_ce' | 'ce' | 'focal'
    ctx_class_weights    list[float] | None — per-class weights actually
                                   used by weighted_ce (None for other
                                   loss types)
    focal_gamma          float   — focusing parameter (only meaningful
                                   when loss_type='focal')
    clean_logits_*       router-scale diagnostics (mean/std/abs_mean/min/
                                                     max/lse_mean)
    noisy_logits_*       same, when noisy_logits != clean_logits
    noise_std_*          mean/min/max, when noise_std is not None
    noise_scale          float   — global multiplier on sampled gate noise
                                   (NoisyTopkGate only; 1.0 if absent)
    noise_epsilon        float   — pre-softplus epsilon on the noise std
                                   head (NoisyTopkGate only)
    noise_to_clean_std_ratio
                         scalar  — (noise_scale · noise_std_mean) /
                                   clean_logits_std.  Multiplying by
                                   ``noise_scale`` is critical: the actual
                                   noise injected into noisy_logits is
                                   ``noise_scale · randn · noise_std``, so
                                   the effective exploration std is the
                                   scaled product, not the bare softplus
                                   output of the noise head.  Target: ≲ 1
                                   under NoisyTopkGate (≈ 0.5 is a healthy
                                   exploration regime).
    context_summary_type            str  — 'BEVResSummaryEncoder'.
    context_summary_stem_channels   int  — width of stem and first two res blocks.
    context_summary_out_channels    int  — width of third res block and pooled rep.
    context_summary_out_dim         int  — projected descriptor dimension (256).
    context_summary_num_res_blocks  int  — number of residual blocks (3).
    context_summary_dropout         float
    context_summary_pool_size       int  — avg+max grid pool side length P
                                           (current default 2 → 4-cell summary).
    gate_feat_dim            int  — context_summary.out_dim (256); the gate
                                    consumes either z_ctx.detach() or z_ctx
                                    depending on gate_input_detach.
    z_ctx_detached_for_gate  bool — value of self.gate_input_detach.
    gate_input                str  — 'z_ctx_detach' (context-supervised
                                    routing) or 'z_ctx' (task-driven routing).
    context_head_type        str  — 'mlp' (when a context_head is configured)
                                    or 'none' (when context_aux_cfg=None).
    moe_insertion_point      str  — 'post_secondfpn'.
    moe_input_channels       int  — input channel count (512 for Variant D).
    gate_type                str  — 'topk' | 'noisy_topk' | 'dense'.
    dense_dispatch           bool — True when every expert always runs
                                    (gate_type='dense'); False under top-k.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .losses import (importance_loss, load_loss, router_z_loss,
                      switch_balance_loss)
from .routing import (BEVResSummaryEncoder, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


# ── Built-in inverse-frequency class-weight fallbacks ────────────────────
# Normalised to mean 1.  Computed from the ZOD training-split distribution
# of each context field (see also ``compute_inverse_frequency_weights``
# for doing this from your own data offline).  Used when a block is
# configured with ``class_weights='inverse_frequency'`` but no explicit
# weight list is provided — the values below are close to what you get
# from the training pkl today (e.g. road_type: city dominates, smaller-
# rural is rare).
_INVERSE_FREQUENCY_FALLBACK: Dict[str, List[float]] = {
    # ZOD_FIELD_REGISTRY order:
    # ['arterial-rural', 'arterial-urban', 'city', 'highway', 'smaller-rural']
    'road_type': [1.13, 0.47, 0.18, 1.8, 3.0],
}


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


def focal_ce_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
    """Focal cross-entropy loss (Lin et al. 2017).

    Modulates standard cross-entropy by ``(1 - p_t) ** gamma`` where
    ``p_t`` is the probability assigned to the true class.  Easy
    majority-class examples (high ``p_t``) are down-weighted, allowing
    the model to focus on harder minority samples without any explicit
    class-frequency weighting.

    Args:
        logits:  Raw unnormalised scores, shape (B, C).
        targets: Integer class indices, shape (B,).
        gamma:   Focusing parameter ≥ 0.  gamma=0 recovers plain CE.
                 Default 2.0 (standard from the focal-loss paper).

    Returns:
        Scalar mean focal loss.
    """
    ce = F.cross_entropy(logits, targets, reduction='none')  # (B,)
    pt = torch.exp(-ce)                                       # probability of true class
    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


@MODELS.register_module()
class BEVMoEBlock(nn.Module):
    """Single-modality BEV Mixture-of-Experts block with context-supervised
    routing.

    Args:
        channels:             Number of BEV feature channels (input == output).
        num_experts:          Number of expert modules.
        k:                    Top-k experts selected per sample (default 2).
        num_convs:            Conv layers inside each legacy
                              ``BEVResidualExpert``.  Ignored when
                              ``expert_type='bottleneck'`` (default), which
                              has a fixed reduce/spatial/expand structure.
        expert_type:          ``'bottleneck'`` (default,
                              :class:`BEVBottleneckResidualExpert`,
                              identity-init residual adapter, ~12× cheaper
                              under dense dispatch) or ``'full'`` (legacy
                              :class:`BEVResidualExpert` operating at the
                              full channel count ``C``).
        expert_hidden_channels:
                              Bottleneck width for the bottleneck expert.
                              Default 128.  Ignored when
                              ``expert_type='full'``.
        importance_coef:      Weight α for the Shazeer importance loss.
        load_coef:            Weight α for the Shazeer load loss.
        switch_balance_coef:  Weight α for the Switch balance loss, computed
                              from ``GateOutput.clean_topk_idx`` so it
                              disciplines the clean (validation-time)
                              router rather than the noisy training
                              dispatch.  Pass 0 to disable.  Default 0.0.
        z_loss_coef:          Weight λ_z for ``router_z_loss(clean_logits)``.
                              Pass 0 to disable.  Default 1e-4.
        residual_gain:        Scalar multiplier on the routed expert delta in
                              the residual-delta dispatch.  Default 1.0.
        context_aux_cfg:      Dict configuring the context-supervised
                              auxiliary loss.  Recognised keys / defaults::

                                target_field      str   (no default; required)
                                loss_coef         float = 0.05
                                loss_type         str   = 'weighted_ce'
                                                         # 'weighted_ce' |
                                                         # 'ce' | 'focal'
                                class_weights     list[float] | 'inverse_frequency' | None
                                                         # weighted_ce only
                                focal_gamma       float = 2.0
                                label_smoothing   float = 0.0

                              ``target_field`` must be a key of
                              :data:`ZOD_FIELD_REGISTRY`.  When
                              ``loss_type='weighted_ce'``,
                              ``class_weights`` either enumerates the
                              per-class weights in the order of the
                              registry vocab or equals
                              ``'inverse_frequency'`` (fallback weights
                              from the ZOD training distribution, see
                              :data:`_INVERSE_FREQUENCY_FALLBACK`).  When
                              ``loss_type='focal'``, ``focal_ce_loss``
                              is used and ``label_smoothing`` /
                              ``class_weights`` are ignored.  Pass
                              ``None`` as the whole config to disable
                              context supervision.
        gate_type:            ``'topk'`` (deterministic top-k),
                              ``'noisy_topk'`` (Shazeer noisy top-k), or
                              ``'dense'`` (every expert always runs,
                              mixed by ``full_softmax_probs``).  When
                              ``'dense'``, ``k`` is forced to
                              ``num_experts`` and the top-k cliff is
                              bypassed; see the "Dense dispatch" section
                              of this module's docstring.  Default
                              ``'topk'``.
        gate_cfg:             Extra kwargs forwarded to NoisyTopkGate
                              (``temperature``, ``noise_epsilon``,
                              ``noise_scale``).  The ``temperature`` value in
                              this dict is the *post-warmup* steady-state
                              temperature (usually 1.0).
        ctx_gate_warmup_epochs: Number of epochs over which the gate's
                              softmax temperature is linearly annealed from
                              ``ctx_gate_temp_high`` down to 1.0.  During
                              epoch ``e`` the gate receives::

                                  T(e) = temp_high + (1 - temp_high)
                                         * min(1, e / ctx_gate_warmup_epochs)

                              High temperature → flat softmax → routing stays
                              balanced while ``z_ctx`` is still weak.  After
                              ``ctx_gate_warmup_epochs`` epochs the temperature
                              is 1.0 and routing is fully normal.  Balance
                              losses remain active throughout.  Set to 0
                              (default) to disable (temperature always stays
                              at whatever ``gate_cfg.temperature`` specifies).
                              Call ``set_epoch(epoch)`` each epoch (e.g. from
                              :class:`MoERoutingHook`) to advance the
                              schedule.
        ctx_gate_temp_high:   Starting temperature for the warmup schedule.
                              Default 5.0.  Ignored when
                              ``ctx_gate_warmup_epochs == 0``.
        gate_input_detach:    If ``True`` (default, context-supervised
                              routing) the gate consumes ``z_ctx.detach()``
                              and the BEVResSummaryEncoder is shaped only
                              by the auxiliary context CE — detection
                              gradient never reaches it.  If ``False``
                              (task-driven routing) the gate consumes
                              ``z_ctx`` directly so detection-loss
                              gradient flows back through the gate's
                              softmax → expert dispatch → expert outputs
                              all the way into ``context_summary``.  The
                              encoder learns whatever feature the routing
                              decision needs to depend on for the task,
                              with no auxiliary supervision.

                              Validation: when ``gate_input_detach=True``
                              you must also pass a ``context_aux_cfg`` —
                              otherwise the encoder receives no gradient
                              at all and stays at its random init for
                              the whole run.  Conversely setting
                              ``gate_input_detach=False`` together with a
                              ``context_aux_cfg`` is allowed (encoder
                              gets gradient from both sources) but is
                              effectively a hybrid mode and not the
                              intended use of this flag.
    """

    # Architecture dimensions for the context_summary BEVResSummaryEncoder.
    # The gate consumes z_ctx.detach() directly, so its input dim equals
    # _SUMMARY_OUT_DIM (no separate router descriptor / no concat).
    _SUMMARY_STEM_CHANNELS  = 128
    _SUMMARY_OUT_CHANNELS   = 256
    _SUMMARY_OUT_DIM        = 256
    _SUMMARY_DROPOUT        = 0.2
    # Avg+max grid pool side length P.  Was 4 (16-cell summary, projection
    # Linear ≈ 2.1M params); reduced to 2 (4-cell summary, projection
    # Linear ≈ 524K params, ~1.6M fewer parameters in context_summary).
    # The 2×2 grid still preserves the front/back × left/right gross
    # spatial layout that drove the routing-separability win when GAP
    # was replaced with grid pooling, but cuts the projection's share
    # of context_summary capacity by 4×.  Empirically (canvas dense-moe-
    # vs-baseline-4562173-4562168 §5) the routing decision is dominated
    # by aggregate channel statistics within each spatial cell rather
    # than fine-grained 4×4 spatial layout, so the smaller summary
    # carries essentially the same routing signal at lower cost.
    _SUMMARY_POOL_SIZE      = 2

    def __init__(
        self,
        channels: int,
        num_experts: int = 6,
        k: int = 2,
        num_convs: int = 1,
        importance_coef: float = 0.02,
        load_coef: float = 0.002,
        switch_balance_coef: float = 0.0,
        z_loss_coef: float = 1e-4,
        residual_gain: float = 1.0,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'topk',
        gate_cfg: Optional[dict] = None,
        ctx_gate_warmup_epochs: int = 0,
        ctx_gate_temp_high: float = 5.0,
        gate_input_detach: bool = True,
        expert_type: str = 'bottleneck',
        expert_hidden_channels: int = 128,
        expert_norm_type: str = 'bn',
    ):
        super().__init__()
        self.channels = channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = float(importance_coef)
        self.load_coef = float(load_coef)
        self.switch_balance_coef = float(switch_balance_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.residual_gain = float(residual_gain)
        self.ctx_gate_warmup_epochs = int(ctx_gate_warmup_epochs)
        self.ctx_gate_temp_high = float(ctx_gate_temp_high)
        # Whether to stop-gradient on the gate's input (default True =
        # historical context-supervised design; False = task-driven
        # routing where the BEVResSummaryEncoder is shaped by detection
        # loss flowing back through the gate → expert dispatch path).
        # Validation below ensures the encoder always has *some* gradient
        # source — either the auxiliary CE (detach=True path) or the
        # detection-via-gate path (detach=False path).
        self.gate_input_detach = bool(gate_input_detach)
        if self.gate_input_detach and context_aux_cfg is None:
            raise ValueError(
                'BEVMoEBlock: gate_input_detach=True with context_aux_cfg='
                'None means the BEVResSummaryEncoder receives no gradient '
                'from any source (context CE is disabled and detection '
                'gradient is blocked at the detach).  The encoder would '
                'remain at its random initialisation throughout training. '
                'Either set gate_input_detach=False (task-driven routing) '
                'or provide a context_aux_cfg (context-supervised '
                'routing).')
        self._current_epoch: int = 0

        self.expert_type = str(expert_type).lower()
        self.expert_hidden_channels = int(expert_hidden_channels)
        self.expert_norm_type = str(expert_norm_type).lower()
        # ``expert_norm_type`` selects the in-expert normalisation
        # flavour for ``expert_type='full'``.  Default ``'bn'``
        # preserves the lidar-only MoE behaviour; ``'gn'`` is
        # required for fusion-then-MoE where the fused (cam ⊕ lidar)
        # BEV channel distribution drives BN ``running_var → 0`` and
        # the resulting BN output saturates fp16 (see run 4613034
        # epoch 6 in the run notes).  Ignored when
        # ``expert_type='bottleneck'``.
        self.experts = make_bev_experts(
            num_experts, channels,
            num_convs=num_convs,
            expert_type=self.expert_type,
            hidden_channels=self.expert_hidden_channels,
            norm_type=self.expert_norm_type)

        # ── Single BEVResSummaryEncoder branch ────────────────────────
        # context_summary is the *only* descriptor encoder; it is trained
        # by the auxiliary context CE.  The gate consumes z_ctx.detach()
        # so context-CE gradients never reach the gate's Linear and the
        # gate's task signal travels exclusively through dispatch (its
        # Linear weights → softmax → top-k weights → expert outputs →
        # detection loss).
        #
        # Architecture: stem conv → 3 residual blocks → avg + max grid
        # pooling at (P, P) (concatenated along channel axis) → Linear
        # → LayerNorm.  Spatial structure is learned at full BEV
        # resolution and then summarised at a fixed (P, P) grid rather
        # than being collapsed to a single global average vector.
        self.context_summary = BEVResSummaryEncoder(
            channels=channels,
            stem_channels=self._SUMMARY_STEM_CHANNELS,
            out_channels=self._SUMMARY_OUT_CHANNELS,
            out_dim=self._SUMMARY_OUT_DIM,
            dropout=self._SUMMARY_DROPOUT,
            pool_size=self._SUMMARY_POOL_SIZE,
        )

        # ── Context auxiliary classification head ─────────────────────
        # Configured via ``context_aux_cfg``; see ``_build_context_head``
        # for the accepted keys.  The head is an MLP wired to
        # ``self.context_summary.out_dim`` (z_ctx only — not z_gate).
        # Class weights, when used, are registered as a buffer so they
        # follow device transitions automatically.
        self.context_aux_cfg: Optional[dict] = None
        self.context_head: Optional[nn.Module] = None
        self._ctx_vocab_map: Optional[Dict[str, int]] = None
        self._ctx_target_field: Optional[str] = None
        self._ctx_loss_coef: float = 0.0
        self._ctx_loss_type: str = 'ce'
        self._ctx_focal_gamma: float = 2.0
        self._ctx_label_smoothing: float = 0.05
        # ``_ctx_class_weights_list`` is the plain Python list (None when
        # not used) kept for logging.  The tensor version is registered
        # as a buffer in ``_build_context_head`` (named
        # ``_ctx_class_weights``) so it moves with the module.
        self._ctx_class_weights_list: Optional[List[float]] = None
        if context_aux_cfg is not None:
            self._build_context_head(context_aux_cfg)

        # Gate input = z_ctx.detach(); its dim equals context_summary.out_dim.
        gate_feat_dim = self.context_summary.out_dim
        extra_gate_kwargs = gate_cfg or {}
        gate_type_norm = str(gate_type).lower()
        if gate_type_norm not in ('topk', 'noisy_topk', 'dense'):
            raise ValueError(
                "BEVMoEBlock.gate_type must be 'topk', 'noisy_topk' or "
                f"'dense', got '{gate_type}'.")
        # Dense dispatch: every expert always runs, weighted by the full
        # pre-top-k softmax.  We still construct a TopkGate with
        # k=num_experts so that GateOutput.topk_idx / topk_weights carry
        # the experts sorted by dense probability for downstream
        # diagnostics; the dispatch math itself uses full_softmax_probs
        # directly (see forward()).
        self._dense_dispatch = (gate_type_norm == 'dense')
        if self._dense_dispatch:
            self.k = num_experts
            self.gate = TopkGate(
                feat_dim=gate_feat_dim,
                num_experts=num_experts,
                k=num_experts,
                **extra_gate_kwargs,
            )
        elif gate_type_norm == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=gate_feat_dim,
                num_experts=num_experts,
                k=k,
                **extra_gate_kwargs,
            )
        else:
            self.gate = TopkGate(
                feat_dim=gate_feat_dim,
                num_experts=num_experts,
                k=k,
                **extra_gate_kwargs,
            )
        self.gate_type = gate_type_norm

        # Prime the gate at the warmup starting temperature so that the very
        # first forward pass (before before_train_epoch fires) is already warm.
        if self.ctx_gate_warmup_epochs > 0 and self.ctx_gate_temp_high > 1.0:
            self.gate.set_temperature(self.ctx_gate_temp_high)

        # Sanity check: gate must consume z_ctx.detach() directly.
        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        expected_gate_in = self.context_summary.out_dim
        assert gate_in == expected_gate_in, (
            f'BEVMoEBlock: gate input dim ({gate_in}) must equal '
            f'context_summary.out_dim ({expected_gate_in}).')

        self._moe_info: Optional[Dict[str, Any]] = None

    # ── Epoch schedule ─────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Advance the temperature-annealing schedule to ``epoch``.

        Should be called at the start of each training epoch (e.g. from
        :class:`MoERoutingHook`).  Has no effect when
        ``ctx_gate_warmup_epochs == 0`` (warmup disabled).
        """
        self._current_epoch = int(epoch)

    @property
    def router_temperature(self) -> float:
        """Current gate softmax temperature for the active epoch.

        Linearly decays from ``ctx_gate_temp_high`` at epoch 0 to 1.0 at
        epoch ``ctx_gate_warmup_epochs``.  Returns 1.0 unconditionally when
        ``ctx_gate_warmup_epochs <= 0`` or ``ctx_gate_temp_high <= 1.0``.
        """
        if self.ctx_gate_warmup_epochs <= 0 or self.ctx_gate_temp_high <= 1.0:
            return 1.0
        alpha = min(1.0, self._current_epoch / self.ctx_gate_warmup_epochs)
        return self.ctx_gate_temp_high + (1.0 - self.ctx_gate_temp_high) * alpha

    # ── Construction helpers ───────────────────────────────────────────

    def _build_context_head(self, cfg: dict) -> None:
        """Configure and instantiate the context auxiliary classifier.

        Recognised ``cfg`` keys
        -----------------------
        ``target_field``     — required; must be a key of
                               ``ZOD_FIELD_REGISTRY``.
        ``loss_coef``        — scalar weight on the context loss in the
                               block's total auxiliary loss (default 0.05).
        ``loss_type``        — ``'weighted_ce'`` (default), ``'ce'`` or
                               ``'focal'``.
        ``class_weights``    — required only for ``'weighted_ce'``: either
                               a list of floats (one per class, in the
                               registry vocab order) or the string
                               ``'inverse_frequency'`` (use built-in
                               fallback weights from the ZOD training
                               distribution; see
                               :data:`_INVERSE_FREQUENCY_FALLBACK`).
                               The weights are normalised to mean 1 so
                               ``loss_coef`` retains its usual scale.
        ``focal_gamma``      — focusing parameter for ``'focal'``
                               (default 2.0).  Ignored otherwise.
        ``label_smoothing``  — forwarded to ``F.cross_entropy`` for
                               ``'ce'`` / ``'weighted_ce'``.  Ignored by
                               focal.
        """
        cfg = dict(cfg)  # don't mutate caller's dict
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg must include a 'target_field' "
                "(e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        loss_type = str(cfg.pop('loss_type', 'weighted_ce')).lower()
        if loss_type not in ('weighted_ce', 'ce', 'focal'):
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg: loss_type must be "
                "'weighted_ce', 'ce' or 'focal', got "
                f"'{loss_type}'.")
        class_weights_cfg = cfg.pop('class_weights', None)
        focal_gamma = float(cfg.pop('focal_gamma', 2.0))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"BEVMoEBlock.context_aux_cfg got unexpected keys: {list(cfg)}")

        vocab = get_context_vocab(target_field)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        num_classes = len(vocab)

        # MLP context head wired to context_summary.out_dim (z_ctx).
        # z_ctx is produced by the dedicated context branch and never
        # enters the gate directly.  Full gradient flows through z_ctx.
        z_dim = self.context_summary.out_dim
        self.context_head = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(z_dim),
            nn.Dropout(0.2),
            nn.Linear(z_dim, num_classes),
        )

        # Resolve per-class weights for the 'weighted_ce' path.
        class_weights_list: Optional[List[float]] = None
        if loss_type == 'weighted_ce':
            class_weights_list = self._resolve_class_weights(
                class_weights_cfg, target_field, num_classes)
            # Buffer so device transitions (e.g. .cuda()) are automatic.
            w = torch.tensor(class_weights_list, dtype=torch.float32)
            self.register_buffer('_ctx_class_weights', w, persistent=False)
        elif class_weights_cfg is not None:
            # Gently warn through a ValueError so configs stay explicit.
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg: 'class_weights' is only "
                "meaningful with loss_type='weighted_ce' (got "
                f"'{loss_type}').")

        self._ctx_vocab_map = vocab_map
        self._ctx_target_field = target_field
        self._ctx_loss_coef = loss_coef
        self._ctx_loss_type = loss_type
        self._ctx_focal_gamma = focal_gamma
        self._ctx_label_smoothing = label_smoothing
        self._ctx_class_weights_list = class_weights_list
        self.context_aux_cfg = dict(
            target_field=target_field,
            loss_coef=loss_coef,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=(list(class_weights_list)
                           if class_weights_list is not None else None),
            num_classes=num_classes,
        )

    @staticmethod
    def _resolve_class_weights(
        cfg_val: Any,
        target_field: str,
        num_classes: int,
    ) -> List[float]:
        """Resolve the ``class_weights`` cfg value into a concrete list.

        Accepts an explicit list of floats or the string
        ``'inverse_frequency'`` which pulls from
        :data:`_INVERSE_FREQUENCY_FALLBACK`.  The returned list is
        normalised to mean 1 so ``loss_coef`` keeps its usual scale
        regardless of the absolute weight magnitudes.
        """
        if cfg_val is None:
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg: loss_type='weighted_ce' "
                "requires 'class_weights' (list or 'inverse_frequency').")
        if isinstance(cfg_val, str):
            key = cfg_val.lower()
            if key != 'inverse_frequency':
                raise ValueError(
                    "BEVMoEBlock.context_aux_cfg: unknown "
                    f"class_weights='{cfg_val}'. Accepted strings: "
                    "'inverse_frequency'.")
            if target_field not in _INVERSE_FREQUENCY_FALLBACK:
                raise ValueError(
                    "BEVMoEBlock.context_aux_cfg: no built-in "
                    f"inverse-frequency fallback for '{target_field}'. "
                    "Provide an explicit class_weights list.")
            raw = list(_INVERSE_FREQUENCY_FALLBACK[target_field])
        else:
            try:
                raw = [float(w) for w in cfg_val]
            except TypeError as e:
                raise ValueError(
                    "BEVMoEBlock.context_aux_cfg: class_weights must be "
                    "an iterable of floats or 'inverse_frequency'.") from e

        if len(raw) != num_classes:
            raise ValueError(
                f"BEVMoEBlock.context_aux_cfg: class_weights length "
                f"{len(raw)} does not match num_context_classes "
                f"{num_classes} for target_field='{target_field}'.")
        if any(w < 0 for w in raw):
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg: class_weights must be "
                "non-negative.")
        mean = sum(raw) / len(raw)
        if mean <= 0:
            raise ValueError(
                "BEVMoEBlock.context_aux_cfg: class_weights have zero "
                "mean; cannot normalise.")
        return [w / mean for w in raw]

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

        # ── Steps 1–2: Routing path (context_summary + gate) in fp32 ──
        # The routing path is small relative to the experts but
        # numerically fragile: a 3-block residual CNN at full BEV
        # resolution feeds an avg+max-pool projection ending in
        # ``LayerNorm`` and is then fed to a small ``Linear → softmax``
        # gate.  Running it under the outer ``AmpOptimWrapper`` fp16
        # autocast produced non-finite ``z_ctx`` values from iter 1
        # onwards in the LiDAR-only runs (e.g. ``lidar-moe_4594168``
        # stderr: 70k+ ``NaN/Inf in z_ctx`` events starting at the
        # first training step).  Each event used to be masked by a
        # ``torch.nan_to_num`` here, which created a *worse* failure
        # mode than the original NaN:
        #
        #   * the gate received an exact-zero descriptor → uniform
        #     softmax every iter → router decoupled from the experts;
        #   * ``nan_to_num`` backward returns 0 wherever input was
        #     non-finite, so the autograd path through
        #     ``context_summary`` was killed and the encoder stayed at
        #     its random initialisation for the entire run;
        #   * the total loss stayed finite (because the masked aux CE
        #     was finite), so ``GradScaler`` never skipped the step,
        #     ``parse_losses`` never warned, and
        #     ``_repair_bn_stats`` never fired.  The model trained
        #     happily through corruption until backbone / expert
        #     weights themselves overflowed at peak LR and the mAP
        #     curve cliffed (epoch 7–8 collapse in the same run).
        #
        # The fix is structural, not defensive: promote the entire
        # routing block to fp32.  context_summary + gate +
        # context_head + aux losses together are a small fraction of
        # one expert in FLOPs, so the AMP memory / speed win on the
        # dominant compute (the experts and backbone) is preserved.
        # No fallback ``nan_to_num`` is layered on top — if a
        # non-finite value still appears (truly catastrophic upstream
        # corruption) we *want* it to propagate to the total loss so
        # the existing ``BEVFusion.parse_losses`` /
        # ``train_step`` defences (warn, skip step, repair BN stats)
        # can do their job.  Masking it would re-introduce the silent
        # corruption mode this rewrite eliminates.
        with torch.autocast('cuda', enabled=False):
            x_bev_fp32 = x_bev.float()

            # z_ctx:  context branch — its gradient sources depend on
            #         the training mode:
            #           * gate_input_detach=True  (default, context-
            #             supervised): z_ctx is shaped by the auxiliary
            #             context CE only; detection-loss gradient is
            #             blocked from reaching context_summary by the
            #             detach below.
            #           * gate_input_detach=False (task-driven): z_ctx
            #             is shaped by the detection loss flowing back
            #             through gate → expert dispatch → expert
            #             outputs.  The encoder learns whatever
            #             feature the routing decision needs to depend
            #             on, with no auxiliary supervision.
            # z_gate: feeds the gate.  Detached when
            #         gate_input_detach=True so the gate's input space
            #         is fixed by context CE alone (gate Linear is
            #         still task-driven via dispatch).  Same tensor as
            #         z_ctx when gate_input_detach=False so detection
            #         gradient flows all the way back into
            #         context_summary.
            # z_ctx itself is NOT detached for context_head (full grad
            # there whenever a context_head exists).
            z_ctx = self.context_summary(x_bev_fp32)   # (B, 256), fp32

            if self.gate_input_detach:
                z_gate = z_ctx.detach()           # (B, 256), stop-gradient
            else:
                z_gate = z_ctx                    # (B, 256), full grad through gate

            # Apply temperature annealing: T decays from
            # ctx_gate_temp_high (e.g. 5.0) to 1.0 over
            # ctx_gate_warmup_epochs.  High T keeps the softmax flat
            # so routing stays balanced while z_ctx is still weak in
            # early epochs.  Balance losses remain active throughout.
            if self.ctx_gate_warmup_epochs > 0:
                self.gate.set_temperature(self.router_temperature)
            gate_out = self.gate(z_gate)

        # ── Step 3: Dispatch to selected experts ──────────────────────
        # Dense path: every expert always runs, weighted by the full
        # pre-top-k softmax (no top-k cliff, no gradient starvation, no
        # train/val routing mismatch).  See module docstring "Dense
        # dispatch" section.  The Shazeer top-k path is preserved below
        # for ``gate_type ∈ {'topk', 'noisy_topk'}``.
        #
        # Batched-vectorised dense mixing: each expert is called *once*
        # per step on the full ``(B, C, H, W)`` input rather than once
        # per sample with batch_size=1.  This restores normal BN batch
        # statistics (per-sample loops were effectively InstanceNorm in
        # train and updated running stats with effective momentum
        # ``B × configured_momentum`` — both producing a real train↔eval
        # mismatch *inside* the experts, see canvas dense-moe-vs-baseline-
        # 4562173-4562168 §4).  It also drops wall-clock per step ~1/B
        # because we skip ``B − 1`` redundant per-sample forward passes
        # per expert.  The math is identical to the previous loop:
        #
        #     x_out = x_bev + g · Σ_e probs[:, e] · (expert_e(x_bev) − x_bev)
        if self._dense_dispatch:
            probs = gate_out.full_softmax_probs                       # (B, E)
            delta_sum = torch.zeros_like(x_bev)
            for e in range(self.num_experts):
                # delta_e ∈ (B, C, H, W); expert_e(x_bev) sees full batch
                # so its BN behaves exactly like any other batched layer.
                delta_e = self.experts[e](x_bev) - x_bev
                # probs[:, e] ∈ (B,) → broadcast to (B, 1, 1, 1) for
                # per-sample dispatch weighting on the spatial map.
                w = probs[:, e].view(-1, 1, 1, 1).to(delta_e.dtype)
                delta_sum = delta_sum + w * delta_e
            x_out = x_bev + self.residual_gain * delta_sum
        else:
            # Top-k path (k < E): different samples can route to
            # different expert subsets, so a fully-batched implementation
            # would have to run *every* expert on *every* sample
            # (defeating the point of top-k).  We keep the per-sample
            # loop for correctness, accepting the BN-batch-size-1 quirk
            # because all current production runs use ``gate_type=
            # 'dense'``.  Treat this branch as legacy / ablation only.
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

        # ── Step 4: Auxiliary losses (fp32, same rationale as Step 1–2)
        # ``context_head`` is a tiny ``Linear → ReLU → LayerNorm →
        # Dropout → Linear`` MLP and the balance losses are scalar
        # reductions over ``(B, E)`` tensors — both need to be fp32 for
        # the same numerical-robustness reason that the routing block
        # above is fp32.  The previous ``nan_to_num`` guard on
        # ``ctx_logits`` is deliberately not reinstated here: under
        # fp32 the only way to reach a non-finite ``ctx_logits`` is via
        # corruption upstream of this block, and in that case we want
        # the NaN to propagate to the total loss so GradScaler can
        # skip the step and ``_repair_bn_stats`` can run.
        with torch.autocast('cuda', enabled=False):
            imp_loss = importance_loss(
                gate_out.full_softmax_probs, self.importance_coef)
            ld_loss  = load_loss(
                gate_out.clean_logits, gate_out.noisy_logits,
                gate_out.noise_std, self.k, self.load_coef)
            z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)

            # Switch balance loss — Fedus et al. (2022),
            # α · E · Σ f_e · P_e.  Fed with the *clean* top-k
            # (``clean_topk_idx``) so it disciplines the deterministic
            # validation-time router rather than the noisy training
            # dispatch; see losses.py docstring.  Under
            # :class:`TopkGate` and under :class:`NoisyTopkGate` in
            # eval, ``clean_topk_idx`` equals ``topk_idx`` so this is
            # a no-op relative to the classical Switch formulation.
            #
            # In dense dispatch (``gate_type='dense'``, k=E) the
            # switch loss collapses to the constant α (every expert is
            # "selected" on every sample so f_e = 1/E uniformly,
            # giving E·Σ(1/E)·P_e = 1).  The gradient through P_e is a
            # uniform additive bias on every logit and provides no
            # specialisation signal — so we short-circuit the term to
            # zero regardless of the configured coefficient and rely
            # on ``importance_loss`` for soft balance.
            if self._dense_dispatch:
                sw_loss = z_ctx.new_zeros(())
            elif self.switch_balance_coef > 0.0:
                clean_idx = gate_out.clean_topk_idx
                if clean_idx is None:                         # legacy safety
                    clean_idx = gate_out.topk_idx
                sw_loss = switch_balance_loss(
                    gate_out.full_softmax_probs,
                    clean_idx,
                    self.num_experts,
                    self.switch_balance_coef,
                )
            else:
                sw_loss = z_ctx.new_zeros(())

            # Context auxiliary classification ─────────────────────────
            ctx_loss_raw = z_ctx.new_zeros(())
            ctx_loss_weighted = z_ctx.new_zeros(())
            ctx_acc = z_ctx.new_zeros(())
            ctx_pred_hist: List[int] = []
            ctx_label_hist: List[int] = []
            ctx_logits_mean_abs = 0.0

            if self.context_head is not None:
                if batch_input_metas is None:
                    raise RuntimeError(
                        'BEVMoEBlock: context_aux_cfg is configured but '
                        'batch_input_metas was not passed to forward().')
                ctx_logits = self.context_head(z_ctx)              # (B, K), fp32
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
                    z_ctx.device,
                )
                assert ctx_labels.dtype == torch.long and ctx_labels.shape == (B,)
                if self._ctx_loss_type == 'focal':
                    ctx_loss_raw = focal_ce_loss(
                        ctx_logits, ctx_labels, gamma=self._ctx_focal_gamma)
                elif self._ctx_loss_type == 'weighted_ce':
                    # Weights registered as buffer → follows device moves.
                    w = self._ctx_class_weights.to(
                        device=ctx_logits.device, dtype=ctx_logits.dtype)
                    ctx_loss_raw = F.cross_entropy(
                        ctx_logits, ctx_labels,
                        weight=w,
                        label_smoothing=self._ctx_label_smoothing)
                else:  # 'ce'
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

            aux = imp_loss + ld_loss + z_loss + sw_loss + ctx_loss_weighted

        # ── Step 5: Build moe_info ────────────────────────────────────
        clean_topk_idx_detached = (
            gate_out.clean_topk_idx.detach()
            if gate_out.clean_topk_idx is not None
            else gate_out.topk_idx.detach())

        moe_info: Dict[str, Any] = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'clean_topk_idx':       clean_topk_idx_detached,
            'clean_logits':         gate_out.clean_logits.detach(),
            'noisy_logits':         gate_out.noisy_logits.detach(),
            'noise_std':            (gate_out.noise_std.detach()
                                     if gate_out.noise_std is not None else None),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
            # Fedus Switch balance over clean_topk_idx (0 when disabled).
            # Kept WITH grad so BEVFusion.loss can log it as its own
            # entry without breaking gradient flow through aux_loss.
            'switch_balance_loss':  sw_loss,
            'router_z_loss':        z_loss,
            # ctx_aux_loss is the unweighted context loss
            # (weighted_ce / ce / focal), detached for inspection only
            # (does NOT enter gradient flow via this dict).
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
            'ctx_loss_type':        self._ctx_loss_type,
            'ctx_class_weights':    (list(self._ctx_class_weights_list)
                                     if self._ctx_class_weights_list is not None
                                     else None),
            'focal_gamma':          self._ctx_focal_gamma,
            # Single BEVResSummaryEncoder config (z_ctx-only gate input).
            'context_summary_type':            'BEVResSummaryEncoder',
            'context_summary_stem_channels':   self.context_summary.stem_channels,
            'context_summary_out_channels':    self.context_summary.out_channels,
            'context_summary_out_dim':         self.context_summary.out_dim,
            'context_summary_num_res_blocks':  self.context_summary.num_res_blocks,
            'context_summary_dropout':         self.context_summary.dropout,
            'context_summary_pool_size':       self.context_summary.pool_size,
            'gate_feat_dim':                   self.context_summary.out_dim,
            'z_ctx_detached_for_gate':         self.gate_input_detach,
            'gate_input':                      ('z_ctx_detach'
                                                if self.gate_input_detach
                                                else 'z_ctx'),
            'context_head_type':               ('mlp'
                                                if self.context_head is not None
                                                else 'none'),
            'moe_input_channels':              self.channels,
            'ctx_gate_warmup_epochs':          self.ctx_gate_warmup_epochs,
            'router_temperature':              self.router_temperature,
            'gate_type':                       self.gate_type,
            'dense_dispatch':                  self._dense_dispatch,
        }

        # Router-scale diagnostics from clean / noisy logits.
        clean_diag = _logit_diagnostics('clean_logits', gate_out.clean_logits)
        moe_info.update(clean_diag)
        noisy_logits_diff_from_clean = (
            gate_out.noisy_logits is not gate_out.clean_logits and
            not torch.equal(gate_out.noisy_logits, gate_out.clean_logits))
        if noisy_logits_diff_from_clean:
            moe_info.update(
                _logit_diagnostics('noisy_logits', gate_out.noisy_logits))

        # Noise diagnostics + noise-to-clean-std ratio.  The ratio targets
        # values ≲ 1 — when larger the training-time gate is noise-driven
        # and the deterministic validation router sees an essentially
        # different distribution (hence the collapse-on-val failure
        # mode).  The ratio multiplies ``noise_std_mean`` by the active
        # ``noise_scale`` because the noise actually injected into
        # ``noisy_logits`` has std = ``noise_scale · noise_std`` — using
        # the bare ``noise_std_mean`` over-reports exploration whenever
        # ``noise_scale ≠ 1`` (e.g. the 4.19 reading on the previous run
        # corresponded to a real ratio of 0.34 with noise_scale=0.08).
        if gate_out.noise_std is not None:
            moe_info.update(_noise_diagnostics(gate_out.noise_std))
            with torch.no_grad():
                clean_std = float(
                    gate_out.clean_logits.detach().std(
                        unbiased=False).item())
                noise_mean = float(moe_info['noise_std_mean'])
                noise_scale = (float(self.gate.noise_scale)
                               if isinstance(self.gate, NoisyTopkGate)
                               else 1.0)
                moe_info['noise_to_clean_std_ratio'] = round(
                    (noise_scale * noise_mean) / (clean_std + 1e-8), 6)

        # Gate-config diagnostics (NoisyTopkGate only; echo the active
        # ``noise_scale`` / ``noise_epsilon`` so the training log
        # captures them).  For TopkGate they simply aren't reported.
        if isinstance(self.gate, NoisyTopkGate):
            moe_info['noise_scale']   = float(self.gate.noise_scale)
            moe_info['noise_epsilon'] = float(self.gate.noise_epsilon)

        self._moe_info = moe_info
        return x_out, moe_info

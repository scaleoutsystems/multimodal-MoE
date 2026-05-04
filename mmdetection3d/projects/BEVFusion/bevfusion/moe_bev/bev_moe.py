"""Single-input BEV Mixture-of-Experts block (context-supervised routing).

BEVMoEBlock is a single-input MoE block used for:
  - Variant C: fusion-then-MoE — applied to the fused BEV after ConvFuser
    and before pts_backbone.
  - Variant D: LiDAR-only MoE — applied after SECONDFPN (post-neck,
    pre-bbox_head), operating on the semantically rich 512-ch FPN output.

  For modality-specific experts (joint gate over cam + lidar expert pools),
  use ModalitySpecificMoEBlock instead.

Dual-summary design (residual-CNN encoders)
--------------------------------------------
Two independent ``BEVResSummaryEncoder`` branches (stem + 3 residual conv
blocks + global average pooling) produce separate descriptors:

    z_router  — task / routing branch; optimised by detection + MoE losses.
    z_ctx     — context branch; optimised by context CE loss only.

The gate receives a concatenation of both, but with ``z_ctx`` stop-grad
so detection/router gradients cannot corrupt the context descriptor:

    z_router  = self.router_summary(x_bev)   # (B, 256)
    z_ctx     = self.context_summary(x_bev)  # (B, 256)

    z_gate    = torch.cat([z_router, z_ctx.detach()], dim=1)  # (B, 512)
    gate_out  = self.gate(z_gate)            # NoisyTopkGate — dispatch
    ctx_logits = self.context_head(z_ctx)    # MLP head on z_ctx (full grad)

Each encoder learns: spatial structure first (residual conv blocks at full
BEV resolution) → global compression (AdaptiveAvgPool2d(1)) → projected
descriptor.  This is the same principle as ResNet feature extraction used
in the reference CIFAR MoE experiment.

This design lets:

- ``z_ctx`` learn road_type cleanly under weighted CE without detection
  gradients competing inside the same descriptor.
- The router read context information through ``z_ctx.detach()`` — the gate
  *sees* context but cannot differentiate through it.
- ``z_router`` remain free to learn task/routing-specific structure.

Context-supervised routing
--------------------------
Context labels are NOT concatenated into the gate input.  The block's
``context_head`` (a small MLP — ``Linear → ReLU → LayerNorm → Dropout →
Linear``) consumes ``z_ctx`` exclusively and is supervised by one of:

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

    # x_bev is (B, 512, H, W) from SECONDFPN
    z_router   = self.router_summary(x_bev)   # BEVResSummaryEncoder
    z_ctx      = self.context_summary(x_bev)  # BEVResSummaryEncoder
    z_gate     = torch.cat([z_router, z_ctx.detach()], dim=1)
    gate_out   = self.gate(z_gate)            # NoisyTopkGate — dispatch
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
                         scalar  — noise_std_mean / clean_logits_std
                                   (target: ≲ 1 under NoisyTopkGate)
    router_summary_type             str  — 'BEVResSummaryEncoder'.
    router_summary_stem_channels    int  — width of stem and first two res blocks.
    router_summary_out_channels     int  — width of third res block and pooled rep.
    router_summary_out_dim          int  — projected descriptor dimension (256).
    router_summary_num_res_blocks   int  — number of residual blocks (3).
    router_summary_dropout          float
    context_summary_type            str  — 'BEVResSummaryEncoder'.
    context_summary_stem_channels   int
    context_summary_out_channels    int
    context_summary_out_dim         int
    context_summary_num_res_blocks  int
    context_summary_dropout         float
    gate_feat_dim            int  — 512 (= router_out_dim + context_out_dim).
    z_ctx_detached_for_gate  bool — always True in this design.
    context_head_type        str  — 'mlp' (MLP head on z_ctx).
    moe_insertion_point      str  — 'post_secondfpn'.
    moe_input_channels       int  — input channel count (512 for Variant D).
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
        num_convs:            Conv layers inside each BEVResidualExpert.
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
        gate_type:            ``'topk'`` (deterministic) or ``'noisy_topk'``
                              (Shazeer noisy gate).  Default ``'topk'``.
        gate_cfg:             Extra kwargs forwarded to NoisyTopkGate
                              (``temperature``, ``noise_epsilon``,
                              ``noise_scale``).
    """

    # Shared architecture dimensions for both BEVResSummaryEncoder branches.
    # Both router_summary and context_summary are built with these params.
    _SUMMARY_STEM_CHANNELS  = 128
    _SUMMARY_OUT_CHANNELS   = 256
    _SUMMARY_OUT_DIM        = 256
    _SUMMARY_DROPOUT        = 0.2
    # Gate input = concat([z_router, z_ctx.detach()]) = 256 + 256.
    _GATE_FEAT_DIM          = _SUMMARY_OUT_DIM * 2  # 512

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

        self.experts = make_bev_experts(num_experts, channels, num_convs)

        # ── Dual BEVResSummaryEncoder branches ────────────────────────
        # Both encoders share the same architecture but have separate
        # weights; they are trained by independent loss signals.
        #
        # router_summary:  optimised by detection + MoE routing losses.
        # context_summary: optimised by context CE loss only.
        #
        # Each encoder: stem conv → 3 residual blocks → global avg pool
        # → Linear → LayerNorm.  Spatial structure is learned at full BEV
        # resolution before global compression.
        #
        # The gate receives cat([z_router, z_ctx.detach()]) so the router
        # can read context structure without context CE corrupting z_router.
        _enc_kwargs = dict(
            channels=channels,
            stem_channels=self._SUMMARY_STEM_CHANNELS,
            out_channels=self._SUMMARY_OUT_CHANNELS,
            out_dim=self._SUMMARY_OUT_DIM,
            dropout=self._SUMMARY_DROPOUT,
        )
        self.router_summary  = BEVResSummaryEncoder(**_enc_kwargs)
        self.context_summary = BEVResSummaryEncoder(**_enc_kwargs)

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

        # Gate input = cat([z_router, z_ctx.detach()]) = 512.
        gate_feat_dim = self._GATE_FEAT_DIM
        extra_gate_kwargs = gate_cfg or {}
        if gate_type == 'noisy_topk':
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
            )

        # Sanity check: gate must consume concat([z_router, z_ctx]).
        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        expected_gate_in = (self.router_summary.out_dim
                            + self.context_summary.out_dim)
        assert gate_in == expected_gate_in, (
            f'BEVMoEBlock: gate input dim ({gate_in}) must equal '
            f'router_summary.out_dim + context_summary.out_dim '
            f'({expected_gate_in}).')

        self._moe_info: Optional[Dict[str, Any]] = None

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

        # ── Step 1: Build dual BEV descriptors ────────────────────────
        # z_router: task/routing branch — shaped by detection + MoE losses.
        # z_ctx:    context branch — shaped by context CE only.
        # z_gate:   cat([z_router, z_ctx.detach()]) fed to the gate so the
        #           router reads context structure without context CE gradients
        #           corrupting z_router.  z_ctx is NOT detached for context_head.
        z_router = self.router_summary(x_bev)   # (B, 256)
        z_ctx    = self.context_summary(x_bev)  # (B, 256)
        z_gate   = torch.cat([z_router, z_ctx.detach()], dim=1)  # (B, 512)

        # ── Step 2: Gate → top-k expert selection ─────────────────────
        gate_out = self.gate(z_gate)

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

        # Switch balance loss — Fedus et al. (2022), α · E · Σ f_e · P_e.
        # Fed with the *clean* top-k (``clean_topk_idx``) so it
        # disciplines the deterministic validation-time router rather
        # than the noisy training dispatch; see losses.py docstring.
        # Under :class:`TopkGate` and under :class:`NoisyTopkGate` in
        # eval, ``clean_topk_idx`` equals ``topk_idx`` so this is a no-op
        # relative to the classical Switch formulation.
        if self.switch_balance_coef > 0.0:
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
            sw_loss = z_router.new_zeros(())

        # Context auxiliary classification ----------------------------------
        ctx_loss_raw = z_router.new_zeros(())
        ctx_loss_weighted = z_router.new_zeros(())
        ctx_acc = z_router.new_zeros(())
        ctx_pred_hist: List[int] = []
        ctx_label_hist: List[int] = []
        ctx_logits_mean_abs = 0.0

        if self.context_head is not None:
            if batch_input_metas is None:
                raise RuntimeError(
                    'BEVMoEBlock: context_aux_cfg is configured but '
                    'batch_input_metas was not passed to forward().')
            ctx_logits = self.context_head(z_ctx)                  # (B, K)
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
            # Dual BEVResSummaryEncoder config.
            'router_summary_type':             'BEVResSummaryEncoder',
            'router_summary_stem_channels':    self.router_summary.stem_channels,
            'router_summary_out_channels':     self.router_summary.out_channels,
            'router_summary_out_dim':          self.router_summary.out_dim,
            'router_summary_num_res_blocks':   self.router_summary.num_res_blocks,
            'router_summary_dropout':          self.router_summary.dropout,
            'context_summary_type':            'BEVResSummaryEncoder',
            'context_summary_stem_channels':   self.context_summary.stem_channels,
            'context_summary_out_channels':    self.context_summary.out_channels,
            'context_summary_out_dim':         self.context_summary.out_dim,
            'context_summary_num_res_blocks':  self.context_summary.num_res_blocks,
            'context_summary_dropout':         self.context_summary.dropout,
            'gate_feat_dim':                   self._GATE_FEAT_DIM,
            'z_ctx_detached_for_gate':         True,
            'context_head_type':               'mlp',
            'moe_insertion_point':             'post_secondfpn',
            'moe_input_channels':              self.channels,
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
        # mode).  Exposed even when noise_std is None (value is None) so
        # downstream hooks can still record the key.
        if gate_out.noise_std is not None:
            moe_info.update(_noise_diagnostics(gate_out.noise_std))
            with torch.no_grad():
                clean_std = float(
                    gate_out.clean_logits.detach().std(
                        unbiased=False).item())
                noise_mean = float(moe_info['noise_std_mean'])
                moe_info['noise_to_clean_std_ratio'] = round(
                    noise_mean / (clean_std + 1e-8), 6)

        # Gate-config diagnostics (NoisyTopkGate only; echo the active
        # ``noise_scale`` / ``noise_epsilon`` so the training log
        # captures them).  For TopkGate they simply aren't reported.
        if isinstance(self.gate, NoisyTopkGate):
            moe_info['noise_scale']   = float(self.gate.noise_scale)
            moe_info['noise_epsilon'] = float(self.gate.noise_epsilon)

        self._moe_info = moe_info
        return x_out, moe_info

"""Routing modules for BEV Mixture-of-Experts.

Contents
--------
BEV summary encoders
    :class:`BasicBEVResBlock`      — building block for BEVResSummaryEncoder.
    :class:`BEVResSummaryEncoder`  — residual-CNN descriptor used by all variants.

Gate modules
    :class:`TopkGate`              — deterministic top-k gate.
    :class:`NoisyTopkGate`         — Shazeer noisy top-k gate.
    :class:`GateOutput`            — dataclass returned by both gates.

Context utilities
    :data:`ZOD_FIELD_REGISTRY`     — vocabulary registry for ZOD context fields.
    :func:`get_context_vocab`      — look up a field's vocab list.
    :func:`extract_context_labels` — convert batch metadata into integer labels.

Context-supervised routing
--------------------------
Context labels (``road_type``, ``weather_group``, …) are **not** concatenated
into the router input.  Instead, an auxiliary context head is supervised by
those labels and shapes the BEV summary descriptor via its gradient — the gate
never reads context labels directly.  This keeps expert dispatch task-driven
while biasing the descriptor toward context-relevant BEV structure.

All variants use :class:`BEVResSummaryEncoder` for descriptor extraction.

Variant D (LiDAR-only, ``BEVMoEBlock``) uses a single context-CE encoder
and feeds ``z_ctx.detach()`` straight to the gate::

    z_ctx      = context_summary(x_bev)           # context-CE branch
    z_gate     = z_ctx.detach()                   # fixed input space for gate
    gate_out   = gate(z_gate)                     # task-driven Linear → top-k
    ctx_logits = context_head(z_ctx)              # full grad through z_ctx

The gate's ``Linear`` weights still receive task gradient via the dispatch
path (top-k weights → expert outputs → detection loss), so the *router*
is task-driven; what's anchored to context is the *input space* of the
gate, not the routing decision itself.

Variants A/B (joint/modality-specific) use one encoder per modality::

    z_C  = cam_summary(cam_bev)
    z_L  = lidar_summary(lidar_bev)
    z_LC = cat([z_C, z_L])
    gate_out = gate(z_LC)

GateOutput contract
-------------------
Both gate classes return a :class:`GateOutput` with these fields:

    full_softmax_probs   (B, E)  — softmax(clean_logits/T); consumed by
                                   importance_loss and dense-prob diagnostics.
    sparse_softmax_probs (B, E)  — top-k mixture scattered back to (B, E);
                                   zeros off-topk.  Diagnostics only.
    topk_idx             (B, k)  — dispatch indices (noisy for NoisyTopkGate
                                   during training, else clean).
    topk_weights         (B, k)  — Shazeer weights: softmax(topk_vals/T),
                                   Σ_j = 1 per sample.  Used for dispatch.
    clean_logits         (B, E)  — pre-noise logits; used by router_z_loss,
                                   load_loss, and switch_balance_loss.
    noisy_logits         (B, E)  — logits used for the top-k selection;
                                   equals clean_logits at eval or for TopkGate.
    noise_std            (B, E) or None
                                 — per-element noise std (training only,
                                   NoisyTopkGate only); None signals
                                   load_loss to return zero.
    clean_topk_idx       (B, k)  — deterministic top-k of clean_logits;
                                   fed to switch_balance_loss so it
                                   disciplines the validation-time router.
                                   Equals topk_idx at eval or for TopkGate.

Top-k routing
-------------
Default ``k = 2`` is intentional:

* Top-1 makes assignment brittle; a single wrong decision has full impact.
* Top-2 enables cooperative specialisation and, for ModalitySpecificMoEBlock,
  allows mixed-modality (LiDAR + Camera) expert pairs.

Do not hard-code ``num_experts == num_context_classes`` and do not switch to
top-1 "context-class" routing — expert dispatch is task-driven.

Dense dispatch (``gate_type='dense'``)
--------------------------------------
When the router's dense softmax probabilities are bunched within a few
percent of uniform (typical of small-E setups with mild specialisation),
top-k dispatch acts as a discontinuous cliff: tiny logit perturbations
flip top-k membership, the bottom experts get zero detection gradient on
that sample, and validation routing oscillates because of the
noisy-vs-clean / train-vs-val mismatch around the cliff.  ``BEVMoEBlock``
exposes ``gate_type='dense'`` to bypass the cliff by mixing all experts
with their full softmax probabilities (``k`` is forced to ``num_experts``
internally).  Trade-offs:

* **Pro**: every expert always receives gradient (weight ``p_e``, never
  zero), no discontinuous switching, identical routing in train and val.
* **Con**: ``num_experts`` × the FLOPs of a single-expert forward —
  budget for that or reduce E if compute is tight.

Under dense, ``importance_loss`` and ``router_z_loss`` still bite (soft
balance and logit-scale anchoring); ``switch_balance_loss`` becomes a
useless constant α and ``BEVMoEBlock`` short-circuits it to zero
regardless of the configured coefficient.  ``load_loss`` is a no-op
under :class:`TopkGate` either way (no Gaussian noise to integrate).

LayerNorm as final activation
------------------------------
All summary encoders end in ``LayerNorm`` rather than ``ReLU``.  Signed,
unit-variance descriptors let the gate's linear projection produce logits of
both signs; a final ReLU leaves dead units at 0, which combined with weight
decay on the gate prevents logit magnitudes from growing (observed as a
dead-gate failure mode in earlier runs).

GroupNorm in the descriptor path
--------------------------------
The summary encoders use :class:`torch.nn.GroupNorm` for all intermediate
normalisation layers (stem and residual blocks).  BatchNorm is deliberately
*not* used here even though it's standard in detection backbones:

* BatchNorm normalises with batch statistics in train mode and EMA running
  statistics in eval mode.  With small per-GPU batches (4 on this setup) and
  class-imbalanced sampling, the running stats end up biased toward the
  dominant scene type seen during training (e.g. 'city' for ZOD road_type).
* This leaves a residual *direction* offset in the descriptor ``z`` between
  train and eval that the trailing ``LayerNorm(out_dim)`` cannot remove
  (LayerNorm only fixes magnitude, not direction).  The gate's linear
  projection then maps that offset to a constant additive bias on every
  expert logit — empirically observed as ``clean_logits_mean`` shifting
  by +0.6 between train and val while ``std`` stays unchanged, which
  causes whichever expert column is closest to the bias direction to
  dominate dispatch at validation time.
* GroupNorm has no running stats and behaves identically in train and
  eval, which removes this train↔eval descriptor drift entirely.  It also
  keeps the encoder robust to the small per-GPU batch sizes used here.

GroupNorm group size is fixed at 32 groups across the encoder (`stem_channels
= 128` → 4 ch/group; `out_channels = 256` → 8 ch/group).  See also
:func:`_make_group_norm` for the divisibility-safe wrapper used to build
each instance.

The experts (``BEVResidualExpert``) intentionally keep BatchNorm with a
small-random-init last gamma (N(0, 0.005)) — they sit on the *output* of the
gate where dispatch is balanced at train time, so their running stats don't
suffer the class-imbalance bias the descriptor encoder did.  The tiny gamma
std breaks expert symmetry from step 1 (so routing gradients carry per-expert
signal immediately) while keeping the initial expert perturbation at O(0.005),
safely within FP16 representable range on pretrained FPN features.  See
``bev_experts._LAST_BN_GAMMA_STD``.

Context target configuration
-----------------------------
Each MoE block takes a ``context_aux_cfg`` dict:

    context_aux_cfg = dict(
        target_field='road_type',   # key in ZOD_FIELD_REGISTRY
        loss_coef=0.05,
        loss_type='weighted_ce',    # 'weighted_ce' | 'ce' | 'focal'
        label_smoothing=0.05,
        class_weights='inverse_frequency',
    )

``target_field`` must be a key of :data:`ZOD_FIELD_REGISTRY`.  Use
:func:`extract_context_labels` to convert ``batch_input_metas`` to the
integer ``LongTensor`` consumed by the context CE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Normalisation helper ──────────────────────────────────────────────────

def _make_group_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Create a :class:`GroupNorm` whose group count divides ``num_channels``.

    Falls back to the largest divisor of ``num_channels`` not exceeding
    ``num_groups`` when 32 doesn't divide evenly (defensive against
    non-power-of-two channel counts that callers might use).
    """
    # in practice we have num_channels = 512, num_groups = 32
    # --> group size = 512/32 = 16
    # G=32 or so sits in a sweet spot. It's not so coarse that 
    # outlier channels dominate (Layer Norm), not so fine that
    # you lose cross-channel information (Instance Norm).
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels)


# ── BasicBEVResBlock ──────────────────────────────────────────────────────

class BasicBEVResBlock(nn.Module):
    """Two-layer residual conv block for BEV feature maps.

    Main path:
        Conv2d(in_channels → out_channels, 3×3, pad=1) → GroupNorm → ReLU
        Conv2d(out_channels → out_channels, 3×3, pad=1) → GroupNorm

    Residual path (identity when in_channels == out_channels, 1×1 conv
    + GroupNorm otherwise).

    Output: ReLU(main + residual)

    Normalisation
    -------------
    Uses :class:`GroupNorm` rather than :class:`BatchNorm2d` because this
    block sits inside the router/context descriptor path where the
    train --> eval mode-switch in BN running statistics produces a
    train-vs-validation descriptor drift that downstream LayerNorm cannot
    correct (it normalises descriptor magnitude but not direction).  See
    the module docstring for details.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = _make_group_norm(out_channels)

        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                _make_group_norm(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ── BEVResSummaryEncoder ──────────────────────────────────────────────────

class BEVResSummaryEncoder(nn.Module):
    """Residual-CNN BEV summary encoder producing a routing descriptor ``z``.

    Architecture
    ------------
    1. **Stem**: Conv2d(C → stem_channels, 3×3) → GroupNorm → ReLU — projects
       input channels without changing spatial resolution.

    2. **Residual blocks** (3 × :class:`BasicBEVResBlock`):
           BasicBEVResBlock(stem_channels, stem_channels)
           BasicBEVResBlock(stem_channels, stem_channels)
           BasicBEVResBlock(stem_channels, out_channels)
       The last block widens channels if ``stem_channels != out_channels``.
       Spatial structure is preserved (no pooling/striding) so all three
       blocks learn spatial patterns before grid pooling.

    3. **Avg + max grid pooling / vectorisation** (replaces the previous
       global average pooling):
           AdaptiveAvgPool2d((P, P))   → (B, out_channels, P, P)
           AdaptiveMaxPool2d((P, P))   → (B, out_channels, P, P)
           cat along channel dim       → (B, 2·out_channels, P, P)
           Flatten                    → (B, 2·out_channels·P·P)
           Linear(2·out_channels·P·P → out_dim) → (B, out_dim)
           LayerNorm(out_dim)         → signed, unit-variance descriptor
           Dropout(dropout)           → regularisation

    Rationale
    ---------
    The previous design used a single ``AdaptiveAvgPool2d(1)`` which
    collapses the entire BEV feature map into one vector per channel
    before projection.  That global compression discards routing-
    relevant spatial structure such as front/back occupancy patterns,
    sparse-vs-dense scene layout, long-range highway geometry, urban
    clutter distribution, and coarse spatial context differences
    between road types.  Replacing the GAP step with avg + max grid
    pooling at a small ``P × P`` (default 4 × 4) keeps the residual
    CNN exactly as-is while preserving a coarse, fixed-size spatial
    summary of the BEV.  Concatenating both pooled tensors along the
    channel axis lets the projection see *average activation* and
    *peak activation* per cell, which together carry both density and
    salience cues that a single mean over the map cannot express.

    The intent is to improve routing/context separability for different
    driving regimes (highway vs city vs rural) without adding any
    attention pooling, extra CNN stages, stride/downsampling changes,
    transformer layers, or metadata fusion — only the pooling and
    projection input dimension change.

    All intermediate normalisation is GroupNorm (no running statistics)
    so the descriptor is identical in train and eval mode.  See the
    module docstring "GroupNorm in the descriptor path" section for the
    full rationale.

    Args:
        channels:      Number of input BEV channels ``C``.
        stem_channels: Width of the stem and first two residual blocks.
                       Default 128.
        out_channels:  Width of the final residual block and pooled
                       representation before the projection MLP.
                       Default 256.
        out_dim:       Output descriptor dimension (same as gate input
                       dim per branch).  Exposed as ``self.out_dim``.
                       Default 256.
        dropout:       Dropout probability after LayerNorm.  Default 0.2.
        pool_size:     Spatial side length ``P`` of the avg + max grid
                       pooling output.  The descriptor is computed from
                       a ``(B, 2·out_channels, P, P)`` tensor.  Default
                       4 (i.e. a 4 × 4 grid summary, 16 spatial cells).
                       Set to 1 to recover a near-GAP behaviour (avg+max
                       over the whole map, still doubled by the max
                       branch).  Larger values preserve more spatial
                       layout at the cost of a wider projection Linear.
    """

    def __init__(
        self,
        channels: int,
        stem_channels: int = 128,
        out_channels: int = 256,
        out_dim: int = 256,
        dropout: float = 0.2,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        assert pool_size >= 1, f'pool_size must be ≥ 1, got {pool_size}'

        self.stem = nn.Sequential(
            nn.Conv2d(channels, stem_channels, kernel_size=3,
                      padding=1, bias=False),
            _make_group_norm(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            BasicBEVResBlock(stem_channels, stem_channels),
            BasicBEVResBlock(stem_channels, stem_channels),
            BasicBEVResBlock(stem_channels, out_channels),
        )

        # Avg + max grid pooling: produces two (B, out_channels, P, P)
        # tensors concatenated along the channel axis.  The avg branch
        # carries mean activation per cell (density / occupancy cue);
        # the max branch carries peak activation per cell (salience /
        # presence cue).  Together they give the projection both kinds
        # of spatial summary at a fixed (P × P) resolution.
        self.avg_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.max_pool = nn.AdaptiveMaxPool2d((pool_size, pool_size))

        # Projection input width: 2 · out_channels (avg ⊕ max) · P · P.
        pooled_dim = 2 * out_channels * pool_size * pool_size
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

        self.out_dim = out_dim
        self.stem_channels = stem_channels
        self.out_channels = out_channels
        self.num_res_blocks = 3
        self.dropout = dropout
        self.pool_size = pool_size
        self.pooled_dim = pooled_dim

    def forward(self, x: Tensor) -> Tensor:
        """Encode a BEV feature map into a fixed-size descriptor.

        Spatial structure is preserved through the residual conv blocks
        and then summarised by avg + max grid pooling at a fixed
        ``(pool_size, pool_size)`` resolution; no global average pooling
        is used.  Concatenating along the channel axis keeps the
        per-cell mean and per-cell max signals jointly available to the
        projection MLP.

        Args:
            x: BEV feature map ``(B, C, H, W)``.

        Returns:
            Routing/context descriptor ``(B, out_dim)``.
        """
        x = self.stem(x)         # (B, stem_channels, H, W)
        x = self.res_blocks(x)   # (B, out_channels, H, W)

        x_avg = self.avg_pool(x)  # (B, out_channels, P, P)
        x_max = self.max_pool(x)  # (B, out_channels, P, P)

        x = torch.cat([x_avg, x_max], dim=1)  # (B, 2·out_channels, P, P)

        return self.proj(x)      # (B, out_dim)


# ── Gate output container ─────────────────────────────────────────────────

@dataclass
class GateOutput:
    full_softmax_probs:   Tensor            # (B, E) softmax over clean_logits
    sparse_softmax_probs: Tensor            # (B, E) top-k mixture scattered
                                            #         into (B, E); zero off-topk
    topk_idx:             Tensor            # (B, k) dispatch indices
                                            #         (noisy for NoisyTopkGate
                                            #         during training)
    topk_weights:         Tensor            # (B, k) Shazeer top-k mixture
                                            #         weights, Σ_j = 1 per sample
    clean_logits:         Tensor            # (B, E) pre-noise gate logits
    noisy_logits:         Tensor            # (B, E) logits used for top-k
    noise_std:            Optional[Tensor]  # (B, E) or None — see GateOutput
                                            #         contract in the module
                                            #         docstring.
    clean_topk_idx:       Optional[Tensor] = None
                                            # (B, k) deterministic top-k of
                                            #         clean_logits.  Equals
                                            #         ``topk_idx`` for
                                            #         :class:`TopkGate` and for
                                            #         :class:`NoisyTopkGate` in
                                            #         eval mode; differs in
                                            #         training for the noisy
                                            #         gate.  See module
                                            #         docstring — used by
                                            #         ``switch_balance_loss``
                                            #         and the clean-routing
                                            #         selection-frequency
                                            #         diagnostics.


# ── ZOD field registry ────────────────────────────────────────────────────
# Single source of truth for context vocabularies.  Vocabs match the values
# present in the ZOD parquet + infos .pkl files (zod_nuscenes dataset).

ZOD_FIELD_REGISTRY: Dict[str, List[str]] = {
    'complexity_bin':    ['empty', 'low', 'medium', 'high'],
    'road_condition':    ['normal', 'wet', 'snow'],
    'road_type':         ['arterial-rural', 'arterial-urban', 'city',
                          'highway', 'smaller-rural'],
    'scraped_weather':   ['clear-day', 'clear-night', 'cloudy', 'fog',
                          'partly-cloudy-day', 'partly-cloudy-night',
                          'rain', 'snow', 'wind'],
    'solar_context_bin': ['day', 'night', 'twilight'],
    'weather_group':     ['clear_like', 'cloud_like', 'fog',
                          'precipitation', 'wind'],
}

ALL_FIELDS: List[str] = list(ZOD_FIELD_REGISTRY.keys())


# ── Context label utilities ───────────────────────────────────────────────

def get_context_vocab(target_field: str) -> List[str]:
    """Return the vocabulary list for a registered context field.

    Raises ``KeyError`` if the field is not in :data:`ZOD_FIELD_REGISTRY`.
    """
    if target_field not in ZOD_FIELD_REGISTRY:
        raise KeyError(
            f"Unknown context target_field '{target_field}'. "
            f"Known fields: {ALL_FIELDS}")
    return list(ZOD_FIELD_REGISTRY[target_field])


def extract_context_labels(
    batch_input_metas: List[dict],
    target_field: str,
    vocab_map: Dict[str, int],
    device: torch.device,
) -> Tensor:
    """Convert a batch of metadata dicts into integer context labels.

    Each meta is expected to carry a ``'context'`` sub-dict (added via
    ``Pack3DDetInputs(meta_keys=[..., 'context'])`` in the data pipeline).

    Args:
        batch_input_metas: Per-sample metadata, length B.
        target_field:      Name of the context field to extract (e.g.
            ``'road_type'``).  Must exist in every sample's ``'context'``
            sub-dict.
        vocab_map:         Mapping ``{value_string: integer_class_id}``.
        device:            Device on which to allocate the output tensor.

    Returns:
        A ``LongTensor`` of shape ``(B,)`` containing class ids.

    Raises:
        KeyError: If a sample's context value is missing from ``vocab_map``.
    """
    labels: List[int] = []
    for meta in batch_input_metas:
        ctx = meta.get('context', {}) or {}
        val = str(ctx.get(target_field, ''))
        if val not in vocab_map:
            raise KeyError(
                f"extract_context_labels: unexpected value '{val}' for "
                f"field '{target_field}'. Known values: {list(vocab_map)}")
        labels.append(vocab_map[val])
    return torch.tensor(labels, dtype=torch.long, device=device)


# ── TopkGate ──────────────────────────────────────────────────────────────

class TopkGate(nn.Module):
    """Deterministic top-k gate over experts (Shazeer-style dispatch weights).

    Routing procedure:
      1. ``logits = W_gate(z)``   — z is the BEV summary descriptor.
      2. ``full_softmax_probs = softmax(logits)`` — pre-top-k router belief
         over all experts (consumed by ``importance_loss`` and diagnostics).
      3. Select top-k expert indices and their logit values.
      4. ``topk_weights = softmax(topk_vals)`` — renormalised softmax over
         just the top-k logits, so ``Σ_j topk_weights = 1`` per sample.
         This is the standard Shazeer MoE dispatch weight.
      5. ``sparse_softmax_probs`` is the same mixture placed back into
         ``(B, E)`` with zeros off-topk.  Diagnostics only.

    Gradient flow
    -------------
    • ``k ≥ 2``: task loss flows through each ``topk_weights[b, j]`` —
      pushing the router toward the expert that best reduces task loss.
    • ``k = 1``: ``topk_weights ≡ 1`` is constant w.r.t. ``l_top`` so task
      loss cannot select between experts via the weight.  Use k ≥ 2 if
      you want task-driven specialisation.

    Args:
        feat_dim:    Dimension of the BEV summary descriptor (gate input).
        num_experts: Number of experts to route over.
        k:           Top-k experts selected per sample.
        temperature: Softmax temperature applied to both ``full_softmax_probs``
                     (consumed by ``importance_loss``) and ``topk_weights``
                     (dispatch mixing coefficients).  T > 1 keeps the dispatch
                     weights more balanced between the k selected experts,
                     maximising the softmax Jacobian term ``w₁·w₂`` that
                     carries the per-expert task-loss gradient through the gate.
                     With T=1 and a logit spread of 2 (typical after ep0), the
                     top-2 weights are ~[0.88, 0.12]; with T=2 they become
                     ~[0.73, 0.27]; with T=4 they are ~[0.62, 0.38].  A more
                     balanced second weight means more gradient reaching the
                     second expert at each step, fighting the winner-take-all
                     collapse observed in run 4542759 (E0 top-1 = 100% at ep0
                     end).  Default 1.0 for backward compatibility.
    """

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        assert temperature > 0.0, 'temperature must be positive'
        self.num_experts = num_experts
        self.k = k
        self.temperature = float(temperature)
        self.gate = nn.Linear(feat_dim, num_experts)

    def set_temperature(self, temperature: float) -> None:
        """Update the active softmax temperature (used for warmup schedules).

        Called by :class:`BEVMoEBlock` each forward pass to apply a
        linearly-decaying temperature that keeps routing high-entropy while
        the context descriptor is still weak in early epochs.
        """
        self.temperature = max(1e-6, float(temperature))

    def forward(self, feat: Tensor) -> GateOutput:
        """Route based on the BEV summary descriptor ``feat``.

        Args:
            feat: ``(B, feat_dim)`` BEV summary descriptor.

        Returns:
            :class:`GateOutput` with placeholder noise fields
            (``noise_std=None``).
        """
        logits = self.gate(feat)                                       # (B, E)

        # Guard against NaN/inf logits (can arise if gate weights are
        # corrupted by a gradient update that slipped through GradScaler
        # in a multi-GPU DDP run).  Clamping to ±30 keeps softmax
        # numerically safe while preserving relative expert ordering.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = logits.clamp(-30.0, 30.0)

        full_softmax_probs = F.softmax(
            logits / self.temperature, dim=-1)                         # (B, E)

        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)       # (B, k)
        topk_weights = F.softmax(
            topk_vals / self.temperature, dim=-1)                      # (B, k)

        sparse_softmax_probs = torch.zeros_like(logits)
        sparse_softmax_probs.scatter_(
            1, topk_idx, topk_weights.to(sparse_softmax_probs.dtype))

        return GateOutput(
            full_softmax_probs=full_softmax_probs,
            sparse_softmax_probs=sparse_softmax_probs,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            clean_logits=logits,
            noisy_logits=logits,
            noise_std=None,
            # Deterministic gate: clean top-k is identical to dispatch top-k.
            clean_topk_idx=topk_idx,
        )


# ── NoisyTopkGate ─────────────────────────────────────────────────────────

class NoisyTopkGate(nn.Module):
    """Shazeer et al. (2017) noisy top-k gate.

    Adds learned, **input-dependent** Gaussian noise to gate logits during
    training so non-dominant experts still receive gradient signal,
    improving load balance.  At inference the gate is deterministic.

    Exact formulation (paper §2.1, with our ``noise_scale`` multiplier)::

        clean_logits = z · W_gate
        noise_std    = softplus( z · W_noise + noise_epsilon )        (training)
        noisy_logits = clean_logits + noise_scale · StdNormal · noise_std
                                                                      (training)
        noisy_logits = clean_logits                                   (eval)

    The noise std is produced by its own learned linear head ``W_noise``,
    then passed through ``softplus`` so it is strictly positive and can
    smoothly shrink to ~0 as the network becomes confident.  This is the
    paper's built-in annealing mechanism.  The constant ``noise_epsilon``
    keeps the std bounded away from 0 for the Gaussian-CDF ``load_loss``.

    ``noise_scale`` is a global deterministic multiplier on the sampled
    noise.  The standard Shazeer formulation corresponds to
    ``noise_scale = 1.0``.  Values ``< 1`` reduce training-time routing
    noise relative to the clean-logit spread, which is useful when the
    network's natural per-expert ``noise_std`` grows large enough to swamp
    the clean logit gaps (monitor ``noise_std_mean / clean_logits_std``:
    should be ≲ 1).  This keeps exploration active while reducing the
    train/eval routing mismatch.

    Routing procedure
    -----------------
      1. ``clean_logits = W_gate(z)`` — no normalisation.
      2. Training: ``noise_std = softplus(W_noise(z) + noise_epsilon)``;
         ``noisy_logits = clean_logits + noise_scale · randn · noise_std``.
         Eval:    ``noisy_logits = clean_logits``; ``noise_std = None``.
      3. ``full_softmax_probs = softmax(clean_logits / T)`` — clean pre-top-k
        router belief.  This is intentionally computed from clean logits, not
        noisy logits, so ``importance_loss`` regularizes the learned
        deterministic router preference.  Training-time exploration still comes
        from selecting top-k on ``noisy_logits``, and hard-dispatch balancing is
        handled by ``load_loss``.
      4. Top-k selected on ``noisy_logits`` (rank-invariant w.r.t. T>0).
         In parallel, the deterministic clean top-k is also computed from
         ``clean_logits`` and exposed as ``GateOutput.clean_topk_idx`` so
         callers can feed :func:`switch_balance_loss` with the clean
         selection rather than the noisy dispatch.  In eval mode the two
         top-ks are identical.
      5. ``topk_weights = softmax(topk_vals / T)`` — renormalised over the
         top-k, Σ_j = 1 per sample.

    Args:
        feat_dim:      Dimension of the BEV summary descriptor.
        num_experts:   Number of experts to route over.
        k:             Top-k experts selected per sample.
        temperature:   Softmax temperature.  T < 1 sharpens, T > 1 flattens;
                       top-k selection is invariant for T > 0.  Default 1.0.
        noise_epsilon: Constant added before softplus for the noise std,
                       giving a small positive floor.  Default 1e-3 — small
                       enough to barely shift ``softplus`` output while still
                       keeping it strictly above zero for the Gaussian-CDF
                       ``load_loss``.
        noise_scale:   Global multiplier on the sampled Gaussian noise.
                       Default 1.0 (paper formulation).  Set to < 1 (e.g.
                       0.5) to reduce training noise when the learned
                       per-expert noise std is large relative to the
                       clean-logit std.
        max_noise_to_clean_ratio: Optional hard upper-bound on the per-element
                       ratio ``noise_scale * noise_std / clean_logits.std()``.
                       When the learned ``noise_std`` inflates beyond this
                       multiple of the clean-logit spread (e.g. run 4540532
                       reached a ratio of 1.4–1.8 at epoch 1), the gate
                       essentially performs random top-k selection at training
                       time while the deterministic clean router gets nearly
                       no useful gradient.  Setting this to e.g. 0.5 clamps
                       ``noise_std`` to ``0.5 * clean_logits.detach().std()
                       / noise_scale`` before sampling, preventing the failure
                       mode without disabling adaptive noise entirely.
                       ``None`` (default) disables the clamp for backward
                       compatibility.
        noise_bias_init: If not ``None``, initialise the bias of ``w_noise``
                       to this constant value.  A negative value (e.g. −2.0)
                       starts the noise std at ``softplus(−2 + ε) ≈ 0.13``
                       at zero input, which keeps early-training noise small
                       and avoids the epoch-0 spike seen in run 4540532
                       where ``noise_std_mean`` jumped from 1.44 → 2.62 in
                       the first epoch.  ``None`` (default) keeps PyTorch's
                       default uniform initialisation.
    """

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        noise_epsilon: float = 1e-3,
        noise_scale: float = 1.0,
        max_noise_to_clean_ratio: Optional[float] = None,
        noise_bias_init: Optional[float] = None,
    ):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        assert temperature > 0.0, 'temperature must be positive'
        assert noise_scale >= 0.0, 'noise_scale must be non-negative'
        if max_noise_to_clean_ratio is not None:
            assert max_noise_to_clean_ratio > 0.0, \
                'max_noise_to_clean_ratio must be positive'

        self.num_experts = num_experts
        self.k = k
        self.temperature = float(temperature)
        self.noise_epsilon = float(noise_epsilon)
        self.noise_scale = float(noise_scale)
        self.max_noise_to_clean_ratio = max_noise_to_clean_ratio

        self.w_gate  = nn.Linear(feat_dim, num_experts)
        # Shazeer's second linear head producing per-sample per-expert noise
        # pre-activation.  Same shape as ``w_gate``; bias kept so the
        # network can learn a baseline noise level independent of input.
        self.w_noise = nn.Linear(feat_dim, num_experts)

        if noise_bias_init is not None:
            nn.init.constant_(self.w_noise.bias, noise_bias_init)

    def set_temperature(self, temperature: float) -> None:
        """Update the active softmax temperature (used for warmup schedules).

        Called by :class:`BEVMoEBlock` each forward pass to apply a
        linearly-decaying temperature that keeps routing high-entropy while
        the context descriptor is still weak in early epochs.
        """
        self.temperature = max(1e-6, float(temperature))

    def forward(self, feat: Tensor) -> GateOutput:
        """Route based on the BEV summary descriptor ``feat``.

        Args:
            feat: ``(B, feat_dim)`` BEV summary descriptor.

        Returns:
            :class:`GateOutput` with ``noise_std`` populated during
            training (signalling that ``load_loss`` is computable) and
            ``None`` at eval.
        """
        clean_logits = self.w_gate(feat)                                # (B, E)

        if self.training:
            # Shazeer input-dependent noise: std = softplus(W_noise·z + ε).
            raw = self.w_noise(feat) + self.noise_epsilon              # (B, E)
            noise_std = F.softplus(raw)                                # (B, E)

            # Optional hard clamp: prevent noise_std from exceeding
            # max_noise_to_clean_ratio * clean_logits.std() / noise_scale.
            # Guards against the failure mode (run 4540532) where the optimiser
            # inflated noise_std to ~2.6 (ratio 1.4–1.8) rather than flattening
            # clean logits, rendering training-time top-k effectively random.
            if self.max_noise_to_clean_ratio is not None and self.noise_scale > 0.0:
                clean_std = clean_logits.detach().std(dim=-1, keepdim=True).clamp(min=1e-6)
                max_std = self.max_noise_to_clean_ratio * clean_std / self.noise_scale
                noise_std = noise_std.clamp(max=max_std)

            # ``noise_scale`` globally scales the sampled noise so the
            # exploration term stays comparable to clean-logit variation
            # even when the learned per-expert ``noise_std`` drifts large.
            noisy_logits = (clean_logits
                            + self.noise_scale
                            * torch.randn_like(clean_logits)
                            * noise_std)
        else:
            noise_std = None
            noisy_logits = clean_logits

        # Clean pre-top-k softmax.  We intentionally compute this from clean_logits
        # so importance_loss and dense diagnostics regularize the deterministic
        # router preference rather than a single noisy sample.  Noisy logits are
        # still used below for top-k exploration and by load_loss for expected
        # hard-dispatch balancing.
        full_softmax_probs = F.softmax(
            clean_logits / self.temperature, dim=-1)                    # (B, E)

        topk_vals, topk_idx = torch.topk(noisy_logits, k=self.k, dim=-1)  # (B, k)
        topk_weights = F.softmax(topk_vals / self.temperature, dim=-1)    # (B, k)

        # Deterministic clean top-k — what the router would pick at eval
        # time.  Consumed by ``switch_balance_loss`` and the
        # clean-routing selection-frequency diagnostics.  In eval mode
        # ``noisy_logits is clean_logits`` so the two top-k ops yield the
        # same indices.
        if self.training:
            _, clean_topk_idx = torch.topk(clean_logits, k=self.k, dim=-1)
        else:
            clean_topk_idx = topk_idx

        sparse_softmax_probs = torch.zeros_like(clean_logits)
        sparse_softmax_probs.scatter_(
            1, topk_idx, topk_weights.to(sparse_softmax_probs.dtype))

        return GateOutput(
            full_softmax_probs=full_softmax_probs,
            sparse_softmax_probs=sparse_softmax_probs,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            noise_std=noise_std,
            clean_topk_idx=clean_topk_idx,
        )

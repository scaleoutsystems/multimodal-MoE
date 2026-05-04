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

Variant D (LiDAR-only, ``BEVMoEBlock``) uses two independent branches::

    z_router   = router_summary(x_bev)            # task / routing branch
    z_ctx      = context_summary(x_bev)           # context-CE branch
    z_gate     = cat([z_router, z_ctx.detach()])  # gate sees both; no ctx grad
    gate_out   = gate(z_gate)
    ctx_logits = context_head(z_ctx)              # full grad through z_ctx

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

LayerNorm as final activation
------------------------------
All summary encoders end in ``LayerNorm`` rather than ``ReLU``.  Signed,
unit-variance descriptors let the gate's linear projection produce logits of
both signs; a final ReLU leaves dead units at 0, which combined with weight
decay on the gate prevents logit magnitudes from growing (observed as a
dead-gate failure mode in earlier runs).

Context target configuration
-----------------------------
Each MoE block takes a ``context_aux_cfg`` dict::

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


# ── BasicBEVResBlock ──────────────────────────────────────────────────────

class BasicBEVResBlock(nn.Module):
    """Two-layer residual conv block for BEV feature maps.

    Main path:
        Conv2d(in_channels → out_channels, 3×3, pad=1) → BN → ReLU
        Conv2d(out_channels → out_channels, 3×3, pad=1) → BN

    Residual path (identity when in_channels == out_channels, 1×1 conv
    + BN otherwise).

    Output: ReLU(main + residual)

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ── BEVResSummaryEncoder ──────────────────────────────────────────────────

class BEVResSummaryEncoder(nn.Module):
    """Residual-CNN BEV summary encoder producing a routing descriptor ``z``.

    Architecture
    ------------
    1. **Stem**: Conv2d(C → stem_channels, 3×3) → BN → ReLU — projects
       input channels without changing spatial resolution.

    2. **Residual blocks** (3 × :class:`BasicBEVResBlock`):
           BasicBEVResBlock(stem_channels, stem_channels)
           BasicBEVResBlock(stem_channels, stem_channels)
           BasicBEVResBlock(stem_channels, out_channels)
       The last block widens channels if ``stem_channels != out_channels``.
       Spatial structure is preserved (no pooling/striding) so all three
       blocks learn spatial patterns before global compression.

    3. **Pooling / vectorisation**:
           AdaptiveAvgPool2d(1)      → (B, out_channels, 1, 1)
           Flatten                  → (B, out_channels)
           Linear(out_channels → out_dim) → (B, out_dim)
           LayerNorm(out_dim)       → signed, unit-variance descriptor
           Dropout(dropout)         → regularisation

    Rationale: spatial reasoning first → global pooling → descriptor
    vector.  This follows the same pattern as ResNet feature extraction
    used in the CIFAR MoE reference experiment.

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
    """

    def __init__(
        self,
        channels: int,
        stem_channels: int = 128,
        out_channels: int = 256,
        out_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, stem_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            BasicBEVResBlock(stem_channels, stem_channels),
            BasicBEVResBlock(stem_channels, stem_channels),
            BasicBEVResBlock(stem_channels, out_channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
        )

        self.out_dim = out_dim
        self.stem_channels = stem_channels
        self.out_channels = out_channels
        self.num_res_blocks = 3
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Encode a BEV feature map into a fixed-size descriptor.

        Args:
            x: BEV feature map ``(B, C, H, W)``.

        Returns:
            Routing/context descriptor ``(B, out_dim)``.
        """
        x = self.stem(x)         # (B, stem_channels, H, W)
        x = self.res_blocks(x)   # (B, out_channels, H, W)
        x = self.pool(x)         # (B, out_channels, 1, 1)
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
    """

    def __init__(self, feat_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(feat_dim, num_experts)

    def forward(self, feat: Tensor) -> GateOutput:
        """Route based on the BEV summary descriptor ``feat``.

        Args:
            feat: ``(B, feat_dim)`` BEV summary descriptor.

        Returns:
            :class:`GateOutput` with placeholder noise fields
            (``noise_std=None``).
        """
        logits = self.gate(feat)                                       # (B, E)

        full_softmax_probs = torch.softmax(logits, dim=-1)             # (B, E)

        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)       # (B, k)
        topk_weights = F.softmax(topk_vals, dim=-1)                    # (B, k)

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
    """

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        noise_epsilon: float = 1e-3,
        noise_scale: float = 1.0,
    ):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        assert temperature > 0.0, 'temperature must be positive'
        assert noise_scale >= 0.0, 'noise_scale must be non-negative'

        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.noise_epsilon = float(noise_epsilon)
        self.noise_scale = float(noise_scale)

        self.w_gate  = nn.Linear(feat_dim, num_experts)
        # Shazeer's second linear head producing per-sample per-expert noise
        # pre-activation.  Same shape as ``w_gate``; bias kept so the
        # network can learn a baseline noise level independent of input.
        self.w_noise = nn.Linear(feat_dim, num_experts)

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

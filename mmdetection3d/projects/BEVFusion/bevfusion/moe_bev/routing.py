"""Routing modules: BEVSummaryHead, TopkGate, NoisyTopkGate, context utils.

Context-supervised routing — the design in this module
======================================================
Context labels (road_type, weather_group, …) are no longer concatenated into
the router input.  ``ContextEncoder`` has been removed.  The router sees only
the learned BEV summary descriptor ``z = BEVSummaryHead(x)``.

For every MoE variant the pattern is::

    z          = BEVSummaryHead(BEV_features)        # (B, out_dim)
    gate_out   = gate(z)                             # router logits + dispatch
    ctx_logits = context_head(z)                     # (B, num_context_classes)
    ctx_loss   = F.cross_entropy(ctx_logits, ctx_label)
    z_loss     = router_z_loss(gate_out.clean_logits, coef_z)

The auxiliary loss the block emits is the sum of:

    importance_loss + load_loss + ctx_loss_coef · ctx_loss + z_loss
    (+ group_balance_loss for ModalitySpecificMoEBlock)

Why no metadata in the router input
-----------------------------------
Concatenating an embedded context vector into the router input "leaks" the
metadata into expert dispatch — the gate can short-circuit context-aware
specialisation by reading the label directly.  That makes it impossible to
tell whether the BEV features themselves can support context-aware routing.

By contrast, the ``context_head`` is supervised by the same labels but only
*shapes* the descriptor ``z``: gradients flow back through the summary head,
encouraging it to organise BEV features along context-relevant directions.
Expert dispatch remains task-driven (top-k over learned router logits), so
the model can still pick experts based on local BEV evidence — but its
internal representation is biased toward the context structure.

The ``context_head`` is **not** the gate.  Its output ``ctx_logits`` is used
only for the auxiliary CE loss; it never affects dispatch and is not used
by ``router_z_loss``.

GateOutput contract
-------------------
All gate modules return a ``GateOutput`` with the following fields:

    full_softmax_probs   (B, E)  — softmax over clean_logits: the clean router
                               belief over the full expert pool, before
                               noise and before top-k. Consumed by
                               ``importance_loss`` and the
                               ``dense_mean_prob_per_expert`` diagnostic.
                               This intentionally regularizes the learned
                               deterministic router preference, while
                               noisy_logits are used for training-time
                               top-k exploration and load_loss.

    sparse_softmax_probs (B, E)  — top-k mixture (renormalised over the
                                   selected experts) scattered into the
                                   full (B, E) shape with zeros off-topk.
                                   Diagnostics only.

    topk_idx             (B, k)  — selected expert indices.

    topk_weights         (B, k)  — Shazeer top-k mixture weights:
                                   ``softmax(topk_vals / T)`` with
                                   ``Σ_j topk_weights = 1`` per sample.
                                   Consumed directly by dispatch.

    clean_logits         (B, E)  — pre-noise gate logits.  Consumed by
                                   ``router_z_loss`` and ``load_loss``.

    noisy_logits         (B, E)  — logits used for the actual top-k.
                                   Equals ``clean_logits`` for ``TopkGate``
                                   and for ``NoisyTopkGate`` in eval mode;
                                   for ``NoisyTopkGate`` in training these
                                   are ``clean_logits + randn · noise_std``.

    noise_std            (B, E) or None
                                 — per-sample per-expert noise std used at
                                   this forward pass.  ``None`` for
                                   deterministic gates / eval forwards;
                                   signals to ``load_loss`` that no
                                   Gaussian-CDF integral can be computed.

Top-k routing
-------------
Default ``k = 2`` is preserved deliberately:

    • Top-1 makes expert assignment very brittle.
    • Top-2 enables cooperative / compositional specialisation.
    • For ModalitySpecificMoEBlock, top-2 is the only way the router can
      pick mixed (LiDAR + Camera) combinations as well as same-modality
      pairs.

Do **not** switch the main design to top-1 "context-class" routing, and do
**not** force ``num_experts == num_context_classes``.  Expert dispatch is
task-driven; context supervision shapes ``z`` but does not hard-code any
expert/context mapping.

Final activation of BEVSummaryHead
----------------------------------
The summary head ends in LayerNorm rather than ReLU so the router descriptor
is signed and unit-variance.  Signed inputs let the gate's linear projection
produce expert logits of both signs and let logit magnitudes grow over
training; a final ReLU instead leaves "dead" descriptor units stuck at 0
(which combined with weight decay on the gate prevents logit magnitudes from
growing — observed as the dead-gate failure mode in earlier runs).

Selecting context targets per run
---------------------------------
Each MoE block takes a single ``context_aux_cfg`` of the form::

    context_aux_cfg = dict(
        target_field='road_type',
        loss_coef=0.05,
        label_smoothing=0.0,
    )

``target_field`` must be a key of ``ZOD_FIELD_REGISTRY`` (the single source
of truth for vocabularies).  Use :func:`extract_context_labels` to convert
``batch_input_metas`` into the integer ``LongTensor`` consumed by the
context CE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── BEVSummaryHead ────────────────────────────────────────────────────────

class BEVSummaryHead(nn.Module):
    """Spatial-aware summary head producing a routing descriptor ``z``.

    Pipeline
    --------
    Input ``x`` of shape ``(B, C, H, W)`` is summarised in three stages:

    1. **Coarse spatial pooling**: ``AdaptiveAvgPool2d(P)`` and
       ``AdaptiveMaxPool2d(P)`` produce two ``(B, C, P, P)`` maps that
       split the BEV into a coarse near/far × left/right grid.  Both the
       average and the peak per cell are kept so the router sees both
       "is this region typically active?" and "is there a sharp peak
       anywhere in this region?".  The two are concatenated channel-wise
       to give ``(B, 2C, P, P)``.

    2. **Spatial mixer**: a tiny 1×1 + 3×3 conv stack mixes information
       both within and across pooled cells:

           Conv2d(2C → spatial_dim, k=1) → ReLU
           Conv2d(spatial_dim → spatial_dim, k=3, padding=1) → ReLU

       The 1×1 conv mixes channels per cell; the 3×3 conv mixes
       neighbouring pooled BEV regions so the descriptor encodes
       relational structure (e.g. "dense in near-range cells, sparse in
        far-range cells", or left/right asymmetry) rather than independent
        per-cell summaries.

    3. **Descriptor MLP**: flatten and project::

           Linear(spatial_dim · P · P → hidden_dim) → ReLU
           Linear(hidden_dim → out_dim) → LayerNorm(out_dim)

    The final LayerNorm gives a signed, unit-variance routing descriptor
    suitable for a downstream linear gate (see module docstring for why a
    final ReLU is harmful).

    The MLP here is **not** the gate.  It produces ``z``; the gate is the
    later linear map ``W_gate · z`` (and the context head is ``W_ctx · z``)
    inside the MoE block.

    Args:
        channels:    Number of input BEV channels.
        pool_size:   Pooling grid resolution P (default 4 → 4×4 grid).
        spatial_dim: Number of channels in the spatial-mixer convs
            (default 128).
        hidden_dim:  Hidden width of the descriptor MLP (default 256).
        out_dim:     Final descriptor dimension; the gate's input dim
            (default 128).  Exposed as ``self.out_dim``.
    """

    def __init__(
        self,
        channels: int,
        pool_size: int = 4,
        spatial_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)  # (B, C, P, P)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)  # (B, C, P, P)

        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(2 * channels, spatial_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        flat_dim = spatial_dim * pool_size * pool_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Summarise a BEV feature map into a routing descriptor.

        Args:
            x: BEV feature map ``(B, C, H, W)``.

        Returns:
            Routing descriptor ``(B, out_dim)``.
        """
        avg = self.avg_pool(x)              # (B, C, P, P)
        mx  = self.max_pool(x)              # (B, C, P, P)
        feat = torch.cat([avg, mx], dim=1)  # (B, 2C, P, P)
        feat = self.spatial_mixer(feat)     # (B, spatial_dim, P, P)
        feat = feat.flatten(1)              # (B, spatial_dim · P · P)
        return self.mlp(feat)               # (B, out_dim)


# ── Gate output container ─────────────────────────────────────────────────

@dataclass
class GateOutput:
    full_softmax_probs:   Tensor            # (B, E) softmax over clean_logits
    sparse_softmax_probs: Tensor            # (B, E) top-k mixture scattered
                                            #         into (B, E); zero off-topk
    topk_idx:             Tensor            # (B, k) selected expert indices
    topk_weights:         Tensor            # (B, k) Shazeer top-k mixture
                                            #         weights, Σ_j = 1 per sample
    clean_logits:         Tensor            # (B, E) pre-noise gate logits
    noisy_logits:         Tensor            # (B, E) logits used for top-k
    noise_std:            Optional[Tensor]  # (B, E) or None — see GateOutput
                                            #         contract in the module
                                            #         docstring.


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
        )


# ── NoisyTopkGate ─────────────────────────────────────────────────────────

class NoisyTopkGate(nn.Module):
    """Shazeer et al. (2017) noisy top-k gate.

    Adds learned, **input-dependent** Gaussian noise to gate logits during
    training so non-dominant experts still receive gradient signal,
    improving load balance.  At inference the gate is deterministic.

    Exact formulation (paper §2.1)::

        clean_logits = z · W_gate
        noise_std    = softplus( z · W_noise + noise_epsilon )        (training)
        noisy_logits = clean_logits + StandardNormal() · noise_std    (training)
        noisy_logits = clean_logits                                   (eval)

    The noise std is produced by its own learned linear head ``W_noise``,
    then passed through ``softplus`` so it is strictly positive and can
    smoothly shrink to ~0 as the network becomes confident.  This is the
    paper's built-in annealing mechanism.  The constant ``noise_epsilon``
    keeps the std bounded away from 0 for the Gaussian-CDF ``load_loss``.

    Routing procedure
    -----------------
      1. ``clean_logits = W_gate(z)`` — no normalisation.
      2. Training: ``noise_std = softplus(W_noise(z) + noise_epsilon)``;
         ``noisy_logits = clean_logits + randn · noise_std``.
         Eval:    ``noisy_logits = clean_logits``; ``noise_std = None``.
      3. ``full_softmax_probs = softmax(clean_logits / T)`` — clean pre-top-k
        router belief.  This is intentionally computed from clean logits, not
        noisy logits, so ``importance_loss`` regularizes the learned
        deterministic router preference.  Training-time exploration still comes
        from selecting top-k on ``noisy_logits``, and hard-dispatch balancing is
        handled by ``load_loss``.
      4. Top-k selected on ``noisy_logits`` (rank-invariant w.r.t. T>0).
      5. ``topk_weights = softmax(topk_vals / T)`` — renormalised over the
         top-k, Σ_j = 1 per sample.

    Args:
        feat_dim:      Dimension of the BEV summary descriptor.
        num_experts:   Number of experts to route over.
        k:             Top-k experts selected per sample.
        temperature:   Softmax temperature.  T < 1 sharpens, T > 1 flattens;
                       top-k selection is invariant for T > 0.  Default 1.0.
        noise_epsilon: Constant added before softplus for the noise std,
                       giving a small positive floor.  Default 1e-2.
    """

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        k: int = 2,
        temperature: float = 1.0,
        noise_epsilon: float = 1e-2,
    ):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        assert temperature > 0.0, 'temperature must be positive'

        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.noise_epsilon = noise_epsilon

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
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
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
        )

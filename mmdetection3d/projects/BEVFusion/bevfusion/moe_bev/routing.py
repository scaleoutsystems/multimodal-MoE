"""Routing modules: BEVSummaryHead, ContextEncoder, TopkGate, NoisyTopkGate.

Overview
--------
ContextEncoder converts per-sample metadata dicts (weather, road type, …) into
a fixed-size context vector that is concatenated to the BEV summary descriptor
before the gate.

TopkGate (and the noisy variant NoisyTopkGate) selects the top-k experts from
the resulting descriptor and returns a GateOutput dataclass.

GateOutput contract
-------------------
All gate modules return a ``GateOutput`` with the following fields:

    full_softmax_probs   (B, E)  — softmax(logits) computed over ALL experts
                                   (no masking).  This is the router's "belief"
                                   over the full pool and is used for:
                                     • importance_loss (CV² of per-expert mass)
                                     • dense_mean_prob_per_expert diagnostics

    sparse_softmax_probs (B, E)  — the renormalised top-k mixture laid back
                                   into ``(B, E)`` with zeros off-topk, so the
                                   non-zero entries on each row sum to 1.
                                   Diagnostics/analysis only — dispatch uses
                                   ``topk_weights`` directly.

    topk_idx             (B, k)  — indices of the k selected experts per sample.
                                   Used for selection-frequency diagnostics.

    topk_weights         (B, k)  — Shazeer-style dispatch weights:
                                       topk_weights = softmax(topk_vals / T)
                                   i.e. the softmax of just the top-k logits.
                                   Σ_j topk_weights[b, j] = 1 per sample.  This
                                   is the canonical Shazeer et al. (2017) MoE
                                   gate output and what every downstream block
                                   dispatches through.

    clean_logits         (B, E)  — pre-noise gate logits (identical to the
                                   post-noise logits for deterministic gates).
                                   Consumed by ``load_loss`` to compute Shazeer
                                   et al.'s Gaussian-CDF load estimator.

    noisy_logits         (B, E)  — logits actually used for top-k selection.
                                   For TopkGate this equals ``clean_logits``;
                                   for NoisyTopkGate these are the noisy logits
                                   (clean + randn · noise_std) during training,
                                   and equal ``clean_logits`` at eval.

    noise_std            (B, E) or None
                                 — per-sample per-expert noise std used at this
                                   forward pass.  ``None`` for deterministic
                                   gates (TopkGate) or for NoisyTopkGate in eval
                                   mode, signalling that ``load_loss`` cannot
                                   be computed.

Why renormalised top-k softmax (standard Shazeer) rather than gathered
full-softmax values
----------------------------------------------------------------------
Earlier revisions used "Switch-style" dispatch weights ``topk_weights =
full_softmax_probs.gather(dim=1, index=topk_idx)``: the router's confidence
at each selected expert, NOT renormalised.  That form has an attractive
property at k=1 (a live gradient through the softmax even when only one
expert is active), but a bad property under residual-delta dispatch:

    Σ_j topk_weights ∈ [k/E,  1]      (scales with router peakiness)

so the effective magnitude of the expert residual ``g · Σ w · Δ`` *grows*
as the router sharpens.  Early training gets weak expert signal, late
training gets disproportionately strong signal; stability varies over a
single run.  A ``residual_gain = num_experts`` kludge was required just to
compensate the early-training deficit, and it had to track E.

Renormalised top-k softmax (this implementation) computes

    topk_weights = softmax(topk_vals / T)      # over k values only
    Σ_j topk_weights = 1   always

giving a constant-magnitude mixture at any router peakiness.  ``residual_gain``
reverts to a plain scalar in (0, 1] — unit by default — with no dependence
on ``num_experts``.

Gradient flow with renormalised weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    • k ≥ 2: task loss flows through each ``topk_weights[b, j] =
      σ(l_j)/(σ(l_1)+…+σ(l_k))``, pushing the router toward the expert that
      best reduces the task loss for each sample.  Full specialisation.
    • k = 1: ``topk_weights ≡ 1`` is constant, so task loss cannot select
      between experts via the weight.  The gate is still trained via
      ``importance_loss`` (full-softmax CV²) and ``load_loss`` (Gaussian-CDF
      CV²), which together provide balance pressure but not task-specific
      specialisation through dispatch.  Use k ≥ 2 if task-driven
      specialisation is desired.

All BEV MoE blocks assume Σ w = 1 and therefore take the renormalised
weights directly; no block-local renormalisation is needed anymore.

BEVSummaryHead final activation
--------------------------------
The final activation of BEVSummaryHead was changed from ReLU to LayerNorm.
A final ReLU forced non-negative, sparse routing descriptors which, combined
with weight decay, caused logit magnitudes to stay near zero ("dead gate").
LayerNorm produces a signed, unit-variance descriptor so gate logits can grow
and carry meaningful preference signals.

Selecting context fields per run
---------------------------------
Pass a ``fields`` list to ``ContextEncoder`` with just the field names you
want active.  Vocabs are looked up from ``ZOD_FIELD_REGISTRY`` so you never
need to repeat them in configs.  All registered fields are categorical.

Example config snippet::

    context_cfg = dict(
        fields=['weather_group', 'road_type', 'complexity_bin'],
        embed_dim=16,
        out_dim=64,
    )

To use ALL available fields, omit ``fields`` (or pass ``fields=None``).
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
    """Lightweight spatial summary head that feeds the router gate.

    Motivation
    ----------
    Plain global average pooling (GAP) collapses all spatial information
    to a single C-dimensional vector.  For routing, this loses whether the
    scene has interesting objects in the front/back/left/right quadrants —
    information that is directly relevant to which expert should process
    the BEV map.

    This head retains a coarse (P×P) spatial structure by computing both
    average-pool and max-pool at that resolution (preserving both the
    "typical" and "peak" activation per region), concatenating them, and
    projecting through a tiny MLP to a compact routing descriptor.

    Shape trace (default: pool_size=2, hidden_dim=128, out_dim=64)
    --------------------------------------------------------------
    Input   : (B, C, H, W)
    avg_pool: (B, C, 2, 2)   — average activation per 2×2 quadrant
    max_pool: (B, C, 2, 2)   — peak activation per 2×2 quadrant
    cat     : (B, 2C, 2, 2)  — avg and max side-by-side per channel
    flatten : (B, 8C)        — 8C when pool_size=2: 2 * C * 2 * 2
    Linear  : (B, 128)       — ReLU
    Linear  : (B, 64)
    LayerNorm: (B, 64)       — signed, unit-variance routing descriptor

    Final LayerNorm (no final ReLU) rationale
    -----------------------------------------
    A router descriptor fed into the gate must be **signed** so that the
    gate's linear projection can produce expert logits of both signs, and
    so that rich-get-richer dynamics can actually develop magnitude.
    A final ReLU makes the descriptor non-negative, creates dead units
    (gate input dims stuck at 0), and — combined with weight decay on the
    gate — prevents logit magnitudes from growing, leaving the softmax
    distribution near-uniform even when hard top-1 selections are skewed.
    LayerNorm keeps the descriptor bounded (stable) while remaining signed.

    Args:
        channels:   Number of input BEV feature channels.
        pool_size:  Spatial resolution of the pooled grid (default 2 → 2×2).
        hidden_dim: Width of the intermediate MLP layer (default 128).
        out_dim:    Dimension of the output routing descriptor (default 64).
    """

    def __init__(
        self,
        channels: int,
        pool_size: int = 2,
        hidden_dim: int = 128,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)  # (B, C, P, P)
        self.max_pool = nn.AdaptiveMaxPool2d(pool_size)  # (B, C, P, P)

        # After concat(avg, max) and flattening: 2 * C * pool_size^2 dims.
        flat_dim = 2 * channels * pool_size * pool_size

        # Two-layer MLP: flat_dim → hidden_dim → out_dim, followed by
        # LayerNorm (see class docstring).  Intentionally small — the
        # summary head should be easy to explain, not a heavy learnable
        # module in its own right.
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Summarise a BEV feature map into a compact routing descriptor.

        Args:
            x: BEV feature map (B, C, H, W).

        Returns:
            Routing descriptor (B, out_dim).
        """
        avg = self.avg_pool(x)              # (B, C, P, P)
        mx  = self.max_pool(x)              # (B, C, P, P)
        feat = torch.cat([avg, mx], dim=1)  # (B, 2C, P, P)
        feat = feat.flatten(1)              # (B, 2C·P²) = (B, 8C) when P=2
        return self.mlp(feat)               # (B, out_dim)


# ── Gate output container ─────────────────────────────────────────────────

@dataclass
class GateOutput:
    full_softmax_probs:   Tensor            # (B, E) pre-top-k softmax — router belief over all experts
    sparse_softmax_probs: Tensor            # (B, E) top-k mixture scattered into (B, E); zero off-topk
    topk_idx:             Tensor            # (B, k) selected expert indices
    topk_weights:         Tensor            # (B, k) Shazeer top-k mixture weights:
                                            #         softmax(topk_vals / T), Σ_j=1 per sample.
                                            #         Consumed directly by dispatch.
    clean_logits:         Tensor            # (B, E) pre-noise gate logits (consumed by load_loss)
    noisy_logits:         Tensor            # (B, E) logits actually fed to top-k
    noise_std:            Optional[Tensor]  # (B, E) noise std used this step,
                                            #         or None for deterministic gates
                                            #         / eval forward passes


# ── ZOD field registry ────────────────────────────────────────────────────
# Single source of truth for field types and categorical vocabularies.
# Vocabs contain exactly the values present in the ZOD parquet + infos .pkl files (from zod_nuscenes dataset)

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

# Ordered list of all field names, preserved for deterministic default order.
# [complexity_bin, road_condition, road_type, scraped_weather, solar_context_bin, weather_group]
ALL_FIELDS: List[str] = list(ZOD_FIELD_REGISTRY.keys())


# ── ContextEncoder ────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """Encode per-sample ZOD context metadata into a fixed-size vector.

    Field types and vocabularies are looked up from ``ZOD_FIELD_REGISTRY``,
    so you only need to name the fields you want — no vocab duplication in
    configs.

    Args:
        fields: List of field names to encode.  Must all exist in
            ``ZOD_FIELD_REGISTRY``.  Pass ``None`` to use all registered
            fields.  Order determines the concatenation order (no effect on
            semantics, but keep it stable across runs for reproducibility).
        embed_dim: Embedding dimension for each categorical field.
        out_dim: Dimension of the output context vector.

    Example config:

        context_cfg = dict(
            fields=['weather_group', 'road_type'],
            embed_dim=16,
            out_dim=64,
        )
    """

    def __init__(self,
                 fields: Optional[List[str]] = None,
                 embed_dim: int = 16,
                 out_dim: int = 64):
        super().__init__()

        active_fields = fields if fields is not None else ALL_FIELDS
        unknown = [f for f in active_fields if f not in ZOD_FIELD_REGISTRY]
        if unknown:
            raise ValueError(
                f'ContextEncoder: unknown field(s) {unknown}. '
                f'Available: {ALL_FIELDS}')

        self.field_names: List[str] = list(active_fields)
        self.embeddings = nn.ModuleDict()
        self.vocab_maps: Dict[str, Dict[str, int]] = {}

        for name in active_fields:
            vocab = ZOD_FIELD_REGISTRY[name]
            # vocab_maps: dictionary (words → indices)
            self.vocab_maps[name] = {v: i for i, v in enumerate(vocab)}
            #    self.vocab_maps["weather_group"] = {'clear_like': 0,
            #    'cloud_like': 1,
            #    'fog': 2,
            #    'precipitation': 3,
            #    'wind': 4}
            
            # self.embeddings["weather_group"] = nn.Embedding(5, 16) --> 5x16 learned embedding lookup table
            # embedding (numbers → vectors) in a lookup table (vocab size x embed_dim)
            # ex: vocab = vocab = ['clear_like', 'cloud_like', 'fog', 'precipitation', 'wind']
            # embed_dim = 16 --> embedding lookup table: 5x16
            # essentially, we are mapping each word to a vector of length embed_dim
            # these vectors are learned during training.
            self.embeddings[name] = nn.Embedding(len(vocab), embed_dim)
        # All fields are categorical, each category contributing embed_dim floats.
        # so if weather_group = "fog" --> index = 2 --> embedding = self.embeddings["weather_group"](2) --> 5x16 lookup table --> 16 floats
        # total dimension after concatenation = N * embed_dim
        raw_dim = len(active_fields) * embed_dim
        # Projection network:
        # takes the concatenated raw context vector and transforms it
        # into the final context representation of size out_dim = 64
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def _embed_field(self, batch_ctx: List[dict], name: str,
                     device: torch.device) -> Tensor:
        """Look up the embedding for one categorical field across the batch.
        example: 
        batch_ctx = [
        {"weather_group": "fog", "road_type": "city"},
        {"weather_group": "clear_like", "road_type": "highway"}]

        vmap = self.vocab_maps["weather_group"] = {
            'clear_like': 0,
            'cloud_like': 1,
            'fog': 2,
            'precipitation': 3,
            'wind': 4}
        """
        vmap = self.vocab_maps[name]
        indices = []
        for ctx in batch_ctx:
            # example: val = "fog"
            val = str(ctx.get(name, ''))
            if val not in vmap:
                raise KeyError(
                    f"ContextEncoder: unexpected value '{val}' for field "
                    f"'{name}'. Known values: {list(vmap)}")
            indices.append(vmap[val])
        idx_t = torch.tensor(indices, dtype=torch.long, device=device) # (B, 1) indices of the embedding for each sample
        return self.embeddings[name](idx_t)  # (B, embed_dim)

    def forward(self, batch_input_metas: List[dict]) -> Tensor:
        """Encode context for a batch.
        workflow: 
        - pull out the context dictionary from each sample in the batch
        - embed each categorical field separately
        - concatenate the embeddings
        - project the concatenated embeddings to the final context representation of size out_dim = 64
        batch_input_metas --> batch_ctx --> _embed_field for each field (B, embed_dim)
        --> concatenate (B, num_active_fields * embed_dim) 
        --> project (linear + relu) --> (B, out_dim) final context vectors
        Args:
            batch_input_metas: List of length B.  Each dict must contain
                a ``'context'`` key mapping to the per-sample metadata dict
                (added via ``Pack3DDetInputs`` ``meta_keys``).

        Returns:
            ``(B, out_dim)`` context vector.
        """
        # pull out the context dictionary from each sample in the batch
        batch_ctx = [m.get('context', {}) for m in batch_input_metas]
        device = next(self.parameters()).device

        # Embed each categorical field → (B, embed_dim), concat all, project.
        parts = [self._embed_field(batch_ctx, name, device)
                 for name in self.field_names]
        #example for fields = ['weather_group', 'road_type'], B: 2, embed_dim: 3
        #parts = [
        #  [-0.4, 1.0, 0.1],      "fog"
        #  [ 0.2,-0.1, 0.5],      "city"
        #],

        #[
        #  [ 0.6,-0.2, 0.3],     "highway"
        #   [-0.5, 0.7,-0.1],    "arterial-rural"
        #  ]
        #]

        # --> concatenate per sample (B, 2 * embed_dim) --> (B, 6)
        # [
        #[-0.4, 1.0, 0.1,  0.6, -0.2, 0.3], "fog" + "highway"
        #[ 0.2,-0.1, 0.5, -0.5,  0.7,-0.1], "city" + "arterial-rural"
        #] 
        # --> (B, 6) representation of the context for each sample
        # --> project (linear + relu) --> (B, out_dim) final context vectors

        return self.proj(torch.cat(parts, dim=1))


# ── TopkGate ──────────────────────────────────────────────────────────────

class TopkGate(nn.Module):
    """Deterministic top-k gate over experts (Shazeer-style dispatch weights).

    Routing procedure:
      1. ``logits = W_gate(x)``.
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
    • ``k ≥ 2``: task loss flows through each ``topk_weights[b, j]`` via the
      ratio ``σ(l_j)/Σ_i σ(l_i)`` over the top-k — the router is pushed
      toward the expert that best reduces the task loss for each sample.
    • ``k = 1``: ``topk_weights ≡ 1`` is constant w.r.t. ``l_top`` so task
      loss cannot select between experts via the weight.  Use k ≥ 2 if you
      want task-driven specialisation.  The aux losses still train the gate.

    Args:
        feat_dim: Dimension of the pooled BEV feature vector.
        num_experts: Number of experts to route over.
        k: Number of experts selected per sample.
        context_dim: If > 0, context vector is concatenated with the
            feature vector before the gate linear layer.
    """

    def __init__(self, feat_dim: int, num_experts: int, k: int = 1,
                 context_dim: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(feat_dim + context_dim, num_experts)

    def forward(self, feat: Tensor,
                ctx: Optional[Tensor] = None) -> GateOutput:
        """Route based on feature (+ optional context) vector.

        Args:
            feat: ``(B, feat_dim)`` pooled BEV features.
            ctx:  ``(B, context_dim)`` optional context vector.

        Returns:
            :class:`GateOutput` with ``full_softmax_probs``,
            ``sparse_softmax_probs``, ``topk_idx``, ``topk_weights``
            (Σ_j weights = 1 per sample), and placeholder noise fields.
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)

        logits = self.gate(feat)                                       # (B, E)

        # Pre-top-k softmax — consumed by importance_loss and diagnostics.
        full_softmax_probs = torch.softmax(logits, dim=-1)             # (B, E)

        # Select top-k and compute the renormalised mixture over them.
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)       # (B, k)
        topk_weights = F.softmax(topk_vals, dim=-1)                    # (B, k), sum=1

        # Diagnostic sparse view: the same mixture laid back into (B, E).
        sparse_softmax_probs = torch.zeros_like(logits)
        sparse_softmax_probs.scatter_(1, topk_idx, topk_weights)

        # TopkGate is deterministic — no noise, so Shazeer-style load_loss
        # cannot be computed.  Emit clean==noisy logits and noise_std=None.
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

    Adds learned, input-dependent Gaussian noise to gate logits during
    training so that non-dominant experts still receive gradient signal,
    improving load balance.  At inference the gate is deterministic.

    Crucially, logits are **normalised per sample** before noise is added.
    Without this, a single large logit dominates regardless of noise magnitude,
    making the noise ineffective at redistributing expert selection.  This
    normalisation is not in the original paper but is essential in practice.

    Routing procedure:
      1. ``clean_logits = W_gate(x)`` then z-score normalise per sample.
      2. Training: ``noisy_logits = clean_logits + ε · softplus(W_noise(x)) + noise_floor``.
         Eval:    ``noisy_logits = clean_logits``.
      3. ``full_softmax_probs = softmax(noisy_logits / T)`` — pre-top-k
         router belief (consumed by ``importance_loss``).
      4. Top-k selected on ``noisy_logits`` (rank-invariant w.r.t. T>0).
      5. ``topk_weights = softmax(topk_vals / T)`` — standard Shazeer MoE
         dispatch weights renormalised over the top-k, Σ_j = 1 per sample.
      6. ``sparse_softmax_probs`` is the same mixture placed into ``(B, E)``
         with zeros off-topk (diagnostics only — blocks dispatch through
         ``topk_weights`` directly).

    Args:
        feat_dim: Dimension of the pooled BEV feature vector.
        num_experts: Number of experts to route over.
        k: Number of experts selected per sample.
        context_dim: If > 0, context vector is concatenated with the feature
            vector before the gate linear layers (same as :class:`TopkGate`).
        temperature: Softmax temperature applied to BOTH the full (dispatch /
            importance_loss) and the sparse (diagnostic) softmaxes.  Values
            < 1 sharpen the distribution and amplify the effective noise
            impact on probabilities (since noise is added pre-scaling);
            values > 1 flatten it.  Top-k selection is unaffected (T>0
            preserves rank).
        noise_floor: Minimum noise std added on top of the learned component.
            Prevents the network from collapsing noise to zero.  Default: 0.3.
        input_dropout: Dropout probability applied to the gate input.
        logit_dropout: Dropout probability applied to the noisy logits.
    """

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        k: int = 1,
        context_dim: int = 0,
        temperature: float = 1.0,
        noise_floor: float = 0.3,
        input_dropout: float = 0.0,
        logit_dropout: float = 0.0,
    ):
        super().__init__()
        assert 1 <= k <= num_experts, f'k must be in [1, num_experts], got {k}'
        assert temperature > 0.0, 'temperature must be positive'

        self.num_experts = num_experts
        self.k = k
        self.temperature = temperature
        self.noise_floor = noise_floor

        in_dim = feat_dim + context_dim
        self.w_gate  = nn.Linear(in_dim, num_experts)
        self.w_noise = nn.Linear(in_dim, num_experts)

        self.in_drop  = nn.Dropout(p=input_dropout)  if input_dropout  > 0 else None
        self.log_drop = nn.Dropout(p=logit_dropout)  if logit_dropout  > 0 else None

    def forward(self, feat: Tensor,
                ctx: Optional[Tensor] = None) -> GateOutput:
        """Route based on feature (+ optional context) vector.

        Args:
            feat: ``(B, feat_dim)`` pooled BEV features.
            ctx:  ``(B, context_dim)`` optional context vector.

        Returns:
            :class:`GateOutput` with ``full_softmax_probs`` ``(B, E)``,
            ``sparse_softmax_probs`` ``(B, E)``, ``topk_idx`` ``(B, k)``,
            and ``topk_weights`` ``(B, k)``.



        NOTE: z-score caps softmax peak (logits always std=1 → softmax max ≈ 0.3–0.5 for E=6).
         The router can never fully "commit" to one expert by growing logit magnitude. 
         This is a feature, not a bug — it's exactly what prevented the weight-decay 
         collapse described in the BEVSummaryHead docstring. Dispatch magnitude is 
         instead controlled by residual_gain=num_experts, which compensates for the ~1/E weights.
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)  # (B, feat_dim + context_dim)

        h = self.in_drop(feat) if self.in_drop is not None else feat

        # Clean logits H_clean(x) = W_gate · h, then z-score normalised per
        # sample so all experts start on the same scale (see class docstring).
        # These are the "mean" of the noise distribution used in the Shazeer
        # load_loss Gaussian CDF — do NOT add noise here.
        clean_logits = self.w_gate(h)                                        # (B, E)
        clean_logits = (clean_logits - clean_logits.mean(dim=-1, keepdim=True)) / \
                       (clean_logits.std(dim=-1, keepdim=True) + 1e-5)

        if self.training:
            # Learned noise std (strictly positive) plus a constant noise floor
            # so the network cannot suppress exploration entirely.
            noise_std = F.softplus(self.w_noise(h)) + self.noise_floor       # (B, E)
            noisy_logits = clean_logits + torch.randn_like(noise_std) * noise_std
        else:
            # Eval path: no noise added, so Shazeer load_loss cannot be
            # computed (signalled to callers by noise_std=None).
            noise_std = None
            noisy_logits = clean_logits

        if self.log_drop is not None:
            noisy_logits = self.log_drop(noisy_logits)

        # Temperature controls softmax sharpness consistently for
        # full_softmax_probs (consumed by importance_loss + diagnostics) and
        # topk_weights (consumed by dispatch).  T < 1 sharpens, T > 1
        # flattens.  Top-k selection is unaffected (rank-invariant for T>0).
        # Because T is applied AFTER noise addition, the effective noise
        # magnitude in the softmax is noise_std / T — lowering T therefore
        # also amplifies the impact of noise on probabilities.

        # Pre-top-k softmax — consumed by importance_loss and diagnostics.
        full_softmax_probs = F.softmax(noisy_logits / self.temperature, dim=-1)  # (B, E)

        # Select top-k from the noisy logits and compute the renormalised
        # mixture over those k values only — standard Shazeer MoE dispatch.
        # Σ_j topk_weights = 1 per sample, regardless of router peakiness.
        topk_vals, topk_idx = torch.topk(noisy_logits, k=self.k, dim=-1)     # (B, k)
        topk_weights = F.softmax(topk_vals / self.temperature, dim=-1)       # (B, k), sum=1

        # Diagnostic sparse view: the same mixture laid back into (B, E).
        sparse_softmax_probs = torch.zeros_like(noisy_logits)
        sparse_softmax_probs.scatter_(1, topk_idx, topk_weights)

        return GateOutput(
            full_softmax_probs=full_softmax_probs,
            sparse_softmax_probs=sparse_softmax_probs,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            clean_logits=clean_logits,
            noisy_logits=noisy_logits,
            noise_std=noise_std,
        )

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
All gate modules return a ``GateOutput`` with four fields:

    full_softmax_probs   (B, E)  — softmax(logits) computed BEFORE top-k
                                   masking.  This is the router's "belief"
                                   over all experts and is used for:
                                     • switch_balance_loss (P_e term)
                                     • dense_mean_prob_per_expert diagnostics
                                     • Switch-style dispatch weights (see below)

    sparse_softmax_probs (B, E)  — softmax(logits with non-top-k set to -inf).
                                   Zero for non-selected experts.  Returned for
                                   diagnostics/analysis only; NOT used for
                                   dispatch (see rationale below).

    topk_idx             (B, k)  — indices of the k selected experts per sample.
                                   Used for selection-frequency diagnostics and
                                   for computing f_e in switch_balance_loss.

    topk_weights         (B, k)  — Switch-style dispatch weights:
                                       topk_weights = full_softmax_probs.gather(
                                           dim=1, index=topk_idx)
                                   These are the router's confidence values at
                                   the selected expert positions, gathered from
                                   the PRE-top-k softmax.  They are NOT
                                   renormalised to sum to 1.

Why Switch-style weights (NOT renormalised masked softmax)
----------------------------------------------------------
The previous implementation computed topk_weights as:
    1. mask non-top-k logits to -inf
    2. sparse_softmax_probs = softmax(masked_logits)
    3. topk_weights = sparse_softmax_probs.gather(...)
    4. renormalise: topk_weights /= sum(topk_weights)

For k=1, step 2 produces a lone non-masked logit whose softmax is always 1.0,
and step 4 leaves it at 1.0.  The constant 1.0 has zero gradient, so no
signal from the detection loss ever reaches the gate.  This breaks the
rich-get-richer dynamics that allow the gate to specialise.

Using full_softmax_probs for topk_weights instead:
    • topk_weights ∈ (0, 1) with a live gradient through the softmax
    • valid for any k — no special-casing needed
    • restores end-to-end gradient flow from the task loss through dispatch
      to the gate weights

NOTE: Because topk_weights no longer sum to 1, downstream dispatch blocks
MUST handle the scale themselves:
    • BEVMoEBlock     — residual-delta dispatch (scale-safe; see bev_moe.py)
    • ModalitySpecificMoEBlock — delta dispatch (scale-safe; see modality_specific_moe.py)
    • JointModalityMoEBlock   — locally renormalises topk_weights for dispatch
                                 because it has no natural residual (see joint_modality_moe.py)

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
    full_softmax_probs:   Tensor  # (B, E) pre-top-k softmax — router belief over all experts
    sparse_softmax_probs: Tensor  # (B, E) post-top-k masked softmax — zero on non-selected experts
    topk_idx:             Tensor  # (B, k) selected expert indices
    topk_weights:         Tensor  # (B, k) Switch-style dispatch weights (router
                                  #         confidence at selected indices; NOT
                                  #         renormalised to sum to 1 — downstream
                                  #         blocks handle residual/scale)


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
    """Deterministic top-k gate over experts.

    Routing procedure:
      1. Compute logits from the gate linear layer.
      2. full_softmax_probs = softmax(logits) — pre-top-k router belief over all experts.
      3. Select top-k expert indices from logits.
      4. sparse_softmax_probs = softmax(masked_logits) where non-top-k → −∞.
         Used for diagnostics only.
      5. topk_weights = full_softmax_probs gathered at top-k indices
         (Switch Transformer style — NOT renormalised to sum to 1).

    Why Switch-style topk_weights instead of renormalised masked softmax
    -------------------------------------------------------------------
    Using softmax(masked_logits).gather(...) and renormalising to sum to 1
    gives ``topk_weights ≡ 1.0`` when k=1, because softmax of a single finite
    value (with the rest at −∞) is identically 1.  Its derivative w.r.t. the
    selected logit is zero, so NO gradient from the detection loss flows back
    to the gate — the router cannot learn preferences, only balance.

    Using full_softmax_probs directly gives ``topk_weights ∈ (0, 1)`` that
    scales continuously with the router's confidence in the selected expert
    (≈ 1/E when uniform, → 1 as the router sharpens).  The detection loss
    then provides gradient through the weight back to the gate logits,
    restoring rich-get-richer dynamics and genuine specialization.

    Trade-off: weights do not sum to 1, so for residual experts the
    expert delta is dampened by the router confidence — which is the
    desired behaviour (less disruption when the router is uncertain).

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
            ``sparse_softmax_probs``, ``topk_idx``, ``topk_weights``.
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)

        logits = self.gate(feat)                               # (B, E)

        # Pre-top-k: router belief over all experts (unmasked).
        full_softmax_probs = torch.softmax(logits, dim=-1)     # (B, E)

        # Select top-k indices from logits.
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)  # (B, k)

        # Sparse post-top-k softmax — diagnostics only (NOT used for dispatch).
        # For k=1 this is identically 1.0 on the selected expert and carries
        # no gradient; see class docstring for why this is unsuitable as the
        # dispatch weight.
        threshold = topk_vals[:, -1:]                          # (B, 1) smallest kept logit
        masked = torch.where(
            logits >= threshold,
            logits,
            torch.full_like(logits, float('-inf')),
        )
        sparse_softmax_probs = torch.softmax(masked, dim=-1)   # (B, E)

        # Switch-style dispatch weights: full softmax confidence at the
        # selected indices, WITHOUT renormalisation.  Preserves gradient
        # flow to the gate for any k (including k=1).
        topk_weights = full_softmax_probs.gather(dim=1, index=topk_idx)  # (B, k)

        return GateOutput(full_softmax_probs=full_softmax_probs,
                          sparse_softmax_probs=sparse_softmax_probs,
                          topk_idx=topk_idx, topk_weights=topk_weights)


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
      1. ``logits = W_gate(x)`` then z-score normalise per sample.
      2. Training: add ``ε · softplus(W_noise(x)) + noise_floor``.
      3. ``full_softmax_probs = softmax(logits)`` — pre-top-k router belief.
      4. Keep only the top-k logits; set all others to ``−∞``.
      5. ``sparse_softmax_probs = softmax(masked_logits / temperature)``

    ``full_softmax_probs`` is the router's belief over all experts given the
    (noisy at train-time) logits, before any masking.
    ``sparse_softmax_probs`` is zero on non-selected experts.

    Args:
        feat_dim: Dimension of the pooled BEV feature vector.
        num_experts: Number of experts to route over.
        k: Number of experts selected per sample.
        context_dim: If > 0, context vector is concatenated with the feature
            vector before the gate linear layers (same as :class:`TopkGate`).
        temperature: Softmax temperature.  Values < 1 sharpen the distribution
            over selected experts; values > 1 flatten it.
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
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)  # (B, feat_dim + context_dim)

        h = self.in_drop(feat) if self.in_drop is not None else feat

        logits = self.w_gate(h)  # (B, E)

        # Normalise logits per sample so all experts start on the same scale.
        # Without this a dominant logit always wins and noise has no effect on
        # load balancing — the whole point of the noisy gate.
        logits = (logits - logits.mean(dim=-1, keepdim=True)) / \
                 (logits.std(dim=-1, keepdim=True) + 1e-5)

        if self.training:
            # Learned noise std (strictly positive) plus a constant noise floor
            # so the network cannot suppress exploration entirely.
            noise_std = F.softplus(self.w_noise(h)) + self.noise_floor  # (B, E)
            logits = logits + torch.randn_like(noise_std) * noise_std    # (B, E)

        if self.log_drop is not None:
            logits = self.log_drop(logits)

        # Pre-top-k: router belief over all experts given (noisy) logits.
        full_softmax_probs = F.softmax(logits, dim=-1)                # (B, E)

        # Select top-k indices from the (noisy) logits.
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)   # (B, k)

        # Sparse post-top-k softmax — diagnostics only (NOT used for dispatch).
        # For k=1 this is identically 1.0 on the selected expert and carries
        # no gradient; see TopkGate docstring for the full rationale.
        threshold = topk_vals[:, -1:]                                  # (B, 1)
        masked = torch.where(
            logits >= threshold,
            logits,
            torch.full_like(logits, float('-inf')),
        )  # (B, E)
        sparse_softmax_probs = F.softmax(masked / self.temperature, dim=-1)  # (B, E)

        # Switch-style dispatch weights: full softmax confidence at the
        # selected indices, NOT renormalised.  Preserves gradient flow
        # from the detection loss to the gate logits for any k.
        topk_weights = full_softmax_probs.gather(dim=1, index=topk_idx)      # (B, k)

        return GateOutput(full_softmax_probs=full_softmax_probs,
                          sparse_softmax_probs=sparse_softmax_probs,
                          topk_idx=topk_idx, topk_weights=topk_weights)

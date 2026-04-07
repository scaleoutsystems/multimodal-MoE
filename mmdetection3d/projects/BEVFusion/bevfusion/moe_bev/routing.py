"""Routing modules: ContextEncoder and TopkGate.

ContextEncoder converts per-sample metadata dicts (weather, road type, etc.)
into a fixed-size context vector.  
TopkGate selects the top-k experts from a pooled BEV feature vector optionally
concatenated with context.

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


# ── Gate output container ─────────────────────────────────────────────────

@dataclass
class GateOutput:
    probs: Tensor         # (B, E) full softmax distribution
    topk_idx: Tensor      # (B, k) selected expert indices
    topk_weights: Tensor  # (B, k) softmax weights for selected experts


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

    Example config::

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
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
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
        #  [-0.4, 1.0, 0.1], "fog"
        #  [ 0.2,-0.1, 0.5], "city"
        #],

        #[
        #  [ 0.6,-0.2, 0.3], "highway"
        #   [-0.5, 0.7,-0.1], "arterial-rural"
        #  ]
        #]

        # --> concatenate per sample (B, 2 * embed_dim) --> (B, 6)
        # [
        #[-0.4, 1.0, 0.1,  0.6, -0.2, 0.3], "fog" + "highway"
        #[ 0.2,-0.1, 0.5, -0.5,  0.7,-0.1], "city" + "arterial-rural"
        #] 

        return self.proj(torch.cat(parts, dim=1))


# ── TopkGate ──────────────────────────────────────────────────────────────

class TopkGate(nn.Module):
    """Deterministic top-k gate over experts.

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
            :class:`GateOutput` with ``probs``, ``topk_idx``,
            ``topk_weights``.
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)
        logits = self.gate(feat)                      # (B, E)
        probs = torch.softmax(logits, dim=1)          # (B, E)
        topk_weights, topk_idx = probs.topk(self.k, dim=1)  # (B, k) each
        # Re-normalise selected weights so they sum to 1 per sample.
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True)
                                       + 1e-8)
        return GateOutput(probs=probs, topk_idx=topk_idx,
                          topk_weights=topk_weights)


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
      3. Keep only the top-k logits; set all others to ``−∞``.
      4. ``probs = softmax(masked_logits / temperature)``

    This is in contrast to :class:`TopkGate` which applies softmax first and
    then takes topk — the noisy gate applies topk *before* softmax so that the
    softmax distribution is concentrated on the k selected experts.

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
            :class:`GateOutput` with ``probs`` ``(B, E)``, ``topk_idx``
            ``(B, k)``, and ``topk_weights`` ``(B, k)``.
            ``probs`` is zero on non-selected experts; ``topk_weights`` are
            the re-normalised softmax weights for the k selected experts.
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

        # Mask all non-top-k logits to −∞ BEFORE softmax so that softmax
        # probability mass is concentrated entirely on the k selected experts.
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)  # (B, k)
        threshold = topk_vals[:, -1:]                                 # (B, 1) smallest kept logit
        masked = torch.where(
            logits >= threshold,
            logits,
            torch.full_like(logits, float('-inf')),
        )  # (B, E)

        probs = F.softmax(masked / self.temperature, dim=-1)  # (B, E), 0 on non-topk

        topk_weights = probs.gather(dim=1, index=topk_idx)    # (B, k)
        # Re-normalise so selected weights sum to 1 (guards against fp edge cases).
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)

        return GateOutput(probs=probs, topk_idx=topk_idx,
                          topk_weights=topk_weights)

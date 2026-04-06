"""Routing modules: ContextEncoder and TopkGate.

ContextEncoder converts per-sample metadata dicts (weather, road type, etc.)
into a fixed-size context vector.  TopkGate selects the top-k experts from
a pooled BEV feature vector optionally concatenated with context.

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
from torch import Tensor


# ── Gate output container ─────────────────────────────────────────────────

@dataclass
class GateOutput:
    probs: Tensor         # (B, E) full softmax distribution
    topk_idx: Tensor      # (B, k) selected expert indices
    topk_weights: Tensor  # (B, k) softmax weights for selected experts


# ── ZOD field registry ────────────────────────────────────────────────────
# Single source of truth for field types and categorical vocabularies.
# Vocabs contain exactly the values present in the ZOD parquet — no padding
# or unknown tokens.  Add new fields here; never duplicate specs in configs.

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
            self.vocab_maps[name] = {v: i for i, v in enumerate(vocab)}
            self.embeddings[name] = nn.Embedding(len(vocab), embed_dim)

        # All fields are categorical, each contributing embed_dim floats.
        raw_dim = len(active_fields) * embed_dim
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def _embed_field(self, batch_ctx: List[dict], name: str,
                     device: torch.device) -> Tensor:
        """Look up the embedding for one categorical field across the batch."""
        vmap = self.vocab_maps[name]
        indices = []
        for ctx in batch_ctx:
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

        Args:
            batch_input_metas: List of length B.  Each dict must contain
                a ``'context'`` key mapping to the per-sample metadata dict
                (added via ``Pack3DDetInputs`` ``meta_keys``).

        Returns:
            ``(B, out_dim)`` context vector.
        """
        batch_ctx = [m.get('context', {}) for m in batch_input_metas]
        device = next(self.parameters()).device

        # Embed each categorical field → (B, embed_dim), concat all, project.
        parts = [self._embed_field(batch_ctx, name, device)
                 for name in self.field_names]

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

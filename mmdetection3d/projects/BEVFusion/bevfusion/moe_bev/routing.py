"""Routing modules: ContextEncoder and TopkGate.

ContextEncoder converts per-sample metadata dicts (weather, road type, etc.)
into a fixed-size context vector.  TopkGate selects the top-k experts from
a pooled BEV feature vector optionally concatenated with context.
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
    probs: Tensor       # (B, E) full softmax distribution
    topk_idx: Tensor    # (B, k) selected expert indices
    topk_weights: Tensor  # (B, k) softmax weights for selected experts


# ── ContextEncoder ────────────────────────────────────────────────────────

# Default vocabulary for ZOD context fields.  Index 0 is reserved for
# unknown / missing values ("" or unseen strings).
DEFAULT_CONTEXT_FIELDS = {
    'weather_group': {
        'type': 'categorical',
        'vocab': ['', 'clear_like', 'cloud_like', 'precipitation',
                  'fog', 'wind', 'unknown'],
    },
    'road_type': {
        'type': 'categorical',
        'vocab': ['', 'highway', 'city', 'arterial-urban',
                  'arterial-rural', 'smaller-rural'],
    },
    'time_of_day': {
        'type': 'categorical',
        'vocab': ['', 'day', 'night', 'twilight'],
    },
    'solar_context_bin': {
        'type': 'categorical',
        'vocab': ['', 'night', 'day', 'missing'],
    },
    'complexity_bin': {
        'type': 'categorical',
        'vocab': ['', 'empty', 'low', 'medium', 'high'],
    },
    'solar_angle_elevation': {'type': 'scalar'},
    'complexity_score': {'type': 'scalar'},
    'num_pedestrians_final': {'type': 'scalar'},
}


class ContextEncoder(nn.Module):
    """Encode per-sample context metadata into a fixed-size vector.

    Categorical fields are embedded; scalar fields are normalised and
    passed through a small linear layer.  All embeddings / projections
    are concatenated and projected to ``out_dim``.

    Args:
        context_fields: Mapping from field name to field spec.  Each spec
            is a dict with ``'type'`` (``'categorical'`` or ``'scalar'``)
            and, for categoricals, a ``'vocab'`` list where index 0 is the
            unknown token.  Defaults to ``DEFAULT_CONTEXT_FIELDS``.
        embed_dim: Embedding dimension per categorical field.
        out_dim: Output context vector dimension.
    """

    def __init__(self,
                 context_fields: Optional[Dict] = None,
                 embed_dim: int = 16,
                 out_dim: int = 64):
        super().__init__()
        if context_fields is None:
            context_fields = DEFAULT_CONTEXT_FIELDS

        self.field_names: List[str] = []
        self.field_types: List[str] = []
        self.embeddings = nn.ModuleDict()
        self.vocab_maps: Dict[str, Dict[str, int]] = {}

        raw_dim = 0
        for name, spec in context_fields.items():
            self.field_names.append(name)
            self.field_types.append(spec['type'])
            if spec['type'] == 'categorical':
                vocab = spec['vocab']
                self.vocab_maps[name] = {v: i for i, v in enumerate(vocab)}
                self.embeddings[name] = nn.Embedding(len(vocab), embed_dim,
                                                     padding_idx=0)
                raw_dim += embed_dim
            else:
                raw_dim += 1

        self.proj = nn.Sequential(
            nn.Linear(raw_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def _encode_field(self, batch_ctx: List[dict], name: str,
                      ftype: str, device: torch.device) -> Tensor:
        B = len(batch_ctx)
        if ftype == 'categorical':
            vmap = self.vocab_maps[name]
            indices = []
            for ctx in batch_ctx:
                val = ctx.get(name, '')
                if val is None:
                    val = ''
                indices.append(vmap.get(str(val), 0))
            idx_t = torch.tensor(indices, dtype=torch.long, device=device)
            return self.embeddings[name](idx_t)  # (B, embed_dim)
        else:
            vals = []
            for ctx in batch_ctx:
                v = ctx.get(name, None)
                vals.append(float(v) if v is not None else 0.0)
            return torch.tensor(vals, dtype=torch.float32,
                                device=device).unsqueeze(1)  # (B, 1)

    def forward(self, batch_input_metas: List[dict]) -> Tensor:
        """Encode context from a list of per-sample metadata dicts.

        Args:
            batch_input_metas: List of length B, each containing a
                ``'context'`` key with the metadata dict.

        Returns:
            Context vector of shape ``(B, out_dim)``.
        """
        batch_ctx = [m.get('context', {}) for m in batch_input_metas]
        device = next(self.parameters()).device

        parts = []
        for name, ftype in zip(self.field_names, self.field_types):
            parts.append(self._encode_field(batch_ctx, name, ftype, device))

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
            feat: (B, feat_dim) pooled BEV features.
            ctx: (B, context_dim) optional context vector.

        Returns:
            GateOutput with probs, topk_idx, topk_weights.
        """
        if ctx is not None:
            feat = torch.cat([feat, ctx], dim=1)
        logits = self.gate(feat)                     # (B, E)
        probs = torch.softmax(logits, dim=1)         # (B, E)
        topk_weights, topk_idx = probs.topk(self.k, dim=1)  # (B, k) each
        # Re-normalise selected weights so they sum to 1 per sample.
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True)
                                       + 1e-8)
        return GateOutput(probs=probs, topk_idx=topk_idx,
                          topk_weights=topk_weights)

"""Joint-modality Mixture-of-Experts block (Variant A).

JointModalityMoEBlock replaces ConvFuser entirely.  Each expert receives
both camera and LiDAR BEV maps as input and learns its own fusion strategy
(concat → 3×3 conv).  A TopkGate routes samples to experts based on
spatial-aware summary descriptors from both modalities plus optional context.

Required graph (no ConvFuser anywhere)::

    cam_bev   ─┐
               ├─→ JointModalityMoEBlock ─→ fused_bev ─→ pts_backbone
    lidar_bev ─┘

Router input
------------
    cam_bev   → BEVSummaryHead(cam_channels)    → (B, 64)
    lidar_bev → BEVSummaryHead(lidar_channels)  → (B, 64)
    cat                                          → (B, 128)  [+ ctx]
    TopkGate                                     → GateOutput

Dispatch strategy: Shazeer top-k mixture (Σ w = 1)
--------------------------------------------------
TopkGate returns standard Shazeer top-k mixture weights — renormalised over
the top-k selections so ``Σ_j topk_weights = 1`` per sample (see routing.py).

Joint experts produce a fresh fused BEV map (not a residual correction on an
existing feature map), so the dispatched output must be a proper weighted
mixture that sums to 1 — otherwise the backbone sees a scale-corrupted
feature.  With Shazeer top-k this constraint is satisfied by the gate
output directly, so the block simply does:

    out[b] = Σ_j  topk_weights[b, j] · expert_j(cam_bev[b], lidar_bev[b])

No local renormalisation is needed.

Gradient flow
-------------
  • k ≥ 2: task loss flows through each ``topk_weights[b, j]`` (softmax
    ratio over the top-k logits).  Full specialisation possible.
  • k = 1: ``topk_weights ≡ 1`` is constant, so task loss cannot push the
    router toward a specific expert via the weight.  Gate training relies
    on ``importance_loss`` + ``load_loss``.  Use k ≥ 2 for task-driven
    specialisation in this block.

residual_gain parameter
-----------------------
``residual_gain`` is accepted for config parity with BEVMoEBlock and
ModalitySpecificMoEBlock but is a no-op in this block — the fused output
already sums to 1 by construction.  A warning is issued if a non-1.0 value
is passed so the caller knows it is ignored.

moe_info contract
-----------------
self._moe_info is written after every forward() with:
    full_softmax_probs   (B, E)  — pre-top-k softmax (router belief over all experts).
                                   Used by importance_loss and the
                                   dense_mean_prob_per_expert diagnostics.
    sparse_softmax_probs (B, E)  — top-k mixture laid back into (B, E), zero
                                   off-topk.  Diagnostics only.
    topk_idx             (B, k)  — selected expert indices per sample.
    topk_weights         (B, k)  — Shazeer top-k mixture weights, Σ_j = 1.
                                   Used directly for dispatch and for
                                   dispatch_mass_per_expert diagnostics.
    aux_loss             scalar  — importance_loss + load_loss.
    importance_loss      scalar  — Shazeer importance term (logged as
                                   moe_importance_loss).  Differentiable.
    load_loss            scalar  — Shazeer Gaussian-CDF load term (logged as
                                   moe_load_loss).  Differentiable when the
                                   gate provides noise_std; zero otherwise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .losses import importance_loss, load_loss
from .routing import BEVSummaryHead, ContextEncoder, TopkGate


@MODELS.register_module()
class JointModalityExpert(nn.Module):
    """Single joint-modality expert: concatenate two BEV maps and convolve.

    Each expert has independent weights so it can learn a distinct
    fusion strategy (e.g., camera-dominant vs. LiDAR-dominant).

    Args:
        cam_channels:   Camera BEV channels (e.g. 80).
        lidar_channels: LiDAR BEV channels (e.g. 256).
        out_channels:   Output fused BEV channels (e.g. 256).
    """

    def __init__(self, cam_channels: int, lidar_channels: int,
                 out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(cam_channels + lidar_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cam_bev: Tensor, lidar_bev: Tensor) -> Tensor:
        return self.fuse(torch.cat([cam_bev, lidar_bev], dim=1))


@MODELS.register_module()
class JointModalityMoEBlock(nn.Module):
    """Joint-modality MoE block — Variant A.

    Replaces ConvFuser entirely.  Routes each sample to k fusion experts
    based on spatial-aware routing descriptors from both BEV maps, plus
    optional context metadata.  Each expert receives both cam_bev and
    lidar_bev and produces a single fused output.

    Args:
        cam_channels:      Camera BEV channels.
        lidar_channels:    LiDAR BEV channels.
        out_channels:      Output fused BEV channels.
        num_experts:       Number of fusion experts.
        k:                 Top-k experts per sample.
        importance_coef:   Weight for importance balancing loss.
        load_coef:         Weight for load balancing loss.
        residual_gain:     Accepted for config parity with BEVMoEBlock /
                           ModalitySpecificMoEBlock but is a NO-OP here
                           (see module docstring for why).  A warning is
                           issued if a value other than 1.0 is passed.
                           Default 1.0.
        router_pool_size:  Spatial size for BEVSummaryHead pooling grid.
        router_hidden_dim: Hidden dim of the MLP inside BEVSummaryHead.
        router_out_dim:    Output dim per modality BEVSummaryHead.
                           Gate sees 2 × router_out_dim (+ ctx_dim) as input.
        context_cfg:       If provided, build a ContextEncoder with these kwargs.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_experts: int = 6,
        k: int = 1,
        importance_coef: float = 0.02,
        load_coef: float = 0.01,
        residual_gain: float = 1.0,
        router_pool_size: int = 2,
        router_hidden_dim: int = 128,
        router_out_dim: int = 64,
        context_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef
        self.residual_gain = float(residual_gain)
        if self.residual_gain != 1.0:
            import warnings
            warnings.warn(
                f'JointModalityMoEBlock received residual_gain='
                f'{self.residual_gain} but this parameter is a no-op for '
                f'this block (joint experts produce fresh fused BEVs; '
                f'there is no natural residual reference).  The value is '
                f'silently ignored — use BEVMoEBlock or '
                f'ModalitySpecificMoEBlock if you need residual_gain to '
                f'have effect.',
                stacklevel=2)

        self.experts = nn.ModuleList([
            JointModalityExpert(cam_channels, lidar_channels, out_channels)
            for _ in range(num_experts)
        ])

        self.cam_summary = BEVSummaryHead(cam_channels, router_pool_size,
                                          router_hidden_dim, router_out_dim)
        self.lidar_summary = BEVSummaryHead(lidar_channels, router_pool_size,
                                            router_hidden_dim, router_out_dim)

        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        self.gate = TopkGate(
            feat_dim=2 * router_out_dim,
            num_experts=num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        cam_bev: Tensor,
        lidar_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Fuse camera and LiDAR BEV maps via routed experts.

        Args:
            cam_bev:            Camera BEV (B, Cc, H, W).
            lidar_bev:          LiDAR BEV (B, Cl, H, W).
            batch_input_metas:  Per-sample metadata (for context routing).

        Returns:
            fused_bev: (B, out_channels, H, W).
            moe_info:  Dict with routing statistics and aux_loss.
        """
        B = cam_bev.shape[0]

        cam_feat = self.cam_summary(cam_bev)
        lidar_feat = self.lidar_summary(lidar_bev)
        feat = torch.cat([cam_feat, lidar_feat], dim=1)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)

        gate_out = self.gate(feat, ctx)

        out = cam_bev.new_zeros(B, self.out_channels,
                                cam_bev.shape[2], cam_bev.shape[3])

        # Shazeer top-k mixture weights already sum to 1 per sample (see
        # routing.py), so the fused output is automatically at unit scale.
        # No local renormalisation needed.  With k=1 the single weight is
        # identically 1.0 and carries no task-loss gradient; gate training
        # for k=1 relies on importance_loss + load_loss.  Use k ≥ 2 for
        # task-driven specialisation in this block.
        for b in range(B):
            sample_out = torch.zeros_like(out[b:b + 1])
            for j in range(self.k):
                eidx = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](cam_bev[b:b + 1],
                                                lidar_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
            out[b] = sample_out[0]

        imp_loss = importance_loss(
            gate_out.full_softmax_probs,
            self.importance_coef,
        )
        ld_loss  = load_loss(
            gate_out.clean_logits,
            gate_out.noisy_logits,
            gate_out.noise_std,
            self.k,
            self.load_coef,
        )
        aux = imp_loss + ld_loss

        moe_info = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
        }
        self._moe_info = moe_info
        return out, moe_info

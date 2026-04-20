"""Modality-specific Mixture-of-Experts block (Variant B).

ModalitySpecificMoEBlock replaces ConvFuser entirely.  Camera and LiDAR
BEV maps are processed independently by dedicated expert pools, then the
refined modality outputs are fused via a simple concat + 1×1 conv projection.

Required graph (no ConvFuser anywhere):

    cam_bev   → camera experts  ─┐
                                 ├─→ concat → 1×1 proj → 3×3 conv (spatial alignment) → fused_bev → pts_backbone
    lidar_bev → lidar experts   ─┘

Architecture overview
---------------------
Inputs:
    cam_bev   (B, Cc, H, W)
    lidar_bev (B, Cl, H, W)

Routing signal:
    cam_summary   = BEVSummaryHead(cam_channels)  → (B, router_out_dim)
    lidar_summary = BEVSummaryHead(lidar_channels) → (B, router_out_dim)
    feat = cat([cam_summary, lidar_summary])        → (B, 2·router_out_dim)
    [+ optional context]
    → TopkGate over E = num_cam_experts + num_lidar_experts experts

Expert dispatch:
    Experts 0 … num_cam_experts-1   are camera experts  (BEVResidualExpert, Cc ch)
    Experts num_cam_experts … E-1   are LiDAR experts   (BEVResidualExpert, Cl ch)

    For each sample, the k selected experts are dispatched to their respective
    modality's BEV map. The output for each modality starts from the original
    BEV (identity passthrough) and accumulates weighted expert deltas scaled
    by ``residual_gain``:
        cam_out   = cam_bev   + residual_gain · Σ_j∈cam  w_j · (expert_j(cam_bev) − cam_bev)
        lidar_out = lidar_bev + residual_gain · Σ_j∈lid  w_j · (expert_j(lidar_bev) − lidar_bev)
    The w_j come from the gate's Shazeer top-k mixture (Σ_j w_j = 1 across
    the joint pool), so an individual modality accumulates only the fraction
    of mass assigned to its experts.  ``residual_gain`` therefore does not
    need to track num_experts — default 1.0 applies the weighted delta at
    full scale.  See BEVMoEBlock docstring for tuning guidance.

Fusion:
    After expert dispatch, cam_out (Cc ch) and lidar_out (Cl ch) are
    concatenated channel-wise, then fused in two steps:
      1. 1×1 conv projection → out_channels ch (collapses channel dimension)
      2. 3×3 conv → out_channels ch (compensates for cam/LiDAR spatial
         misalignment from the lift-splat-shoot view transform)
    This is the ONLY fusion step — no ConvFuser is used in this variant.

Regularisation:
    - importance_loss:     CV² of per-expert mean soft probability mass.
                           Catches dead experts (logged as moe_importance_loss).
    - load_loss:           CV² of the Gaussian-CDF dispatch estimator.
                           Catches hard-dispatch collapse (logged as moe_load_loss).
    - group_balance_loss:  equal total routing mass to camera-group vs LiDAR-group.

moe_info contract
-----------------
self._moe_info is written after every forward():
    full_softmax_probs   (B, E)  — full pre-top-k softmax over all experts
    sparse_softmax_probs (B, E)  — post-top-k masked softmax (diagnostics only)
    topk_idx             (B, k)  — selected expert indices
    topk_weights         (B, k)  — Shazeer top-k mixture weights (Σ_j = 1
                                   across the joint pool)
    cam_expert_ids       list[int]
    lidar_expert_ids     list[int]
    cam_group_mass       float   — total routing mass to camera group (this batch)
    lidar_group_mass     float
    aux_loss             scalar  — importance_loss + load_loss + group_balance_loss
    importance_loss      scalar  — Shazeer importance term (moe_importance_loss)
    load_loss            scalar  — Shazeer load term (moe_load_loss)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .losses import group_balance_loss, importance_loss, load_loss
from .routing import BEVSummaryHead, ContextEncoder, TopkGate


@MODELS.register_module()
class ModalitySpecificMoEBlock(nn.Module):
    """Modality-specific MoE block with built-in fusion — Variant B.

    Camera and LiDAR BEV maps are routed independently through separate
    expert pools, then fused via a two-step projection: concat → 1×1 conv
    (channel collapse) → 3×3 conv (spatial alignment).  No ConvFuser is used.

    Args:
        cam_channels:          Camera BEV channels (e.g. 80).
        lidar_channels:        LiDAR BEV channels (e.g. 256).
        out_channels:          Output fused BEV channels (e.g. 256).
        num_cam_experts:       Number of camera-specific experts.
        num_lidar_experts:     Number of LiDAR-specific experts.
        k:                     Top-k experts selected per sample (over all E).
        num_convs:             Conv layers per BEVResidualExpert.
        importance_coef:       Weight for importance loss.
        load_coef:             Weight for load loss.
        group_balance_coef:    Weight for camera-vs-LiDAR group balance loss.
        residual_gain:         Scalar multiplier on the routed expert delta
                               (applied independently to cam and LiDAR paths):
                                 cam_out   = cam_bev   + residual_gain · Σ w·Δ_cam
                                 lidar_out = lidar_bev + residual_gain · Σ w·Δ_lidar
                               With Shazeer top-k mixture (Σ_j w_j = 1 across
                               the joint pool) the default 1.0 applies the
                               weighted delta at full scale with no dependence
                               on num_experts.  See BEVMoEBlock module docstring
                               for tuning guidance.
        router_pool_size:      Spatial pooling size for BEVSummaryHead.
        router_hidden_dim:     Hidden dim of BEVSummaryHead MLP.
        router_out_dim:        Output dim per modality BEVSummaryHead.
                               Gate sees 2 × router_out_dim (+ ctx_dim).
        context_cfg:           Optional ContextEncoder config dict.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_cam_experts: int = 3,
        num_lidar_experts: int = 3,
        k: int = 2,
        num_convs: int = 1,
        importance_coef: float = 0.02,
        load_coef: float = 0.01,
        group_balance_coef: float = 0.005,
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
        self.num_cam_experts = num_cam_experts
        self.num_lidar_experts = num_lidar_experts
        self.num_experts = num_cam_experts + num_lidar_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef
        self.group_balance_coef = group_balance_coef
        self.residual_gain = float(residual_gain)

        self.cam_expert_ids: List[int] = list(range(num_cam_experts))
        self.lidar_expert_ids: List[int] = list(range(num_cam_experts,
                                                      self.num_experts))

        self.cam_experts = make_bev_experts(
            num_cam_experts, cam_channels, num_convs)
        self.lidar_experts = make_bev_experts(
            num_lidar_experts, lidar_channels, num_convs)

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
            num_experts=self.num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        # Two-step fusion — this is the ONLY fusion in this variant
        # (no ConvFuser used):
        #
        # Step 1 — 1×1 channel projection:
        #   Concat cam_out (Cc ch) and lidar_out (Cl ch) and project to
        #   out_channels with a 1×1 conv.  This collapses the channel
        #   dimension without mixing spatial information across pixels,
        #   producing a combined representation at each spatial location.
        #
        # Step 2 — 3×3 spatial alignment conv:
        #   A single 3×3 conv + BN + ReLU applied after the projection.
        #   Camera and LiDAR BEV features are not perfectly registered
        #   spatially due to projection errors in the lift-splat-shoot
        #   view transform.  The 3×3 conv gives the network a small local
        #   receptive field to compensate for these sub-pixel misalignments
        #   and to smooth the boundary between cam and lidar feature regions,
        #   without adding deep capacity or requiring ConvFuser.
        self.fusion_proj = nn.Sequential(
            # 1×1 projection: (Cc + Cl) ch → out_channels ch
            nn.Conv2d(cam_channels + lidar_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 3×3 spatial alignment: compensates for cam/LiDAR misalignment
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        cam_bev: Tensor,
        lidar_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Route modality BEVs through experts, then fuse.

        Args:
            cam_bev:            Camera BEV feature map (B, Cc, H, W).
            lidar_bev:          LiDAR BEV feature map  (B, Cl, H, W).
            batch_input_metas:  Per-sample metadata (for context routing).

        Returns:
            fused_bev: (B, out_channels, H, W) — single fused tensor.
            moe_info:  Dict with routing statistics and aux_loss.
        """
        B = cam_bev.shape[0]

        # ── Step 1: Build joint routing descriptor ────────────────────
        cam_feat = self.cam_summary(cam_bev)
        lidar_feat = self.lidar_summary(lidar_bev)
        feat = torch.cat([cam_feat, lidar_feat], dim=1)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)

        # ── Step 2: Single gate over all E experts ────────────────────
        gate_out = self.gate(feat, ctx)

        # ── Step 3: Modality-aware dispatch ───────────────────────────
        cam_out = cam_bev.clone()
        lidar_out = lidar_bev.clone()

        for b in range(B):
            for j in range(self.k):
                eidx = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]

                if eidx in self.cam_expert_ids:
                    exp_out = self.cam_experts[eidx](cam_bev[b:b + 1])
                    delta = exp_out - cam_bev[b:b + 1]
                    cam_out[b] = cam_out[b] + self.residual_gain * weight * delta[0]
                else:
                    lidar_eidx = eidx - self.num_cam_experts
                    exp_out = self.lidar_experts[lidar_eidx](
                        lidar_bev[b:b + 1])
                    delta = exp_out - lidar_bev[b:b + 1]
                    lidar_out[b] = lidar_out[b] + self.residual_gain * weight * delta[0]

        # ── Step 4: Fusion — concat → 1×1 proj → 3×3 align ──────────
        fused_bev = self.fusion_proj(
            torch.cat([cam_out, lidar_out], dim=1))

        # ── Step 5: Auxiliary losses ──────────────────────────────────
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
        gb_loss  = group_balance_loss(
            gate_out.full_softmax_probs,
            self.cam_expert_ids,
            self.lidar_expert_ids,
            self.group_balance_coef,
        )
        aux = imp_loss + ld_loss + gb_loss

        with torch.no_grad():
            cam_mass   = gate_out.full_softmax_probs[:, self.cam_expert_ids].sum().item()
            lidar_mass = gate_out.full_softmax_probs[:, self.lidar_expert_ids].sum().item()

        moe_info = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'cam_expert_ids':       self.cam_expert_ids,
            'lidar_expert_ids':     self.lidar_expert_ids,
            'cam_group_mass':       cam_mass,
            'lidar_group_mass':     lidar_mass,
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
        }
        self._moe_info = moe_info
        return fused_bev, moe_info

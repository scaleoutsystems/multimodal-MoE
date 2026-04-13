"""Modality-specific Mixture-of-Experts block.

ModalitySpecificMoEBlock (Variant A with joint gate)
----------------------------------------------------
This module handles the case where expert modules are *specialised* to one
modality — some experts process camera BEV, others process LiDAR BEV — but
the **routing decision is made jointly** using spatial summary descriptors
from *both* modalities concatenated together.

Architecture overview
---------------------
Inputs:
    cam_bev   (B, Cc, H, W)
    lidar_bev (B, Cl, H, W)

Routing signal:
    cam_summary   = BEVSummaryHead(cam_channels)  → (B, router_out_dim)
    lidar_summary = BEVSummaryHead(lidar_channels)→ (B, router_out_dim)
    feat = cat([cam_summary, lidar_summary])       → (B, 2·router_out_dim)
    [+ optional context]
    → TopkGate over E = num_cam_experts + num_lidar_experts experts

Expert dispatch:
    Experts 0 … num_cam_experts-1   are camera experts  (BEVResidualExpert, Cc ch)
    Experts num_cam_experts … E-1   are LiDAR experts   (BEVResidualExpert, Cl ch)

    For each sample, the k selected experts are dispatched to their respective
    modality's BEV map. The output for each modality starts from the original
    BEV (passthrough identity) and accumulates weighted expert deltas:
        cam_out[b]   = cam_bev[b]   + Σ_{j: cam expert} weight_j · delta_j(cam_bev[b])
        lidar_out[b] = lidar_bev[b] + Σ_{j: lidar expert} weight_j · delta_j(lidar_bev[b])
    If no expert of a given modality is selected for sample b, that modality's
    output is the unchanged input (identity passthrough).

Regularisation:
    - importance_loss: equal gate-prob mass across all E experts. (differentiable)
    - load_loss:       equal hard selection counts across all E experts. (detached)
    - group_balance_loss: equal total routing mass to camera-group vs LiDAR-group.
      Coefficient: moe_group_balance_loss_weight (default 5e-3).

moe_info contract
-----------------
self._moe_info is written after every forward():
    probs          (B, E)    — full softmax over all experts
    topk_idx       (B, k)    — selected expert indices
    topk_weights   (B, k)    — re-normalised weights
    cam_expert_ids list[int] — indices of camera experts (for Hook C)
    lidar_expert_ids list[int]
    cam_group_mass  float    — total routing mass to camera group (this batch)
    lidar_group_mass float
    aux_loss       scalar

Why a joint gate?
-----------------
A separate gate per modality (as in cam_moe/lidar_moe independent BEVMoEBlocks)
cannot see both modalities when making its routing decision.  A joint gate lets
the router trade off between camera and LiDAR experts based on scene content
visible in BOTH modality summaries — e.g. route to a LiDAR expert when the
LiDAR BEV is rich (high-density scan) even if the camera BEV is average.
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
    """Joint-gate modality-specific MoE block operating on pre-fusion BEV maps.

    Args:
        cam_channels:          Camera BEV channels (e.g. 80).
        lidar_channels:        LiDAR BEV channels (e.g. 256).
        num_cam_experts:       Number of camera-specific experts.
        num_lidar_experts:     Number of LiDAR-specific experts.
        k:                     Top-k experts selected per sample (over all E).
        num_convs:             Conv layers per BEVResidualExpert.
        importance_coef:       Weight for importance loss
                               (config: moe_importance_loss_weight). Default 1e-2.
        load_coef:             Weight for load loss. Default 1e-2.
        group_balance_coef:    Weight for camera-vs-LiDAR group balance loss
                               (config: moe_group_balance_loss_weight). Default 5e-3.
        router_pool_size:      Spatial pooling size for BEVSummaryHead. Default 2.
        router_hidden_dim:     Hidden dim of BEVSummaryHead MLP. Default 128.
        router_out_dim:        Output dim per modality BEVSummaryHead. Default 64.
                               Gate sees 2 × router_out_dim (+ ctx_dim) as input.
        context_cfg:           Optional ContextEncoder config dict.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        num_cam_experts: int = 2,
        num_lidar_experts: int = 2,
        k: int = 1,
        num_convs: int = 2,
        importance_coef: float = 0.01,        # moe_importance_loss_weight
        load_coef: float = 0.01,
        group_balance_coef: float = 0.005,    # moe_group_balance_loss_weight
        router_pool_size: int = 2,
        router_hidden_dim: int = 128,
        router_out_dim: int = 64,
        context_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.num_cam_experts = num_cam_experts
        self.num_lidar_experts = num_lidar_experts
        self.num_experts = num_cam_experts + num_lidar_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef
        self.group_balance_coef = group_balance_coef

        # Expert ID ranges (fixed layout: cam first, then lidar).
        # These lists are stored on self so the routing hook can read them
        # without needing to know the expert split separately.
        self.cam_expert_ids:   List[int] = list(range(num_cam_experts))
        self.lidar_expert_ids: List[int] = list(range(num_cam_experts,
                                                      self.num_experts))

        # Camera experts process cam_bev (Cc channels).
        self.cam_experts   = make_bev_experts(num_cam_experts,   cam_channels,   num_convs)
        # LiDAR experts process lidar_bev (Cl channels).
        self.lidar_experts = make_bev_experts(num_lidar_experts, lidar_channels, num_convs)

        # Separate BEVSummaryHeads — camera and LiDAR may have different
        # channel counts so they need independent projection MLPs.
        # Both project to the same router_out_dim for symmetric gate input.
        self.cam_summary   = BEVSummaryHead(cam_channels,   router_pool_size,
                                            router_hidden_dim, router_out_dim)
        self.lidar_summary = BEVSummaryHead(lidar_channels, router_pool_size,
                                            router_hidden_dim, router_out_dim)

        ctx_dim = 0
        if context_cfg is not None:
            self.context_encoder = ContextEncoder(**context_cfg)
            ctx_dim = self.context_encoder.out_dim
        else:
            self.context_encoder = None

        # Single gate over all E = num_cam + num_lidar experts.
        # Sees [cam_summary, lidar_summary, ctx] → expert logits.
        # Replacing TopkGate with NoisyTopkGate requires changing this line only.
        self.gate = TopkGate(
            feat_dim=2 * router_out_dim,
            num_experts=self.num_experts,
            k=k,
            context_dim=ctx_dim,
        )

        # Written after every forward(); read by MoERoutingHook (Hooks A and C).
        self._moe_info: Optional[Dict[str, Any]] = None

    def forward(
        self,
        cam_bev: Tensor,
        lidar_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Apply modality-specific expert routing to pre-fusion BEV maps.

        Args:
            cam_bev:            Camera BEV feature map (B, Cc, H, W).
            lidar_bev:          LiDAR BEV feature map  (B, Cl, H, W).
            batch_input_metas:  Per-sample metadata (for context routing).

        Returns:
            cam_out:   (B, Cc, H, W) — camera BEV after expert modulation.
            lidar_out: (B, Cl, H, W) — LiDAR BEV after expert modulation.
            moe_info:  Dict with routing statistics and aux_loss.
        """
        B = cam_bev.shape[0]

        # ── Step 1: Build joint routing descriptor ────────────────────
        # Each BEVSummaryHead: (B, C, H, W) → avg+max 2×2 → MLP → (B, 64)
        cam_feat   = self.cam_summary(cam_bev)     # (B, router_out_dim)
        lidar_feat = self.lidar_summary(lidar_bev)  # (B, router_out_dim)
        feat = torch.cat([cam_feat, lidar_feat], dim=1)  # (B, 2·router_out_dim)

        ctx = None
        if self.context_encoder is not None and batch_input_metas is not None:
            ctx = self.context_encoder(batch_input_metas)  # (B, ctx_dim)

        # ── Step 2: Single gate over all E experts ─────────────────────
        gate_out = self.gate(feat, ctx)
        # gate_out.probs        : (B, E)  — softmax over all cam+lidar experts
        # gate_out.topk_idx     : (B, k)  — selected expert indices
        # gate_out.topk_weights : (B, k)  — re-normalised weights

        # ── Step 3: Modality-aware dispatch ───────────────────────────
        # Outputs start as the original BEV (identity passthrough).
        # For each selected expert, we ADD the weighted delta to the
        # appropriate modality output. Modality not touched by any selected
        # expert simply passes through unchanged.
        #
        # cam experts  → modify cam_bev  only (Cc channels)
        # lidar experts → modify lidar_bev only (Cl channels)
        #
        # For residual experts: expert_out = x + block(x)
        #   → delta = expert_out - x = block(x)
        #   → cam_out[b] = cam_bev[b] + Σ weight_j * delta_j(cam_bev[b])
        #
        # This ensures identity passthrough when no expert of that modality
        # is selected (weights sum to zero for that modality → no change).
        cam_out   = cam_bev.clone()    # (B, Cc, H, W) identity passthrough
        lidar_out = lidar_bev.clone()  # (B, Cl, H, W)

        expert_counts = torch.zeros(self.num_experts, device=cam_bev.device)

        for b in range(B):
            for j in range(self.k):
                eidx   = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_counts[eidx] += 1

                if eidx in self.cam_expert_ids:
                    # Camera expert: process cam_bev, leave lidar_bev untouched
                    exp_out = self.cam_experts[eidx](cam_bev[b:b + 1])   # (1,Cc,H,W)
                    delta   = exp_out - cam_bev[b:b + 1]                  # residual delta
                    cam_out[b] = cam_out[b] + weight * delta[0]
                else:
                    # LiDAR expert: process lidar_bev, leave cam_bev untouched
                    lidar_eidx = eidx - self.num_cam_experts
                    exp_out = self.lidar_experts[lidar_eidx](lidar_bev[b:b + 1])
                    delta   = exp_out - lidar_bev[b:b + 1]
                    lidar_out[b] = lidar_out[b] + weight * delta[0]

        # ── Step 4: Auxiliary losses ───────────────────────────────────
        # Importance and load balance over ALL experts (cam + lidar together).
        aux = importance_loss(gate_out.probs, self.importance_coef)
        aux = aux + load_loss(expert_counts, self.load_coef)

        # Group balance: soft penalty for unequal routing mass between
        # camera group and LiDAR group.  Uses gate_probs (differentiable).
        aux = aux + group_balance_loss(
            gate_out.probs,
            self.cam_expert_ids,
            self.lidar_expert_ids,
            self.group_balance_coef,
        )

        # Compute group masses for logging (Hook C).
        # These are the raw batch-level sums from the gate probability distribution.
        with torch.no_grad():
            cam_mass   = gate_out.probs[:, self.cam_expert_ids].sum().item()
            lidar_mass = gate_out.probs[:, self.lidar_expert_ids].sum().item()

        moe_info = {
            'probs':            gate_out.probs.detach(),
            'topk_idx':         gate_out.topk_idx.detach(),
            'topk_weights':     gate_out.topk_weights.detach(),  # Hook A
            'cam_expert_ids':   self.cam_expert_ids,
            'lidar_expert_ids': self.lidar_expert_ids,
            'cam_group_mass':   cam_mass,    # Hook C: camera group mass this batch
            'lidar_group_mass': lidar_mass,  # Hook C: LiDAR group mass this batch
            'aux_loss':         aux,
        }
        self._moe_info = moe_info
        return cam_out, lidar_out, moe_info

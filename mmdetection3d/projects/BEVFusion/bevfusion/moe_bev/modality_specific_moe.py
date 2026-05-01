"""Modality-specific Mixture-of-Experts block (Variant B).

ModalitySpecificMoEBlock replaces ConvFuser entirely.  Camera and LiDAR
BEV maps are processed independently by dedicated expert pools, then the
refined modality outputs are fused via a simple concat + 1×1 + 3×3 conv
projection.

Required graph (no ConvFuser anywhere)::

    cam_bev   → camera experts  ─┐
                                 ├─→ concat → 1×1 proj → 3×3 conv → fused_bev
    lidar_bev → lidar experts   ─┘

Routing definition
------------------
Separate expert pools::

    LiDAR experts:  num_lidar_experts
    Camera experts: num_cam_experts
    total experts:  num_lidar_experts + num_cam_experts

The router is a single gate over the joint pool, with ``k = 2`` so it can
select any of:

    LiDAR + LiDAR
    Camera + Camera
    LiDAR + Camera

Router descriptor::

    z_L  = lidar_summary(lidar_bev)
    z_C  = cam_summary(cam_bev)
    z_MC = torch.cat([z_C, z_L], dim=1)

    gate_out   = gate(z_MC)
    ctx_logits = context_head(z_MC)

Context-supervised routing
--------------------------
Same pattern as ``BEVMoEBlock`` and ``JointModalityMoEBlock``.  Context
labels are NOT concatenated into the gate input — a separate
``context_head`` is supervised by ``F.cross_entropy(ctx_logits, ctx_label)``
and gradients only shape ``z_MC``.  The number of experts is independent
of the number of context classes; do not force expert 0 = city, expert 1
= highway, etc.

Expert dispatch
---------------
Experts ``0 … num_cam_experts-1``     are camera experts.
Experts ``num_cam_experts … E-1``     are LiDAR experts.

For each sample, the k selected experts dispatch to their respective
modality's BEV.  Each modality starts from the original BEV (identity
passthrough) and accumulates weighted expert deltas scaled by
``residual_gain``::

    cam_out   = cam_bev   + g · Σ_{j∈cam-topk}  w_j · (expert_j(cam_bev)   − cam_bev)
    lidar_out = lidar_bev + g · Σ_{j∈lid-topk}  w_j · (expert_j(lidar_bev) − lidar_bev)

The ``w_j`` come from the gate's Shazeer top-k mixture (``Σ_j w_j = 1``
across the joint pool), so an individual modality accumulates only the
fraction of mass assigned to its experts.

Fusion
------
After expert dispatch, ``cam_out`` and ``lidar_out`` are concatenated and
fused in two steps (concat → 1×1 conv channel projection → 3×3 conv
spatial alignment).

Auxiliary losses
----------------
``importance_loss`` + ``load_loss`` + ``router_z_loss`` +
``group_balance_loss`` + ``ctx_loss_coef · ctx_loss``.

moe_info contract
-----------------
Same fields as ``BEVMoEBlock`` plus modality-group fields:
``cam_expert_ids``, ``lidar_expert_ids``, ``cam_group_mass``,
``lidar_group_mass``, ``group_balance_loss``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .bev_moe import _logit_diagnostics, _noise_diagnostics
from .losses import (group_balance_loss, importance_loss, load_loss,
                     router_z_loss)
from .routing import (BEVSummaryHead, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


@MODELS.register_module()
class ModalitySpecificMoEBlock(nn.Module):
    """Modality-specific MoE block — Variant B.

    Args:
        cam_channels:        Camera BEV channels.
        lidar_channels:      LiDAR BEV channels.
        out_channels:        Output fused BEV channels.
        num_cam_experts:     Number of camera-specific experts.
        num_lidar_experts:   Number of LiDAR-specific experts.
        k:                   Top-k experts per sample over the joint pool.
                             Default 2 (allows mixed-modality dispatch).
        num_convs:           Conv layers per BEVResidualExpert.
        importance_coef:     Weight for importance loss.
        load_coef:           Weight for load loss.
        z_loss_coef:         Weight for ``router_z_loss(clean_logits)``.
        group_balance_coef:  Weight for the camera-vs-LiDAR group balance loss.
        residual_gain:       Scalar multiplier on routed expert deltas.
        router_pool_size:    BEVSummaryHead pooling resolution.
        router_spatial_dim:  BEVSummaryHead spatial-mixer width.
        router_hidden_dim:   BEVSummaryHead MLP hidden width.
        router_out_dim:      Output dim per modality summary head.  Gate
                             sees ``2 · router_out_dim`` as input.
        context_aux_cfg:     Same as ``BEVMoEBlock.context_aux_cfg``.
        gate_type:           ``'topk'`` (default) or ``'noisy_topk'``.
        gate_cfg:            Extra kwargs forwarded to ``NoisyTopkGate``.
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
        load_coef: float = 0.002,
        z_loss_coef: float = 1e-4,
        group_balance_coef: float = 0.01,
        residual_gain: float = 1.0,
        router_pool_size: int = 4,
        router_spatial_dim: int = 128,
        router_hidden_dim: int = 256,
        router_out_dim: int = 128,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'topk',
        gate_cfg: Optional[dict] = None,
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
        self.z_loss_coef = float(z_loss_coef)
        self.group_balance_coef = group_balance_coef
        self.residual_gain = float(residual_gain)

        self.cam_expert_ids: List[int] = list(range(num_cam_experts))
        self.lidar_expert_ids: List[int] = list(
            range(num_cam_experts, self.num_experts))

        self.cam_experts = make_bev_experts(
            num_cam_experts, cam_channels, num_convs)
        self.lidar_experts = make_bev_experts(
            num_lidar_experts, lidar_channels, num_convs)

        self.cam_summary = BEVSummaryHead(
            cam_channels, router_pool_size, router_spatial_dim,
            router_hidden_dim, router_out_dim)
        self.lidar_summary = BEVSummaryHead(
            lidar_channels, router_pool_size, router_spatial_dim,
            router_hidden_dim, router_out_dim)

        joint_dim = self.cam_summary.out_dim + self.lidar_summary.out_dim

        self.context_aux_cfg: Optional[dict] = None
        self.context_head: Optional[nn.Linear] = None
        self._ctx_vocab_map: Optional[Dict[str, int]] = None
        self._ctx_target_field: Optional[str] = None
        self._ctx_loss_coef: float = 0.0
        self._ctx_label_smoothing: float = 0.0
        if context_aux_cfg is not None:
            self._build_context_head(context_aux_cfg, joint_dim)

        extra_gate_kwargs = gate_cfg or {}
        if gate_type == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=joint_dim, num_experts=self.num_experts, k=k,
                **extra_gate_kwargs)
        else:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=self.num_experts, k=k)

        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        assert gate_in == joint_dim, (
            f'ModalitySpecificMoEBlock: gate input dim ({gate_in}) must '
            f'equal sum of summary out_dims ({joint_dim}) — context vector '
            f'must NOT be concatenated into the router input.')

        # Two-step fusion (concat → 1×1 channel projection → 3×3 spatial
        # alignment).  Compensates for sub-pixel cam/LiDAR misalignment from
        # the lift-splat-shoot view transform.
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(cam_channels + lidar_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._moe_info: Optional[Dict[str, Any]] = None

    def _build_context_head(self, cfg: dict, in_dim: int) -> None:
        cfg = dict(cfg)
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "ModalitySpecificMoEBlock.context_aux_cfg must include a "
                "'target_field' (e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"ModalitySpecificMoEBlock.context_aux_cfg got unexpected "
                f"keys: {list(cfg)}")

        vocab = get_context_vocab(target_field)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        self.context_head = nn.Linear(in_dim, len(vocab))
        self._ctx_vocab_map = vocab_map
        self._ctx_target_field = target_field
        self._ctx_loss_coef = loss_coef
        self._ctx_label_smoothing = label_smoothing
        self.context_aux_cfg = dict(
            target_field=target_field, loss_coef=loss_coef,
            label_smoothing=label_smoothing, num_classes=len(vocab))

    def forward(
        self,
        cam_bev: Tensor,
        lidar_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        B = cam_bev.shape[0]

        z_C = self.cam_summary(cam_bev)
        z_L = self.lidar_summary(lidar_bev)
        z_MC = torch.cat([z_C, z_L], dim=1)

        gate_out = self.gate(z_MC)

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

        fused_bev = self.fusion_proj(
            torch.cat([cam_out, lidar_out], dim=1))

        imp_loss = importance_loss(
            gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(
            gate_out.clean_logits, gate_out.noisy_logits,
            gate_out.noise_std, self.k, self.load_coef)
        z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)
        gb_loss  = group_balance_loss(
            gate_out.full_softmax_probs,
            self.cam_expert_ids, self.lidar_expert_ids,
            self.group_balance_coef)

        ctx_loss_raw = z_MC.new_zeros(())
        ctx_loss_weighted = z_MC.new_zeros(())
        ctx_acc = z_MC.new_zeros(())
        ctx_pred_hist: List[int] = []
        ctx_label_hist: List[int] = []
        ctx_logits_mean_abs = 0.0

        if self.context_head is not None:
            if batch_input_metas is None:
                raise RuntimeError(
                    'ModalitySpecificMoEBlock: context_aux_cfg is configured '
                    'but batch_input_metas was not passed to forward().')
            ctx_logits = self.context_head(z_MC)
            ctx_labels = extract_context_labels(
                batch_input_metas, self._ctx_target_field,
                self._ctx_vocab_map, z_MC.device)
            assert ctx_labels.dtype == torch.long and ctx_labels.shape == (B,)
            ctx_loss_raw = F.cross_entropy(
                ctx_logits, ctx_labels,
                label_smoothing=self._ctx_label_smoothing)
            ctx_loss_weighted = self._ctx_loss_coef * ctx_loss_raw
            with torch.no_grad():
                pred = ctx_logits.argmax(dim=-1)
                ctx_acc = (pred == ctx_labels).float().mean()
                num_classes = self.context_aux_cfg['num_classes']
                ctx_pred_hist = torch.bincount(
                    pred, minlength=num_classes).cpu().tolist()
                ctx_label_hist = torch.bincount(
                    ctx_labels, minlength=num_classes).cpu().tolist()
                ctx_logits_mean_abs = float(ctx_logits.abs().mean().item())

        aux = imp_loss + ld_loss + z_loss + gb_loss + ctx_loss_weighted

        with torch.no_grad():
            cam_mass   = float(gate_out.full_softmax_probs[
                :, self.cam_expert_ids].sum().item())
            lidar_mass = float(gate_out.full_softmax_probs[
                :, self.lidar_expert_ids].sum().item())

        moe_info: Dict[str, Any] = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'clean_logits':         gate_out.clean_logits.detach(),
            'noisy_logits':         gate_out.noisy_logits.detach(),
            'noise_std':            (gate_out.noise_std.detach()
                                     if gate_out.noise_std is not None else None),
            'cam_expert_ids':       self.cam_expert_ids,
            'lidar_expert_ids':     self.lidar_expert_ids,
            'cam_group_mass':       cam_mass,
            'lidar_group_mass':     lidar_mass,
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
            'router_z_loss':        z_loss,
            'group_balance_loss':   gb_loss,
            'ctx_aux_loss':         (ctx_loss_raw.detach()
                                     if isinstance(ctx_loss_raw, Tensor)
                                     else ctx_loss_raw),
            'ctx_aux_loss_weighted': ctx_loss_weighted,
            'ctx_aux_acc':          ctx_acc.detach()
                                    if isinstance(ctx_acc, Tensor) else ctx_acc,
            'ctx_target_field':     self._ctx_target_field,
            'ctx_pred_hist':        ctx_pred_hist,
            'ctx_label_hist':       ctx_label_hist,
            'ctx_logits_mean_abs':  ctx_logits_mean_abs,
        }
        moe_info.update(_logit_diagnostics('clean_logits', gate_out.clean_logits))
        if gate_out.noisy_logits is not gate_out.clean_logits and \
                not torch.equal(gate_out.noisy_logits, gate_out.clean_logits):
            moe_info.update(
                _logit_diagnostics('noisy_logits', gate_out.noisy_logits))
        if gate_out.noise_std is not None:
            moe_info.update(_noise_diagnostics(gate_out.noise_std))

        self._moe_info = moe_info
        return fused_bev, moe_info

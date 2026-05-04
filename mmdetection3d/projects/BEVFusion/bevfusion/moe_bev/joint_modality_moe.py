"""Joint-modality Mixture-of-Experts block (Variant A).

JointModalityMoEBlock replaces ConvFuser entirely.  Each expert receives
both camera and LiDAR BEV maps as input and learns its own fusion strategy
(concat → 3×3 conv).  A gate routes samples to experts based on
residual-CNN summary descriptors from both modalities.

Required graph (no ConvFuser anywhere)::

    cam_bev   ─┐
               ├─→ JointModalityMoEBlock ─→ fused_bev ─→ pts_backbone
    lidar_bev ─┘

Router descriptor
-----------------
Each modality is summarised independently by a :class:`BEVResSummaryEncoder`
(stem + 3 residual blocks + global avg pool → 256-d descriptor).  The two
descriptors are concatenated before the gate — no feature-level fusion::

    z_L  = lidar_summary(lidar_bev)        # (B, out_dim)
    z_C  = cam_summary(cam_bev)            # (B, out_dim)
    z_LC = torch.cat([z_C, z_L], dim=1)   # (B, 2 · out_dim)

    gate_out   = gate(z_LC)
    ctx_logits = context_head(z_LC)

Experts then receive the full ``cam_bev`` and ``lidar_bev`` features.
Concatenating the two summaries is descriptor-level conditioning, not
feature-level fusion.

Context-supervised routing
--------------------------
Same pattern as ``BEVMoEBlock``: a separate ``context_head`` is supervised
by ``F.cross_entropy(ctx_logits, ctx_label)``.  Context labels are NEVER
concatenated into the gate input.

Dispatch strategy
-----------------
Joint experts produce a fresh fused BEV (not a residual correction), so
the dispatched output must be a proper convex combination.  Shazeer top-k
mixture weights satisfy ``Σ_j w_j = 1`` per sample, so

    out[b] = Σ_j  topk_weights[b, j] · expert_j(cam_bev[b], lidar_bev[b])

is automatically at unit scale.  ``residual_gain`` is accepted for config
parity with the other blocks but is a no-op here.

moe_info contract
-----------------
Same fields as ``BEVMoEBlock`` (see ``bev_moe.py`` docstring); no
modality-group fields (those belong to ``ModalitySpecificMoEBlock``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_moe import _logit_diagnostics, _noise_diagnostics
from .losses import importance_loss, load_loss, router_z_loss
from .routing import (BEVResSummaryEncoder, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


@MODELS.register_module()
class JointModalityExpert(nn.Module):
    """Single joint-modality expert: concatenate two BEV maps and convolve.

    Each expert has independent weights so it can learn a distinct fusion
    strategy (e.g. camera-dominant vs LiDAR-dominant).
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

    Args:
        cam_channels:        Camera BEV channels.
        lidar_channels:      LiDAR BEV channels.
        out_channels:        Output fused BEV channels.
        num_experts:         Number of fusion experts.
        k:                   Top-k experts per sample (default 2).
        importance_coef:     Weight for the Shazeer importance loss.
        load_coef:           Weight for the Shazeer load loss.
        z_loss_coef:         Weight for ``router_z_loss(clean_logits)``.
        residual_gain:       Accepted for config parity; no-op here.
        router_out_dim:      Output dim per modality BEVResSummaryEncoder.
                             Gate sees ``2 · router_out_dim`` as input.
                             Default 256.
        context_aux_cfg:     Same as ``BEVMoEBlock.context_aux_cfg``.
        gate_type:           ``'topk'`` (default) or ``'noisy_topk'``.
        gate_cfg:            Extra kwargs forwarded to ``NoisyTopkGate``.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_experts: int = 6,
        k: int = 2,
        importance_coef: float = 0.02,
        load_coef: float = 0.002,
        z_loss_coef: float = 1e-4,
        residual_gain: float = 1.0,
        router_out_dim: int = 256,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'topk',
        gate_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.k = k
        self.importance_coef = importance_coef
        self.load_coef = load_coef
        self.z_loss_coef = float(z_loss_coef)
        self.residual_gain = float(residual_gain)
        if self.residual_gain != 1.0:
            import warnings
            warnings.warn(
                f'JointModalityMoEBlock received residual_gain='
                f'{self.residual_gain} but this parameter is a no-op for '
                f'this block (joint experts produce fresh fused BEVs). '
                f'The value is silently ignored.',
                stacklevel=2)

        self.experts = nn.ModuleList([
            JointModalityExpert(cam_channels, lidar_channels, out_channels)
            for _ in range(num_experts)
        ])

        self.cam_summary = BEVResSummaryEncoder(
            channels=cam_channels, out_dim=router_out_dim)
        self.lidar_summary = BEVResSummaryEncoder(
            channels=lidar_channels, out_dim=router_out_dim)

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
                feat_dim=joint_dim, num_experts=num_experts, k=k,
                **extra_gate_kwargs)
        else:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=num_experts, k=k)

        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        assert gate_in == joint_dim, (
            f'JointModalityMoEBlock: gate input dim ({gate_in}) must equal '
            f'sum of summary out_dims ({joint_dim}) — context vector must '
            f'NOT be concatenated into the router input.')

        self._moe_info: Optional[Dict[str, Any]] = None

    def _build_context_head(self, cfg: dict, in_dim: int) -> None:
        cfg = dict(cfg)
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "JointModalityMoEBlock.context_aux_cfg must include a "
                "'target_field' (e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"JointModalityMoEBlock.context_aux_cfg got unexpected keys: "
                f"{list(cfg)}")

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
        z_LC = torch.cat([z_C, z_L], dim=1)

        gate_out = self.gate(z_LC)

        out = cam_bev.new_zeros(B, self.out_channels,
                                cam_bev.shape[2], cam_bev.shape[3])
        for b in range(B):
            sample_out = torch.zeros_like(out[b:b + 1])
            for j in range(self.k):
                eidx = gate_out.topk_idx[b, j].item()
                weight = gate_out.topk_weights[b, j]
                expert_out = self.experts[eidx](
                    cam_bev[b:b + 1], lidar_bev[b:b + 1])
                sample_out = sample_out + weight * expert_out
            out[b] = sample_out[0]

        imp_loss = importance_loss(
            gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(
            gate_out.clean_logits, gate_out.noisy_logits,
            gate_out.noise_std, self.k, self.load_coef)
        z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)

        ctx_loss_raw = z_LC.new_zeros(())
        ctx_loss_weighted = z_LC.new_zeros(())
        ctx_acc = z_LC.new_zeros(())
        ctx_pred_hist: List[int] = []
        ctx_label_hist: List[int] = []
        ctx_logits_mean_abs = 0.0

        if self.context_head is not None:
            if batch_input_metas is None:
                raise RuntimeError(
                    'JointModalityMoEBlock: context_aux_cfg is configured '
                    'but batch_input_metas was not passed to forward().')
            ctx_logits = self.context_head(z_LC)
            ctx_labels = extract_context_labels(
                batch_input_metas, self._ctx_target_field,
                self._ctx_vocab_map, z_LC.device)
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

        aux = imp_loss + ld_loss + z_loss + ctx_loss_weighted

        moe_info: Dict[str, Any] = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'clean_logits':         gate_out.clean_logits.detach(),
            'noisy_logits':         gate_out.noisy_logits.detach(),
            'noise_std':            (gate_out.noise_std.detach()
                                     if gate_out.noise_std is not None else None),
            'aux_loss':             aux,
            'importance_loss':      imp_loss,
            'load_loss':            ld_loss,
            'router_z_loss':        z_loss,
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
        return out, moe_info

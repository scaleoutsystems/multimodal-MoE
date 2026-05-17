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
Two separate expert pools share one dense gate::

    Camera experts: num_cam_experts   (default 2)
    LiDAR experts:  num_lidar_experts (default 2)
    Total experts:  E = num_cam_experts + num_lidar_experts (default 4)

The router is a single gate over the joint pool of size ``E``.  Under
the production dense routing it runs every expert, weighted by
``full_softmax_probs``; the legacy top-k branch is kept only for
ablations.

Each modality is summarised by a :class:`BEVResSummaryEncoder` (stem +
3 residual blocks + avg+max grid pool → ``router_out_dim``-d
descriptor)::

    z_C  = cam_summary(cam_bev)
    z_L  = lidar_summary(lidar_bev)
    z_MC = torch.cat([z_C, z_L], dim=1)

    z_gate     = z_MC.detach()  if gate_input_detach else z_MC
    gate_out   = gate(z_gate)
    ctx_logits = context_head(z_MC)        # always full-grad through z_MC

Context-supervised routing
--------------------------
Same pattern as :class:`BEVMoEBlock` (run 4577584): a separate
``context_head`` MLP is supervised by ``F.cross_entropy(ctx_logits,
ctx_label)`` with ``loss_type='weighted_ce'`` (inverse-frequency class
weights) and label smoothing.  The road-type label is never
concatenated into the gate input.

Expert dispatch (dense soft-MoE)
--------------------------------
With ``gate_type='dense'`` (default), the gate produces a softmax over
all ``E`` experts and every expert contributes to its modality's
output via the convex residual::

    probs = softmax(W_gate(z_gate), dim=-1)        # (B, E)

    cam_out   = cam_bev   + g · Σ_{e∈cam}   probs[:,e]·(expert_e(cam_bev) − cam_bev)
    lidar_out = lidar_bev + g · Σ_{e∈lidar} probs[:,e]·(expert_e(lidar_bev) − lidar_bev)

where ``g = residual_gain`` and the modality groups are::

    cam_expert_ids   = [0 … num_cam_experts-1]
    lidar_expert_ids = [num_cam_experts … E-1]

The weights ``probs`` are dense (every sample touches every expert) so
``Σ_e p_e = 1`` per sample but ``Σ_{e∈cam} p_e`` and
``Σ_{e∈lidar} p_e`` can drift away from 0.5 — which is where
``group_balance_loss`` enters (see below).

Fusion
------
After expert dispatch, ``cam_out`` and ``lidar_out`` are concatenated
and fused in two steps (concat → 1×1 conv channel projection → 3×3
conv spatial alignment).

Auxiliary losses
----------------
``importance_coef·importance_loss``
``+ load_coef·load_loss``           (0 under dense routing)
``+ z_loss_coef·router_z_loss``
``+ group_balance_coef·group_balance_loss``
``+ ctx_loss_coef·CE(ctx_logits, road_type)``       (weighted_ce)

moe_info contract
-----------------
Same fields as :class:`BEVMoEBlock` plus modality-group fields:
``cam_expert_ids``, ``lidar_expert_ids``, ``cam_group_mass``,
``lidar_group_mass``, ``group_balance_loss``.  ``topk_idx`` /
``topk_weights`` are populated even under dense routing (experts
sorted by full softmax probability) so existing diagnostic hooks
work unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_experts import make_bev_experts
from .bev_moe import (BEVMoEBlock, _logit_diagnostics, _noise_diagnostics,
                      focal_ce_loss)
from .losses import (group_balance_loss, importance_loss, load_loss,
                     router_z_loss)
from .routing import (BEVResSummaryEncoder, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


@MODELS.register_module()
class ModalitySpecificMoEBlock(nn.Module):
    """Modality-specific MoE block — Variant B.

    Args:
        cam_channels:        Camera BEV channels.
        lidar_channels:      LiDAR BEV channels.
        out_channels:        Output fused BEV channels.
        num_cam_experts:     Number of camera-specific experts.  Default 2.
        num_lidar_experts:   Number of LiDAR-specific experts.  Default 2.
                             ``E = num_cam_experts + num_lidar_experts``
                             defaults to 4 to match the LiDAR-only MoE
                             4577584 expert count.
        k:                   Top-k experts per sample.  Ignored when
                             ``gate_type='dense'`` (effective k is forced
                             to ``E`` so every expert always runs).
        num_convs:           Conv layers per legacy
                             :class:`BEVResidualExpert`.  Ignored when
                             ``expert_type='bottleneck'`` (default).
        expert_type:         ``'bottleneck'`` (default,
                             :class:`BEVBottleneckResidualExpert` —
                             identity-init residual adapter, ~12×
                             cheaper than the legacy full-channel
                             expert under dense dispatch) or ``'full'``
                             (legacy :class:`BEVResidualExpert`).
        expert_hidden_channels:
                             Bottleneck width for the bottleneck expert.
                             Default 128.  Ignored when
                             ``expert_type='full'``.
        importance_coef:     Weight for importance loss.
        load_coef:           Weight for load loss.  Set to 0.0 under
                             dense routing.
        z_loss_coef:         Weight for ``router_z_loss(clean_logits)``.
        group_balance_coef:  Weight for the camera-vs-LiDAR group
                             balance loss (kept active under dense
                             routing to discourage modality collapse).
        residual_gain:       Scalar multiplier on routed expert deltas.
        router_out_dim:      Output dim per modality BEVResSummaryEncoder.
                             Gate sees ``2 · router_out_dim`` as input.
                             Default 128 (→ 256-d joint descriptor,
                             matching the LiDAR-only MoE gate input).
        context_aux_cfg:     Same as ``BEVMoEBlock.context_aux_cfg``.
        gate_type:           ``'dense'`` (default, dense soft-MoE) or
                             ``'topk'`` / ``'noisy_topk'`` (legacy).
        gate_cfg:            Extra kwargs forwarded to the gate.
        gate_input_detach:   If True (default) the gate consumes
                             ``z_MC.detach()`` (mirrors BEVMoEBlock).
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_cam_experts: int = 2,
        num_lidar_experts: int = 2,
        k: Optional[int] = None,
        num_convs: int = 2,
        importance_coef: float = 0.005,
        load_coef: float = 0.0,
        z_loss_coef: float = 0.002,
        group_balance_coef: float = 0.002,
        residual_gain: float = 1.0,
        router_out_dim: int = 128,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'dense',
        gate_cfg: Optional[dict] = None,
        gate_input_detach: bool = True,
        expert_type: str = 'bottleneck',
        expert_hidden_channels: int = 128,
    ):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.num_cam_experts = num_cam_experts
        self.num_lidar_experts = num_lidar_experts
        self.num_experts = num_cam_experts + num_lidar_experts
        self.importance_coef = float(importance_coef)
        self.load_coef = float(load_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.group_balance_coef = float(group_balance_coef)
        self.residual_gain = float(residual_gain)
        self.gate_input_detach = bool(gate_input_detach)
        if self.gate_input_detach and context_aux_cfg is None:
            raise ValueError(
                'ModalitySpecificMoEBlock: gate_input_detach=True with '
                'context_aux_cfg=None means the BEVResSummaryEncoder '
                'branches receive no gradient from any source.  Either '
                'set gate_input_detach=False or provide a context_aux_cfg.')

        self.cam_expert_ids: List[int] = list(range(num_cam_experts))
        self.lidar_expert_ids: List[int] = list(
            range(num_cam_experts, self.num_experts))

        self.expert_type = str(expert_type).lower()
        self.expert_hidden_channels = int(expert_hidden_channels)
        self.cam_experts = make_bev_experts(
            num_cam_experts, cam_channels,
            num_convs=num_convs,
            expert_type=self.expert_type,
            hidden_channels=self.expert_hidden_channels)
        self.lidar_experts = make_bev_experts(
            num_lidar_experts, lidar_channels,
            num_convs=num_convs,
            expert_type=self.expert_type,
            hidden_channels=self.expert_hidden_channels)

        self.cam_summary = BEVResSummaryEncoder(
            channels=cam_channels, out_dim=router_out_dim)
        self.lidar_summary = BEVResSummaryEncoder(
            channels=lidar_channels, out_dim=router_out_dim)

        joint_dim = self.cam_summary.out_dim + self.lidar_summary.out_dim

        self.context_aux_cfg: Optional[dict] = None
        self.context_head: Optional[nn.Module] = None
        self._ctx_vocab_map: Optional[Dict[str, int]] = None
        self._ctx_target_field: Optional[str] = None
        self._ctx_loss_coef: float = 0.0
        self._ctx_loss_type: str = 'weighted_ce'
        self._ctx_focal_gamma: float = 2.0
        self._ctx_label_smoothing: float = 0.0
        self._ctx_class_weights_list: Optional[List[float]] = None
        if context_aux_cfg is not None:
            self._build_context_head(context_aux_cfg, joint_dim)

        # ── Gate construction ─────────────────────────────────────────
        # Dense soft-MoE collapses to a ``TopkGate(k=E)``: the gate's
        # ``full_softmax_probs`` carry all the dispatch mass and
        # ``topk_idx``/``topk_weights`` are simply the experts sorted by
        # probability, kept populated so downstream hooks work unchanged.
        gate_type_norm = str(gate_type).lower()
        if gate_type_norm not in ('topk', 'noisy_topk', 'dense'):
            raise ValueError(
                "ModalitySpecificMoEBlock.gate_type must be 'topk', "
                f"'noisy_topk' or 'dense', got '{gate_type}'.")
        self._dense_dispatch = (gate_type_norm == 'dense')
        if self._dense_dispatch:
            self.k = self.num_experts
        else:
            self.k = int(k) if k is not None else 2

        extra_gate_kwargs = gate_cfg or {}
        if self._dense_dispatch:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=self.num_experts,
                k=self.num_experts, **extra_gate_kwargs)
        elif gate_type_norm == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=joint_dim, num_experts=self.num_experts, k=self.k,
                **extra_gate_kwargs)
        else:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=self.num_experts, k=self.k,
                **extra_gate_kwargs)
        self.gate_type = gate_type_norm

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
        """Configure the context auxiliary classifier (mirrors BEVMoEBlock).

        See :meth:`JointModalityMoEBlock._build_context_head` for the full
        list of accepted cfg keys.  Builds the same MLP head structure as
        :class:`BEVMoEBlock` so all three multimodal variants share the
        LiDAR-only run's context supervision behaviour.
        """
        cfg = dict(cfg)
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "ModalitySpecificMoEBlock.context_aux_cfg must include a "
                "'target_field' (e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        loss_type = str(cfg.pop('loss_type', 'weighted_ce')).lower()
        if loss_type not in ('weighted_ce', 'ce', 'focal'):
            raise ValueError(
                "ModalitySpecificMoEBlock.context_aux_cfg: loss_type must "
                f"be 'weighted_ce', 'ce' or 'focal', got '{loss_type}'.")
        class_weights_cfg = cfg.pop('class_weights', None)
        focal_gamma = float(cfg.pop('focal_gamma', 2.0))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"ModalitySpecificMoEBlock.context_aux_cfg got unexpected "
                f"keys: {list(cfg)}")

        vocab = get_context_vocab(target_field)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        num_classes = len(vocab)

        self.context_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(in_dim),
            nn.Dropout(0.2),
            nn.Linear(in_dim, num_classes),
        )

        class_weights_list: Optional[List[float]] = None
        if loss_type == 'weighted_ce':
            class_weights_list = BEVMoEBlock._resolve_class_weights(
                class_weights_cfg, target_field, num_classes)
            w = torch.tensor(class_weights_list, dtype=torch.float32)
            self.register_buffer('_ctx_class_weights', w, persistent=False)
        elif class_weights_cfg is not None:
            raise ValueError(
                "ModalitySpecificMoEBlock.context_aux_cfg: 'class_weights' "
                "is only meaningful with loss_type='weighted_ce' (got "
                f"'{loss_type}').")

        self._ctx_vocab_map = vocab_map
        self._ctx_target_field = target_field
        self._ctx_loss_coef = loss_coef
        self._ctx_loss_type = loss_type
        self._ctx_focal_gamma = focal_gamma
        self._ctx_label_smoothing = label_smoothing
        self._ctx_class_weights_list = class_weights_list
        self.context_aux_cfg = dict(
            target_field=target_field,
            loss_coef=loss_coef,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            class_weights=(list(class_weights_list)
                           if class_weights_list is not None else None),
            num_classes=num_classes,
        )

    def forward(
        self,
        cam_bev: Tensor,
        lidar_bev: Tensor,
        batch_input_metas: Optional[List[dict]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        # NaN/Inf guard on the incoming BEV feature maps.  See the
        # equivalent comment in BEVMoEBlock.forward for the full
        # rationale.
        if not torch.isfinite(cam_bev).all():
            import logging as _logging
            _logging.getLogger('mmengine').warning(
                'ModalitySpecificMoEBlock: NaN/Inf in cam_bev — '
                'replacing with zeros before MoE forward.')
            cam_bev = torch.nan_to_num(
                cam_bev, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(lidar_bev).all():
            import logging as _logging
            _logging.getLogger('mmengine').warning(
                'ModalitySpecificMoEBlock: NaN/Inf in lidar_bev — '
                'replacing with zeros before MoE forward.')
            lidar_bev = torch.nan_to_num(
                lidar_bev, nan=0.0, posinf=0.0, neginf=0.0)

        B = cam_bev.shape[0]

        # ── Step 1: Joint descriptor ──────────────────────────────────
        z_C  = self.cam_summary(cam_bev)
        z_L  = self.lidar_summary(lidar_bev)
        z_MC = torch.cat([z_C, z_L], dim=1)
        z_gate = z_MC.detach() if self.gate_input_detach else z_MC

        gate_out = self.gate(z_gate)

        # ── Step 2: Dispatch ──────────────────────────────────────────
        # Dense path: every camera expert contributes a weighted residual
        # to ``cam_out``; every LiDAR expert contributes a weighted
        # residual to ``lidar_out``.  Each expert is called once on the
        # full batch (preserves BN batch statistics) and weighted by its
        # softmax probability:
        #
        #   cam_out   = cam_bev   + g · Σ_{e∈cam}   p_e · (cam_e(cam_bev)   − cam_bev)
        #   lidar_out = lidar_bev + g · Σ_{e∈lidar} p_e · (lidar_e(lidar_bev) − lidar_bev)
        if self._dense_dispatch:
            probs = gate_out.full_softmax_probs                          # (B, E)
            cam_delta_sum = torch.zeros_like(cam_bev)
            for j, eidx in enumerate(self.cam_expert_ids):
                local = self.cam_experts[j](cam_bev) - cam_bev
                w = probs[:, eidx].view(-1, 1, 1, 1).to(local.dtype)
                cam_delta_sum = cam_delta_sum + w * local
            lidar_delta_sum = torch.zeros_like(lidar_bev)
            for j, eidx in enumerate(self.lidar_expert_ids):
                local = self.lidar_experts[j](lidar_bev) - lidar_bev
                w = probs[:, eidx].view(-1, 1, 1, 1).to(local.dtype)
                lidar_delta_sum = lidar_delta_sum + w * local
            cam_out   = cam_bev   + self.residual_gain * cam_delta_sum
            lidar_out = lidar_bev + self.residual_gain * lidar_delta_sum
        else:
            # Legacy top-k / noisy-top-k path: per-sample dispatch.
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

        # ── Step 3: Auxiliary losses ──────────────────────────────────
        imp_loss = importance_loss(
            gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(
            gate_out.clean_logits, gate_out.noisy_logits,
            gate_out.noise_std, self.k, self.load_coef)
        z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)
        # Group balance is computed from the soft router belief — works
        # identically for dense and top-k routing because both share the
        # same ``full_softmax_probs`` tensor.
        gb_loss  = group_balance_loss(
            gate_out.full_softmax_probs,
            self.cam_expert_ids, self.lidar_expert_ids,
            self.group_balance_coef)

        # Context auxiliary classification ----------------------------------
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
            if not torch.isfinite(ctx_logits).all():
                import logging as _logging
                _logging.getLogger('mmengine').warning(
                    'ModalitySpecificMoEBlock: NaN/Inf in ctx_logits — '
                    'replacing with zeros before context-aux CE.')
                ctx_logits = torch.nan_to_num(
                    ctx_logits, nan=0.0, posinf=0.0, neginf=0.0)
            ctx_labels = extract_context_labels(
                batch_input_metas, self._ctx_target_field,
                self._ctx_vocab_map, z_MC.device)
            assert ctx_labels.dtype == torch.long and ctx_labels.shape == (B,)
            if self._ctx_loss_type == 'focal':
                ctx_loss_raw = focal_ce_loss(
                    ctx_logits, ctx_labels, gamma=self._ctx_focal_gamma)
            elif self._ctx_loss_type == 'weighted_ce':
                w = self._ctx_class_weights.to(
                    device=ctx_logits.device, dtype=ctx_logits.dtype)
                ctx_loss_raw = F.cross_entropy(
                    ctx_logits, ctx_labels,
                    weight=w,
                    label_smoothing=self._ctx_label_smoothing)
            else:  # 'ce'
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

        # ── Step 4: Diagnostics / moe_info ────────────────────────────
        with torch.no_grad():
            # Per-sample group masses then batch mean → comparable across
            # batch sizes.  Stored as Python floats for logging.
            cam_mass   = float(gate_out.full_softmax_probs[
                :, self.cam_expert_ids].sum(dim=1).mean().item())
            lidar_mass = float(gate_out.full_softmax_probs[
                :, self.lidar_expert_ids].sum(dim=1).mean().item())

        clean_topk_idx_detached = (
            gate_out.clean_topk_idx.detach()
            if gate_out.clean_topk_idx is not None
            else gate_out.topk_idx.detach())

        moe_info: Dict[str, Any] = {
            'full_softmax_probs':   gate_out.full_softmax_probs.detach(),
            'sparse_softmax_probs': gate_out.sparse_softmax_probs.detach(),
            'topk_idx':             gate_out.topk_idx.detach(),
            'topk_weights':         gate_out.topk_weights.detach(),
            'clean_topk_idx':       clean_topk_idx_detached,
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
            'ctx_loss_type':        self._ctx_loss_type,
            'ctx_class_weights':    (list(self._ctx_class_weights_list)
                                     if self._ctx_class_weights_list is not None
                                     else None),
            'focal_gamma':          self._ctx_focal_gamma,
            'gate_feat_dim':        (self.cam_summary.out_dim
                                     + self.lidar_summary.out_dim),
            'z_ctx_detached_for_gate': self.gate_input_detach,
            'gate_input':           ('z_MC_detach'
                                     if self.gate_input_detach else 'z_MC'),
            'context_head_type':    ('mlp'
                                     if self.context_head is not None
                                     else 'none'),
            'gate_type':            self.gate_type,
            'dense_dispatch':       self._dense_dispatch,
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

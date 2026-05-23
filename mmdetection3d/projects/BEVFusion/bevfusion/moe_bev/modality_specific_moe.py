"""Modality-specific Mixture-of-Experts block (Variant B).

Symmetric output-space modality-specific MoE.  Camera is projected
from 80 channels to the shared 256-channel fused BEV space; LiDAR is
already 256 channels and is used directly.  Both expert pools — built
from the existing :class:`BEVBottleneckResidualExpert` factory —
operate in the common 256-channel output space.  A single flat gate
over all ``E = num_cam_experts + num_lidar_experts`` experts weights
both the *direct* modality contributions and the *routed* expert
residual deltas.  Routing is still flat — one gate over all experts,
no separate modality gate, no hierarchical routing — but unlike the
old concat-then-fuse design, the gate now *directly* controls the
camera vs LiDAR contribution to the fused BEV.

Required graph (no ConvFuser anywhere)::

    cam_bev   ─→ cam_direct_proj ─→ cam_direct                     ─┐
                                                                    │
    lidar_bev ──────────────────────────────────────── lidar_direct ┤
                                                                    │
        m_C = Σ_(e∈cam)   p_e                                       │
        m_L = Σ_(e∈lidar) p_e                                       │
        direct_mix = m_C · cam_direct + m_L · lidar_direct          │
                                                                    │
    cam_direct   ─→ cam_experts   ─→ Σ_(e∈cam)   p_e · (E(x) − x) ──┤
    lidar_direct ─→ lidar_experts ─→ Σ_(e∈lidar) p_e · (E(x) − x) ──┤
                                                                    │
              fused = refine(direct_mix + residual_gain · delta_sum)
                                                                    ▼

Shapes::

    cam_bev:      (B,  80, H, W)
    lidar_bev:    (B, 256, H, W)
    cam_direct:   (B, 256, H, W)
    lidar_direct: (B, 256, H, W)
    fused_bev:    (B, 256, H, W)

Symmetry, not LiDAR-anchoring
-----------------------------
Camera and LiDAR enter the fused BEV through structurally identical
paths: one direct projection (1×1 → BN → ReLU, no-op channel-wise for
LiDAR but kept implicit) plus a pool of bottleneck experts that emit
identity-init residuals.  Modality dominance is decided by the gate
and detection gradients during training, not hard-coded at
initialisation.  This is *not* a LiDAR-anchored design — there is no
``lidar_base_proj`` learned base path.

Routing definition
------------------
Two separate expert pools share one dense gate::

    Camera experts: num_cam_experts   (default 2)
    LiDAR experts:  num_lidar_experts (default 2)
    Total experts:  E = num_cam_experts + num_lidar_experts (default 4)

Each modality is summarised by a :class:`BEVResSummaryEncoder` (stem +
3 residual blocks + avg+max grid pool → ``router_out_dim``-d
descriptor)::

    z_C  = cam_summary(cam_bev)
    z_L  = lidar_summary(lidar_bev)
    z_MC = torch.cat([z_C, z_L], dim=1)

    z_gate     = z_MC.detach() if gate_input_detach else z_MC
    gate_out   = gate(z_gate)
    ctx_logits = context_head(z_MC)        # always full-grad through z_MC

The gate produces a softmax over all ``E`` experts; ``cam_mass`` and
``lidar_mass`` are the per-sample sums of softmax probability over
each modality's expert ids.

Context-supervised routing
--------------------------
Same pattern as :class:`BEVMoEBlock`: a separate ``context_head`` MLP
is supervised by ``F.cross_entropy(ctx_logits, ctx_label)`` with
``loss_type='weighted_ce'`` (inverse-frequency class weights) and
label smoothing.  The road-type label is never concatenated into the
gate input.

Expert dispatch (dense soft-MoE)
--------------------------------
With ``gate_type='dense'`` (default), the gate produces a softmax over
all ``E`` experts.  Camera experts run on ``cam_direct``, LiDAR
experts on ``lidar_direct``; both expert pools operate in the shared
256-channel width::

    probs   = softmax(W_gate(z_gate), dim=-1)        # (B, E)
    m_C     = probs[:, cam_expert_ids].sum(dim=1)    # (B,)
    m_L     = probs[:, lidar_expert_ids].sum(dim=1)  # (B,)

    cam_direct   = cam_direct_proj(cam_bev)
    lidar_direct = lidar_bev

    direct_mix = m_C · cam_direct + m_L · lidar_direct

    delta_sum  = Σ_(e∈cam)   p_e · (cam_e(cam_direct)     − cam_direct)
               + Σ_(e∈lidar) p_e · (lidar_e(lidar_direct) − lidar_direct)

    fused      = refine(direct_mix + residual_gain · delta_sum)

Equivalent conceptual form::

    fused = refine(
        Σ_(e∈cam)   p_e · (cam_direct   + g · (cam_e(cam_direct)     − cam_direct))
      + Σ_(e∈lidar) p_e · (lidar_direct + g · (lidar_e(lidar_direct) − lidar_direct))
    )

Identity-at-init behaviour
--------------------------
Each :class:`BEVBottleneckResidualExpert` is initialised so that its
adapter branch emits an exact-zero residual at step 0 (the final BN's
affine parameters are zero-initialised).  Therefore::

    expert(x) = ReLU(x + 0) = ReLU(x)   at step 0
    delta(x)  = expert(x) − x = 0       whenever x ≥ 0

Both ``cam_direct`` (post-ReLU from the 1×1 projection) and
``lidar_direct`` (post-ReLU from the upstream view-transform/SECOND
neck) are non-negative, so ``delta_sum ≈ 0`` at step 0 and::

    fused ≈ refine(m_C · cam_direct + m_L · lidar_direct)

Camera has an active direct path from the very first iteration — it
is NOT zero at init — so the camera contribution does not need to
"grow from zero".

Auxiliary losses
----------------
``importance_coef·importance_loss``
``+ load_coef·load_loss``           (0 under dense routing)
``+ z_loss_coef·router_z_loss``
``+ group_balance_coef·group_balance_loss``
``+ ctx_loss_coef·CE(ctx_logits, road_type)``       (weighted_ce)

The group balance loss remains active (default coef 0.004 in the
production config) because the implementation is now symmetric and we
want to discourage early routing collapse to a single modality group.
If one modality is genuinely stronger, the gate can still drift —
0.004 is small enough that the optimisation can override it.

moe_info contract
-----------------
Same fields as :class:`BEVMoEBlock` plus modality-group fields:
``cam_expert_ids``, ``lidar_expert_ids``, ``cam_group_mass``,
``lidar_group_mass``, ``group_balance_loss``.  ``topk_idx`` /
``topk_weights`` are populated even under dense routing (experts
sorted by full softmax probability) so existing diagnostic hooks
work unchanged.  Four new fields advertise the symmetric design:

* ``modality_specific_design = 'symmetric_output_space_bottleneck'``
* ``cam_direct_channels      = out_channels``
* ``lidar_direct_channels    = out_channels``
* ``expert_input_channels    = out_channels``
* ``expert_output_channels   = out_channels``
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
    """Symmetric output-space modality-specific MoE block (Variant B).

    See the module docstring for the full architecture diagram and the
    identity-at-init analysis.

    Args:
        cam_channels:        Camera BEV channels.  Default 80.
        lidar_channels:      LiDAR BEV channels.   Default 256.  Must
                             equal ``out_channels`` (LiDAR uses the
                             direct feature path without projection).
        out_channels:        Fused output channels.  Also the shared
                             channel width that every expert runs in.
                             Default 256.
        num_cam_experts:     Number of camera-specific experts.
                             Default 2.
        num_lidar_experts:   Number of LiDAR-specific experts.
                             Default 2.
        k:                   Top-k experts per sample.  Ignored when
                             ``gate_type='dense'``.  Legacy non-dense
                             paths are NOT implemented and raise
                             ``NotImplementedError``.
        num_convs:           Number of Conv→BN layers per legacy
                             :class:`BEVResidualExpert`; ignored when
                             ``expert_type='bottleneck'`` (default).
                             Kept in the signature so the existing
                             ``make_bev_experts`` factory call works
                             unchanged.
        importance_coef:     Weight for importance loss.
        load_coef:           Weight for load loss.  Set to 0.0 under
                             dense routing.
        z_loss_coef:         Weight for ``router_z_loss(clean_logits)``.
        group_balance_coef:  Weight for the camera-vs-LiDAR group
                             balance loss.  Kept active (default
                             0.004 in the production config) under the
                             symmetric design to avoid early routing
                             collapse to a single modality group.
        residual_gain:       Scalar multiplier on the summed routed
                             expert delta.  Default 1.0.
        router_out_dim:      Output dim per modality
                             ``BEVResSummaryEncoder``.  Gate sees
                             ``2 · router_out_dim`` as input.
                             Default 128 (→ 256-d joint descriptor).
        context_aux_cfg:     Same as ``BEVMoEBlock.context_aux_cfg``.
        gate_type:           ``'dense'`` (default, dense soft-MoE).
                             Legacy ``'topk'`` / ``'noisy_topk'`` are
                             not implemented in this variant.
        gate_cfg:            Extra kwargs forwarded to the gate.
        gate_input_detach:   If True (default) the gate consumes
                             ``z_MC.detach()`` (mirrors BEVMoEBlock).
        expert_type:         ``'bottleneck'`` (default) or ``'full'``.
                             Passed straight to
                             :func:`make_bev_experts`.
        expert_hidden_channels:
                             Bottleneck width.  Default 128.  Ignored
                             when ``expert_type='full'``.
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
        group_balance_coef: float = 0.004,
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
        self.cam_channels = int(cam_channels)
        self.lidar_channels = int(lidar_channels)
        self.out_channels = int(out_channels)
        self.num_cam_experts = int(num_cam_experts)
        self.num_lidar_experts = int(num_lidar_experts)
        self.num_experts = self.num_cam_experts + self.num_lidar_experts
        self.importance_coef = float(importance_coef)
        self.load_coef = float(load_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.group_balance_coef = float(group_balance_coef)
        self.residual_gain = float(residual_gain)
        self.gate_input_detach = bool(gate_input_detach)
        self.expert_type = str(expert_type).lower()
        self.expert_hidden_channels = int(expert_hidden_channels)

        if self.gate_input_detach and context_aux_cfg is None:
            raise ValueError(
                'ModalitySpecificMoEBlock: gate_input_detach=True with '
                'context_aux_cfg=None means the BEVResSummaryEncoder '
                'branches receive no gradient from any source.  Either '
                'set gate_input_detach=False or provide a context_aux_cfg.')

        # Symmetric design requires LiDAR to be already at the shared
        # output width so the direct path is a no-op channel-wise.
        # We deliberately do NOT add a learned ``lidar_base_proj`` —
        # that would re-introduce a LiDAR-privileged base path.
        if self.lidar_channels != self.out_channels:
            raise ValueError(
                'Symmetric output-space ModalitySpecificMoEBlock expects '
                'lidar_channels == out_channels so LiDAR can use the direct '
                'feature path without projection.  Got '
                f'lidar_channels={self.lidar_channels}, '
                f'out_channels={self.out_channels}.')

        self.cam_expert_ids: List[int] = list(range(self.num_cam_experts))
        self.lidar_expert_ids: List[int] = list(
            range(self.num_cam_experts, self.num_experts))

        # ── Camera direct projection (the only modality-specific proj) ───
        # Brings cam_bev from cam_channels (80) into the shared
        # out_channels (256) width so the rest of the block (experts,
        # direct mix, refine) can operate in a single channel space.
        # NOT zero-initialised: camera must have an active direct path
        # from step 0; the only identity-at-init contract comes from the
        # experts' adapter branches (BEVBottleneckResidualExpert's
        # zero-init final BN), which together with x ≥ 0 inputs makes
        # delta_sum ≈ 0 at init.
        self.cam_direct_proj = nn.Sequential(
            nn.Conv2d(self.cam_channels, self.out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # ── Modality expert pools — shared output-space width ─────────
        # Both pools operate on tensors of shape (B, out_channels, H, W):
        # camera experts consume ``cam_direct``, LiDAR experts consume
        # ``lidar_direct``.  We reuse the existing
        # :class:`BEVBottleneckResidualExpert` factory so we don't
        # duplicate expert code; both pools are constructed with
        # ``channels=out_channels`` (not cam_channels / lidar_channels).
        self.cam_experts = make_bev_experts(
            self.num_cam_experts,
            channels=self.out_channels,
            num_convs=num_convs,
            expert_type=self.expert_type,
            hidden_channels=self.expert_hidden_channels,
        )
        self.lidar_experts = make_bev_experts(
            self.num_lidar_experts,
            channels=self.out_channels,
            num_convs=num_convs,
            expert_type=self.expert_type,
            hidden_channels=self.expert_hidden_channels,
        )

        # ── Post-mixture refinement ───────────────────────────────────
        # Single 3×3 conv smooths over (direct_mix + g · delta_sum).
        # Replaces the old concat → 1×1 → 3×3 fuser; no 1×1 needed
        # because direct_mix and delta_sum already live in the shared
        # 256-channel space.
        self.refine = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # ── Per-modality routing descriptors ──────────────────────────
        self.cam_summary = BEVResSummaryEncoder(
            channels=self.cam_channels, out_dim=router_out_dim)
        self.lidar_summary = BEVResSummaryEncoder(
            channels=self.lidar_channels, out_dim=router_out_dim)

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
        # Shape contract — lightweight asserts up front.
        assert cam_bev.dim() == 4 and lidar_bev.dim() == 4, (
            f'ModalitySpecificMoEBlock: cam_bev/lidar_bev must be 4-D BEV '
            f'tensors, got cam_bev.dim()={cam_bev.dim()} and '
            f'lidar_bev.dim()={lidar_bev.dim()}.')
        assert cam_bev.shape[0] == lidar_bev.shape[0], (
            f'ModalitySpecificMoEBlock: batch dims must match — '
            f'cam_bev.shape[0]={cam_bev.shape[0]} vs '
            f'lidar_bev.shape[0]={lidar_bev.shape[0]}.')
        assert cam_bev.shape[-2:] == lidar_bev.shape[-2:], (
            f'ModalitySpecificMoEBlock: spatial dims must match — '
            f'cam_bev.shape[-2:]={tuple(cam_bev.shape[-2:])} vs '
            f'lidar_bev.shape[-2:]={tuple(lidar_bev.shape[-2:])}.')
        assert lidar_bev.shape[1] == self.out_channels, (
            f'ModalitySpecificMoEBlock: lidar_bev.shape[1]'
            f'={lidar_bev.shape[1]} must equal out_channels='
            f'{self.out_channels} (symmetric design uses lidar_bev '
            f'directly as lidar_direct without projection).')

        B = cam_bev.shape[0]

        # ── Step 1: Joint descriptor (fp32 routing path) ──────────────
        # See ``BEVMoEBlock.forward`` Step 1–2 comment for the full
        # diagnosis of why the routing block must run in fp32 and why
        # the previous ``nan_to_num`` masking was removed: under fp16
        # autocast the summary encoders produced NaN/Inf in the
        # descriptors which were masked to zero, killing the autograd
        # gradient through them and freezing the router at random
        # init.  Promoting this small auxiliary path to fp32 removes
        # the only place where this fault can be introduced; any
        # genuinely non-finite value now propagates to the total loss
        # so ``GradScaler`` / ``_repair_bn_stats`` can recover.
        with torch.autocast('cuda', enabled=False):
            z_C  = self.cam_summary(cam_bev.float())
            z_L  = self.lidar_summary(lidar_bev.float())
            z_MC = torch.cat([z_C, z_L], dim=1)
            z_gate = z_MC.detach() if self.gate_input_detach else z_MC
            gate_out = self.gate(z_gate)

        # ── Step 2: Dispatch ──────────────────────────────────────────
        # Dense path — symmetric output-space mixture:
        #
        #   cam_direct   = cam_direct_proj(cam_bev)
        #   lidar_direct = lidar_bev
        #
        #   m_C = probs[:, cam_expert_ids].sum(dim=1)
        #   m_L = probs[:, lidar_expert_ids].sum(dim=1)
        #
        #   direct_mix = m_C · cam_direct + m_L · lidar_direct
        #
        #   delta_sum  = Σ_(e∈cam)   p_e · (cam_e(cam_direct)     − cam_direct)
        #              + Σ_(e∈lidar) p_e · (lidar_e(lidar_direct) − lidar_direct)
        #
        #   fused = refine(direct_mix + residual_gain · delta_sum)
        #
        # Each expert is called once on the full batch (preserves BN
        # batch statistics).  Identity-init bottleneck experts give
        # delta_sum ≈ 0 at step 0 whenever cam_direct/lidar_direct are
        # non-negative (which they are — both come from a ReLU).
        if not self._dense_dispatch:
            raise NotImplementedError(
                "ModalitySpecificMoEBlock symmetric output-space "
                "implementation currently supports only gate_type='dense'. "
                f"Got gate_type='{self.gate_type}'.")

        probs = gate_out.full_softmax_probs                              # (B, E)

        cam_direct   = self.cam_direct_proj(cam_bev)                     # (B, C_out, H, W)
        lidar_direct = lidar_bev                                         # (B, C_out, H, W)

        # Shape contract on the direct paths.
        assert cam_direct.shape[1] == self.out_channels
        assert lidar_direct.shape[1] == self.out_channels
        assert cam_direct.shape == lidar_direct.shape

        cam_mass   = probs[:, self.cam_expert_ids].sum(dim=1)            # (B,)
        lidar_mass = probs[:, self.lidar_expert_ids].sum(dim=1)          # (B,)

        cam_mass_map   = cam_mass.view(B, 1, 1, 1).to(cam_direct.dtype)
        lidar_mass_map = lidar_mass.view(B, 1, 1, 1).to(lidar_direct.dtype)

        direct_mix = (cam_mass_map * cam_direct
                      + lidar_mass_map * lidar_direct)

        delta_sum = torch.zeros_like(direct_mix)

        for j, eidx in enumerate(self.cam_expert_ids):
            y = self.cam_experts[j](cam_direct)                          # (B, C_out, H, W)
            delta = y - cam_direct
            w = probs[:, eidx].view(B, 1, 1, 1).to(delta.dtype)
            delta_sum = delta_sum + w * delta

        for j, eidx in enumerate(self.lidar_expert_ids):
            y = self.lidar_experts[j](lidar_direct)                      # (B, C_out, H, W)
            delta = y - lidar_direct
            w = probs[:, eidx].view(B, 1, 1, 1).to(delta.dtype)
            delta_sum = delta_sum + w * delta

        fused_bev = direct_mix + self.residual_gain * delta_sum
        fused_bev = self.refine(fused_bev)

        # Output shape contract.
        assert fused_bev.shape[1] == self.out_channels, (
            f'ModalitySpecificMoEBlock: fused channels '
            f'({fused_bev.shape[1]}) != out_channels ({self.out_channels}).')
        assert fused_bev.shape[-2:] == lidar_bev.shape[-2:], (
            f'ModalitySpecificMoEBlock: fused spatial dims '
            f'({tuple(fused_bev.shape[-2:])}) != lidar_bev spatial dims '
            f'({tuple(lidar_bev.shape[-2:])}).')

        # ── Step 3: Auxiliary losses (fp32, same rationale as Step 1) ─
        # ``context_head`` + balance losses run in fp32; no
        # ``nan_to_num`` masking — propagate any non-finite value to
        # the total loss so GradScaler can skip the step.
        with torch.autocast('cuda', enabled=False):
            imp_loss = importance_loss(
                gate_out.full_softmax_probs, self.importance_coef)
            ld_loss  = load_loss(
                gate_out.clean_logits, gate_out.noisy_logits,
                gate_out.noise_std, self.k, self.load_coef)
            z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)
            # Group balance kept ACTIVE under the symmetric design — see
            # the production config for the default coefficient.  Stops
            # early routing collapse to a single modality group while
            # remaining small enough that the gate can still drift
            # toward the genuinely stronger modality during training.
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
            # Per-sample group masses then batch mean.  Under the
            # symmetric design these are *routed modality contribution
            # masses* (both the direct mix and the routed deltas are
            # weighted by p_e) rather than evidence of modality balance.
            cam_mass_mean   = float(cam_mass.mean().item())
            lidar_mass_mean = float(lidar_mass.mean().item())

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
            'cam_group_mass':       cam_mass_mean,
            'lidar_group_mass':     lidar_mass_mean,
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
            # ── Symmetric output-space design fields ──────────────────
            'modality_specific_design': 'symmetric_output_space_bottleneck',
            'cam_direct_channels':      self.out_channels,
            'lidar_direct_channels':    self.out_channels,
            'expert_input_channels':    self.out_channels,
            'expert_output_channels':   self.out_channels,
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

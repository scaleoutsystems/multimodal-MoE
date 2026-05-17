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
(stem + 3 residual blocks + grid pool → ``router_out_dim``-d descriptor).
The two descriptors are concatenated to form the joint descriptor that
feeds both the gate and the context head::

    z_C  = cam_summary(cam_bev)            # (B, router_out_dim)
    z_L  = lidar_summary(lidar_bev)        # (B, router_out_dim)
    z_LC = torch.cat([z_C, z_L], dim=1)    # (B, 2 · router_out_dim)

    z_gate     = z_LC.detach()  if gate_input_detach else z_LC
    gate_out   = gate(z_gate)
    ctx_logits = context_head(z_LC)        # always full-grad through z_LC

Experts then receive the full ``cam_bev`` and ``lidar_bev`` features.
Concatenating the two summaries is descriptor-level conditioning, not
feature-level fusion.  The road-type label is NEVER concatenated into
the gate input — context supervision shapes ``z_LC`` only.

Context-supervised routing
--------------------------
Same pattern as :class:`BEVMoEBlock` (run 4577584): a separate
``context_head`` MLP is supervised by ``F.cross_entropy(ctx_logits,
ctx_label)`` with ``loss_type='weighted_ce'`` (inverse-frequency class
weights) and label smoothing.

Dispatch strategy
-----------------
Dense soft-MoE (default ``gate_type='dense'``): every expert always
runs; outputs are mixed by the full softmax over the gate logits::

    probs = softmax(W_gate(z_gate), dim=-1)        # (B, E)
    out = Σ_j  probs[:, j].view(B, 1, 1, 1) · expert_j(cam_bev, lidar_bev)

Each expert produces a fresh fused BEV (not a residual delta) so the
``Σ_j probs[:, j] = 1`` invariant already keeps the mixture at unit
scale.  ``residual_gain`` is accepted for config parity with the other
blocks but is a no-op for joint experts.

The legacy per-sample top-k loop is preserved only for ``gate_type ∈
{'topk', 'noisy_topk'}`` so existing ablations still work; production
configs (Joint-Modality / Modality-Specific / Fusion-then-MoE 28-epoch
thesis variants) all use ``gate_type='dense'``.

moe_info contract
-----------------
Same fields as ``BEVMoEBlock`` (see ``bev_moe.py`` docstring); no
modality-group fields (those belong to ``ModalitySpecificMoEBlock``).
``topk_idx`` / ``topk_weights`` are populated even under dense routing
(experts sorted by full softmax probability) so existing
:class:`MoERoutingHook` / :class:`ContextRoutingStatsHook` /
:class:`ExpertRespawnHook` consumers keep working without a dense-
specific branch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS

from .bev_moe import (BEVMoEBlock, _logit_diagnostics, _noise_diagnostics,
                      focal_ce_loss)
from .losses import importance_loss, load_loss, router_z_loss
from .routing import (BEVResSummaryEncoder, NoisyTopkGate, TopkGate,
                       extract_context_labels, get_context_vocab)


@MODELS.register_module()
class JointModalityExpert(nn.Module):
    """Single joint-modality expert — lightweight bottleneck fusion.

    Each expert independently fuses ``cam_bev`` and ``lidar_bev`` into a
    fresh BEV feature map (no identity residual — this is *not* a residual
    adapter; the expert produces a new fused representation that replaces
    the per-modality BEVs).  The bottleneck structure reduces the dense-MoE
    compute relative to a full-channel ``3×3 (C_cam + C_lidar) → C_out``
    fusion conv while preserving the same I/O contract.

    Architecture::

        x = concat([cam_bev, lidar_bev], dim=1)        # (B, C_cam+C_lidar, H, W)

        x = 1×1 Conv (C_cam+C_lidar → hidden_channels) ─ BN ─ ReLU
        x = 3×3 Conv (hidden_channels  → hidden_channels) ─ BN ─ ReLU
        x = 1×1 Conv (hidden_channels  → out_channels)   ─ BN ─ ReLU

        fused_bev = x                                  # (B, C_out, H, W)

    Compute (per BEV cell), default hidden_channels=128, cam=80, lidar=256,
    out=256::

        Legacy 3×3 (C_cam+C_lidar) → C_out:
            (80 + 256) · 256 · 9 = 774 144 FLOPs/cell
        Bottleneck (1×1 + 3×3 + 1×1) at H=128:
            (80 + 256) · 128 + 128 · 128 · 9 + 128 · 256
          = 43 008 + 147 456 + 32 768 ≈ 223 232 FLOPs/cell

    ~3.5× cheaper per expert; with 4 dense experts the saving is ~2 MFLOPs/cell.

    Important:
        * Input shapes: ``cam_bev (B, C_cam, H, W)``, ``lidar_bev (B, C_lidar, H, W)``.
        * Output shape: ``(B, out_channels, H, W)``.  ``out_channels`` is
          chosen by the parent block to match the downstream
          ``pts_backbone`` input channel count (256 in the BEVFusion stack).
        * No identity residual: the joint expert produces a fresh fused
          BEV from two independent inputs (the cam/lidar maps do not share
          a single canonical "input to add back").  Each forward pass
          returns a full fused tensor that the parent block weights into
          the dense softmax mixture (``Σ_e p_e · expert_e(cam, lidar)``).

    Args:
        cam_channels:    Camera BEV channel count.
        lidar_channels:  LiDAR BEV channel count.
        out_channels:    Fused BEV output channel count (matches the
                         downstream ``pts_backbone`` input).
        hidden_channels: Bottleneck width.  Default 128.
    """

    def __init__(self, cam_channels: int, lidar_channels: int,
                 out_channels: int, hidden_channels: int = 128):
        super().__init__()
        assert (cam_channels > 0 and lidar_channels > 0
                and out_channels > 0 and hidden_channels > 0), (
            f'channels must all be positive, got cam={cam_channels}, '
            f'lidar={lidar_channels}, out={out_channels}, '
            f'hidden={hidden_channels}')
        self.cam_channels    = int(cam_channels)
        self.lidar_channels  = int(lidar_channels)
        self.out_channels    = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        in_channels = cam_channels + lidar_channels
        # Reduce → spatial → expand bottleneck fusion.  Output is a fresh
        # fused BEV (no residual add) so a final ReLU after the expand BN
        # is appropriate.  BN epsilon/momentum mirror the BEV-residual
        # expert defaults for consistency.
        self.fuse = nn.Sequential(
            # 1×1 reduce: project the concatenated cam+lidar tensor down
            # to the bottleneck width.  Cheapest spatially-trivial way to
            # combine the two modalities before the 3×3 spatial mixing.
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            # 3×3 spatial mixing at the bottleneck width.
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            # 1×1 expand: project up to the requested fused-BEV channel
            # count (matches ``pts_backbone`` input).
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
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
        num_experts:         Number of fusion experts.  Default 4 (matches
                             the LiDAR-only MoE 4577584 expert count).
        k:                   Top-k experts per sample.  Ignored when
                             ``gate_type='dense'`` (effective k is forced
                             to ``num_experts`` so every expert always
                             runs).  Default ``num_experts`` (dense).
        importance_coef:     Weight for the Shazeer importance loss.
        load_coef:           Weight for the Shazeer load loss.  Set to 0.0
                             under dense routing (no top-k cliff, so the
                             smooth-CDF load surrogate does not apply).
        z_loss_coef:         Weight for ``router_z_loss(clean_logits)``.
        residual_gain:       Accepted for config parity; no-op here.
        router_out_dim:      Output dim per modality BEVResSummaryEncoder.
                             Gate sees ``2 · router_out_dim`` as input.
                             Default 128 (→ 256-d joint descriptor,
                             matching the LiDAR-only MoE gate input).
        context_aux_cfg:     Same as ``BEVMoEBlock.context_aux_cfg``.
                             Accepted keys: ``target_field`` (required),
                             ``loss_coef`` (default 0.05),
                             ``loss_type`` ∈ {``'weighted_ce'`` (default),
                             ``'ce'``, ``'focal'``}, ``class_weights``
                             (required for ``'weighted_ce'``: explicit
                             per-class list or ``'inverse_frequency'``),
                             ``focal_gamma`` (default 2.0),
                             ``label_smoothing`` (default 0.0).
        gate_type:           ``'dense'`` (default, dense soft-MoE — every
                             expert always runs, weighted by the full
                             softmax probabilities), ``'topk'`` or
                             ``'noisy_topk'`` (legacy ablations).
        gate_cfg:            Extra kwargs forwarded to the gate
                             (``temperature``, plus noise/scale kwargs
                             for the noisy variant).
        gate_input_detach:   If True (default, context-supervised
                             routing) the gate consumes
                             ``z_LC.detach()``; the context head still
                             sees full-grad ``z_LC`` so the summary
                             encoders are shaped purely by the CE on
                             ``road_type``.  If False the gate consumes
                             ``z_LC`` directly and detection-loss
                             gradient flows through the dispatch path
                             into the summary encoders.
        expert_hidden_channels:
                             Bottleneck width for each
                             :class:`JointModalityExpert`'s fusion
                             pathway.  Default 128 — matches the
                             single-input BEV bottleneck experts in
                             :class:`BEVMoEBlock` /
                             :class:`ModalitySpecificMoEBlock` so the
                             three MoE variants stay roughly
                             compute-comparable.
    """

    def __init__(
        self,
        cam_channels: int = 80,
        lidar_channels: int = 256,
        out_channels: int = 256,
        num_experts: int = 4,
        k: Optional[int] = None,
        importance_coef: float = 0.005,
        load_coef: float = 0.0,
        z_loss_coef: float = 0.002,
        residual_gain: float = 1.0,
        router_out_dim: int = 128,
        context_aux_cfg: Optional[dict] = None,
        gate_type: str = 'dense',
        gate_cfg: Optional[dict] = None,
        gate_input_detach: bool = True,
        expert_hidden_channels: int = 128,
    ):
        super().__init__()
        self.cam_channels = cam_channels
        self.lidar_channels = lidar_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.importance_coef = float(importance_coef)
        self.load_coef = float(load_coef)
        self.z_loss_coef = float(z_loss_coef)
        self.residual_gain = float(residual_gain)
        self.gate_input_detach = bool(gate_input_detach)
        if self.gate_input_detach and context_aux_cfg is None:
            raise ValueError(
                'JointModalityMoEBlock: gate_input_detach=True with '
                'context_aux_cfg=None means the BEVResSummaryEncoder '
                'branches receive no gradient from any source.  Either '
                'set gate_input_detach=False (task-driven routing) or '
                'provide a context_aux_cfg (context-supervised routing).')
        if self.residual_gain != 1.0:
            import warnings
            warnings.warn(
                f'JointModalityMoEBlock received residual_gain='
                f'{self.residual_gain} but this parameter is a no-op for '
                f'this block (joint experts produce fresh fused BEVs). '
                f'The value is silently ignored.',
                stacklevel=2)

        self.expert_hidden_channels = int(expert_hidden_channels)
        self.experts = nn.ModuleList([
            JointModalityExpert(
                cam_channels, lidar_channels, out_channels,
                hidden_channels=self.expert_hidden_channels)
            for _ in range(num_experts)
        ])

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
        # Dense soft-MoE collapses to a ``TopkGate(k=num_experts)``: the
        # gate's ``full_softmax_probs`` carry all the dispatch mass and
        # ``topk_idx``/``topk_weights`` are simply the experts sorted by
        # probability, kept populated so downstream hooks
        # (:class:`MoERoutingHook`, :class:`ExpertRespawnHook`,
        # :class:`ContextRoutingStatsHook`) work unchanged.
        gate_type_norm = str(gate_type).lower()
        if gate_type_norm not in ('topk', 'noisy_topk', 'dense'):
            raise ValueError(
                "JointModalityMoEBlock.gate_type must be 'topk', "
                f"'noisy_topk' or 'dense', got '{gate_type}'.")
        self._dense_dispatch = (gate_type_norm == 'dense')
        if self._dense_dispatch:
            self.k = num_experts
        else:
            self.k = int(k) if k is not None else 2

        extra_gate_kwargs = gate_cfg or {}
        if self._dense_dispatch:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=num_experts,
                k=num_experts, **extra_gate_kwargs)
        elif gate_type_norm == 'noisy_topk':
            self.gate = NoisyTopkGate(
                feat_dim=joint_dim, num_experts=num_experts, k=self.k,
                **extra_gate_kwargs)
        else:
            self.gate = TopkGate(
                feat_dim=joint_dim, num_experts=num_experts, k=self.k,
                **extra_gate_kwargs)
        self.gate_type = gate_type_norm

        gate_in = (self.gate.gate.in_features
                   if isinstance(self.gate, TopkGate)
                   else self.gate.w_gate.in_features)
        assert gate_in == joint_dim, (
            f'JointModalityMoEBlock: gate input dim ({gate_in}) must equal '
            f'sum of summary out_dims ({joint_dim}) — context vector must '
            f'NOT be concatenated into the router input.')

        self._moe_info: Optional[Dict[str, Any]] = None

    def _build_context_head(self, cfg: dict, in_dim: int) -> None:
        """Configure the context auxiliary classifier (mirrors BEVMoEBlock).

        Accepts ``target_field`` (required), ``loss_coef``, ``loss_type``
        ∈ {``'weighted_ce'`` (default), ``'ce'``, ``'focal'``},
        ``class_weights`` (required for ``'weighted_ce'``: explicit list
        or ``'inverse_frequency'``), ``focal_gamma`` and
        ``label_smoothing``.  Builds the same MLP head structure as
        ``BEVMoEBlock`` (Linear → ReLU → LayerNorm → Dropout → Linear)
        so the multimodal MoE variants share the LiDAR-only run's
        context-head behaviour.
        """
        cfg = dict(cfg)
        target_field = cfg.pop('target_field', None)
        if target_field is None:
            raise ValueError(
                "JointModalityMoEBlock.context_aux_cfg must include a "
                "'target_field' (e.g. 'road_type').")
        loss_coef = float(cfg.pop('loss_coef', 0.05))
        loss_type = str(cfg.pop('loss_type', 'weighted_ce')).lower()
        if loss_type not in ('weighted_ce', 'ce', 'focal'):
            raise ValueError(
                "JointModalityMoEBlock.context_aux_cfg: loss_type must be "
                f"'weighted_ce', 'ce' or 'focal', got '{loss_type}'.")
        class_weights_cfg = cfg.pop('class_weights', None)
        focal_gamma = float(cfg.pop('focal_gamma', 2.0))
        label_smoothing = float(cfg.pop('label_smoothing', 0.0))
        if cfg:
            raise ValueError(
                f"JointModalityMoEBlock.context_aux_cfg got unexpected "
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
                "JointModalityMoEBlock.context_aux_cfg: 'class_weights' "
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
        # rationale: a transient fp16 overflow upstream can leave
        # NaN/Inf in the BEV maps, which then poison the experts' BN
        # running stats and the context head's CE, contaminating
        # ``moe_ctx_aux_loss_weighted`` for the rest of the run.
        if not torch.isfinite(cam_bev).all():
            import logging as _logging
            _logging.getLogger('mmengine').warning(
                'JointModalityMoEBlock: NaN/Inf in cam_bev — replacing '
                'with zeros before MoE forward.')
            cam_bev = torch.nan_to_num(
                cam_bev, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(lidar_bev).all():
            import logging as _logging
            _logging.getLogger('mmengine').warning(
                'JointModalityMoEBlock: NaN/Inf in lidar_bev — replacing '
                'with zeros before MoE forward.')
            lidar_bev = torch.nan_to_num(
                lidar_bev, nan=0.0, posinf=0.0, neginf=0.0)

        B = cam_bev.shape[0]

        # ── Step 1: Joint descriptor ──────────────────────────────────
        # z_LC drives both the gate (via z_gate) and the context head.
        # When gate_input_detach=True the gate consumes z_LC.detach()
        # so the summary encoders are shaped only by the auxiliary
        # context CE (mirrors BEVMoEBlock).
        z_C  = self.cam_summary(cam_bev)
        z_L  = self.lidar_summary(lidar_bev)
        z_LC = torch.cat([z_C, z_L], dim=1)
        z_gate = z_LC.detach() if self.gate_input_detach else z_LC

        gate_out = self.gate(z_gate)

        # ── Step 2: Dispatch ──────────────────────────────────────────
        # Dense path (production for thesis Variant A): every expert
        # runs once on the full batch, mixed by ``full_softmax_probs``.
        # Each joint expert produces a fresh fused BEV so we directly
        # form the convex combination ``Σ_e p_e · expert_e(cam, lidar)``
        # without the residual-delta dance used by BEVMoEBlock /
        # ModalitySpecificMoEBlock.
        if self._dense_dispatch:
            probs = gate_out.full_softmax_probs                     # (B, E)
            out: Optional[Tensor] = None
            for e in range(self.num_experts):
                h_e = self.experts[e](cam_bev, lidar_bev)            # (B, C, H, W)
                w = probs[:, e].view(-1, 1, 1, 1).to(h_e.dtype)
                contribution = w * h_e
                out = contribution if out is None else out + contribution
            assert out is not None
        else:
            # Legacy top-k / noisy-top-k path: per-sample dispatch.
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

        # ── Step 3: Auxiliary losses ──────────────────────────────────
        imp_loss = importance_loss(
            gate_out.full_softmax_probs, self.importance_coef)
        ld_loss  = load_loss(
            gate_out.clean_logits, gate_out.noisy_logits,
            gate_out.noise_std, self.k, self.load_coef)
        z_loss   = router_z_loss(gate_out.clean_logits, self.z_loss_coef)

        # Context auxiliary classification ----------------------------------
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
            if not torch.isfinite(ctx_logits).all():
                import logging as _logging
                _logging.getLogger('mmengine').warning(
                    'JointModalityMoEBlock: NaN/Inf in ctx_logits — '
                    'replacing with zeros before context-aux CE.')
                ctx_logits = torch.nan_to_num(
                    ctx_logits, nan=0.0, posinf=0.0, neginf=0.0)
            ctx_labels = extract_context_labels(
                batch_input_metas, self._ctx_target_field,
                self._ctx_vocab_map, z_LC.device)
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

        aux = imp_loss + ld_loss + z_loss + ctx_loss_weighted

        # ── Step 4: Build moe_info ────────────────────────────────────
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
            'ctx_loss_type':        self._ctx_loss_type,
            'ctx_class_weights':    (list(self._ctx_class_weights_list)
                                     if self._ctx_class_weights_list is not None
                                     else None),
            'focal_gamma':          self._ctx_focal_gamma,
            'gate_feat_dim':        (self.cam_summary.out_dim
                                     + self.lidar_summary.out_dim),
            'z_ctx_detached_for_gate': self.gate_input_detach,
            'gate_input':           ('z_LC_detach'
                                     if self.gate_input_detach else 'z_LC'),
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
        return out, moe_info

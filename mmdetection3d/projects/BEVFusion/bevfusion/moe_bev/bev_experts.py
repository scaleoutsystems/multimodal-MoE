"""BEV-space expert modules.

Two expert flavours are available; both preserve the input tensor shape
``(B, C, H, W) → (B, C, H, W)`` and use an identity residual path so the
pretrained BEV representation is preserved at initialisation:

1. :class:`BEVResidualExpert` (legacy, ``expert_type='full'``):
   ``num_convs × [Conv2d(C, C, 3×3) → BN]`` with inter-layer ReLU, then
   ``+ input → ReLU``.  Operating at the full channel count ``C`` (typically
   256–512 in BEVFusion) makes a single expert FLOP-equivalent to a full
   pts_backbone conv block; with dense dispatch and ``num_experts=4`` this
   dominates the MoE-block cost.  Kept for backwards compatibility and
   ablations.

2. :class:`BEVBottleneckResidualExpert` (default, ``expert_type='bottleneck'``):
   A ResNet-style bottleneck adapter::

       x ──┬─────────────────────────────────────────────────────────────┐
           │                                                              │
           └─ 1×1 Conv C → H ─ BN ─ ReLU                                  │
              ─ 3×3 Conv H → H ─ BN ─ ReLU                                │
              ─ 1×1 Conv H → C ─ BN(gamma=0, bias=0)                      │
                                                                          │
                                                          ──── (+ x) → ReLU

   With ``hidden_channels=128`` and ``C=512``, the residual branch costs
   roughly ``(128² + 128² × 9 + 128 × 512) / (512² × 9) ≈ 0.085`` of one
   full ``3×3 C→C`` conv, i.e. ~12× cheaper per expert.  The expand BN's
   ``weight`` and ``bias`` are zero-initialised so each expert produces an
   exact-zero residual at step 0 → ``expert(x) ≡ x`` (identity).  This is
   the "ResNet zero-init last BN" trick: the forward output is identical
   across experts, but the *gradient* w.r.t. the expand BN's gamma is
   ``dL/dy · (normalised input)`` which is non-zero and different per
   expert from iter 1 (each expert's reduce/spatial/expand_conv have
   independent random init), so expert symmetry breaks through the
   gradient path within a handful of iterations.

Why this matters for dense MoE
------------------------------
Under dense soft-MoE every expert runs on the full BEV map for every
sample (no token sparsity).  With the legacy full-channel expert, the
MoE block adds ``num_experts × Conv2d(C, C, 3×3)`` FLOPs to the
detector and the activation memory of ``num_experts`` full residual
graphs (which is why dense-MoE at batch=4 OOMs on a 40 GB A100 and we
had to drop to batch=2 + grad-accum=2).  The bottleneck expert keeps
the same input/output contract but reduces both compute and activation
memory enough that the dense-MoE overhead is roughly proportional to
the context_summary cost rather than dominating it.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS


# Small-random init std for the last BN gamma of the legacy full-channel
# expert.  Pure zero-init would make every expert produce ``block(x) ≡ 0``
# at step 0; combined with full-channel convs (and no per-expert hidden
# axis to differentiate them via the gradient path) this empirically
# took 4+ epochs to break out of the degenerate symmetric state (run
# 4540532).  A ~5% perturbation is FP16-safe and lets the gate see a
# distinct per-expert signal from iter 1.  This rationale is specific
# to :class:`BEVResidualExpert`; :class:`BEVBottleneckResidualExpert`
# uses pure zero-init on the expand BN because its random reduce/
# spatial/expand_conv weights provide the per-expert gradient asymmetry.
_LAST_BN_GAMMA_STD: float = 0.05


# ── fp32-safe BatchNorm2d ─────────────────────────────────────────────────

class _FP32BatchNorm2d(nn.BatchNorm2d):
    """:class:`nn.BatchNorm2d` that always computes in fp32, irrespective
    of the surrounding ``torch.amp.autocast`` region.

    Why this matters for the MoE experts
    ------------------------------------
    Under dense soft-MoE every expert runs on every sample at the full
    BEV resolution (post-SECONDFPN, 512 channels, ~180×180 spatial), so
    each expert BN sees a *large* per-iter reduction window.  When the
    surrounding region is ``autocast(dtype=fp16)`` and the post-FPN
    feature map carries one or two outlier samples (e.g. dense
    near-range LiDAR returns shrinking ``var → 0`` in some channels),
    the BN's batch-variance computation can yield a *huge but finite*
    ``batch_var`` (~1e3–1e5).  That value enters the running-stat EMA
    update verbatim and, because the buffer is fp32, accumulates over
    epochs without ever crossing into NaN/Inf — invisible to the
    existing :meth:`BEVFusion._repair_bn_stats` guard which only checks
    ``torch.isfinite``.

    Run 4585570 epoch 15 ended with
    ``experts.{0..3}.block.4.running_var ∈ [7.6e4, 2.1e5]`` and
    ``running_mean ∈ [1.3e3, 2.8e3]``; combined with the gate-Linear's
    ``nan_to_num`` defensive cast (which silently zeros the gate
    gradient when ``z_ctx`` overflows), this is the upstream of the
    mAP→0 collapse documented in :mod:`moe_bev.bev_moe`.

    Forcing the BN op to fp32 while leaving the surrounding convs in
    fp16 keeps the dense-MoE memory budget intact (BN activations are
    a small fraction of total) and removes the only spot in the expert
    pipeline where the variance reduction can pick up an fp16-rounded
    contribution.  The cast back to ``input_dtype`` on the output
    preserves the activation-tensor dtype contract for the rest of the
    expert's residual branch, so this is a drop-in replacement for
    :class:`nn.BatchNorm2d` everywhere it appears inside the experts.

    Running stats buffers and affine parameters are already fp32 in
    mmengine regardless of model dtype, so checkpoints are
    bit-compatible — a model trained with the previous
    :class:`nn.BatchNorm2d` loads into a model using this class
    without any state-dict renaming.
    """

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        # Disable autocast for the BN op itself.  ``x.float()`` is a
        # no-op when ``x`` is already fp32; under fp16 autocast it
        # ensures the variance reduction sees full-precision inputs.
        with torch.autocast('cuda', enabled=False):
            out = super().forward(x.float())
        return out.to(input_dtype)


@MODELS.register_module()
class BEVResidualExpert(nn.Module):
    """Legacy full-channel residual conv expert (kept for backwards compat).

    Architecture (num_convs=2):
        Conv2d(C, C, 3×3) → BN → ReLU → Conv2d(C, C, 3×3) → BN → (+ x) → ReLU

    The last BN's gamma is initialised to a small independent random value
    N(0, _LAST_BN_GAMMA_STD) so experts are distinguishable from step 1
    without producing large enough perturbations to overflow AMP fp16.

    Note:
        New MoE blocks should prefer :class:`BEVBottleneckResidualExpert`
        (default in :func:`make_bev_experts`) which has the same I/O
        contract but is dramatically cheaper under dense dispatch.

    Args:
        channels:  Number of input/output channels.
        num_convs: Number of Conv→Norm layers in the block (default 1).
        norm_type: ``'bn'`` (default) → :class:`_FP32BatchNorm2d` —
                   matches the original full expert used by the
                   lidar-only MoE runs.
                   ``'gn'``           → ``nn.GroupNorm(32, channels)``
                   — drop-in replacement with no running statistics.
                   Required for the fusion-then-MoE variant: the
                   fused (camera ⊕ lidar) BEV has narrow per-channel
                   modes (camera depth-splat shells are dense in
                   small depth bands), which drives the BN running
                   ``var → ~1e-5`` after a few epochs.  Once that
                   happens the BN output ``γ·(x − μ)/√(var + ε)``
                   blows up to ``~3·10³``, which then saturates fp16
                   in the next conv → ``inf`` → ``NaN`` in
                   TransFusionHead's softmax (see run 4613034 epoch 6
                   iter 2550 in the run notes).  GroupNorm computes
                   the denominator from per-sample current
                   activations, so it cannot degenerate the same way.
                   Activation-tensor shape and dtype contract are
                   identical for both choices, so the rest of the
                   MoE pipeline is unaffected.
    """

    def __init__(self,
                 channels: int,
                 num_convs: int = 1,
                 norm_type: str = 'bn'):
        super().__init__()
        self.norm_type = str(norm_type).lower()
        if self.norm_type not in ('bn', 'gn'):
            raise ValueError(
                f"BEVResidualExpert.norm_type must be 'bn' or 'gn', "
                f"got '{norm_type}'.")

        def _make_norm(num_channels: int) -> nn.Module:
            if self.norm_type == 'gn':
                # GroupNorm(32, C): see Args docstring for why fused
                # BEV needs per-sample normalisation instead of EMA
                # running stats.  PyTorch AMP autocasts GroupNorm to
                # fp32 automatically (default policy) so we do not
                # need an _FP32GroupNorm wrapper.
                return nn.GroupNorm(32, num_channels)
            # _FP32BatchNorm2d: fp16-overflow-safe drop-in for the
            # legacy full-channel expert.  See its docstring for the
            # AMP fp16 reduction-window rationale.
            return _FP32BatchNorm2d(num_channels, eps=1e-3, momentum=0.01)

        layers: List[nn.Module] = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            layers.append(_make_norm(channels))
            if i < num_convs - 1:
                layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

        # Small-random init on the last norm's ``weight`` parameter
        # for per-expert asymmetry — both BatchNorm and GroupNorm
        # expose a per-channel ``weight`` (== gamma) so this works
        # for either norm choice.
        last_norm = next(
            m for m in reversed(list(self.block.modules()))
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)))
        nn.init.normal_(last_norm.weight, mean=0.0, std=_LAST_BN_GAMMA_STD)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))


@MODELS.register_module()
class BEVBottleneckResidualExpert(nn.Module):
    """Lightweight residual-bottleneck BEV expert (default).

    Replaces the full ``C → C`` 3×3 convolution of :class:`BEVResidualExpert`
    with a ResNet-style bottleneck adapter that operates almost entirely at
    ``hidden_channels`` (default 128, << C).  The identity path is preserved
    bit-for-bit so a freshly constructed expert behaves as ``expert(x) ≡ x``
    and the pretrained BEV representation is untouched at the start of
    training.

    Architecture::

        residual = 1×1 Conv(C → H) ─ BN ─ ReLU
                 → 3×3 Conv(H → H) ─ BN ─ ReLU
                 → 1×1 Conv(H → C) ─ BN(gamma=0, bias=0)

        output = ReLU(x + residual)

    Important:
        * Input/output shape: ``(B, C, H_BEV, W_BEV) → (B, C, H_BEV, W_BEV)``
          — preserved exactly regardless of ``channels`` or
          ``hidden_channels``.
        * The expand projection's BN is zero-initialised
          (``weight = bias = 0``) so the residual branch outputs exact-zero
          tensors at step 0 → ``output = ReLU(x) = x`` for all post-FPN
          ReLU-activated BEV maps.  Backbones whose features are not
          non-negative would see a ReLU applied here on the identity
          path; this is consistent with the legacy expert (which also
          ends in ReLU after the residual add) and matches the
          assumption that BEV experts sit after a ReLU-activated FPN or
          ConvFuser block.
        * No activation is applied between the final expand projection
          and the residual add — that ReLU is the *output* activation,
          applied after summation as in the legacy expert.

    Args:
        channels:        Input/output channel count ``C``.  Must match the
                         feature map at the MoE insertion point (typically
                         256 for fused BEV or 512 for post-SECONDFPN FPN
                         output).
        hidden_channels: Bottleneck width ``H``.  Default 128.

    Per-expert FLOPs (approx., per BEV cell):
        ``2 · C · H  +  9 · H²  +  2 · H · C``
        ≈ ``4 · C · H + 9 · H²``.
        At ``C=512, H=128``: ≈ 4 · 512 · 128 + 9 · 128² ≈ 3.6 × 10⁵
        FLOPs/cell, vs 2.4 × 10⁶ for one full 3×3 ``C → C`` conv ⇒
        ~6.7× cheaper per residual branch, ~12× cheaper overall after
        the dropped second ReLU/BN of ``num_convs=2`` legacy mode.
    """

    def __init__(self, channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        assert channels > 0 and hidden_channels > 0, (
            f'channels and hidden_channels must be positive, '
            f'got channels={channels}, hidden_channels={hidden_channels}')
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)

        # All three BNs use _FP32BatchNorm2d: dense MoE dispatch keeps
        # every expert on every batch at the full BEV resolution, so the
        # AMP fp16 variance reduction needs to be safe against outlier
        # batches.  See the _FP32BatchNorm2d docstring for the full
        # rationale — running stats buffers and affine params remain
        # fp32, so this is a state-dict-compatible swap.

        # Reduce: project to hidden width with a cheap 1×1 conv.
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            _FP32BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Spatial: a single 3×3 conv at the bottleneck width — this is the
        # only spatial-mixing layer in the expert and accounts for the bulk
        # of the bottleneck's expressivity.
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=1, bias=False),
            _FP32BatchNorm2d(hidden_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        # Expand: project back to the full channel count.  The BN's affine
        # parameters are zero-initialised so the residual branch starts
        # producing exact zeros and the expert begins as an exact identity.
        # No activation between expand and the residual add — the output
        # ReLU is applied after summation.
        self.expand_conv = nn.Conv2d(hidden_channels, channels,
                                     kernel_size=1, bias=False)
        self.expand_bn = _FP32BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        nn.init.zeros_(self.expand_bn.weight)
        nn.init.zeros_(self.expand_bn.bias)

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.reduce(x)
        residual = self.spatial(residual)
        residual = self.expand_conv(residual)
        residual = self.expand_bn(residual)
        return self.out_act(x + residual)


def make_bev_experts(num_experts: int,
                     channels: int,
                     num_convs: int = 1,
                     expert_type: str = 'bottleneck',
                     hidden_channels: int = 128,
                     norm_type: str = 'bn') -> nn.ModuleList:
    """Factory: build a list of independent BEV expert modules.

    All produced experts share the same I/O contract:
    ``(B, channels, H, W) → (B, channels, H, W)``.  Each expert has
    independent weights (no sharing).

    Args:
        num_experts:     How many experts to create.
        channels:        Input/output channel count for each expert
                         (must match the feature map at the MoE
                         insertion point).
        num_convs:       Number of Conv→Norm layers in the legacy
                         :class:`BEVResidualExpert`; ignored for
                         ``expert_type='bottleneck'`` (which has a
                         fixed reduce / spatial / expand structure).
        expert_type:     ``'bottleneck'`` (default) → cheap
                         :class:`BEVBottleneckResidualExpert`;
                         ``'full'`` → legacy :class:`BEVResidualExpert`.
        hidden_channels: Bottleneck width for the bottleneck expert.
                         Default 128.  Ignored for ``expert_type='full'``.
        norm_type:       Normalisation flavour passed to the full
                         expert (``'bn'`` default, ``'gn'`` for fused
                         BEV — see :class:`BEVResidualExpert` Args).
                         Ignored for ``expert_type='bottleneck'``
                         which always uses :class:`_FP32BatchNorm2d`.

    Returns:
        ``nn.ModuleList`` of expert modules.
    """
    expert_type = str(expert_type).lower()
    if expert_type == 'bottleneck':
        return nn.ModuleList([
            BEVBottleneckResidualExpert(channels,
                                        hidden_channels=hidden_channels)
            for _ in range(num_experts)
        ])
    if expert_type == 'full':
        return nn.ModuleList([
            BEVResidualExpert(channels, num_convs, norm_type=norm_type)
            for _ in range(num_experts)
        ])
    raise ValueError(
        f"make_bev_experts: expert_type must be 'bottleneck' or 'full', "
        f"got '{expert_type}'.")

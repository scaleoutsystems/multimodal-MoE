"""BEV-space expert modules.

Each expert is a residual conv block (num_convs × [Conv2d→BN], with
inter-layer ReLUs, then +input → ReLU) that processes a BEV feature map
(B, C, H, W) and returns a feature map of the same shape.

Expert initialisation strategy
--------------------------------
The last BatchNorm in each expert's block has its weight (gamma) initialised
to a small independent random perturbation drawn from N(0, _LAST_BN_GAMMA_STD).

**Why not zero-init?**
Pure zero-init (gamma=0) makes block(x) ≡ 0 for all experts at step 0.  This
is safe for AMP stability but breaks expert symmetry: the gate sees identical
outputs from every expert throughout the first few hundred iterations, so
routing gradients carry no per-expert signal and all experts receive the same
cumulative update.  Empirically (run 4540532) this results in 4+ epochs of
routing churn before any stable specialisation emerges.

**Why small random, not full random?**
With standard Kaiming init on gamma, the expert output at step 0 is a random
perturbation on pretrained backbone features.  Post-SECONDFPN activations can
be large; the intermediate activations inside the block are BN-normalised to
unit variance, so the final output magnitude is |gamma| × O(1).  With gamma
drawn from N(0, std=0.005), the initial expert perturbation is O(0.005 × 1)
≈ 5 × 10⁻³, which is safely within FP16 representable range (the fp16 minimum
normal is ~6 × 10⁻⁵; values this small are exact to machine precision).  The
residual ``x + block(x)`` ≈ ``x + 0.005 ε`` is numerically identical to the
pretrained backbone output, preventing the AMP overflow / NaN observed with
larger initialisation in earlier runs.

**Effect on routing**
Each expert's block produces a *distinct*, tiny, input-dependent perturbation
from step 1.  The gate can immediately distinguish expert outputs, so routing
gradients start carrying per-expert signal without requiring the gamma values
to "escape" from the degenerate zero state first.  Combined with the switch to
deterministic TopkGate (no noise inflation) and Fix #2 (filter_empty_gt=False),
this should eliminate the multi-epoch latency before specialisation.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS


# Small-random init std for the last BN gamma of each expert block.
# Set to 0.05: each expert produces a ~5% perturbation on pretrained FPN
# features at step 0, giving the gate a meaningful per-expert task-loss
# signal from the first iteration.  0.005 (previous value) produced only
# a ~0.5% perturbation, which was too small relative to the balance-loss
# gradient — the gate took the lazy "always use E0" solution before any
# per-expert task signal could compete.  0.05 is still three orders of
# magnitude below the fp16 overflow threshold (~65504) and the dynamic
# loss scaler handles the slightly larger initial gradient noise.
# See module docstring for the full rationale.
_LAST_BN_GAMMA_STD: float = 0.05


@MODELS.register_module()
class BEVResidualExpert(nn.Module):
    """Residual conv block operating on BEV feature maps.

    Architecture (num_convs=2):
        Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → (+ input) → ReLU

    The last BN's gamma is initialised to a small independent random value
    N(0, _LAST_BN_GAMMA_STD) so experts are distinguishable from step 1
    without producing large enough perturbations to overflow AMP fp16.
    See module docstring for the full rationale.

    Args:
        channels:  Number of input/output channels.
        num_convs: Number of Conv→BN layers in the block (default 1).
    """

    def __init__(self, channels: int, num_convs: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01))
            if i < num_convs - 1:
                layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

        # Small-random init: break expert symmetry at step 0 while keeping
        # the initial perturbation magnitude tiny (FP16-safe).
        last_bn = next(
            m for m in reversed(list(self.block.modules()))
            if isinstance(m, nn.BatchNorm2d))
        nn.init.normal_(last_bn.weight, mean=0.0, std=_LAST_BN_GAMMA_STD)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.block(x))

def make_bev_experts(num_experts: int, channels: int,
                     num_convs: int = 1) -> nn.ModuleList:
    """Factory: create a list of independent BEVResidualExpert modules.
    Expert channels must match the feature map at the insertion point
    Each expert has its own weights (no sharing).

    Args:
        num_experts: How many experts to create.
        channels: Channel count for each expert.
        num_convs: Number of conv layers per expert.

    Returns:
        nn.ModuleList of BEVResidualExpert instances.
    """
    return nn.ModuleList([
        BEVResidualExpert(channels, num_convs) for _ in range(num_experts)
    ])

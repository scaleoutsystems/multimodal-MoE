"""BEV-space expert modules.

Each expert is a single-conv residual block (Conv2d→BN, then +x→ReLU)
that processes a BEV feature map (B, C, H, W) and returns a feature map
of the same shape.  Parameter count matches ConvFuser, keeping any
performance gains attributable to MoE routing rather than expressiveness.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVResidualExpert(nn.Module):
    """Single-conv residual block operating on BEV feature maps.

    Architecture: Conv2d(3x3) -> BN -> (+ input) -> ReLU.
    Equivalent parameter count to ConvFuser's single conv layer, with a
    residual skip to aid gradient flow during refinement.

    Args:
        channels: Number of input/output channels.
        num_convs: Number of conv-BN layers in the block (default 1).
    """

    def __init__(self, channels: int, num_convs: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            # ReLU between layers only; final activation applied after skip add.
            if i < num_convs - 1:
                layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        # Structure: x -> Conv -> BN -> (+ x) -> ReLU
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

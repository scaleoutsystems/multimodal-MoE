"""BEV-space expert modules.

Each expert is a lightweight residual conv block that processes a full
BEV feature map (B, C, H, W) and returns a feature map of the same shape.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVResidualExpert(nn.Module):
    """Small residual conv block operating on BEV feature maps.

    Architecture per conv layer: Conv2d(3x3) -> BN -> ReLU.
    A residual skip connection adds the input to the output (requires
    input and output channels to be equal).

    Args:
        channels: Number of input/output channels.
        num_convs: Number of conv-BN-ReLU layers in the block.
    """

    def __init__(self, channels: int, num_convs: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_convs):
            layers.extend([
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


def make_bev_experts(num_experts: int, channels: int,
                     num_convs: int = 2) -> nn.ModuleList:
    """Factory: create a list of independent BEVResidualExpert modules.

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

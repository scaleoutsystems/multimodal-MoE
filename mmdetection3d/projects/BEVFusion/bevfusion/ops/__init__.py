# bev_pool_ext is a CUDA extension built via projects/BEVFusion/setup.py. LiDAR-only
# configs do not need it; a failed import must not block Voxelization (MMCV fallback).
try:
    from .bev_pool import bev_pool
except Exception:
    bev_pool = None

try:
    from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization
except Exception:
    # Fallback to MMCV ops when BEVFusion custom voxel extension is unavailable.
    from mmcv.ops import DynamicScatter, Voxelization
    from mmcv.ops import dynamic_scatter, voxelization

__all__ = [
    'bev_pool', 'Voxelization', 'voxelization', 'dynamic_scatter',
    'DynamicScatter'
]

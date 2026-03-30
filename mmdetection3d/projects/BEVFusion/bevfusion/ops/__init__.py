import warnings as _warnings

# bev_pool_ext is a CUDA extension built via projects/BEVFusion/setup.py.
# LiDAR-only configs do not need it, but camera+LiDAR (BEVFusion) configs do.
# Instead of silently setting bev_pool=None (which causes a confusing
# "'NoneType' is not callable" crash later), we warn loudly at import time
# and raise immediately when someone actually tries to call it.
try:
    from .bev_pool import bev_pool
except Exception as _bev_pool_err:
    _warnings.warn(
        'bev_pool CUDA extension failed to import: '
        f'{_bev_pool_err!r}. '
        'Camera+LiDAR BEVFusion will NOT work. '
        'Build the extension: bash projects/BEVFusion/scripts/build_inplace.sh '
        'or: sbatch tools/sbatch/meluxina_build_bevfusion_ops.sbatch',
        stacklevel=1,
    )

    def bev_pool(*args, **kwargs):
        raise RuntimeError(
            'bev_pool CUDA extension is not compiled. '
            'Run: bash projects/BEVFusion/scripts/build_inplace.sh '
            '(on a GPU node with nvcc available). '
            'Or: sbatch tools/sbatch/meluxina_build_bevfusion_ops.sbatch'
        )

# Always use mmcv voxelization ops.  The BEVFusion custom voxel_layer
# extension outputs coordinates in (X, Y, Z) order, but extract_pts_feat
# assumes the mmcv convention of (Z, Y, X).  Using the custom extension
# silently produces wrong coordinate reordering, causing the sparse
# encoder to crash with "N > 0 assert failed".
from mmcv.ops import DynamicScatter, Voxelization
from mmcv.ops import dynamic_scatter, voxelization

__all__ = [
    'bev_pool', 'Voxelization', 'voxelization', 'dynamic_scatter',
    'DynamicScatter'
]

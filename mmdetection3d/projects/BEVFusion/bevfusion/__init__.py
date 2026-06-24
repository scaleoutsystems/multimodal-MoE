from .sparse_encoder import BEVFusionSparseEncoder
from .hooks import MinEpochEarlyStoppingHook

__all__ = ['BEVFusionSparseEncoder', 'MinEpochEarlyStoppingHook']

# Some BEVFusion modules require custom CUDA extensions (e.g., bev_pool_ext).
# Keep sparse encoder importable even if those optional ops are unavailable.
# MoE modules have no CUDA-extension dependencies and must always be
# importable so that @MODELS.register_module() runs unconditionally.
from .moe_bev import (BEVMoEBlock, BEVResidualExpert,
                      JointModalityExpert, JointModalityMoEBlock,
                      ModalitySpecificMoEBlock)

__all__ += [
    'BEVMoEBlock', 'JointModalityMoEBlock', 'BEVResidualExpert',
    'JointModalityExpert', 'ModalitySpecificMoEBlock',
]

# The remaining BEVFusion modules require optional CUDA extensions
# (e.g. bev_pool_ext for the camera LSS path).  Keep them in a
# try/except so a missing extension doesn't break LiDAR-only runs.
try:
    from .bevfusion import BEVFusion
    from .bevfusion_camera_zero import BEVFusionCameraZero
    from .camera_only_bevfusion import CameraOnlyBEVFusion
    from .bevfusion_necks import GeneralizedLSSFPN
    from .depth_lss import DepthLSSTransform, LSSTransform
    from .loading import BEVLoadMultiViewImageFromFiles
    from .transformer import TransformerDecoderLayer
    from .transforms_3d import (BEVFusionGlobalRotScaleTrans,
                                BEVFusionRandomFlip3D, GridMask, ImageAug3D)
    from .transfusion_head import ConvFuser, TransFusionHead
    from .utils import (BBoxBEVL1Cost, HeuristicAssigner3D, HungarianAssigner3D,
                        IoU3DCost)

    __all__ += [
        'BEVFusion', 'BEVFusionCameraZero', 'CameraOnlyBEVFusion',
        'TransFusionHead', 'ConvFuser', 'ImageAug3D', 'GridMask',
        'GeneralizedLSSFPN', 'HungarianAssigner3D', 'BBoxBEVL1Cost',
        'IoU3DCost', 'HeuristicAssigner3D', 'DepthLSSTransform',
        'LSSTransform', 'BEVLoadMultiViewImageFromFiles',
        'TransformerDecoderLayer', 'BEVFusionRandomFlip3D',
        'BEVFusionGlobalRotScaleTrans',
    ]
except Exception:
    pass

from .sparse_encoder import BEVFusionSparseEncoder

__all__ = ['BEVFusionSparseEncoder']

# Some BEVFusion modules require custom CUDA extensions (e.g., bev_pool_ext).
# Keep sparse encoder importable even if those optional ops are unavailable.
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

    # MoE modules — importing the sub-package triggers @MODELS.register_module()
    # for BEVMoEBlock, FusionMoEBlock, BEVResidualExpert, and FusionExpert,
    # making them available to MODELS.build() in configs.
    from .moe_bev import (BEVMoEBlock, BEVResidualExpert, FusionExpert,
                          FusionMoEBlock)

    __all__ += [
        'BEVFusion', 'BEVFusionCameraZero', 'CameraOnlyBEVFusion',
        'TransFusionHead', 'ConvFuser', 'ImageAug3D', 'GridMask',
        'GeneralizedLSSFPN', 'HungarianAssigner3D', 'BBoxBEVL1Cost',
        'IoU3DCost', 'HeuristicAssigner3D', 'DepthLSSTransform',
        'LSSTransform', 'BEVLoadMultiViewImageFromFiles',
        'TransformerDecoderLayer', 'BEVFusionRandomFlip3D',
        'BEVFusionGlobalRotScaleTrans',
        'BEVMoEBlock', 'FusionMoEBlock', 'BEVResidualExpert', 'FusionExpert',
    ]
except Exception:
    pass

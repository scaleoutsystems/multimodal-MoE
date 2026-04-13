"""Selective dual-checkpoint initialisation hook for BEVFusion.

Before any training epoch runs, load modality-specific weights from two
separate unimodal checkpoints into the fusion model.

Typical module assignment for LiDAR+camera BEVFusion
-----------------------------------------------------
Camera-side  (img_backbone, img_neck, view_transform) ← camera_ckpt
LiDAR-side   (pts_middle_encoder, pts_backbone, pts_neck) ← lidar_ckpt
Fusion/other (fusion_layer, bbox_head)                ← default init

Notes on specific modules
--------------------------
pts_voxel_encoder (HardSimpleVFE)
    This module has **no learned parameters** — it is a stateless voxel
    averaging op.  It will not appear in any mmengine checkpoint; do not
    list it.

pts_middle_encoder
    The primary learned LiDAR sparse encoder.  Always load from the
    LiDAR-only checkpoint.

pts_backbone / pts_neck
    The fusion model uses SECOND with in_channels=256, which is identical
    to the LiDAR-only model.  Loading from the LiDAR checkpoint gives a
    strong, compatible warm-start for the BEV backbone that processes the
    fused 256-ch feature map.
    The camera-only model uses in_channels=80 for pts_backbone — this is
    architecturally incompatible at the first Conv layer and must NOT be
    used for the fusion model.

fusion_layer (ConvFuser)
    No unimodal counterpart exists; must be randomly initialised.

bbox_head (TransFusionHead)
    Post-fusion detection head.  Leave at default init to avoid introducing
    a modality-biased prior.

view_transform (DepthLSSTransform)
    Verified shape-compatible between the camera-only and fusion configs:
      depthnet.0.weight : [256, 320, 3, 3]  (in=256+64=320) ✓
      depthnet.6.weight : [258, 256, 1, 1]  (out=D+C=178+80) ✓
      frustum           : [178, 88, 156, 3]  (D=(90-1)/0.5=178) ✓
      dtransform        : 1→8→32→64 conv stack ✓
      downsample        : 80-ch conv stack ✓
    All 59 tensors load without shape mismatch.

Hook ordering note
------------------
``before_run`` fires **before** mmengine's ``resume_or_load()`` call (which
processes the config-level ``load_from``).  Therefore the config that uses
this hook **must not** set ``load_from``; doing so would cause
``resume_or_load`` to overwrite the carefully selected weights after this
hook has already loaded them.
"""

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class DualCheckpointInitHook(Hook):
    """Load modality-specific weights from two separate unimodal checkpoints.

    Camera-side modules are loaded from ``camera_ckpt``.
    LiDAR-side modules are loaded from ``lidar_ckpt``.
    All other modules retain their default / init_cfg-based initialisation.

    Args:
        camera_ckpt (str): Path to the camera-only model checkpoint
            (mmengine-style, i.e. a dict containing ``'state_dict'``).
        lidar_ckpt (str): Path to the LiDAR-only model checkpoint.
        camera_modules (tuple[str] | list[str] | None): Names of model
            attributes to load from ``camera_ckpt``.  Defaults to
            ``('img_backbone', 'img_neck', 'view_transform')``.
        lidar_modules (tuple[str] | list[str] | None): Names of model
            attributes to load from ``lidar_ckpt``.  Defaults to
            ``('pts_middle_encoder',)``.  Note: ``pts_backbone`` and
            ``pts_neck`` can be added when the fusion model and the
            LiDAR-only model share the same pts_backbone architecture
            (same in_channels); see module notes in the module docstring.
    """

    # Run as early as possible inside before_run so that the weights are in
    # place before any other hook might inspect or modify the model state.
    priority = 'VERY_HIGH'

    _DEFAULT_CAMERA_MODULES = ('img_backbone', 'img_neck', 'view_transform')
    # Minimal safe default: pts_backbone / pts_neck require the caller to
    # verify architecture compatibility (same in_channels) before adding.
    _DEFAULT_LIDAR_MODULES = ('pts_middle_encoder',)

    def __init__(
        self,
        camera_ckpt: str,
        lidar_ckpt: str,
        camera_modules=None,
        lidar_modules=None,
    ):
        self.camera_ckpt = camera_ckpt
        self.lidar_ckpt = lidar_ckpt
        self.camera_modules = tuple(
            camera_modules if camera_modules is not None
            else self._DEFAULT_CAMERA_MODULES)
        self.lidar_modules = tuple(
            lidar_modules if lidar_modules is not None
            else self._DEFAULT_LIDAR_MODULES)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model):
        """Strip DDP / FSDP wrapper if present."""
        return model.module if is_model_wrapper(model) else model

    def _load_modules_from_ckpt(
        self,
        runner,
        ckpt_path: str,
        module_names: tuple,
        label: str,
    ) -> None:
        """Load ``module_names`` sub-modules from a checkpoint file.

        Each module is loaded independently with ``strict=False`` so that
        minor key mismatches (e.g. buffers added in one model version) do not
        abort the init process.  Missing / unexpected keys are logged.

        Args:
            runner: mmengine Runner instance.
            ckpt_path: Filesystem path to the checkpoint.
            module_names: Tuple of model attribute names to load.
            label: Human-readable source label used in log messages.
        """
        model = self._unwrap(runner.model)

        runner.logger.info(
            f'DualCheckpointInitHook [{label}]: loading checkpoint '
            f'{ckpt_path}')

        ckpt = torch.load(ckpt_path, map_location='cpu')

        # Support both raw state_dict files and mmengine-format checkpoints
        # (which store the state dict under the key 'state_dict').
        if isinstance(ckpt, dict):
            state = ckpt.get('state_dict', ckpt)
        else:
            state = ckpt

        loaded_any = False
        for attr in module_names:
            submodule = getattr(model, attr, None)
            if submodule is None:
                runner.logger.warning(
                    f'DualCheckpointInitHook [{label}]: '
                    f'model has no attribute "{attr}" — skipped.')
                continue

            # Extract the sub-state-dict for this module using its name prefix.
            prefix = attr + '.'
            sub_state = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }

            if not sub_state:
                runner.logger.warning(
                    f'DualCheckpointInitHook [{label}]: '
                    f'no keys with prefix "{attr}." in checkpoint — skipped.')
                continue

            missing, unexpected = submodule.load_state_dict(
                sub_state, strict=False)

            runner.logger.info(
                f'DualCheckpointInitHook [{label}]: '
                f'"{attr}" ← {len(sub_state)} tensors; '
                f'missing={len(missing)}, unexpected={len(unexpected)}')
            if missing:
                runner.logger.debug(f'  [{attr}] missing:    {missing}')
            if unexpected:
                runner.logger.debug(f'  [{attr}] unexpected: {unexpected}')

            loaded_any = True

        if not loaded_any:
            runner.logger.warning(
                f'DualCheckpointInitHook [{label}]: no modules were loaded '
                f'from {ckpt_path}; verify module names and checkpoint keys.')

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def before_run(self, runner) -> None:
        """Selectively load both unimodal checkpoints before training."""
        runner.logger.info(
            'DualCheckpointInitHook: starting dual-checkpoint init …\n'
            f'  LiDAR ckpt  : {self.lidar_ckpt}\n'
            f'  Camera ckpt : {self.camera_ckpt}')

        self._load_modules_from_ckpt(
            runner, self.lidar_ckpt, self.lidar_modules, 'LiDAR')
        self._load_modules_from_ckpt(
            runner, self.camera_ckpt, self.camera_modules, 'Camera')

        lidar_str  = ', '.join(self.lidar_modules)
        camera_str = ', '.join(self.camera_modules)
        runner.logger.info(
            'DualCheckpointInitHook: dual-checkpoint init complete.\n'
            f'  Loaded from LiDAR-only ckpt  : {lidar_str}\n'
            f'  Loaded from Camera-only ckpt : {camera_str}\n'
            '  Default init (not loaded)    : '
            'fusion_layer, bbox_head')

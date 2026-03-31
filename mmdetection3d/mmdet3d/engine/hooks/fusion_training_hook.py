"""Two-phase training strategy for multi-modal BEVFusion.

Phase 1 (epochs 1..freeze_lidar_epochs):
    Freeze the pretrained LiDAR branch so that 100 % of gradient signal
    flows through the camera encoder, view transform, and fusion layer.
    This forces the camera path to learn to produce useful BEV features
    rather than being suppressed by the already-trained LiDAR path.

Phase 2 (epochs freeze_lidar_epochs+1..end):
    Unfreeze everything for joint end-to-end fine-tuning.  Combined with
    ``paramwise_cfg`` (lr_mult < 1 for LiDAR modules) this lets the
    LiDAR branch adapt gently while the camera branch continues learning.

Usage in config::

    custom_hooks = [
        ...,
        dict(type='FusionTrainingStrategyHook',
             freeze_lidar_epochs=5,
             lidar_prefixes=['pts_voxel_encoder',
                             'pts_middle_encoder',
                             'pts_backbone_2d',
                             'pts_neck']),
    ]
"""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


def _unwrap(runner):
    m = runner.model
    return m.module if is_model_wrapper(m) else m


@HOOKS.register_module()
class FusionTrainingStrategyHook(Hook):
    """Freeze / unfreeze the LiDAR branch during BEVFusion training.

    Args:
        freeze_lidar_epochs: Number of initial epochs during which
            LiDAR parameters are frozen (requires_grad=False).
        lidar_prefixes: Module-name prefixes that identify the LiDAR
            branch parameters to freeze.
    """

    priority = 'ABOVE_NORMAL'

    _LIDAR_DEFAULTS = (
        'pts_middle_encoder',
        'pts_backbone',
        'pts_neck',
    )

    def __init__(
        self,
        freeze_lidar_epochs: int = 5,
        lidar_prefixes: tuple | list | None = None,
    ):
        self.freeze_lidar_epochs = freeze_lidar_epochs
        self.lidar_prefixes = tuple(lidar_prefixes or self._LIDAR_DEFAULTS)
        self._frozen = False

    def _is_lidar_param(self, name: str) -> bool:
        return any(name.startswith(p) for p in self.lidar_prefixes)

    def _set_lidar_grad(self, runner, requires_grad: bool) -> None:
        model = _unwrap(runner)
        count = 0
        for name, param in model.named_parameters():
            if self._is_lidar_param(name):
                param.requires_grad_(requires_grad)
                count += 1
        state = 'unfrozen' if requires_grad else 'frozen'
        runner.logger.info(
            f'FusionTrainingStrategyHook: {state} {count} LiDAR parameters '
            f'(prefixes: {self.lidar_prefixes})')

    def before_train(self, runner) -> None:
        if self.freeze_lidar_epochs > 0:
            self._set_lidar_grad(runner, requires_grad=False)
            self._frozen = True

    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch + 1  # 1-indexed
        if self._frozen and epoch > self.freeze_lidar_epochs:
            self._set_lidar_grad(runner, requires_grad=True)
            self._frozen = False

"""Hard-freeze hook for the LiDAR-only branch of a BEVFusion model."""
from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS

_DEFAULT_LIDAR_MODULES = (
    'pts_voxel_encoder',
    'pts_middle_encoder',
    'pts_backbone',
    'pts_neck',
)


@HOOKS.register_module()
class FreezeLidarBranchHook(Hook):
    """Hard-freeze named LiDAR-only sub-modules in a BEVFusion model.

    Performs two actions during the frozen phase:

    1. ``before_run`` — sets ``requires_grad_(False)`` on every listed
       sub-module so that no gradient is computed or accumulated for their
       parameters during the backward pass.  Also calls ``eval()`` to put
       their BatchNorm layers into inference mode (running stats fixed).

    2. ``before_train_epoch`` — re-applies ``eval()`` after each call to
       ``model.train()`` that the training loop issues at the start of every
       epoch (which would otherwise silently re-enable BN's update of running
       statistics).

    Staged unfreeze
    ---------------
    If ``unfreeze_epoch`` is given (0-indexed, matching ``runner.epoch``),
    the frozen modules are re-enabled with ``requires_grad_(True)`` and
    ``train()`` at the start of that epoch.  From then on the hook no longer
    intervenes, so the normal training loop controls their mode.

    DDP note
    --------
    DDP registers gradient-reduction hooks on all ``requires_grad=True``
    parameters at model-wrap time (before any training hook fires).  After
    this hook sets ``requires_grad_(False)``, those parameters will never
    accumulate a gradient, and DDP would hang waiting for their buckets to
    be marked ready.  To prevent this, the config **must** set::

        env_cfg = dict(dist_cfg=dict(backend='nccl',
                                     find_unused_parameters=True))

    With ``find_unused_parameters=True``, DDP traces the autograd graph
    after each forward pass and marks unfired buckets as ready automatically.
    This also handles the unfreeze transition cleanly — once gradients start
    flowing through the re-enabled modules, DDP picks them up without any
    additional configuration change.

    Args:
        module_names (tuple[str]): Attribute names on the model (or its
            ``module`` wrapper) that should be frozen.  Defaults to the four
            LiDAR-only BEVFusion sub-modules.
        unfreeze_epoch (int | None): 0-indexed epoch at which to permanently
            unfreeze the listed modules.  ``None`` means stay frozen for the
            entire run.
    """

    def __init__(self, module_names=_DEFAULT_LIDAR_MODULES,
                 unfreeze_epoch=None):
        self.module_names = tuple(module_names)
        self.unfreeze_epoch = unfreeze_epoch
        self._unfrozen = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(model):
        """Return the bare model, stripping DDP / FSDP wrappers if present."""
        return model.module if hasattr(model, 'module') else model

    def _apply_freeze(self, runner, *, set_requires_grad):
        model = self._unwrap(runner.model)
        for name in self.module_names:
            m = getattr(model, name, None)
            if m is None:
                runner.logger.warning(
                    f'FreezeLidarBranchHook: "{name}" not found on model — skipped.')
                continue
            if set_requires_grad:
                m.requires_grad_(False)
                runner.logger.info(
                    f'FreezeLidarBranchHook: {name} → requires_grad=False, eval()')
            m.eval()

    def _apply_unfreeze(self, runner):
        """sets self._unfrozen = True so the hook knows it's done and stops intervening.
        Also sets requires_grad_(True) and train() on the modules."""
        model = self._unwrap(runner.model)
        for name in self.module_names:
            m = getattr(model, name, None)
            if m is None:
                continue
            m.requires_grad_(True)
            m.train()
            runner.logger.info(
                f'FreezeLidarBranchHook: {name} → unfrozen '
                f'(requires_grad=True, train()) at epoch {runner.epoch}')
        self._unfrozen = True

    # ------------------------------------------------------------------
    # Hook entry points
    # ------------------------------------------------------------------

    def before_run(self, runner):
        """Called once, before any training starts. Does the full freeze: requires_grad_(False) + eval()."""
        self._apply_freeze(runner, set_requires_grad=True)

    def before_train_epoch(self, runner):
        """Manage freeze/unfreeze state at the start of each epoch."""
        if self._unfrozen:
            return

        if (self.unfreeze_epoch is not None
                and runner.epoch >= self.unfreeze_epoch):
            self._apply_unfreeze(runner)
        else:
            # Re-apply eval() to override the training loop's model.train() call.
            self._apply_freeze(runner, set_requires_grad=False)

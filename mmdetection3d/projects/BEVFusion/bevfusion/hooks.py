from mmengine.hooks import EarlyStoppingHook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MinEpochEarlyStoppingHook(EarlyStoppingHook):
    """Early stopping hook with a minimum-epoch guard.

    Identical to :class:`mmengine.hooks.EarlyStoppingHook` but will not
    trigger before ``min_epochs`` have been completed, regardless of the
    patience counter.

    Args:
        min_epochs (int): Minimum number of epochs that must complete before
            early stopping is allowed to fire.  Defaults to 0 (same behaviour
            as the base class).
        **kwargs: Forwarded verbatim to :class:`EarlyStoppingHook`.
    """

    def __init__(self, min_epochs: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.min_epochs = min_epochs

    def after_val_epoch(self, runner, metrics):
        if runner.epoch < self.min_epochs:
            return
        super().after_val_epoch(runner, metrics)

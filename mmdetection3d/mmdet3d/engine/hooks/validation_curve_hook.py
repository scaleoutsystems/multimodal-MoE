"""Plot validation AP curves over training epochs.

Produces a continuously-updated PNG in ``{work_dir}/visualizations/`` so
progress can be monitored during long runs.  Supports an arbitrary list of
metric keys; defaults to AP@1m from CenterDistanceMetric.
"""

import json
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmengine.hooks import Hook

from mmdet3d.registry import HOOKS


@HOOKS.register_module()
class ValidationCurveHook(Hook):
    """Record and plot validation metrics over epochs.

    After each validation epoch the hook appends the requested metric
    values and re-draws the plot.  A JSON sidecar is also written so the
    data can be loaded for custom post-hoc analysis.

    Args:
        metric_keys: Metric names to track.  Each gets its own line.
        filename: Output filename (without extension) inside
            ``{work_dir}/visualizations/``.
    """

    priority = 'LOW'

    def __init__(
        self,
        metric_keys=('mAP_1.0m',),
        filename='val_curve_ap',
    ):
        self.metric_keys = list(metric_keys)
        self.filename = filename
        self.history: dict[str, list[tuple[int, float]]] = {
            k: [] for k in self.metric_keys
        }

    def before_run(self, runner) -> None:
        """Restore history from an existing JSON sidecar when resuming.

        This lets the plot accumulate correctly across SLURM restarts: the
        previous epochs are loaded back so the curve continues from where it
        left off rather than restarting from zero.
        """
        vis_dir = osp.join(runner.work_dir, 'visualizations')
        json_path = osp.join(vis_dir, f'{self.filename}.json')
        if not osp.isfile(json_path):
            return
        try:
            with open(json_path) as f:
                saved = json.load(f)
            for key in self.metric_keys:
                if key in saved and saved[key]:
                    self.history[key] = [tuple(pt) for pt in saved[key]]
            runner.logger.info(
                f'ValidationCurveHook: restored {len(self.history[self.metric_keys[0]])} '
                f'epoch(s) of history from {json_path}')
        except Exception as exc:
            runner.logger.warning(
                f'ValidationCurveHook: could not restore history from '
                f'{json_path}: {exc}')

    def after_val_epoch(self, runner, metrics=None) -> None:
        if metrics is None:
            return
        epoch = runner.epoch
        updated = False
        for key in self.metric_keys:
            if key in metrics:
                # Replace any existing entry for this epoch so that resumed
                # runs that re-run an epoch don't create duplicate points.
                self.history[key] = [
                    (e, v) for e, v in self.history[key] if e != epoch
                ]
                self.history[key].append((epoch, float(metrics[key])))
                updated = True

        if not updated:
            return

        vis_dir = osp.join(runner.work_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        self._save_json(vis_dir)
        self._save_plot(vis_dir)

    def _save_json(self, vis_dir):
        path = osp.join(vis_dir, f'{self.filename}.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _save_plot(self, vis_dir):
        fig, ax = plt.subplots(figsize=(8, 5))

        for key in self.metric_keys:
            pts = self.history.get(key, [])
            if not pts:
                continue
            epochs, values = zip(*pts)
            ax.plot(epochs, values, marker='o', markersize=4, label=key)

            best_val = max(values)
            best_ep = epochs[values.index(best_val)]
            ax.annotate(
                f'{best_val:.4f}',
                xy=(best_ep, best_val),
                xytext=(0, -14),
                textcoords='offset points',
                fontsize=8,
                ha='center',
                va='top',
                color='green',
                fontweight='bold',
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP')
        ax.set_title('Validation AP over training')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        if self.history.get(self.metric_keys[0]):
            all_epochs = [e for e, _ in self.history[self.metric_keys[0]]]
            ax.set_xticks(all_epochs)

        fig.tight_layout()
        fig.savefig(osp.join(vis_dir, f'{self.filename}.png'), dpi=150)
        plt.close(fig)

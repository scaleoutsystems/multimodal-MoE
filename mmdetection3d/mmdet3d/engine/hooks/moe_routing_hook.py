"""MoE routing diagnostics hook.

Tracks four mathematically distinct routing quantities per epoch (train + val):

    dispatch_mass_per_expert
        Mean post-top-k routed weight per expert, averaged over samples.
        Uses the actual dispatch weights (topk_weights) after top-k masking
        and any renormalisation applied by the model.
        Reflects true expert contribution to the output.

    dense_mean_prob_per_expert
        Mean pre-top-k softmax probability per expert, averaged over samples.
        Computed from the full softmax over all experts before top-k masking.
        Reflects the router's continuous preference signal.

    top1_selection_freq_per_expert
        Fraction of samples for which expert e is the rank-1 selected expert
        (i.e. topk_idx[:, 0] == e).

    topk_selection_freq_per_expert
        Fraction of samples for which expert e appears anywhere in the
        selected top-k set (i.e. in any column of topk_idx).
        Equals top1_selection_freq when k=1.

All four quantities are computed from the same _moe_info dict
(full_softmax_probs, topk_idx, topk_weights) so they are directly comparable.

Outputs
-------
Per epoch, inside <work_dir>/moe_routing/:

    dispatch_mass_train_epochN.json / dispatch_mass_val_epochN.json
    dense_mean_prob_train_epochN.json / dense_mean_prob_val_epochN.json
    dispatch_mass_per_expert.png      — dispatch mass over epochs (line plot)
    dense_mean_prob_per_expert.png    — dense softmax prob over epochs (line plot)
    routing_summary_epochN.json       — val-epoch summary with all 4 metrics + AP

Hook C (modality-specific group mass):
    group_mass_train_epochN.json / group_mass_val_epochN.json
    group_mass.png

Config example
--------------
    dict(
        type='MoERoutingHook',
        num_experts=6,
        ap_metric_keys=['mAP_0.5m', 'mAP_0.50'],   # both APs logged in summary
        enable_hook_c=True,   # only has effect for modality_specific_moe
    )
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _unwrap(model):
    return model.module if is_model_wrapper(model) else model


def _get_moe_modules(model) -> Dict[str, Any]:
    """Return {attr_name: module} for all MoE blocks (any variant)."""
    m = _unwrap(model)
    result = {}
    for name in ('bev_moe', 'modality_specific_moe', 'joint_modality_moe'):
        attr = getattr(m, name, None)
        if attr is not None and hasattr(attr, '_moe_info'):
            result[name] = attr
    return result


@HOOKS.register_module()
class MoERoutingHook(Hook):
    """Routing diagnostics hook — tracks dispatch mass, dense softmax
    probability, top-1 selection frequency, and top-k selection frequency
    per expert over training and validation epochs.

    Args:
        num_experts:      Total expert count (must match model config).
        enable_hook_c:    Enable modality-group mass tracking. Only has effect
                          for modality_specific_moe. Default True.
        ap_metric_keys:   AP metric keys to record in routing_summary_epochN.json.
                          Defaults to ['mAP_0.5m', 'mAP_0.50'].
        out_subdir:       Subdirectory inside runner.work_dir for artifacts.
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        num_experts: int,
        enable_hook_a: bool = True,   # kept for backwards-compat, always active
        enable_hook_b: bool = True,   # kept for backwards-compat, always active
        enable_hook_c: bool = True,
        ap_metric_key: Optional[str] = None,      # legacy single-key compat
        ap_metric_keys: Optional[List[str]] = None,
        out_subdir: str = 'moe_routing',
    ):
        self.num_experts   = num_experts
        self.enable_hook_c = enable_hook_c
        self.out_subdir    = out_subdir

        # Resolve AP keys: new list param takes priority, legacy scalar fallback.
        if ap_metric_keys:
            self.ap_metric_keys: List[str] = list(ap_metric_keys)
        elif ap_metric_key:
            self.ap_metric_keys = [ap_metric_key, 'mAP_0.50']
        else:
            self.ap_metric_keys = ['mAP_0.5m', 'mAP_0.50']

        E = num_experts

        # ── Per-epoch accumulators (reset after each epoch) ─────────────
        # dispatch_mass: sum of topk_weights scattered per expert
        self._tr_dispatch: List[float] = [0.0] * E
        self._va_dispatch: List[float] = [0.0] * E
        # dense_prob: sum of full pre-top-k softmax probs per expert
        self._tr_dense:    List[float] = [0.0] * E
        self._va_dense:    List[float] = [0.0] * E
        # top1_freq: count of samples where expert is rank-1 selection
        self._tr_top1:     List[int]   = [0] * E
        self._va_top1:     List[int]   = [0] * E
        # topk_freq: count of samples where expert appears anywhere in top-k
        self._tr_topk:     List[int]   = [0] * E
        self._va_topk:     List[int]   = [0] * E
        # sample counts
        self._tr_n: int = 0
        self._va_n: int = 0

        # Modality group mass (Hook C; modality_specific_moe only)
        self._tr_cam_mass:   float = 0.0
        self._tr_lidar_mass: float = 0.0
        self._va_cam_mass:   float = 0.0
        self._va_lidar_mass: float = 0.0

        # Epoch-history for line plots
        self._tr_dispatch_hist: Dict[int, List[float]] = {}
        self._va_dispatch_hist: Dict[int, List[float]] = {}
        self._tr_dense_hist:    Dict[int, List[float]] = {}
        self._va_dense_hist:    Dict[int, List[float]] = {}
        self._tr_group_hist:    Dict[int, tuple] = {}
        self._va_group_hist:    Dict[int, tuple] = {}

        self._out_dir: Optional[str] = None

    # ── Setup ──────────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        runner.logger.info(f'MoERoutingHook: artifacts → {self._out_dir}')

    # ── Per-iteration accumulation ─────────────────────────────────────────

    def _accumulate(self, runner, phase: str) -> None:
        """Accumulate all four routing metrics from _moe_info."""
        moe_modules = _get_moe_modules(runner.model)
        if not moe_modules:
            return

        is_train = (phase == 'train')
        dispatch = self._tr_dispatch if is_train else self._va_dispatch
        dense    = self._tr_dense    if is_train else self._va_dense
        top1     = self._tr_top1     if is_train else self._va_top1
        topk     = self._tr_topk     if is_train else self._va_topk

        for attr_name, mod in moe_modules.items():
            info = getattr(mod, '_moe_info', None)
            if info is None:
                continue

            full_probs   = info.get('full_softmax_probs')  # (B, E) pre-top-k softmax
            topk_idx     = info.get('topk_idx')            # (B, k)
            topk_weights = info.get('topk_weights')        # (B, k)
            if full_probs is None or topk_idx is None or topk_weights is None:
                continue

            B, E = full_probs.shape
            k    = topk_idx.shape[1]
            E    = min(E, self.num_experts)

            # dense_mean_prob: sum of full (pre-top-k) softmax probs per expert
            for e in range(E):
                dense[e] += float(full_probs[:, e].sum().item())

            # dispatch_mass: scatter topk_weights per expert
            for b in range(B):
                for j in range(k):
                    eidx = int(topk_idx[b, j].item())
                    if 0 <= eidx < self.num_experts:
                        dispatch[eidx] += float(topk_weights[b, j].item())

            # top1_selection_freq: rank-1 expert per sample = topk_idx[:, 0]
            for b in range(B):
                eidx = int(topk_idx[b, 0].item())
                if 0 <= eidx < self.num_experts:
                    top1[eidx] += 1

            # topk_selection_freq: any expert in the top-k set
            for b in range(B):
                seen = set()
                for j in range(k):
                    eidx = int(topk_idx[b, j].item())
                    if 0 <= eidx < self.num_experts and eidx not in seen:
                        topk[eidx] += 1
                        seen.add(eidx)

            if is_train:
                self._tr_n += B
            else:
                self._va_n += B

            # Hook C: modality group mass (modality_specific_moe only)
            if self.enable_hook_c and attr_name == 'modality_specific_moe':
                cam_m   = info.get('cam_group_mass',   0.0)
                lidar_m = info.get('lidar_group_mass', 0.0)
                if is_train:
                    self._tr_cam_mass   += cam_m
                    self._tr_lidar_mass += lidar_m
                else:
                    self._va_cam_mass   += cam_m
                    self._va_lidar_mass += lidar_m

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        self._accumulate(runner, 'train')

    def after_val_iter(self, runner, batch_idx, data_batch=None,
                       outputs=None) -> None:
        self._accumulate(runner, 'val')

    # ── Helpers: normalise and build metric dicts ──────────────────────────

    def _norm_dispatch(self, raw: List[float], n: int) -> List[float]:
        """Mean dispatch weight per expert (divide raw sum by n_samples)."""
        denom = float(n) + 1e-8
        return [v / denom for v in raw]

    def _norm_dense(self, raw: List[float], n: int) -> List[float]:
        """Mean dense softmax prob per expert."""
        denom = float(n) + 1e-8
        return [v / denom for v in raw]

    def _freq(self, counts: List[int], n: int) -> List[float]:
        """Selection frequency = count / n_samples."""
        denom = float(n) + 1e-8
        return [c / denom for c in counts]

    def _to_expert_dict(self, vals: List[float]) -> Dict[str, float]:
        return {f'expert_{i}': round(v, 8) for i, v in enumerate(vals)}

    def _build_metrics(
        self,
        dispatch: List[float],
        dense: List[float],
        top1: List[int],
        topk: List[int],
        n: int,
    ) -> Dict[str, Any]:
        """Build the four canonical metric dicts for a given split."""
        return {
            'num_samples': n,
            'dispatch_mass_per_expert':         self._to_expert_dict(
                self._norm_dispatch(dispatch, n)),
            'dense_mean_prob_per_expert':        self._to_expert_dict(
                self._norm_dense(dense, n)),
            'top1_selection_freq_per_expert':    self._to_expert_dict(
                self._freq(top1, n)),
            'topk_selection_freq_per_expert':    self._to_expert_dict(
                self._freq(topk, n)),
        }

    # ── Epoch-trend plots ──────────────────────────────────────────────────

    def _save_line_plot(
        self,
        tr_hist: Dict[int, List[float]],
        va_hist: Dict[int, List[float]],
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not _MPL_AVAILABLE:
            return
        if not tr_hist and not va_hist:
            return

        colors = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(8, 5))
        for eidx in range(self.num_experts):
            col = colors[eidx % len(colors)]
            if tr_hist:
                epochs = sorted(tr_hist)
                ax.plot(epochs, [tr_hist[e][eidx] for e in epochs],
                        color=col, linestyle='-',  label=f'E{eidx} train')
            if va_hist:
                epochs = sorted(va_hist)
                ax.plot(epochs, [va_hist[e][eidx] for e in epochs],
                        color=col, linestyle='--', label=f'E{eidx} val')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self._out_dir, filename), dpi=150)
        plt.close(fig)

    def _save_dispatch_mass_plot(self) -> None:
        self._save_line_plot(
            self._tr_dispatch_hist, self._va_dispatch_hist,
            ylabel='Mean dispatch weight (post-top-k, per sample)',
            title='Dispatch Mass per Expert Over Epochs (Post-Top-k)',
            filename='dispatch_mass_per_expert.png',
        )

    def _save_dense_prob_plot(self) -> None:
        self._save_line_plot(
            self._tr_dense_hist, self._va_dense_hist,
            ylabel='Mean softmax probability (pre-top-k, per sample)',
            title='Dense Mean Probability per Expert Over Epochs (Pre-Top-k)',
            filename='dense_mean_prob_per_expert.png',
        )

    def _save_group_mass_plot(self) -> None:
        if not _MPL_AVAILABLE:
            return
        if not self._tr_group_hist and not self._va_group_hist:
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        if self._tr_group_hist:
            ep = sorted(self._tr_group_hist)
            ax.plot(ep, [self._tr_group_hist[e][0] for e in ep], 'b-',
                    label='Camera group (train)')
            ax.plot(ep, [self._tr_group_hist[e][1] for e in ep], 'r-',
                    label='LiDAR group (train)')
        if self._va_group_hist:
            ep = sorted(self._va_group_hist)
            ax.plot(ep, [self._va_group_hist[e][0] for e in ep], 'b--',
                    label='Camera group (val)')
            ax.plot(ep, [self._va_group_hist[e][1] for e in ep], 'r--',
                    label='LiDAR group (val)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Routing mass fraction')
        ax.set_title('Modality-Group Routing Mass Over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self._out_dir, 'group_mass.png'), dpi=150)
        plt.close(fig)

    # ── JSON helpers ───────────────────────────────────────────────────────

    def _save_json(self, data: dict, filename: str) -> None:
        with open(os.path.join(self._out_dir, filename), 'w') as f:
            json.dump(data, f, indent=2)

    # ── End-of-epoch: save and reset ───────────────────────────────────────

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        n = self._tr_n
        if n == 0:
            self._reset_train()
            return

        metrics = self._build_metrics(
            self._tr_dispatch, self._tr_dense,
            self._tr_top1, self._tr_topk, n)

        # Record histories for line plots
        dispatch_vals = list(metrics['dispatch_mass_per_expert'].values())
        dense_vals    = list(metrics['dense_mean_prob_per_expert'].values())
        self._tr_dispatch_hist[epoch] = dispatch_vals
        self._tr_dense_hist[epoch]    = dense_vals

        # Save per-epoch JSONs
        self._save_json(
            {'epoch': epoch, 'split': 'train', **metrics},
            f'dispatch_mass_train_epoch{epoch}.json')
        self._save_json(
            {'epoch': epoch, 'split': 'train', **metrics},
            f'dense_mean_prob_train_epoch{epoch}.json')

        # Redraw epoch-trend plots
        self._save_dispatch_mass_plot()
        self._save_dense_prob_plot()

        # Hook C: modality group mass
        if self.enable_hook_c and (self._tr_cam_mass + self._tr_lidar_mass) > 0:
            total = self._tr_cam_mass + self._tr_lidar_mass + 1e-8
            cam_f   = self._tr_cam_mass   / total
            lidar_f = self._tr_lidar_mass / total
            self._tr_group_hist[epoch] = (cam_f, lidar_f)
            self._save_json(
                {'epoch': epoch, 'split': 'train',
                 'cam_group_mass_frac': cam_f, 'lidar_group_mass_frac': lidar_f},
                f'group_mass_train_epoch{epoch}.json')
            self._save_group_mass_plot()

        self._reset_train()

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        epoch = runner.epoch
        metrics = metrics or {}
        n = self._va_n
        if n == 0:
            self._reset_val()
            return

        routing_metrics = self._build_metrics(
            self._va_dispatch, self._va_dense,
            self._va_top1, self._va_topk, n)

        dispatch_vals = list(routing_metrics['dispatch_mass_per_expert'].values())
        dense_vals    = list(routing_metrics['dense_mean_prob_per_expert'].values())
        self._va_dispatch_hist[epoch] = dispatch_vals
        self._va_dense_hist[epoch]    = dense_vals

        # Per-epoch JSONs
        self._save_json(
            {'epoch': epoch, 'split': 'val', **routing_metrics},
            f'dispatch_mass_val_epoch{epoch}.json')
        self._save_json(
            {'epoch': epoch, 'split': 'val', **routing_metrics},
            f'dense_mean_prob_val_epoch{epoch}.json')

        self._save_dispatch_mass_plot()
        self._save_dense_prob_plot()

        # Val summary: all 4 metrics + all requested AP keys
        ap_values = {
            k: float(metrics.get(k, -1.0))
            for k in self.ap_metric_keys
        }
        summary = {
            'epoch':  epoch,
            'split':  'val',
            **ap_values,
            **routing_metrics,
        }
        # modality group mass (if available)
        if self.enable_hook_c and (self._va_cam_mass + self._va_lidar_mass) > 0:
            total = self._va_cam_mass + self._va_lidar_mass + 1e-8
            summary['cam_group_mass_frac']   = self._va_cam_mass   / total
            summary['lidar_group_mass_frac'] = self._va_lidar_mass / total
        self._save_json(summary, f'routing_summary_epoch{epoch}.json')

        # Hook C: modality group mass
        if self.enable_hook_c and (self._va_cam_mass + self._va_lidar_mass) > 0:
            total = self._va_cam_mass + self._va_lidar_mass + 1e-8
            cam_f   = self._va_cam_mass   / total
            lidar_f = self._va_lidar_mass / total
            self._va_group_hist[epoch] = (cam_f, lidar_f)
            self._save_json(
                {'epoch': epoch, 'split': 'val',
                 'cam_group_mass_frac': cam_f, 'lidar_group_mass_frac': lidar_f},
                f'group_mass_val_epoch{epoch}.json')
            self._save_group_mass_plot()

        self._reset_val()

    # ── Resets ─────────────────────────────────────────────────────────────

    def _reset_train(self) -> None:
        E = self.num_experts
        self._tr_dispatch    = [0.0] * E
        self._tr_dense       = [0.0] * E
        self._tr_top1        = [0]   * E
        self._tr_topk        = [0]   * E
        self._tr_n           = 0
        self._tr_cam_mass    = 0.0
        self._tr_lidar_mass  = 0.0

    def _reset_val(self) -> None:
        E = self.num_experts
        self._va_dispatch    = [0.0] * E
        self._va_dense       = [0.0] * E
        self._va_top1        = [0]   * E
        self._va_topk        = [0]   * E
        self._va_n           = 0
        self._va_cam_mass    = 0.0
        self._va_lidar_mass  = 0.0

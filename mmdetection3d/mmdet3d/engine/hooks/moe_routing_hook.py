"""MoE routing diagnostics hook — Hooks A, B, and C.

Overview
--------
MoERoutingHook implements three complementary diagnostics for MoE routing
analysis during BEVFusion training and validation.  All three are bundled in
one hook to share the per-iteration routing mass accumulation loop.

Hook A — Expert routing mass over epochs (train + val)
    For each epoch, accumulates the *soft* routing mass per expert: the sum of
    top-k routing weights assigned to each expert across all samples.  This
    reflects actual expert contribution, not just selection frequency.

    Saves:
        routing_mass_train_epochN.json   — {expert_id: mass} for training
        routing_mass_val_epochN.json     — {expert_id: mass} for validation
        routing_mass.png                 — line plot, one curve per expert
                                           (train curves solid, val dashed)
    Enabled via: enable_hook_a=True (default True)

Hook B — Performance-aligned routing summary (val only)
    After each validation epoch, saves a compact table of:
        global AP@0.5m (from the mmengine metrics dict)
        average routing mass per expert over the val split
    For modality_specific variant, also includes cam-group and lidar-group mass.

    Saves:
        routing_summary_epochN.json

    Enabled via: enable_hook_b=True (default True)

Hook C — Modality-group routing mass over epochs (modality_specific only)
    For each epoch, accumulates and logs the total routing mass to the camera
    expert group vs. the LiDAR expert group.

    Saves:
        group_mass_train_epochN.json
        group_mass_val_epochN.json
        group_mass.png               — epoch-vs-mass plot, two curves

    Only active when the model has a modality_specific_moe block AND the
    moe_info dict contains cam_group_mass / lidar_group_mass keys.
    Enabled via: enable_hook_c=True (default True)

Which variant uses which hooks
-------------------------------
    joint_modality (FusionMoEBlock as fusion_layer):   A, B
    modality_specific (ModalitySpecificMoEBlock):       A, B, C
    fusion_then_moe (BEVMoEBlock on bev_moe):          A, B

Output location
---------------
All artifacts are saved to:
    <runner.work_dir>/moe_routing/

Config example
--------------
    custom_hooks = [
        ...
        dict(
            type='MoERoutingHook',
            num_experts=4,             # total expert count (must match model)
            enable_hook_a=True,
            enable_hook_b=True,
            enable_hook_c=True,        # only has effect for modality_specific
            ap_metric_key='mAP_0.5m', # key in the mmengine metrics dict for AP
        ),
    ]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

# ──────────────────────────────────────────────────────────────────────────────
# Optional matplotlib import — hooks degrade gracefully if not installed.
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend; safe on compute nodes
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _unwrap(model):
    """Strip DDP / FSDP wrapper."""
    return model.module if is_model_wrapper(model) else model


def _get_moe_modules(model) -> Dict[str, Any]:
    """Return a dict of {attr_name: module} for all MoE blocks on the model.

    Auto-detects by looking for attributes that have a ``_moe_info`` field.
    Covers: cam_moe, lidar_moe, bev_moe, modality_specific_moe, and
    fusion_layer (when it is a FusionMoEBlock).
    """
    m = _unwrap(model)
    result = {}
    for name in ('cam_moe', 'lidar_moe', 'bev_moe',
                 'modality_specific_moe', 'fusion_layer'):
        attr = getattr(m, name, None)
        if attr is not None and hasattr(attr, '_moe_info'):
            result[name] = attr
    return result


def _scatter_routing_mass(
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    accumulator: List[float],
) -> None:
    """Add per-sample, per-expert routing mass to accumulator in-place.

    Args:
        topk_idx:     (B, k) selected expert indices (int tensor).
        topk_weights: (B, k) softmax weights for selected experts (float tensor).
        accumulator:  List of length E (num_experts); modified in-place.
    """
    B, k = topk_idx.shape
    for b in range(B):
        for j in range(k):
            eidx = int(topk_idx[b, j].item())
            w    = float(topk_weights[b, j].item())
            if 0 <= eidx < len(accumulator):
                accumulator[eidx] += w


@HOOKS.register_module()
class MoERoutingHook(Hook):
    """Diagnostics hook for MoE routing — Hooks A, B, and C.

    Args:
        num_experts:    Total number of experts in the MoE block.
                        Must match the model configuration.
        enable_hook_a:  Enable Hook A (routing mass per expert). Default True.
        enable_hook_b:  Enable Hook B (val summary with AP). Default True.
        enable_hook_c:  Enable Hook C (modality group mass). Default True.
                        Only has effect when modality_specific_moe is present.
        ap_metric_key:  Key in the mmengine metrics dict that holds the AP
                        value for Hook B.  Default 'mAP_0.5m'.
        out_subdir:     Subdirectory inside runner.work_dir for artifacts.
                        Default 'moe_routing'.
    """

    # Run after the model forward but before the optimizer step; low priority
    # so other hooks have already had a chance to run.
    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        num_experts: int,
        enable_hook_a: bool = True,
        enable_hook_b: bool = True,
        enable_hook_c: bool = True,
        ap_metric_key: str = 'mAP_0.5m',
        out_subdir: str = 'moe_routing',
    ):
        self.num_experts   = num_experts
        self.enable_hook_a = enable_hook_a
        self.enable_hook_b = enable_hook_b
        self.enable_hook_c = enable_hook_c
        self.ap_metric_key = ap_metric_key
        self.out_subdir    = out_subdir

        # ── Per-epoch accumulators ─────────────────────────────────────
        # Routing mass: sum of topk_weights per expert.
        self._train_mass:  List[float] = [0.0] * num_experts
        self._val_mass:    List[float] = [0.0] * num_experts
        self._train_n:     int = 0  # number of samples seen in training
        self._val_n:       int = 0

        # Modality group mass (Hook C; only for modality_specific_moe).
        self._train_cam_mass:   float = 0.0
        self._train_lidar_mass: float = 0.0
        self._val_cam_mass:     float = 0.0
        self._val_lidar_mass:   float = 0.0

        # History across epochs for plotting.
        # {epoch: [mass_e0, mass_e1, ...]} for each expert.
        self._train_mass_history: Dict[int, List[float]] = {}
        self._val_mass_history:   Dict[int, List[float]] = {}
        # Group history: {epoch: (cam_mass, lidar_mass)}
        self._train_group_history: Dict[int, tuple] = {}
        self._val_group_history:   Dict[int, tuple]  = {}

        self._out_dir: Optional[str] = None  # set in before_run

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        runner.logger.info(
            f'MoERoutingHook: artifacts → {self._out_dir}')

    # ──────────────────────────────────────────────────────────────────────
    # Per-iteration accumulation
    # ──────────────────────────────────────────────────────────────────────

    def _accumulate_from_model(
        self,
        runner,
        mass_acc: List[float],
        n_acc_attr: str,  # attribute name on self to increment sample count
        cam_acc_attr: str,
        lidar_acc_attr: str,
        phase: str,
    ) -> None:
        """Read _moe_info from all MoE blocks and scatter routing mass."""
        moe_modules = _get_moe_modules(runner.model)
        if not moe_modules:
            return

        for attr_name, mod in moe_modules.items():
            info = getattr(mod, '_moe_info', None)
            if info is None:
                continue

            topk_idx     = info.get('topk_idx')
            topk_weights = info.get('topk_weights')
            if topk_idx is None or topk_weights is None:
                continue

            B = int(topk_idx.shape[0])

            # Hook A: scatter soft routing mass into per-expert accumulator.
            if self.enable_hook_a:
                _scatter_routing_mass(topk_idx, topk_weights, mass_acc)
                setattr(self, n_acc_attr, getattr(self, n_acc_attr) + B)

            # Hook C: accumulate modality group mass for modality_specific_moe.
            if self.enable_hook_c and attr_name == 'modality_specific_moe':
                cam_m   = info.get('cam_group_mass',   0.0)
                lidar_m = info.get('lidar_group_mass', 0.0)
                setattr(self, cam_acc_attr,
                        getattr(self, cam_acc_attr) + cam_m)
                setattr(self, lidar_acc_attr,
                        getattr(self, lidar_acc_attr) + lidar_m)

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        self._accumulate_from_model(
            runner,
            mass_acc=self._train_mass,
            n_acc_attr='_train_n',
            cam_acc_attr='_train_cam_mass',
            lidar_acc_attr='_train_lidar_mass',
            phase='train',
        )

    def after_val_iter(self, runner, batch_idx, data_batch=None,
                       outputs=None) -> None:
        self._accumulate_from_model(
            runner,
            mass_acc=self._val_mass,
            n_acc_attr='_val_n',
            cam_acc_attr='_val_cam_mass',
            lidar_acc_attr='_val_lidar_mass',
            phase='val',
        )

    # ──────────────────────────────────────────────────────────────────────
    # End-of-epoch: save artifacts and reset accumulators
    # ──────────────────────────────────────────────────────────────────────

    def _normalised_mass(self, raw_mass: List[float], n_samples: int
                         ) -> List[float]:
        """Normalise routing mass by total expected mass (n_samples × k).

        Each sample contributes k weights summing to 1.0.  Total expected mass
        = n_samples.  Dividing by this gives per-expert fractional share, which
        is directly comparable across epochs regardless of dataset size.
        """
        total = float(n_samples) + 1e-8
        return [m / total for m in raw_mass]

    def _save_epoch_json(self, epoch: int, mass: List[float],
                         filename_prefix: str,
                         extra: Optional[dict] = None) -> None:
        data = {f'expert_{i}': m for i, m in enumerate(mass)}
        if extra:
            data.update(extra)
        path = os.path.join(self._out_dir, f'{filename_prefix}_epoch{epoch}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_routing_mass_plot(self) -> None:
        """Save Hook A line plot: epoch on x-axis, routing mass per expert."""
        if not _MPL_AVAILABLE:
            return
        if not self._train_mass_history and not self._val_mass_history:
            return

        epochs = sorted(set(list(self._train_mass_history.keys()) +
                            list(self._val_mass_history.keys())))
        num_e  = self.num_experts
        colors = plt.cm.tab10.colors  # up to 10 distinct colours

        fig, ax = plt.subplots(figsize=(8, 5))
        for eidx in range(num_e):
            col = colors[eidx % len(colors)]
            # Training curves — solid lines
            if self._train_mass_history:
                tr_epochs = sorted(self._train_mass_history)
                tr_vals   = [self._train_mass_history[e][eidx]
                             for e in tr_epochs]
                ax.plot(tr_epochs, tr_vals, color=col, linestyle='-',
                        label=f'E{eidx} train')
            # Validation curves — dashed lines
            if self._val_mass_history:
                va_epochs = sorted(self._val_mass_history)
                va_vals   = [self._val_mass_history[e][eidx]
                             for e in va_epochs]
                ax.plot(va_epochs, va_vals, color=col, linestyle='--',
                        label=f'E{eidx} val')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalised routing mass (fraction of total)')
        ax.set_title('Hook A — Expert routing mass over epochs')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self._out_dir, 'routing_mass.png'), dpi=150)
        plt.close(fig)

    def _save_group_mass_plot(self) -> None:
        """Save Hook C line plot: epoch vs cam/lidar group routing mass."""
        if not _MPL_AVAILABLE:
            return
        if not self._train_group_history and not self._val_group_history:
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        if self._train_group_history:
            tr_ep  = sorted(self._train_group_history)
            tr_cam = [self._train_group_history[e][0] for e in tr_ep]
            tr_lid = [self._train_group_history[e][1] for e in tr_ep]
            ax.plot(tr_ep, tr_cam, 'b-',  label='Camera group (train)')
            ax.plot(tr_ep, tr_lid, 'r-',  label='LiDAR group (train)')
        if self._val_group_history:
            va_ep  = sorted(self._val_group_history)
            va_cam = [self._val_group_history[e][0] for e in va_ep]
            va_lid = [self._val_group_history[e][1] for e in va_ep]
            ax.plot(va_ep, va_cam, 'b--', label='Camera group (val)')
            ax.plot(va_ep, va_lid, 'r--', label='LiDAR group (val)')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total routing mass (batch-normalised)')
        ax.set_title('Hook C — Modality-group routing mass over epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self._out_dir, 'group_mass.png'), dpi=150)
        plt.close(fig)

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch  # 0-indexed; becomes epoch N after N+1 iters

        if self.enable_hook_a and self._train_n > 0:
            norm_mass = self._normalised_mass(self._train_mass, self._train_n)
            self._train_mass_history[epoch] = norm_mass
            self._save_epoch_json(epoch, norm_mass, 'routing_mass_train')
            self._save_routing_mass_plot()

        if self.enable_hook_c and (self._train_cam_mass + self._train_lidar_mass) > 0:
            total = self._train_cam_mass + self._train_lidar_mass + 1e-8
            cam_frac   = self._train_cam_mass   / total
            lidar_frac = self._train_lidar_mass / total
            self._train_group_history[epoch] = (cam_frac, lidar_frac)
            data = {'cam_group_mass': cam_frac, 'lidar_group_mass': lidar_frac,
                    'cam_raw': self._train_cam_mass,
                    'lidar_raw': self._train_lidar_mass}
            path = os.path.join(
                self._out_dir, f'group_mass_train_epoch{epoch}.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self._save_group_mass_plot()

        self._reset_train_accumulators()

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        epoch = runner.epoch
        metrics = metrics or {}

        if self.enable_hook_a and self._val_n > 0:
            norm_mass = self._normalised_mass(self._val_mass, self._val_n)
            self._val_mass_history[epoch] = norm_mass
            self._save_epoch_json(epoch, norm_mass, 'routing_mass_val')
            self._save_routing_mass_plot()

            # Hook B: combine AP with routing mass into a single summary JSON.
            if self.enable_hook_b:
                ap_val  = float(metrics.get(self.ap_metric_key, -1.0))
                summary = {
                    'epoch':      epoch,
                    'ap_metric':  self.ap_metric_key,
                    'ap_value':   ap_val,
                    'routing_mass_per_expert': {
                        f'expert_{i}': m for i, m in enumerate(norm_mass)
                    },
                }
                # Add modality group masses if available (Hook B for modal_specific).
                if self.enable_hook_c and (self._val_cam_mass + self._val_lidar_mass) > 0:
                    total = self._val_cam_mass + self._val_lidar_mass + 1e-8
                    summary['cam_group_mass_frac']   = self._val_cam_mass   / total
                    summary['lidar_group_mass_frac'] = self._val_lidar_mass / total
                path = os.path.join(
                    self._out_dir, f'routing_summary_epoch{epoch}.json')
                with open(path, 'w') as f:
                    json.dump(summary, f, indent=2)

        if self.enable_hook_c and (self._val_cam_mass + self._val_lidar_mass) > 0:
            total = self._val_cam_mass + self._val_lidar_mass + 1e-8
            cam_frac   = self._val_cam_mass   / total
            lidar_frac = self._val_lidar_mass / total
            self._val_group_history[epoch] = (cam_frac, lidar_frac)
            data = {'cam_group_mass': cam_frac, 'lidar_group_mass': lidar_frac,
                    'cam_raw': self._val_cam_mass,
                    'lidar_raw': self._val_lidar_mass}
            path = os.path.join(
                self._out_dir, f'group_mass_val_epoch{epoch}.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self._save_group_mass_plot()

        self._reset_val_accumulators()

    # ──────────────────────────────────────────────────────────────────────
    # Accumulator resets
    # ──────────────────────────────────────────────────────────────────────

    def _reset_train_accumulators(self) -> None:
        self._train_mass        = [0.0] * self.num_experts
        self._train_n           = 0
        self._train_cam_mass    = 0.0
        self._train_lidar_mass  = 0.0

    def _reset_val_accumulators(self) -> None:
        self._val_mass          = [0.0] * self.num_experts
        self._val_n             = 0
        self._val_cam_mass      = 0.0
        self._val_lidar_mass    = 0.0

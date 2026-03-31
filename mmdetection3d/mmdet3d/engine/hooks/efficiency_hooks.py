"""Training efficiency + final run summary hooks for thesis-ready instrumentation.

TrainingEfficiencyHook
    Lightweight per-epoch training throughput / memory metrics.
    DDP-safe: only rank 0 logs; metric collection is local-only.

RunSummaryHook
    Prints a compact summary block after training ends, consolidating
    run configuration, best/final val metrics, and efficiency numbers.
"""

import time
from collections import defaultdict
from typing import Dict, Optional, Sequence, Union

import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmdet3d.registry import HOOKS


def _is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Training Efficiency Hook
# ─────────────────────────────────────────────────────────────────────────────

@HOOKS.register_module()
class TrainingEfficiencyHook(Hook):
    """Log training throughput and GPU memory at configurable intervals.

    Metrics (rank-0 only, logged via MMEngine logger):
        train/sec_per_iter           – avg seconds per iteration
        train/iters_per_sec          – reciprocal
        train/samples_per_sec        – global throughput
        train/samples_per_sec_per_gpu
        train/max_mem_allocated_mb   – peak allocated (since last reset)
        train/max_mem_reserved_mb    – peak reserved (cache)

    Args:
        log_interval (int): Log every N iterations.  Defaults to the same
            value as LoggerHook (resolved from runner at ``before_train``).
            Set explicitly to override.
    """

    priority = 'BELOW_NORMAL'

    def __init__(self, log_interval: Optional[int] = None):
        self._log_interval = log_interval
        self._epoch_t0: float = 0.0
        self._iter_count: int = 0
        self._batch_size: int = 0
        self._world: int = 1

    # ── lifecycle ────────────────────────────────────────────────────────

    def before_train(self, runner) -> None:
        self._world = _world_size()
        dl = runner.train_dataloader
        self._batch_size = getattr(dl, 'batch_size', None) or 1
        if self._log_interval is None:
            for h in runner.hooks:
                if h.__class__.__name__ == 'LoggerHook':
                    self._log_interval = getattr(h, 'interval', 50)
                    break
            if self._log_interval is None:
                self._log_interval = 50

    def before_train_epoch(self, runner) -> None:
        torch.cuda.reset_peak_memory_stats()
        self._epoch_t0 = time.monotonic()
        self._iter_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        self._iter_count += 1
        if not _is_rank0():
            return
        if self._iter_count % self._log_interval != 0:
            return

        elapsed = time.monotonic() - self._epoch_t0
        sec_per_iter = elapsed / self._iter_count
        iters_per_sec = self._iter_count / max(elapsed, 1e-9)
        samples_per_sec = iters_per_sec * self._batch_size * self._world
        samples_per_sec_gpu = iters_per_sec * self._batch_size

        mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        log_vars = {
            'train/sec_per_iter': round(sec_per_iter, 4),
            'train/iters_per_sec': round(iters_per_sec, 2),
            'train/samples_per_sec': round(samples_per_sec, 2),
            'train/samples_per_sec_per_gpu': round(samples_per_sec_gpu, 2),
            'train/max_mem_allocated_mb': round(mem_alloc, 1),
            'train/max_mem_reserved_mb': round(mem_reserved, 1),
        }
        runner.message_hub.update_scalars(log_vars)

    def after_train_epoch(self, runner) -> None:
        if not _is_rank0():
            return
        elapsed = time.monotonic() - self._epoch_t0
        mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        epoch = runner.epoch + 1
        print_log(
            f'[Efficiency] Epoch {epoch}: '
            f'wall={elapsed:.1f}s  iters={self._iter_count}  '
            f'sec/iter={elapsed / max(self._iter_count, 1):.3f}  '
            f'mem_alloc={mem_alloc:.0f}MB  mem_rsv={mem_reserved:.0f}MB',
            logger='current')


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Final Run Summary Hook
# ─────────────────────────────────────────────────────────────────────────────

@HOOKS.register_module()
class RunSummaryHook(Hook):
    """Print a compact run summary after training finishes.

    Collects:
      A) Run configuration (world size, batch size, global batch size)
      B) Best / final validation metrics (from runner.message_hub)
      C) Training efficiency (wall time, throughput, memory)

    Rank-0 only.  Easy to scrape into comparison tables.
    """

    priority = 'VERY_LOW'

    def __init__(self):
        self._train_t0: float = 0.0
        self._total_iters: int = 0
        self._batch_size: int = 0
        self._world: int = 1
        self._peak_mem_alloc: float = 0.0
        self._peak_mem_reserved: float = 0.0
        self._best_metrics: Dict[str, float] = {}
        self._best_epoch: int = -1
        self._final_metrics: Dict[str, float] = {}

    def before_train(self, runner) -> None:
        self._train_t0 = time.monotonic()
        self._world = _world_size()
        dl = runner.train_dataloader
        self._batch_size = getattr(dl, 'batch_size', None) or 1

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        self._total_iters += 1
        mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        rsv = torch.cuda.max_memory_reserved() / (1024 ** 2)
        self._peak_mem_alloc = max(self._peak_mem_alloc, mem)
        self._peak_mem_reserved = max(self._peak_mem_reserved, rsv)

    # Ordered list of metric keys to try as primary for best-epoch tracking.
    # First match wins.
    _PRIMARY_KEYS = [
        'mAP_1.0m',      # center-distance (preferred for outdoor)
        'mAP_2.0m',      # center-distance (nuScenes leaderboard default)
        'mAP_0.50',      # 3D IoU @ 0.50
        'mAP_0.25',      # 3D IoU @ 0.25
        'mAP',           # generic fallback
    ]

    def after_val_epoch(self, runner, metrics=None) -> None:
        if metrics is None:
            return
        self._final_metrics = dict(metrics)
        primary_key = None
        for k in self._PRIMARY_KEYS:
            if k in metrics:
                primary_key = k
                break
        if primary_key is None:
            return
        cur = metrics[primary_key]
        prev = self._best_metrics.get(primary_key, -1)
        if cur > prev:
            self._best_metrics = dict(metrics)
            self._best_epoch = runner.epoch + 1

    def after_train(self, runner) -> None:
        if not _is_rank0():
            return
        total_time = time.monotonic() - self._train_t0
        sec_per_iter = total_time / max(self._total_iters, 1)
        iters_per_sec = self._total_iters / max(total_time, 1e-9)
        samples_per_sec = iters_per_sec * self._batch_size * self._world
        samples_per_sec_gpu = iters_per_sec * self._batch_size

        sep = '=' * 72
        lines = [
            '',
            sep,
            '  RUN SUMMARY',
            sep,
            '',
            '  A) Run Configuration',
            f'     world_size           : {self._world}',
            f'     per_gpu_batch_size   : {self._batch_size}',
            f'     global_batch_size    : {self._batch_size * self._world}',
            f'     total_epochs         : {runner.epoch + 1}',
            '',
            '  B) Validation Performance',
        ]
        if self._best_metrics:
            lines.append(f'     best_epoch           : {self._best_epoch}')
            for k, v in self._best_metrics.items():
                lines.append(f'     best/{k:18s}: {v:.4f}')
        else:
            lines.append('     (no validation metrics recorded)')
        if self._final_metrics:
            lines.append('')
            for k, v in self._final_metrics.items():
                lines.append(f'     final/{k:17s}: {v:.4f}')
        lines += [
            '',
            '  C) Training Efficiency',
            f'     total_wall_time      : {total_time:.1f}s  '
            f'({total_time / 3600:.2f}h)',
            f'     total_iters          : {self._total_iters}',
            f'     avg_sec_per_iter     : {sec_per_iter:.4f}',
            f'     avg_iters_per_sec    : {iters_per_sec:.2f}',
            f'     avg_samples_per_sec  : {samples_per_sec:.2f}',
            f'     avg_samples/s/gpu    : {samples_per_sec_gpu:.2f}',
            f'     peak_mem_allocated_mb: {self._peak_mem_alloc:.0f}',
            f'     peak_mem_reserved_mb : {self._peak_mem_reserved:.0f}',
            '',
            sep,
        ]
        summary = '\n'.join(lines)
        print_log(summary, logger='current')

        summary_path = f'{runner.work_dir}/run_summary.txt'
        try:
            with open(summary_path, 'w') as f:
                f.write(summary + '\n')
            print_log(f'Run summary saved to {summary_path}', logger='current')
        except OSError:
            pass

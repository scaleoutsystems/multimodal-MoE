"""MOE THESIS CONTEXT ROUTING HOOKS:
Context-aware routing diagnostics — ContextRoutingStatsHook and
ContextExpertUsageVisualizationHook.

These companion hooks analyse MoE routing decisions grouped by the context
metadata keys that the router actively uses (e.g. ``road_type``,
``solar_context_bin``).  Context keys are **auto-detected** from the model's
``ContextEncoder.field_names`` at the start of validation; they can also be
supplied explicitly in the hook config.

Both hooks are **validation-only** and do not alter training or evaluation
behaviour in any way.

ContextRoutingStatsHook
-----------------------
Collects per-sample routing statistics during validation and computes
aggregated summaries grouped by context value for each active context key.

Per sample (collected internally):
    - context label(s) for each active router key
    - gate output probabilities (full softmax over experts)
    - top-k expert indices and re-normalised weights
    - top-1 expert id and probability
    - gate entropy

Aggregated outputs (saved per val epoch):
    - overall: expert usage fractions, mean routing weights, mean top-1
      probability, mean gate entropy
    - per context value: same statistics, grouped by each context key

Saves:
    context_routing_stats_epoch{N}.json
    (optionally) context_routing_samples_epoch{N}.json  (per-sample records)

ContextExpertUsageVisualizationHook
------------------------------------
Produces matplotlib bar charts showing expert usage distribution, overall
and broken down by each active context key.

Saves:
    expert_usage_overall_epoch{N}.png
    expert_usage_by_{key}_epoch{N}.png   — one per active context key

Output location
---------------
    <runner.work_dir>/context_routing/

Config example
--------------
    custom_hooks = [
        ...
        dict(type='ContextRoutingStatsHook'),
        dict(type='ContextExpertUsageVisualizationHook'),
    ]

    To override auto-detected keys::

        dict(type='ContextRoutingStatsHook',
             context_keys=['road_type', 'solar_context_bin'])
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────

def _unwrap(model):
    """Strip DDP / FSDP wrapper."""
    return model.module if is_model_wrapper(model) else model


def _find_moe_modules(model) -> Dict[str, Any]:
    """Return ``{attr_name: module}`` for every MoE block with ``_moe_info``."""
    m = _unwrap(model)
    result = {}
    for name in ('bev_moe', 'modality_specific_moe', 'joint_modality_moe'):
        attr = getattr(m, name, None)
        if attr is not None and hasattr(attr, '_moe_info'):
            result[name] = attr
    return result


def _discover_from_model(
    model,
) -> Tuple[Optional[List[str]], Optional[int], Optional[str]]:
    """Auto-detect context keys and expert count from the model.

    Walks all MoE blocks and returns the ``field_names`` from the first
    ``ContextEncoder`` found, together with the corresponding expert count.

    Returns:
        ``(context_keys, num_experts, block_name)`` — any element may be
        ``None`` if it cannot be determined.
    """
    moe_mods = _find_moe_modules(model)
    context_keys: Optional[List[str]] = None
    num_experts: Optional[int] = None
    block_name: Optional[str] = None

    for name, mod in moe_mods.items():
        if num_experts is None and hasattr(mod, 'num_experts'):
            num_experts = mod.num_experts
            block_name = name
        enc = getattr(mod, 'context_encoder', None)
        if enc is not None and hasattr(enc, 'field_names'):
            context_keys = list(enc.field_names)
            if hasattr(mod, 'num_experts'):
                num_experts = mod.num_experts
                block_name = name
    return context_keys, num_experts, block_name


def _extract_batch_context(
    data_batch,
    context_keys: List[str],
) -> Optional[List[Dict[str, str]]]:
    """Pull per-sample context labels from the data batch's ``data_samples``.

    Handles both the standard mmengine dict-style batch
    (``{'inputs': …, 'data_samples': […]}`` ) and list-of-samples batches.
    """
    if data_batch is None or not context_keys:
        return None

    data_samples = None
    if isinstance(data_batch, dict):
        data_samples = data_batch.get('data_samples')
    elif isinstance(data_batch, (list, tuple)):
        data_samples = data_batch
    if not data_samples:
        return None

    batch_ctx: List[Dict[str, str]] = []
    for ds in data_samples:
        meta = getattr(ds, 'metainfo', {}) if not isinstance(ds, dict) else ds
        ctx_raw = meta.get('context', {})
        batch_ctx.append(
            {k: str(ctx_raw.get(k, 'unknown')) for k in context_keys})
    return batch_ctx or None


def _gate_entropy(probs: List[float]) -> float:
    """Shannon entropy  H = −Σ pᵢ·ln(pᵢ)."""
    return -sum(p * math.log(p) for p in probs if p > 1e-12)


def _collect_iter_records(
    model,
    data_batch,
    context_keys: List[str],
    num_experts: int,
) -> List[dict]:
    """Build per-sample routing records from the current val iteration.

    Reads ``_moe_info`` from the first MoE block that exposes it and pairs
    each sample's routing data with its context labels from *data_batch*.
    """
    moe_mods = _find_moe_modules(model)
    if not moe_mods:
        return []
    batch_ctx = _extract_batch_context(data_batch, context_keys)
    if batch_ctx is None:
        return []

    records: List[dict] = []
    for _name, mod in moe_mods.items():
        info = getattr(mod, '_moe_info', None)
        if info is None:
            continue
        probs = info.get('probs')
        topk_idx = info.get('topk_idx')
        topk_weights = info.get('topk_weights')
        if probs is None or topk_idx is None:
            continue

        B = probs.shape[0]
        for b in range(min(B, len(batch_ctx))):
            p = probs[b].cpu().tolist()
            ti = topk_idx[b].cpu().tolist()
            tw = (topk_weights[b].cpu().tolist()
                  if topk_weights is not None else [])
            records.append({
                'context': batch_ctx[b],
                'probs': p,
                'topk_idx': ti,
                'topk_weights': tw,
                'top1_expert': int(topk_idx[b, 0].item()),
                'top1_prob': float(probs[b].max().item()),
                'entropy': _gate_entropy(p),
            })
        break  # use first block with valid info
    return records


def _aggregate(
    records: List[dict],
    num_experts: int,
    context_keys: List[str],
) -> dict:
    """Compute aggregated statistics from per-sample routing records.

    Returns a dict with ``overall`` stats and ``per_context`` stats keyed by
    each context field name, then by context value.
    """

    def _group_stats(recs: List[dict]) -> dict:
        n = len(recs)
        if n == 0:
            return {'num_samples': 0}

        top1_counts = [0] * num_experts
        prob_sums = [0.0] * num_experts
        topk_mass = [0.0] * num_experts
        topk_cnt = [0] * num_experts
        sum_top1 = 0.0
        sum_entropy = 0.0

        for r in recs:
            eidx = r['top1_expert']
            if 0 <= eidx < num_experts:
                top1_counts[eidx] += 1
            for e, p in enumerate(r['probs'][:num_experts]):
                prob_sums[e] += p
            for idx, w in zip(r['topk_idx'], r['topk_weights']):
                if 0 <= idx < num_experts:
                    topk_mass[idx] += w
                    topk_cnt[idx] += 1
            sum_top1 += r['top1_prob']
            sum_entropy += r['entropy']

        return {
            'num_samples': n,
            'expert_usage_fractions': {
                f'expert_{e}': round(c / n, 6)
                for e, c in enumerate(top1_counts)
            },
            'mean_routing_weights': {
                f'expert_{e}': round(s / n, 6)
                for e, s in enumerate(prob_sums)
            },
            'mean_topk_weight_when_selected': {
                f'expert_{e}': round(
                    topk_mass[e] / max(topk_cnt[e], 1), 6)
                for e in range(num_experts)
            },
            'mean_top1_prob': round(sum_top1 / n, 6),
            'mean_gate_entropy': round(sum_entropy / n, 6),
        }

    result: Dict[str, Any] = {'overall': _group_stats(records)}

    per_ctx: Dict[str, Any] = {}
    for key in context_keys:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in records:
            groups[rec['context'].get(key, 'unknown')].append(rec)
        per_ctx[key] = {
            val: _group_stats(recs)
            for val, recs in sorted(groups.items())
        }
    result['per_context'] = per_ctx
    return result


# ──────────────────────────────────────────────────────────────────────────
# ContextRoutingStatsHook
# ──────────────────────────────────────────────────────────────────────────

@HOOKS.register_module()
class ContextRoutingStatsHook(Hook):
    """Collect and aggregate MoE routing statistics by context group.

    Validation-only.  Reads ``_moe_info`` from MoE blocks and context labels
    from ``data_batch``, then saves per-epoch JSON summaries.

    Args:
        context_keys:    Explicitly specify which context fields to group by.
                         ``None`` → auto-detect from the model's ContextEncoder.
        save_per_sample: Also dump a JSON file with one record per validation
                         sample (useful for custom post-hoc analysis).
        out_subdir:      Sub-directory inside ``runner.work_dir``.
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        context_keys: Optional[List[str]] = None,
        save_per_sample: bool = False,
        out_subdir: str = 'context_routing',
    ):
        self._explicit_keys = context_keys
        self.save_per_sample = save_per_sample
        self.out_subdir = out_subdir

        self._context_keys: Optional[List[str]] = None
        self._num_experts: Optional[int] = None
        self._block_name: Optional[str] = None
        self._val_records: List[dict] = []
        self._out_dir: Optional[str] = None
        self._discovered = False

    # ── Setup ──────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)

    def _ensure_discovered(self, runner) -> None:
        """Lazily discover context keys / num_experts from the model."""
        if self._discovered:
            return
        self._discovered = True
        keys, n_exp, block = _discover_from_model(runner.model)
        self._num_experts = n_exp
        self._block_name = block
        self._context_keys = (
            list(self._explicit_keys) if self._explicit_keys else keys)

        if self._context_keys and self._num_experts:
            runner.logger.info(
                f'ContextRoutingStatsHook: context_keys={self._context_keys}, '
                f'num_experts={self._num_experts}, block={self._block_name}')
        else:
            runner.logger.warning(
                'ContextRoutingStatsHook: could not discover context keys or '
                'num_experts — hook will be inactive.  Set context_keys '
                'explicitly or ensure a ContextEncoder is configured.')

    # ── Per-iteration collection (val only) ────────────────────────────

    def before_val_epoch(self, runner) -> None:
        self._ensure_discovered(runner)
        self._val_records = []

    def after_val_iter(self, runner, batch_idx, data_batch=None,
                       outputs=None) -> None:
        if not self._context_keys or not self._num_experts:
            return
        self._val_records.extend(
            _collect_iter_records(
                runner.model, data_batch,
                self._context_keys, self._num_experts))

    # ── End-of-epoch aggregation and output ────────────────────────────

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._val_records or not self._context_keys:
            return

        epoch = runner.epoch
        stats = _aggregate(
            self._val_records, self._num_experts, self._context_keys)
        stats['epoch'] = epoch
        stats['moe_block'] = self._block_name
        stats['context_keys'] = self._context_keys
        stats['num_experts'] = self._num_experts

        path = os.path.join(
            self._out_dir, f'context_routing_stats_epoch{epoch}.json')
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
        runner.logger.info(f'ContextRoutingStatsHook: saved → {path}')

        if self.save_per_sample:
            ps_path = os.path.join(
                self._out_dir,
                f'context_routing_samples_epoch{epoch}.json')
            with open(ps_path, 'w') as f:
                json.dump(self._val_records, f, indent=1)
            runner.logger.info(
                f'ContextRoutingStatsHook: per-sample → {ps_path}')

        self._log_summary(runner.logger, stats)
        self._val_records = []

    def _log_summary(self, logger, stats: dict) -> None:
        ov = stats.get('overall', {})
        logger.info(
            f"  Overall ({ov.get('num_samples', 0)} samples): "
            f"mean_top1_prob={ov.get('mean_top1_prob', 'N/A')}, "
            f"mean_entropy={ov.get('mean_gate_entropy', 'N/A')}")
        usage = ov.get('expert_usage_fractions', {})
        if usage:
            logger.info(
                '  Expert usage: '
                + ', '.join(f'{k}={v:.3f}' for k, v in usage.items()))
        for key, groups in stats.get('per_context', {}).items():
            parts = []
            for val, g in groups.items():
                u = g.get('expert_usage_fractions', {})
                u_str = '[' + ', '.join(f'{v:.2f}' for v in u.values()) + ']'
                parts.append(f"{val}(n={g['num_samples']}): {u_str}")
            logger.info(f'  {key}: ' + ' | '.join(parts))


# ──────────────────────────────────────────────────────────────────────────
# ContextExpertUsageVisualizationHook
# ──────────────────────────────────────────────────────────────────────────

@HOOKS.register_module()
class ContextExpertUsageVisualizationHook(Hook):
    """Produce bar-chart visualisations of expert usage by context group.

    Validation-only.  Collects the same routing + context data as
    :class:`ContextRoutingStatsHook` (independently, so either hook can
    be used alone) and renders matplotlib figures.

    Outputs per val epoch:
        - overall expert usage (selection frequency + mean routing weight)
        - grouped bar chart per active context key

    Args:
        context_keys:    See :class:`ContextRoutingStatsHook`.
        out_subdir:      Sub-directory inside ``runner.work_dir``.
    """

    priority = 'LOW'

    def __init__(
        self,
        context_keys: Optional[List[str]] = None,
        out_subdir: str = 'context_routing',
    ):
        self._explicit_keys = context_keys
        self.out_subdir = out_subdir

        self._context_keys: Optional[List[str]] = None
        self._num_experts: Optional[int] = None
        self._block_name: Optional[str] = None
        self._val_records: List[dict] = []
        self._out_dir: Optional[str] = None
        self._discovered = False

    # ── Setup ──────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        if not _MPL_AVAILABLE:
            runner.logger.warning(
                'ContextExpertUsageVisualizationHook: matplotlib not '
                'available — plots will be skipped.')

    def _ensure_discovered(self, runner) -> None:
        if self._discovered:
            return
        self._discovered = True
        keys, n_exp, block = _discover_from_model(runner.model)
        self._num_experts = n_exp
        self._block_name = block
        self._context_keys = (
            list(self._explicit_keys) if self._explicit_keys else keys)

        if self._context_keys and self._num_experts:
            runner.logger.info(
                f'ContextExpertUsageVisualizationHook: '
                f'context_keys={self._context_keys}, '
                f'num_experts={self._num_experts}')
        else:
            runner.logger.warning(
                'ContextExpertUsageVisualizationHook: could not discover '
                'context keys — plots will be skipped.')

    # ── Per-iteration collection (val only) ────────────────────────────

    def before_val_epoch(self, runner) -> None:
        self._ensure_discovered(runner)
        self._val_records = []

    def after_val_iter(self, runner, batch_idx, data_batch=None,
                       outputs=None) -> None:
        if not self._context_keys or not self._num_experts:
            return
        if not _MPL_AVAILABLE:
            return
        self._val_records.extend(
            _collect_iter_records(
                runner.model, data_batch,
                self._context_keys, self._num_experts))

    # ── End-of-epoch plotting ──────────────────────────────────────────

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._val_records or not _MPL_AVAILABLE:
            return
        if not self._context_keys or not self._num_experts:
            return

        epoch = runner.epoch
        E = self._num_experts
        stats = _aggregate(self._val_records, E, self._context_keys)

        self._plot_overall(epoch, stats.get('overall', {}), E)

        for key, groups in stats.get('per_context', {}).items():
            self._plot_by_context_key(epoch, key, groups, E)

        runner.logger.info(
            f'ContextExpertUsageVisualizationHook: '
            f'plots saved → {self._out_dir}')
        self._val_records = []

    # ── Plotting helpers ───────────────────────────────────────────────

    def _plot_overall(self, epoch: int, overall: dict, E: int) -> None:
        usage = overall.get('expert_usage_fractions', {})
        weights = overall.get('mean_routing_weights', {})
        if not usage:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10.colors
        experts = list(range(E))
        labels = [f'E{e}' for e in experts]

        # Left panel: top-1 selection frequency
        ax = axes[0]
        vals = [usage.get(f'expert_{e}', 0) for e in experts]
        bars = ax.bar(
            experts, vals,
            color=[colors[e % len(colors)] for e in experts])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Top-1 selection fraction')
        ax.set_title(f'Expert Selection Frequency (Epoch {epoch})')
        ax.set_xticks(experts)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.25 + 0.01)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # Right panel: mean routing weight (soft probability mass)
        ax = axes[1]
        vals = [weights.get(f'expert_{e}', 0) for e in experts]
        bars = ax.bar(
            experts, vals,
            color=[colors[e % len(colors)] for e in experts])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Mean routing weight')
        ax.set_title(f'Mean Routing Weights (Epoch {epoch})')
        ax.set_xticks(experts)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.25 + 0.01)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        n = overall.get('num_samples', '?')
        top1 = overall.get('mean_top1_prob', '?')
        ent = overall.get('mean_gate_entropy', '?')
        fig.suptitle(
            f'Overall Expert Usage — {n} samples  |  '
            f'mean top-1 prob={top1}  |  mean entropy={ent}',
            fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self._out_dir, f'expert_usage_overall_epoch{epoch}.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_by_context_key(
        self, epoch: int, key: str, groups: dict, E: int,
    ) -> None:
        if not groups:
            return

        context_vals = sorted(groups.keys())
        n_vals = len(context_vals)
        colors = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(max(8, n_vals * 1.8 + 2), 5.5))

        x = np.arange(n_vals)
        bar_width = 0.8 / E

        for e in range(E):
            vals = [
                groups[cv].get('expert_usage_fractions', {}).get(
                    f'expert_{e}', 0)
                for cv in context_vals
            ]
            offset = (e - (E - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, vals, bar_width,
                label=f'Expert {e}',
                color=colors[e % len(colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), f'{v:.2f}',
                            ha='center', va='bottom', fontsize=7)

        for i, cv in enumerate(context_vals):
            n = groups[cv].get('num_samples', 0)
            ax.text(i, -0.04, f'n={n}', ha='center', va='top',
                    fontsize=8, color='gray',
                    transform=ax.get_xaxis_transform())

        ax.set_xlabel(key.replace('_', ' ').title())
        ax.set_ylabel('Top-1 expert selection fraction')
        ax.set_title(f'Expert Usage by {key} (Epoch {epoch})')
        ax.set_xticks(x)
        ax.set_xticklabels(
            context_vals,
            rotation=30 if n_vals > 4 else 0,
            ha='right' if n_vals > 4 else 'center')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                self._out_dir,
                f'expert_usage_by_{key}_epoch{epoch}.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

"""Context-aware routing diagnostics hooks.

ContextRoutingStatsHook
-----------------------
Collects per-sample routing statistics during validation and saves aggregated
summaries grouped by context value for each active context key.

Uses these exact metric names in all JSON output and plot titles:

    dense_mean_prob_per_expert
        Mean pre-top-k softmax probability per expert, averaged over samples.

    dispatch_mass_per_expert
        Mean post-top-k routed weight per expert, averaged over samples.

    top1_selection_freq_per_expert
        Fraction of samples for which expert e is the rank-1 selection.

    topk_selection_freq_per_expert
        Fraction of samples for which expert e appears anywhere in the
        selected top-k set. Identical to top1 when k=1.

Saves per val epoch:
    context_routing_stats_epoch{N}.json

ContextExpertUsageVisualizationHook
------------------------------------
Produces matplotlib figures per val epoch:

    routing_overall_epoch{N}.png
        Left:  "Top-1 Selection Frequency" (bar chart)
        Right: "Dense Mean Router Probability (Pre-Top-k)" (bar chart)

    top1_selection_by_{key}_epoch{N}.png
        Grouped bars: top-1 selection frequency per expert, by context group.

    dense_prob_by_{key}_epoch{N}.png
        Grouped bars: dense mean prob per expert, by context group.

    dispatch_mass_by_{key}_epoch{N}.png
        Grouped bars: actual Shazeer topk dispatch mass per expert, by context
        group.  With k>1 this shows both top-1 and secondary selections
        weighted by their renormalized Shazeer weights, making it the most
        informative plot for diagnosing B-team / dead-expert patterns.

    topk_selection_by_{key}_epoch{N}.png
        Grouped bars: top-k (any position) selection frequency per expert, by
        context group.  Separates dead experts (never in top-k) from secondary
        partners (routinely in top-k but rarely top-1).

Output location: <runner.work_dir>/context_routing/
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


# ── Shared utilities ──────────────────────────────────────────────────────

def _unwrap(model):
    return model.module if is_model_wrapper(model) else model


def _find_moe_modules(model) -> Dict[str, Any]:
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
    """Auto-detect context keys and expert count from the model."""
    moe_mods = _find_moe_modules(model)
    context_keys: Optional[List[str]] = None
    num_experts:  Optional[int] = None
    block_name:   Optional[str] = None

    for name, mod in moe_mods.items():
        if num_experts is None and hasattr(mod, 'num_experts'):
            num_experts = mod.num_experts
            block_name  = name
        enc = getattr(mod, 'context_encoder', None)
        if enc is not None and hasattr(enc, 'field_names'):
            context_keys = list(enc.field_names)
            if hasattr(mod, 'num_experts'):
                num_experts = mod.num_experts
                block_name  = name
    return context_keys, num_experts, block_name


def _extract_batch_context(
    data_batch,
    context_keys: List[str],
) -> Optional[List[Dict[str, str]]]:
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
    return -sum(p * math.log(p) for p in probs if p > 1e-12)


def _collect_iter_records(
    model,
    data_batch,
    context_keys: List[str],
    num_experts: int,
) -> List[dict]:
    """Build per-sample routing records from the current val iteration.

    Each record contains:
        context              — {key: value} labels from data_batch
        full_softmax_probs   — pre-top-k softmax over all experts (list, length E)
        topk_idx             — selected expert indices (list, length k)
        topk_weights         — post-top-k dispatch weights (list, length k)
        top1_expert          — rank-1 selected expert index
        entropy              — gate entropy H = -Σ p log p (from full_softmax_probs)
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
        full_probs   = info.get('full_softmax_probs')  # (B, E) pre-top-k
        topk_idx     = info.get('topk_idx')
        topk_weights = info.get('topk_weights')
        if full_probs is None or topk_idx is None or topk_weights is None:
            continue

        B = full_probs.shape[0]
        for b in range(min(B, len(batch_ctx))):
            p  = full_probs[b].cpu().tolist()
            ti = topk_idx[b].cpu().tolist()
            tw = topk_weights[b].cpu().tolist()
            records.append({
                'context':            batch_ctx[b],
                'full_softmax_probs': p,
                'topk_idx':           ti,
                'topk_weights':       tw,
                'top1_expert':        int(topk_idx[b, 0].item()),
                'entropy':            _gate_entropy(p),
            })
        break  # use first block with valid info
    return records


def _group_stats(recs: List[dict], num_experts: int) -> dict:
    """Compute the four canonical routing metrics for a list of records."""
    n = len(recs)
    if n == 0:
        return {'num_samples': 0}

    E = num_experts
    dense_sum    = [0.0] * E   # sum of pre-top-k softmax probs
    dispatch_sum = [0.0] * E   # sum of topk_weights scattered per expert
    top1_cnt     = [0]   * E   # count where expert is rank-1
    topk_cnt     = [0]   * E   # count where expert appears in top-k set
    sum_entropy  = 0.0

    for r in recs:
        # dense_mean_prob: full pre-top-k softmax
        for e, p in enumerate(r['full_softmax_probs'][:E]):
            dense_sum[e] += p

        # dispatch_mass: scatter actual topk_weights
        for idx, w in zip(r['topk_idx'], r['topk_weights']):
            if 0 <= idx < E:
                dispatch_sum[idx] += w

        # top1_selection_freq: rank-1 expert
        eidx = r['top1_expert']
        if 0 <= eidx < E:
            top1_cnt[eidx] += 1

        # topk_selection_freq: any expert in top-k set
        seen = set()
        for idx in r['topk_idx']:
            if 0 <= idx < E and idx not in seen:
                topk_cnt[idx] += 1
                seen.add(idx)

        sum_entropy += r['entropy']

    def _ed(vals, denom=n):  # expert dict, rounded
        return {f'expert_{e}': round(vals[e] / denom, 8) for e in range(E)}

    return {
        'num_samples':                   n,
        'dense_mean_prob_per_expert':     _ed(dense_sum),
        'dispatch_mass_per_expert':       _ed(dispatch_sum),
        'top1_selection_freq_per_expert': _ed(top1_cnt),
        'topk_selection_freq_per_expert': _ed(topk_cnt),
        'mean_gate_entropy':              round(sum_entropy / n, 6),
    }


def _aggregate(
    records: List[dict],
    num_experts: int,
    context_keys: List[str],
) -> dict:
    """Compute overall and per-context routing statistics."""
    result: Dict[str, Any] = {
        'overall': _group_stats(records, num_experts)}

    per_ctx: Dict[str, Any] = {}
    for key in context_keys:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in records:
            groups[rec['context'].get(key, 'unknown')].append(rec)
        per_ctx[key] = {
            val: _group_stats(recs, num_experts)
            for val, recs in sorted(groups.items())
        }
    result['per_context'] = per_ctx
    return result


# ── ContextRoutingStatsHook ───────────────────────────────────────────────

@HOOKS.register_module()
class ContextRoutingStatsHook(Hook):
    """Collect and aggregate MoE routing statistics by context group.

    Validation-only. Saves context_routing_stats_epoch{N}.json with the four
    canonical routing metrics (dense_mean_prob_per_expert,
    dispatch_mass_per_expert, top1_selection_freq_per_expert,
    topk_selection_freq_per_expert) overall and per context group.

    Args:
        context_keys:    Explicitly specify context fields to group by.
                         None → auto-detect from the model's ContextEncoder.
        save_per_sample: Also save a per-sample JSON (for post-hoc analysis).
        out_subdir:      Subdirectory inside runner.work_dir.
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        context_keys: Optional[List[str]] = None,
        save_per_sample: bool = False,
        out_subdir: str = 'context_routing',
    ):
        self._explicit_keys  = context_keys
        self.save_per_sample = save_per_sample
        self.out_subdir      = out_subdir

        self._context_keys: Optional[List[str]] = None
        self._num_experts:  Optional[int] = None
        self._block_name:   Optional[str] = None
        self._val_records:  List[dict] = []
        self._out_dir:      Optional[str] = None
        self._discovered    = False

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)

    def _ensure_discovered(self, runner) -> None:
        if self._discovered:
            return
        self._discovered = True
        keys, n_exp, block = _discover_from_model(runner.model)
        self._num_experts  = n_exp
        self._block_name   = block
        self._context_keys = (
            list(self._explicit_keys) if self._explicit_keys else keys)
        if self._context_keys and self._num_experts:
            runner.logger.info(
                f'ContextRoutingStatsHook: context_keys={self._context_keys}, '
                f'num_experts={self._num_experts}, block={self._block_name}')
        else:
            runner.logger.warning(
                'ContextRoutingStatsHook: could not discover context keys or '
                'num_experts — hook inactive.')

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

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._val_records or not self._context_keys:
            return

        epoch = runner.epoch
        stats = _aggregate(
            self._val_records, self._num_experts, self._context_keys)
        stats['epoch']       = epoch
        stats['moe_block']   = self._block_name
        stats['context_keys']= self._context_keys
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

        self._log_summary(runner.logger, stats)
        self._val_records = []

    def _log_summary(self, logger, stats: dict) -> None:
        ov = stats.get('overall', {})
        logger.info(
            f"  Overall ({ov.get('num_samples', 0)} samples): "
            f"mean_entropy={ov.get('mean_gate_entropy', 'N/A')}")
        top1 = ov.get('top1_selection_freq_per_expert', {})
        if top1:
            logger.info(
                '  top1_selection_freq: '
                + ', '.join(f'{k}={v:.3f}' for k, v in top1.items()))
        dense = ov.get('dense_mean_prob_per_expert', {})
        if dense:
            logger.info(
                '  dense_mean_prob: '
                + ', '.join(f'{k}={v:.3f}' for k, v in dense.items()))
        for key, groups in stats.get('per_context', {}).items():
            parts = []
            for val, g in groups.items():
                u = g.get('top1_selection_freq_per_expert', {})
                u_str = '[' + ', '.join(f'{v:.2f}' for v in u.values()) + ']'
                parts.append(f"{val}(n={g['num_samples']}): {u_str}")
            logger.info(f'  {key}: ' + ' | '.join(parts))


# ── ContextExpertUsageVisualizationHook ───────────────────────────────────

@HOOKS.register_module()
class ContextExpertUsageVisualizationHook(Hook):
    """Produce bar-chart visualisations of routing metrics by context group.

    Validation-only. Outputs per val epoch:

        routing_overall_epoch{N}.png
            Left:  Top-1 Selection Frequency
            Right: Dense Mean Router Probability (Pre-Top-k)

        top1_selection_by_{key}_epoch{N}.png
            Grouped bars: top-1 selection freq per expert, per context value.

        dense_prob_by_{key}_epoch{N}.png
            Grouped bars: dense mean prob per expert, per context value.

        dispatch_mass_by_{key}_epoch{N}.png
            Grouped bars: Shazeer topk dispatch mass per expert, per context
            value.  Captures both top-1 and secondary (top-2+) selections
            weighted by their renormalized weights — the ground truth for
            "how much work does each expert do per context."

        topk_selection_by_{key}_epoch{N}.png
            Grouped bars: any-position top-k selection freq per expert, per
            context value.  Distinguishes dead experts (zero) from B-team
            secondary partners (high topk but low top1).
    """

    priority = 'LOW'

    def __init__(
        self,
        context_keys: Optional[List[str]] = None,
        out_subdir: str = 'context_routing',
    ):
        self._explicit_keys = context_keys
        self.out_subdir     = out_subdir

        self._context_keys: Optional[List[str]] = None
        self._num_experts:  Optional[int] = None
        self._block_name:   Optional[str] = None
        self._val_records:  List[dict] = []
        self._out_dir:      Optional[str] = None
        self._discovered    = False

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        if not _MPL_AVAILABLE:
            runner.logger.warning(
                'ContextExpertUsageVisualizationHook: matplotlib not '
                'available — plots skipped.')

    def _ensure_discovered(self, runner) -> None:
        if self._discovered:
            return
        self._discovered = True
        keys, n_exp, block = _discover_from_model(runner.model)
        self._num_experts  = n_exp
        self._block_name   = block
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
                'context keys — plots skipped.')

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

    def after_val_epoch(self, runner, metrics=None) -> None:
        if not self._val_records or not _MPL_AVAILABLE:
            return
        if not self._context_keys or not self._num_experts:
            return

        epoch = runner.epoch
        E     = self._num_experts
        stats = _aggregate(self._val_records, E, self._context_keys)

        self._plot_overall(epoch, stats.get('overall', {}), E)
        for key, groups in stats.get('per_context', {}).items():
            self._plot_top1_by_context(epoch, key, groups, E)
            self._plot_topk_by_context(epoch, key, groups, E)
            self._plot_dense_prob_by_context(epoch, key, groups, E)
            self._plot_dispatch_mass_by_context(epoch, key, groups, E)

        runner.logger.info(
            f'ContextExpertUsageVisualizationHook: plots → {self._out_dir}')
        self._val_records = []

    # ── Plotting helpers ───────────────────────────────────────────────────

    def _plot_overall(self, epoch: int, overall: dict, E: int) -> None:
        top1  = overall.get('top1_selection_freq_per_expert', {})
        dense = overall.get('dense_mean_prob_per_expert', {})
        if not top1 and not dense:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors  = plt.cm.tab10.colors
        experts = list(range(E))
        labels  = [f'E{e}' for e in experts]

        # Left: top-1 selection frequency
        ax   = axes[0]
        vals = [top1.get(f'expert_{e}', 0) for e in experts]
        bars = ax.bar(experts, vals,
                      color=[colors[e % len(colors)] for e in experts])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Fraction of samples')
        ax.set_title('Top-1 Selection Frequency')
        ax.set_xticks(experts)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.25 + 0.01 if vals else 0.1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        # Right: dense mean router probability (pre-top-k)
        ax   = axes[1]
        vals = [dense.get(f'expert_{e}', 0) for e in experts]
        bars = ax.bar(experts, vals,
                      color=[colors[e % len(colors)] for e in experts])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Mean pre-top-k softmax probability')
        ax.set_title('Dense Mean Router Probability (Pre-Top-k)')
        ax.set_xticks(experts)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(vals) * 1.25 + 0.01 if vals else 0.1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        n   = overall.get('num_samples', '?')
        ent = overall.get('mean_gate_entropy', '?')
        fig.suptitle(
            f'Overall Routing — {n} samples  |  mean entropy={ent}',
            fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir,
                         f'routing_overall_epoch{epoch}.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_grouped_bars(
        self,
        epoch: int,
        key: str,
        groups: dict,
        E: int,
        metric_key: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not groups:
            return

        context_vals = sorted(groups.keys())
        n_vals  = len(context_vals)
        colors  = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(max(8, n_vals * 1.8 + 2), 5.5))
        x         = np.arange(n_vals)
        bar_width = 0.8 / E

        for e in range(E):
            vals = [
                groups[cv].get(metric_key, {}).get(f'expert_{e}', 0)
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
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(
            context_vals,
            rotation=30 if n_vals > 4 else 0,
            ha='right' if n_vals > 4 else 'center')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self._out_dir, filename),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_top1_by_context(
        self, epoch: int, key: str, groups: dict, E: int
    ) -> None:
        self._plot_grouped_bars(
            epoch, key, groups, E,
            metric_key='top1_selection_freq_per_expert',
            ylabel='Fraction of samples (top-1 selection)',
            title=f'Top-1 Selection Frequency by {key.replace("_", " ").title()} (Epoch {epoch})',
            filename=f'top1_selection_by_{key}_epoch{epoch}.png',
        )

    def _plot_topk_by_context(
        self, epoch: int, key: str, groups: dict, E: int
    ) -> None:
        self._plot_grouped_bars(
            epoch, key, groups, E,
            metric_key='topk_selection_freq_per_expert',
            ylabel='Fraction of samples (any top-k position)',
            title=f'Top-k Selection Frequency by {key.replace("_", " ").title()} (Epoch {epoch})',
            filename=f'topk_selection_by_{key}_epoch{epoch}.png',
        )

    def _plot_dense_prob_by_context(
        self, epoch: int, key: str, groups: dict, E: int
    ) -> None:
        self._plot_grouped_bars(
            epoch, key, groups, E,
            metric_key='dense_mean_prob_per_expert',
            ylabel='Mean pre-top-k softmax probability',
            title=f'Dense Mean Probability by {key.replace("_", " ").title()} (Pre-Top-k) (Epoch {epoch})',
            filename=f'dense_prob_by_{key}_epoch{epoch}.png',
        )

    def _plot_dispatch_mass_by_context(
        self, epoch: int, key: str, groups: dict, E: int
    ) -> None:
        self._plot_grouped_bars(
            epoch, key, groups, E,
            metric_key='dispatch_mass_per_expert',
            ylabel='Mean Shazeer dispatch mass (topk weights)',
            title=f'Dispatch Mass by {key.replace("_", " ").title()} (Epoch {epoch})',
            filename=f'dispatch_mass_by_{key}_epoch{epoch}.png',
        )

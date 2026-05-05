"""MoE routing diagnostics hook.

Tracks per-expert routing quantities, router-scale diagnostics, and
context-supervised auxiliary diagnostics per epoch (train + val).

Per-expert routing metrics (computed from each iter's ``_moe_info``):

    dispatch_mass_per_expert
        Mean post-top-k routed weight per expert, averaged over samples.
        Uses the actual dispatch weights (``topk_weights``) after top-k
        masking and any renormalisation applied by the model.

    dense_mean_prob_per_expert
        Mean pre-top-k softmax probability per expert, averaged over
        samples.  Computed from the full softmax over all experts.
        Reflects the router's continuous preference signal.

    top1_selection_freq_per_expert
        Fraction of samples for which expert e is the rank-1 selection
        (``topk_idx[:, 0] == e``).  Computed from the *dispatch* top-k
        (noisy for NoisyTopkGate during training, clean otherwise).

    topk_selection_freq_per_expert
        Fraction of samples for which expert e appears anywhere in the
        dispatch top-k set.  Equals top1_selection_freq when k=1.

    clean_top1_selection_freq_per_expert
        Same as ``top1_selection_freq_per_expert`` but computed from
        ``clean_topk_idx`` — the deterministic top-k of the clean
        logits.  For :class:`TopkGate` and for :class:`NoisyTopkGate`
        in eval mode this is identical to
        ``top1_selection_freq_per_expert``; under a NoisyTopkGate in
        training it reveals what the router *would* dispatch without
        the Gaussian noise perturbation, catching clean rank-collapse
        that is hidden by the noisy dispatch.

    clean_topk_selection_freq_per_expert
        Clean-top-k analog of ``topk_selection_freq_per_expert``.

    mean_gate_entropy
        Average ``-Σ p log p`` of the per-sample full pre-top-k softmax.

Router-scale diagnostics (mean over samples, per epoch):

    clean_logits_{mean,std,abs_mean,min,max,lse_mean}
    noisy_logits_{mean,std,abs_mean,min,max,lse_mean}    (if available)
    noise_std_{mean,min,max}                              (NoisyTopkGate only)

Context-supervised auxiliary diagnostics:

    ctx_aux_loss              UNWEIGHTED F.cross_entropy(ctx_logits, label)
    ctx_aux_loss_weighted     coef · ctx_aux_loss
    router_z_loss             clean-logit z regulariser value
    importance_loss
    load_loss
    switch_balance_loss       Fedus Switch balance (present when
                              switch_balance_coef > 0 on any block;
                              should be computed from clean_topk_idx).
    group_balance_loss        (modality_specific only)
    ctx_aux_acc               Fraction of correct argmax predictions.
    ctx_aux_acc_per_class     (overall val) accuracy per context class.
    ctx_pred_hist             bincount of argmax predictions over the split.
    ctx_label_hist            bincount of context labels over the split.
    ctx_loss_type             Most recent ctx loss type seen
                              ('weighted_ce' | 'ce' | 'focal').
    ctx_class_weights         Normalised class weights used for weighted_ce
                              (None when not used).
    summary_pool_size
    summary_spatial_dim
    summary_hidden_dim
    summary_out_dim           Echo of the BEVSummaryHead config for the run.
    noise_scale               NoisyTopkGate global noise multiplier (train-time
                              only; most recent value seen).
    noise_epsilon             NoisyTopkGate softplus epsilon on the std head.
    noise_to_clean_std_ratio  Mean of ``(noise_scale · noise_std_mean) /
                              clean_logits_std`` across accumulated iterations.
                              Multiplying by ``noise_scale`` is intentional:
                              the actual injected noise std is the scaled
                              product, not the bare softplus output of the
                              noise head.  Target: ≲ 1 (≈ 0.5 is healthy).

Outputs (under ``<work_dir>/moe_routing/``):

    dispatch_mass_train_epochN.json / dispatch_mass_val_epochN.json
    dense_mean_prob_train_epochN.json / dense_mean_prob_val_epochN.json
    routing_summary_epochN.json — val-epoch summary including all routing
                                  metrics, router-scale diagnostics,
                                  context-aux diagnostics, and AP.
    dispatch_mass_per_expert.png — line plot over epochs.
    dense_mean_prob_per_expert.png — line plot over epochs.

Hook C (modality-specific group mass):
    group_mass_train_epochN.json / group_mass_val_epochN.json
    group_mass.png

Config example
--------------
    dict(
        type='MoERoutingHook',
        num_experts=5,
        ap_metric_keys=['mAP_0.5m', 'mAP_0.50'],
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
        self._tr = self._fresh_acc(E)
        self._va = self._fresh_acc(E)

        # Epoch-history for line plots
        self._tr_dispatch_hist: Dict[int, List[float]] = {}
        self._va_dispatch_hist: Dict[int, List[float]] = {}
        self._tr_dense_hist:    Dict[int, List[float]] = {}
        self._va_dense_hist:    Dict[int, List[float]] = {}
        self._tr_group_hist:    Dict[int, tuple] = {}
        self._va_group_hist:    Dict[int, tuple] = {}

        self._out_dir: Optional[str] = None

    @staticmethod
    def _fresh_acc(E: int) -> Dict[str, Any]:
        """Build a fresh accumulator dict for one phase (train or val)."""
        return {
            # Per-expert routing (dispatch = training/noisy top-k)
            'dispatch':       [0.0] * E,
            'dense':          [0.0] * E,
            'top1':           [0]   * E,
            'topk':           [0]   * E,
            # Per-expert routing over the *clean* deterministic top-k
            # (what the router would pick without training-time noise).
            'clean_top1':     [0]   * E,
            'clean_topk':     [0]   * E,
            'clean_n':        0,
            'entropy_sum':    0.0,
            'n':              0,
            # Modality group mass
            'cam_mass':       0.0,
            'lidar_mass':     0.0,
            # Aux loss running sums (loss_value_sum, count) → mean later
            'imp_sum':        0.0, 'imp_n':       0,
            'load_sum':       0.0, 'load_n':      0,
            'switch_sum':     0.0, 'switch_n':    0,
            'router_z_sum':   0.0, 'router_z_n':  0,
            'group_bal_sum':  0.0, 'group_bal_n': 0,
            'ctx_loss_sum':   0.0, 'ctx_loss_n':  0,
            'ctx_w_sum':      0.0, 'ctx_w_n':     0,
            'ctx_acc_sum':    0.0, 'ctx_acc_n':   0,
            # Per-class context confusion (sum over iters): pred[c, t]
            'ctx_pred_hist':  None,   # List[int] | None — sized on first hit
            'ctx_label_hist': None,
            'ctx_correct_per_class': None,   # List[int]
            'ctx_total_per_class':   None,   # List[int]
            'ctx_target_field':      None,
            # Ctx-head config echoes (last seen wins; all blocks share).
            'ctx_loss_type':         None,
            'ctx_class_weights':     None,
            # BEVSummaryHead config echoes (last seen wins; all blocks
            # in a run share).
            'summary_cfg':           {},
            # Router-scale diagnostics (running sums of per-iter scalars)
            'logit_stats':    {},  # key → [sum, n]
        }

    # ── Setup ──────────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        runner.logger.info(f'MoERoutingHook: artifacts → {self._out_dir}')

    # ── Per-iteration accumulation ─────────────────────────────────────────

    @staticmethod
    def _add_scalar_loss(acc: Dict[str, Any], key_sum: str, key_n: str,
                         info: Dict[str, Any], src: str) -> None:
        v = info.get(src)
        if v is None:
            return
        if isinstance(v, torch.Tensor):
            try:
                v = float(v.detach().item())
            except Exception:
                return
        try:
            acc[key_sum] += float(v)
            acc[key_n]   += 1
        except (TypeError, ValueError):
            pass

    def _accumulate(self, runner, phase: str) -> None:
        """Accumulate routing + router-scale + ctx-aux metrics from _moe_info."""
        moe_modules = _get_moe_modules(runner.model)
        if not moe_modules:
            return

        acc = self._tr if phase == 'train' else self._va

        for attr_name, mod in moe_modules.items():
            info = getattr(mod, '_moe_info', None)
            if info is None:
                continue

            full_probs     = info.get('full_softmax_probs')    # (B, E)
            topk_idx       = info.get('topk_idx')              # (B, k)
            topk_weights   = info.get('topk_weights')          # (B, k)
            clean_topk_idx = info.get('clean_topk_idx')        # (B, k) | None
            if full_probs is None or topk_idx is None or topk_weights is None:
                continue

            B, E = full_probs.shape
            k    = topk_idx.shape[1]
            E_use = min(E, self.num_experts)

            # ── Per-expert metrics (dispatch top-k) ───────────────────
            dispatch = acc['dispatch']
            dense    = acc['dense']
            top1     = acc['top1']
            topk_arr = acc['topk']

            for e in range(E_use):
                dense[e] += float(full_probs[:, e].sum().item())

            for b in range(B):
                for j in range(k):
                    eidx = int(topk_idx[b, j].item())
                    if 0 <= eidx < self.num_experts:
                        dispatch[eidx] += float(topk_weights[b, j].item())

            for b in range(B):
                eidx = int(topk_idx[b, 0].item())
                if 0 <= eidx < self.num_experts:
                    top1[eidx] += 1

            for b in range(B):
                seen = set()
                for j in range(k):
                    eidx = int(topk_idx[b, j].item())
                    if 0 <= eidx < self.num_experts and eidx not in seen:
                        topk_arr[eidx] += 1
                        seen.add(eidx)

            # ── Per-expert metrics (clean deterministic top-k) ────────
            # Diagnostic-only: what the router would dispatch without
            # the Gaussian noise term.  Under TopkGate / eval this is
            # identical to the dispatch top-k above.
            if clean_topk_idx is not None:
                c_top1 = acc['clean_top1']
                c_topk = acc['clean_topk']
                k_clean = clean_topk_idx.shape[1]
                for b in range(B):
                    eidx = int(clean_topk_idx[b, 0].item())
                    if 0 <= eidx < self.num_experts:
                        c_top1[eidx] += 1
                for b in range(B):
                    seen = set()
                    for j in range(k_clean):
                        eidx = int(clean_topk_idx[b, j].item())
                        if (0 <= eidx < self.num_experts
                                and eidx not in seen):
                            c_topk[eidx] += 1
                            seen.add(eidx)
                acc['clean_n'] += B

            with torch.no_grad():
                ent = -(full_probs.clamp_min(1e-12) *
                        full_probs.clamp_min(1e-12).log()).sum(dim=-1)
                acc['entropy_sum'] += float(ent.sum().item())
            acc['n'] += B

            # ── Modality group mass (Hook C) ──────────────────────────
            if self.enable_hook_c and attr_name == 'modality_specific_moe':
                acc['cam_mass']   += float(info.get('cam_group_mass',   0.0))
                acc['lidar_mass'] += float(info.get('lidar_group_mass', 0.0))

            # ── Aux loss components ───────────────────────────────────
            self._add_scalar_loss(acc, 'imp_sum',       'imp_n',
                                  info, 'importance_loss')
            self._add_scalar_loss(acc, 'load_sum',      'load_n',
                                  info, 'load_loss')
            self._add_scalar_loss(acc, 'switch_sum',    'switch_n',
                                  info, 'switch_balance_loss')
            self._add_scalar_loss(acc, 'router_z_sum',  'router_z_n',
                                  info, 'router_z_loss')
            self._add_scalar_loss(acc, 'group_bal_sum', 'group_bal_n',
                                  info, 'group_balance_loss')
            self._add_scalar_loss(acc, 'ctx_loss_sum',  'ctx_loss_n',
                                  info, 'ctx_aux_loss')
            self._add_scalar_loss(acc, 'ctx_w_sum',     'ctx_w_n',
                                  info, 'ctx_aux_loss_weighted')
            self._add_scalar_loss(acc, 'ctx_acc_sum',   'ctx_acc_n',
                                  info, 'ctx_aux_acc')

            # ── Ctx-head + BEVSummaryHead configuration echoes ────────
            # Last-seen wins; inside a run the blocks share the same
            # config so overwriting is fine.  Kept as plain fields in
            # the accumulator so the final JSON metric dump carries
            # them.
            ctx_lt = info.get('ctx_loss_type')
            if ctx_lt is not None:
                acc['ctx_loss_type'] = str(ctx_lt)
            ctx_cw = info.get('ctx_class_weights')
            if ctx_cw is not None:
                acc['ctx_class_weights'] = [float(w) for w in ctx_cw]
            for cfg_key in ('summary_pool_size', 'summary_spatial_dim',
                            'summary_hidden_dim', 'summary_out_dim'):
                v = info.get(cfg_key)
                if v is not None:
                    acc['summary_cfg'][cfg_key] = int(v)

            # ── Context histograms / per-class accuracy ───────────────
            ctx_pred  = info.get('ctx_pred_hist')
            ctx_label = info.get('ctx_label_hist')
            target    = info.get('ctx_target_field')
            if ctx_pred and ctx_label and len(ctx_pred) == len(ctx_label):
                C = len(ctx_pred)
                if acc['ctx_pred_hist'] is None:
                    acc['ctx_pred_hist']  = [0] * C
                    acc['ctx_label_hist'] = [0] * C
                    acc['ctx_correct_per_class'] = [0] * C
                    acc['ctx_total_per_class']   = [0] * C
                # Pad if a different MoE block has more classes (rare).
                while len(acc['ctx_pred_hist']) < C:
                    acc['ctx_pred_hist'].append(0)
                    acc['ctx_label_hist'].append(0)
                    acc['ctx_correct_per_class'].append(0)
                    acc['ctx_total_per_class'].append(0)
                for c in range(C):
                    acc['ctx_pred_hist'][c]  += int(ctx_pred[c])
                    acc['ctx_label_hist'][c] += int(ctx_label[c])
                # Per-class correct counts can only be derived per iter
                # from the diagonal of a confusion matrix — we
                # approximate using min(pred, label) per class which is
                # a strict lower bound.  For a precise per-class
                # accuracy use the per-sample analysis in
                # ContextRoutingStatsHook.
                for c in range(C):
                    acc['ctx_correct_per_class'][c] += min(
                        int(ctx_pred[c]), int(ctx_label[c]))
                    acc['ctx_total_per_class'][c]   += int(ctx_label[c])
                if target is not None:
                    acc['ctx_target_field'] = target

            # ── Router-scale diagnostics ──────────────────────────────
            # The ``noise_scale`` and ``noise_epsilon`` fields are
            # constants per block but are accumulated as means for
            # uniformity with the other scale diagnostics — the mean
            # equals the constant for any non-empty window.
            for src_key in (
                'clean_logits_mean', 'clean_logits_std',
                'clean_logits_abs_mean', 'clean_logits_min',
                'clean_logits_max', 'clean_logits_lse_mean',
                'noisy_logits_mean', 'noisy_logits_std',
                'noisy_logits_abs_mean', 'noisy_logits_min',
                'noisy_logits_max', 'noisy_logits_lse_mean',
                'noise_std_mean', 'noise_std_min', 'noise_std_max',
                'noise_scale', 'noise_epsilon',
                'noise_to_clean_std_ratio',
                'ctx_logits_mean_abs',
            ):
                v = info.get(src_key)
                if v is None:
                    continue
                stats = acc['logit_stats'].setdefault(src_key, [0.0, 0])
                stats[0] += float(v)
                stats[1] += 1

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

    def _build_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        """Build the canonical metric dict for a given split's accumulator."""
        n = acc['n']
        out: Dict[str, Any] = {
            'num_samples': n,
            'dispatch_mass_per_expert':       self._to_expert_dict(
                self._norm_dispatch(acc['dispatch'], n)),
            'dense_mean_prob_per_expert':      self._to_expert_dict(
                self._norm_dense(acc['dense'], n)),
            'top1_selection_freq_per_expert':  self._to_expert_dict(
                self._freq(acc['top1'], n)),
            'topk_selection_freq_per_expert':  self._to_expert_dict(
                self._freq(acc['topk'], n)),
            'mean_gate_entropy':              round(
                acc['entropy_sum'] / max(n, 1), 6),
        }

        # Clean-routing selection frequencies (diagnostic-only): what
        # the router would dispatch without the training-time Gaussian
        # noise.  Identical to ``top*_selection_freq_per_expert`` for
        # TopkGate and for NoisyTopkGate in eval.
        clean_n = acc['clean_n']
        if clean_n > 0:
            out['clean_top1_selection_freq_per_expert'] = self._to_expert_dict(
                self._freq(acc['clean_top1'], clean_n))
            out['clean_topk_selection_freq_per_expert'] = self._to_expert_dict(
                self._freq(acc['clean_topk'], clean_n))

        # Aux loss means
        def _mean(s: float, n: int) -> Optional[float]:
            if n == 0:
                return None
            return round(s / n, 8)

        for label, src_sum, src_n in (
            ('importance_loss',       'imp_sum',       'imp_n'),
            ('load_loss',             'load_sum',      'load_n'),
            ('switch_balance_loss',   'switch_sum',    'switch_n'),
            ('router_z_loss',         'router_z_sum',  'router_z_n'),
            ('group_balance_loss',    'group_bal_sum', 'group_bal_n'),
            ('ctx_aux_loss',          'ctx_loss_sum',  'ctx_loss_n'),
            ('ctx_aux_loss_weighted', 'ctx_w_sum',     'ctx_w_n'),
            ('ctx_aux_acc',           'ctx_acc_sum',   'ctx_acc_n'),
        ):
            v = _mean(acc[src_sum], acc[src_n])
            if v is not None:
                out[label] = v

        # Ctx-head and BEVSummaryHead config echoes.
        if acc.get('ctx_loss_type') is not None:
            out['ctx_loss_type'] = acc['ctx_loss_type']
        if acc.get('ctx_class_weights') is not None:
            out['ctx_class_weights'] = list(acc['ctx_class_weights'])
        if acc.get('summary_cfg'):
            out.update(acc['summary_cfg'])

        # Router-scale diagnostics
        scale: Dict[str, float] = {}
        for k, (s, c) in acc['logit_stats'].items():
            if c > 0:
                scale[k] = round(s / c, 8)
        if scale:
            out['router_scale_diagnostics'] = scale

        # Context histograms / per-class accuracy
        if acc['ctx_pred_hist'] is not None:
            out['ctx_target_field'] = acc.get('ctx_target_field')
            out['ctx_pred_hist']    = list(acc['ctx_pred_hist'])
            out['ctx_label_hist']   = list(acc['ctx_label_hist'])
            per_class_acc = []
            for correct, total in zip(acc['ctx_correct_per_class'],
                                      acc['ctx_total_per_class']):
                per_class_acc.append(
                    round(correct / total, 6) if total > 0 else None)
            out['ctx_aux_acc_per_class_lower_bound'] = per_class_acc
            tot_correct = sum(acc['ctx_correct_per_class'])
            tot_total   = sum(acc['ctx_total_per_class'])
            if tot_total > 0:
                out['ctx_aux_acc_overall_lower_bound'] = round(
                    tot_correct / tot_total, 6)

        return out

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
        acc = self._tr
        if acc['n'] == 0:
            self._reset_train()
            return

        metrics = self._build_metrics(acc)

        dispatch_vals = list(metrics['dispatch_mass_per_expert'].values())
        dense_vals    = list(metrics['dense_mean_prob_per_expert'].values())
        self._tr_dispatch_hist[epoch] = dispatch_vals
        self._tr_dense_hist[epoch]    = dense_vals

        self._save_json(
            {'epoch': epoch, 'split': 'train', **metrics},
            f'dispatch_mass_train_epoch{epoch}.json')
        self._save_json(
            {'epoch': epoch, 'split': 'train', **metrics},
            f'dense_mean_prob_train_epoch{epoch}.json')

        self._save_dispatch_mass_plot()
        self._save_dense_prob_plot()

        if self.enable_hook_c and (acc['cam_mass'] + acc['lidar_mass']) > 0:
            total = acc['cam_mass'] + acc['lidar_mass'] + 1e-8
            cam_f   = acc['cam_mass']   / total
            lidar_f = acc['lidar_mass'] / total
            self._tr_group_hist[epoch] = (cam_f, lidar_f)
            self._save_json(
                {'epoch': epoch, 'split': 'train',
                 'cam_group_mass_frac': cam_f,
                 'lidar_group_mass_frac': lidar_f},
                f'group_mass_train_epoch{epoch}.json')
            self._save_group_mass_plot()

        self._reset_train()

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        epoch = runner.epoch
        metrics = metrics or {}
        acc = self._va
        if acc['n'] == 0:
            self._reset_val()
            return

        routing_metrics = self._build_metrics(acc)

        dispatch_vals = list(routing_metrics['dispatch_mass_per_expert'].values())
        dense_vals    = list(routing_metrics['dense_mean_prob_per_expert'].values())
        self._va_dispatch_hist[epoch] = dispatch_vals
        self._va_dense_hist[epoch]    = dense_vals

        self._save_json(
            {'epoch': epoch, 'split': 'val', **routing_metrics},
            f'dispatch_mass_val_epoch{epoch}.json')
        self._save_json(
            {'epoch': epoch, 'split': 'val', **routing_metrics},
            f'dense_mean_prob_val_epoch{epoch}.json')

        self._save_dispatch_mass_plot()
        self._save_dense_prob_plot()

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
        if self.enable_hook_c and (acc['cam_mass'] + acc['lidar_mass']) > 0:
            total = acc['cam_mass'] + acc['lidar_mass'] + 1e-8
            summary['cam_group_mass_frac']   = acc['cam_mass']   / total
            summary['lidar_group_mass_frac'] = acc['lidar_mass'] / total
        self._save_json(summary, f'routing_summary_epoch{epoch}.json')

        if self.enable_hook_c and (acc['cam_mass'] + acc['lidar_mass']) > 0:
            total = acc['cam_mass'] + acc['lidar_mass'] + 1e-8
            cam_f   = acc['cam_mass']   / total
            lidar_f = acc['lidar_mass'] / total
            self._va_group_hist[epoch] = (cam_f, lidar_f)
            self._save_json(
                {'epoch': epoch, 'split': 'val',
                 'cam_group_mass_frac': cam_f,
                 'lidar_group_mass_frac': lidar_f},
                f'group_mass_val_epoch{epoch}.json')
            self._save_group_mass_plot()

        self._reset_val()

    # ── Resets ─────────────────────────────────────────────────────────────

    def _reset_train(self) -> None:
        self._tr = self._fresh_acc(self.num_experts)

    def _reset_val(self) -> None:
        self._va = self._fresh_acc(self.num_experts)

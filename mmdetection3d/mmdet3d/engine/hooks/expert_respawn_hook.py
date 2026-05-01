"""Dead-expert respawn hook for sparse MoE training.

Motivation
----------
Sparse top-k routers (Shazeer / Switch / Mixtral-style) are susceptible to the
"lottery-winner" dead-expert pathology:

  1. At init each expert's gate bias + weight lands in a random spot.
  2. If an expert's logits start even slightly below the others it falls
     out of the top-k rotation early.
  3. Once out, it receives no task-loss gradient through the dispatch
     pathway, so its conv weights (and its softmax probability) drift
     further from the live experts, making it permanently dead.

Soft balance losses (importance_loss, load_loss) can fight this, but only
while the dead expert's softmax probability is still in the region where
the ``p(1-p)`` Jacobian is non-negligible.  Once an expert is at the
softmax floor, these losses can no longer rescue it — the gradient is
numerically zero.

Mechanism
---------
This hook tracks per-expert cumulative dispatch mass during training (same
quantity plotted by :class:`MoERoutingHook`).  At the end of each training
epoch it checks whether any expert's dispatch fraction has fallen below a
configurable threshold (default 10% of the uniform fraction, i.e. 0.1/E).
Every dead expert is reinitialised from the single most-utilised live
expert via weight-copy-with-perturbation:

  • Expert conv + BN parameters: copied from the healthiest expert.  Conv
    weights get small Gaussian perturbation to break symmetry; BN running
    stats are copied verbatim.
  • Router gate column (``w_gate``): copied from the healthiest expert's
    column so the respawned expert enters the next epoch with the same
    routing preferences as its donor — it will immediately start winning
    roughly half of the dispatches that previously went to the donor.
    Perturbation breaks the tie in a random direction.
  • Router noise head (``w_noise``) if present: same treatment as
    ``w_gate`` so noise magnitudes stay reasonable.
  • Optimizer state for all respawned parameters is wiped so that Adam's
    running moments do not instantly undo the reset.

The effect is a "biological" respawn: the dead expert comes back as a
near-clone of the healthiest expert, then rapidly specialises because
(a) they start on top of each other in logit space and the importance
loss + task loss pressure them apart, and (b) the tiny weight perturbation
creates a differential task-loss gradient.

Safety properties
-----------------
  • Only runs once per epoch (after training, before validation).
  • Skips respawn during the very first training epoch — need some stats
    before we can trust the dispatch numbers.
  • No-op if no experts are dead, no-op if there's only one live expert
    (nothing to copy from).
  • Limits total respawns per run via ``max_respawns`` so repeated
    resetting cannot mask a deeper bug.

Config example
--------------
    dict(
        type='ExpertRespawnHook',
        num_experts=5,
        dead_threshold_ratio=0.1,   # dead if dispatch < 0.1 * uniform = 0.02
        perturbation_std=0.02,
        max_respawns=3,
    )

Place this hook *before* ``MoERoutingHook`` in ``custom_hooks`` so that the
routing diagnostic for the next epoch reflects the post-respawn state.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet3d.registry import HOOKS


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if is_model_wrapper(model) else model


def _get_moe_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Return {attr_name: module} for all MoE blocks on the model.

    Mirrors the discovery logic in :class:`MoERoutingHook` so both hooks
    operate on the same set of blocks.
    """
    m = _unwrap(model)
    result: Dict[str, nn.Module] = {}
    for name in ('bev_moe', 'modality_specific_moe', 'joint_modality_moe'):
        attr = getattr(m, name, None)
        if attr is not None and hasattr(attr, '_moe_info'):
            result[name] = attr
    return result


@HOOKS.register_module()
class ExpertRespawnHook(Hook):
    """Respawn dead experts at epoch boundaries by cloning the healthiest
    expert into each dead slot (with small perturbation) and wiping the
    corresponding optimizer state.

    Args:
        num_experts:           Total expert count (must match model config).
        dead_threshold_ratio:  An expert is respawned if its end-of-epoch
            dispatch fraction falls below
            ``dead_threshold_ratio / num_experts``.  Default 0.1 → respawn
            when usage is below 10% of the uniform share (e.g. < 0.02 for
            E=5).  Set to 0 to disable (hook becomes a no-op).
        perturbation_std:      Std of Gaussian noise added to copied conv
            and gate weights to break symmetry between the donor and the
            respawned clone.  Default 0.02 (≈ 2% of a typical Kaiming-init
            weight magnitude); small enough not to destroy the donor's
            learned representation.
        max_respawns:          Safety cap on total respawns across the
            entire run (summed across all MoE blocks and epochs).  Default
            5.  Once exhausted the hook becomes a no-op to avoid masking
            a deeper training pathology.
        skip_first_epoch:      If True, never respawn after epoch 0 — the
            dispatch stats after a single epoch are noisy and may
            prematurely flag an expert that would recover on its own.
            Default True.
        out_subdir:            Subdirectory of ``runner.work_dir`` in
            which to write ``respawn_log.json``.
    """

    # Run before MoERoutingHook so the routing diagnostic for the next
    # epoch reflects the post-respawn state.  Hooks with lower priority
    # numbers run first in mmengine; 'NORMAL' = 50, 'BELOW_NORMAL' = 60
    # (MoERoutingHook uses BELOW_NORMAL).
    priority = 'NORMAL'

    def __init__(
        self,
        num_experts: int,
        dead_threshold_ratio: float = 0.1,
        perturbation_std: float = 0.02,
        max_respawns: int = 5,
        skip_first_epoch: bool = True,
        out_subdir: str = 'moe_routing',
    ) -> None:
        self.num_experts         = num_experts
        self.dead_threshold_ratio = float(dead_threshold_ratio)
        self.perturbation_std    = float(perturbation_std)
        self.max_respawns        = int(max_respawns)
        self.skip_first_epoch    = skip_first_epoch
        self.out_subdir          = out_subdir

        # Cumulative dispatch mass per expert for the current training epoch.
        # Reset at epoch start.  Keys are the MoE block attribute names.
        self._dispatch: Dict[str, List[float]] = {}
        self._n_samples: Dict[str, int]        = {}

        # Total respawns executed so far across this run.
        self._respawn_count: int = 0

        # Event log — one entry per respawn action.  Dumped to
        # respawn_log.json at the end of each respawned epoch.
        self._log: List[Dict[str, Any]] = []

        self._out_dir: Optional[str] = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def before_run(self, runner) -> None:
        self._out_dir = os.path.join(runner.work_dir, self.out_subdir)
        os.makedirs(self._out_dir, exist_ok=True)
        runner.logger.info(
            f'ExpertRespawnHook: dead threshold = '
            f'{self.dead_threshold_ratio / max(self.num_experts, 1):.4f} '
            f'({self.dead_threshold_ratio:.0%} of uniform share), '
            f'perturbation_std={self.perturbation_std}, '
            f'max_respawns={self.max_respawns}')

    # ── Per-iteration dispatch accumulation ───────────────────────────────

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        moe_modules = _get_moe_modules(runner.model)
        for attr_name, mod in moe_modules.items():
            info = getattr(mod, '_moe_info', None)
            if info is None:
                continue
            topk_idx     = info.get('topk_idx')      # (B, k)
            topk_weights = info.get('topk_weights')  # (B, k)
            if topk_idx is None or topk_weights is None:
                continue

            B, k = topk_idx.shape
            if attr_name not in self._dispatch:
                self._dispatch[attr_name] = [0.0] * self.num_experts
                self._n_samples[attr_name] = 0

            # Scatter-add topk_weights onto the per-expert bucket.  Kept on
            # CPU to avoid accumulating a GPU tensor across the epoch.
            d = self._dispatch[attr_name]
            for b in range(B):
                for j in range(k):
                    e = int(topk_idx[b, j].item())
                    if 0 <= e < self.num_experts:
                        d[e] += float(topk_weights[b, j].item())
            self._n_samples[attr_name] += B

    # ── End-of-epoch: detect dead experts and respawn ─────────────────────

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch

        try:
            if self.skip_first_epoch and epoch == 0:
                return

            if self._respawn_count >= self.max_respawns:
                return

            moe_modules = _get_moe_modules(runner.model)

            for attr_name, mod in moe_modules.items():
                if attr_name not in self._dispatch:
                    continue

                raw     = self._dispatch[attr_name]
                total   = sum(raw) + 1e-12
                fracs   = [v / total for v in raw]
                uniform = 1.0 / max(self.num_experts, 1)
                dead_thresh = self.dead_threshold_ratio * uniform

                dead_idxs  = [e for e, f in enumerate(fracs) if f < dead_thresh]
                alive_idxs = [e for e in range(self.num_experts) if e not in dead_idxs]

                if not dead_idxs:
                    continue
                if len(alive_idxs) == 0:
                    runner.logger.warning(
                        f'[ExpertRespawnHook] {attr_name}: all experts dead at '
                        f'epoch {epoch}; skipping (no donor available).')
                    continue

                # Pick the single healthiest live expert as donor.
                donor_idx = max(alive_idxs, key=lambda e: fracs[e])

                for dead_idx in dead_idxs:
                    if self._respawn_count >= self.max_respawns:
                        runner.logger.warning(
                            f'[ExpertRespawnHook] max_respawns={self.max_respawns} '
                            f'reached; skipping further respawns this run.')
                        break

                    self._respawn_expert(
                        runner=runner,
                        block=mod,
                        block_name=attr_name,
                        dead_idx=dead_idx,
                        donor_idx=donor_idx,
                        epoch=epoch,
                        dispatch_fracs=fracs,
                    )
                    self._respawn_count += 1

            if self._log:
                out_path = os.path.join(self._out_dir, 'respawn_log.json')
                with open(out_path, 'w') as f:
                    json.dump(self._log, f, indent=2)
        finally:
            # Always reset accumulators — regardless of whether we acted.
            self._dispatch.clear()
            self._n_samples.clear()

    # ── Respawn primitive ─────────────────────────────────────────────────

    def _respawn_expert(
        self,
        runner,
        block: nn.Module,
        block_name: str,
        dead_idx: int,
        donor_idx: int,
        epoch: int,
        dispatch_fracs: List[float],
    ) -> None:
        """Clone donor expert into the dead slot and reset corresponding
        optimizer state.

        Expert conv/BN parameters belong entirely to the respawned expert,
        so their optimizer state (Adam moments) is wiped wholesale.  Gate
        Linear parameters are *shared* across experts (each expert occupies
        one row of the ``(E, in_dim)`` weight matrix), so we zero only the
        ``dead_idx`` row of each state tensor — preserving the donor's and
        other live experts' accumulated moments.
        """
        optim = runner.optim_wrapper.optimizer
        wiped_whole = 0
        wiped_rows  = 0

        # ── 1. Copy expert weights + BN stats, with perturbation on convs ─
        experts = getattr(block, 'experts', None)
        if experts is not None and len(experts) > max(dead_idx, donor_idx):
            donor_expert = experts[donor_idx]
            dead_expert  = experts[dead_idx]
            with torch.no_grad():
                for (dname, dparam), (_, sparam) in zip(
                        dead_expert.named_parameters(),
                        donor_expert.named_parameters()):
                    dparam.data.copy_(sparam.data)
                    # Perturb everything that looks like a learnable weight
                    # (convs, BN scales).  BN bias also gets a tiny nudge,
                    # which is benign.
                    if dparam.data.dim() >= 1:
                        dparam.data.add_(
                            torch.randn_like(dparam.data) * self.perturbation_std)
                    if dparam in optim.state:
                        del optim.state[dparam]
                        wiped_whole += 1
                # BatchNorm running stats (non-parameter buffers).
                for (dname, dbuf), (_, sbuf) in zip(
                        dead_expert.named_buffers(),
                        donor_expert.named_buffers()):
                    if dbuf.shape == sbuf.shape:
                        dbuf.data.copy_(sbuf.data)

        # ── 2. Copy router gate row(s) and zero only the dead-idx Adam row ─
        gate = getattr(block, 'gate', None)
        if gate is not None:
            wiped_rows += self._copy_linear_row(
                gate, 'w_gate', dead_idx, donor_idx, optim)
            if hasattr(gate, 'w_noise'):
                wiped_rows += self._copy_linear_row(
                    gate, 'w_noise', dead_idx, donor_idx, optim)

        event = {
            'epoch':             epoch,
            'block':             block_name,
            'dead_expert':       dead_idx,
            'donor_expert':      donor_idx,
            'dead_dispatch':     round(dispatch_fracs[dead_idx], 6),
            'donor_dispatch':    round(dispatch_fracs[donor_idx], 6),
            'opt_state_params_wiped': wiped_whole,
            'opt_state_rows_wiped':   wiped_rows,
        }
        self._log.append(event)
        runner.logger.info(
            f'[ExpertRespawnHook] epoch {epoch} {block_name}: respawned '
            f'expert {dead_idx} (dispatch {dispatch_fracs[dead_idx]:.4f}) '
            f'from expert {donor_idx} (dispatch {dispatch_fracs[donor_idx]:.4f}); '
            f'wiped Adam state for {wiped_whole} expert params and '
            f'{wiped_rows} gate rows.')

    def _copy_linear_row(
        self,
        gate: nn.Module,
        attr: str,
        dead_idx: int,
        donor_idx: int,
        optim,
    ) -> int:
        """Clone row ``donor_idx`` into row ``dead_idx`` of a gate Linear,
        with Gaussian perturbation on the weight row (bias copied verbatim).
        Surgically zeros ONLY the dead-idx row of any Adam state tensors
        attached to the weight/bias so the donor's running moments survive.

        The gate Linear weight has shape ``(num_experts, in_dim)`` so each
        expert occupies one row.  Copying the row transplants the donor's
        input-to-logit mapping — the respawned expert starts on top of the
        donor in logit space; the perturbation breaks the tie randomly so
        task loss + importance loss can pry them apart.

        Returns the number of state tensors that had their dead_idx row
        zeroed (for logging only).
        """
        linear: Optional[nn.Linear] = getattr(gate, attr, None)
        if not isinstance(linear, nn.Linear):
            return 0

        W, b = linear.weight, linear.bias
        with torch.no_grad():
            W.data[dead_idx].copy_(W.data[donor_idx])
            W.data[dead_idx].add_(
                torch.randn_like(W.data[dead_idx]) * self.perturbation_std)
            if b is not None:
                b.data[dead_idx].copy_(b.data[donor_idx])

        rows_wiped = 0
        # Adam/AdamW state keys: 'exp_avg', 'exp_avg_sq' (and 'max_exp_avg_sq'
        # if amsgrad=True).  Each has the same shape as the parameter, so
        # we can index the leading dim to zero just the dead row.
        for param in (W, b) if b is not None else (W,):
            st = optim.state.get(param, None)
            if not st:
                continue
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                t = st.get(key, None)
                if t is None or t.dim() == 0:
                    continue
                if t.shape[0] != linear.out_features:
                    continue
                t[dead_idx].zero_()
                rows_wiped += 1
            # ``step`` is a scalar tensor counting the number of updates
            # and is shared across all rows — do NOT reset it.
        return rows_wiped

"""Auxiliary MoE losses for expert balancing.

Primary API (used by the BEV MoE blocks)
----------------------------------------
The two default-active losses are Shazeer et al. (2017)'s "Outrageously Large
Neural Networks" importance + load pair.  They work together and penalise
complementary failure modes; with a NoisyTopkGate both are fully
differentiable.

    importance_loss(gate_probs, coef)
        CV² of the per-expert total soft probability mass
            Importance_e = Σ_b gate_probs[b, e]
        Gradient flows through the softmax back to the gate parameters.
        Penalises **uneven soft probability mass**: the signal that catches
        "dead experts" whose P_e collapses toward 0 even if hard dispatch
        looks uniform (e.g. because noise is drowning the gate signal).

    load_loss(clean_logits, noisy_logits, noise_std, k, coef)
        CV² of the smooth Shazeer load estimator
            Load_e = Σ_b  Φ((clean_logit_e - threshold_{≠e}) / noise_std_e)
        where the threshold is the k-th largest noisy logit excluding position
        e, and Φ is the standard normal CDF.  Load_e is a differentiable
        surrogate for "expected number of samples routed to expert e" under
        the noise distribution; gradient flows via Φ back through both the
        clean logits and the noise std.  Penalises **uneven hard dispatch**,
        which is the failure mode where every expert has similar P_e but only
        1–2 experts actually win top-k.

    group_balance_loss(gate_probs, cam_expert_ids, lidar_expert_ids, coef)
        Used only by the modality-specific variant; penalises deviation from
        equal camera-group vs LiDAR-group routing mass.

Why both importance_loss AND load_loss?
---------------------------------------
``importance_loss`` only sees the softmax, so a router that assigns equal
soft mass to every expert but still hard-selects only 1-2 of them scores
zero importance loss.  ``load_loss`` fills exactly this gap: by estimating
the actual top-k hit probability via the Gaussian CDF, it fires when hard
utilisation is uneven even though the soft distribution is flat.  This is
precisely the "collapsed but uniform P_e" state we saw in run 4487087.

Requirements and fallback
-------------------------
``load_loss`` REQUIRES a noisy gate — the Gaussian CDF closed form only
makes sense when there is a Gaussian noise term to integrate over.
Callers pass ``noise_std=None`` (from a deterministic TopkGate, or from a
NoisyTopkGate in eval mode) to signal "no load_loss computable"; in that
case ``load_loss`` returns a zero tensor attached to the graph so
downstream sums do not break.

Legacy / alternative
--------------------
``switch_balance_loss`` — Fedus et al. (2022) Switch Transformer balance
loss (α · E · Σ f_e · P_e) is still defined and exported for experiments
that want to swap the Shazeer pair for the Switch formulation, but is NOT
called by any block by default.  Run 4487087 showed it sat pinned at its
uniform floor α because noise-dominated f_e was already ~1/E, providing
essentially no gradient signal under the current noisy-gate regime.

Variant usage:
    BEVMoEBlock                 → importance_loss + load_loss
    JointModalityMoEBlock       → importance_loss + load_loss
    ModalitySpecificMoEBlock    → importance_loss + load_loss + group_balance_loss
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor


# ── Shazeer importance loss ───────────────────────────────────────────────

def importance_loss(
    gate_probs: Tensor,
    coef: float,
    eps: float = 1e-8,
) -> Tensor:
    """Shazeer et al. (2017) importance loss — CV² of per-expert soft mass.

    Formula::

        Importance_e = Σ_{b=1..B}  gate_probs[b, e]        # (E,)
        L_importance = α · Var(Importance) / (Mean(Importance)² + ε)

    Variance and mean are taken over the E experts.  The loss is zero iff
    every expert receives equal total soft mass over the batch and grows
    quadratically with imbalance; it is scale-invariant so the ``sum`` vs
    ``mean`` choice for Importance_e does not matter.

    Because it is computed from ``gate_probs`` (the pre-top-k softmax),
    gradient flows through the softmax back to every gate logit.

    Args:
        gate_probs: ``(B, E)`` pre-top-k softmax router probabilities
            (typically ``GateOutput.full_softmax_probs``).
        coef: Scalar weight α applied to the loss.
        eps:  Small constant to avoid division by zero.

    Returns:
        Scalar loss tensor, differentiable through ``gate_probs``.
    """
    importance = gate_probs.sum(dim=0)                       # (E,)
    mean = importance.mean()
    cv_sq = importance.var(unbiased=False) / (mean ** 2 + eps)
    return cv_sq * coef


# ── Shazeer load loss (Gaussian-CDF dispatch estimator) ───────────────────

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def load_loss(
    clean_logits: Optional[Tensor],
    noisy_logits: Optional[Tensor],
    noise_std: Optional[Tensor],
    k: int,
    coef: float,
    eps: float = 1e-8,
) -> Tensor:
    """Shazeer et al. (2017) load loss — CV² of the Gaussian-CDF dispatch estimator.

    For each sample ``b`` and expert ``e`` we compute a smooth surrogate for
    the indicator "expert e was selected into the top-k for sample b":

        P(b, e) = Pr(noisy_logit_e wins a top-k slot | clean, std)
                = Φ((clean_logit_{b,e} - threshold_{b, ≠e}) / noise_std_{b,e})

    where ``threshold_{b, ≠e}`` is the k-th largest value of
    ``noisy_logits[b, :]`` with position e excluded, and Φ is the standard
    normal CDF.  This is the classic Shazeer load estimator (see paper
    appendix A); it equals the probability that the noise sample at position
    e would push the noisy logit above the k-th largest competitor.

    The loss is then::

        Load_e = Σ_b P(b, e)                   # expected dispatch count
        L_load = α · Var(Load) / (Mean(Load)² + ε)

    ``Load_e`` is differentiable w.r.t. both ``clean_logits`` and
    ``noise_std``, so the gradient pushes the router to equalise *actual*
    top-k utilisation rather than just soft probability mass — something
    ``importance_loss`` on its own cannot do.

    Requires a noisy gate
    ---------------------
    A closed-form Gaussian CDF only makes sense when there is an explicit
    Gaussian noise term to integrate over.  If the caller has no noise
    (e.g. deterministic :class:`TopkGate`, or :class:`NoisyTopkGate` in
    eval mode), they pass ``noise_std=None`` and this function returns a
    zero tensor attached to the graph — the loss is simply disabled.
    Callers that want a load-balance signal without a noisy gate should
    use :func:`importance_loss` alone or :func:`switch_balance_loss`.

    Args:
        clean_logits: ``(B, E)`` pre-noise gate logits.  ``None`` → skip.
        noisy_logits: ``(B, E)`` logits actually used for the top-k.
            ``None`` → skip.
        noise_std:    ``(B, E)`` per-sample per-expert noise std used at
            this forward.  ``None`` → skip (no noise to integrate over).
        k:    Top-k value used by the gate.
        coef: Scalar weight α applied to the loss.
        eps:  Small constant for numerical stability.

    Returns:
        Scalar loss tensor.  Differentiable through ``clean_logits`` and
        ``noise_std`` when all three inputs are provided; a detached zero
        scalar otherwise.
    """
    # Graceful no-op when the gate provides no noise signal.
    if noise_std is None or clean_logits is None or noisy_logits is None:
        ref = clean_logits if clean_logits is not None else noisy_logits
        device = ref.device if ref is not None else torch.device('cpu')
        return torch.zeros((), device=device)

    B, E = noisy_logits.shape
    assert 1 <= k < E, (
        f'load_loss requires 1 <= k < num_experts, got k={k}, E={E}. '
        f'With k == E every expert always wins and the CDF threshold is '
        f'undefined.')

    # ── Threshold: k-th largest of noisy_logits excluding each position ──
    # Standard trick: compute top-(k+1) of the full row, then for each
    # position i the threshold_{≠i} is the k-th value if i is NOT already
    # in top-k, or the (k+1)-th value if i IS in top-k.
    topk1_vals, _ = torch.topk(noisy_logits, k + 1, dim=-1)               # (B, k+1)
    threshold_if_out = topk1_vals[:, k - 1:k]                             # (B, 1) — k-th largest
    threshold_if_in  = topk1_vals[:, k:k + 1]                             # (B, 1) — (k+1)-th largest

    # Mask of "position i belongs to the top-k of this row".
    _, topk_idx = torch.topk(noisy_logits, k, dim=-1)                     # (B, k)
    is_in_top_k = torch.zeros_like(noisy_logits, dtype=torch.bool)
    is_in_top_k.scatter_(1, topk_idx, True)                               # (B, E)

    threshold = torch.where(is_in_top_k, threshold_if_in, threshold_if_out)  # (B, E)

    # Threshold is derived from the discrete top-k op; detach it so gradient
    # flows only through the numerator (clean_logits) and denominator
    # (noise_std), matching the standard Shazeer implementation.  Without
    # detach the loss still works but the threshold would receive noisy
    # gradient from its own top-k selection.
    threshold = threshold.detach()

    # ── Gaussian CDF smooth selection probability ────────────────────────
    z = (clean_logits - threshold) / (noise_std + eps)                    # (B, E)
    P = 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))                          # (B, E) = Φ(z)

    load = P.sum(dim=0)                                                   # (E,)
    mean = load.mean()
    cv_sq = load.var(unbiased=False) / (mean ** 2 + eps)
    return cv_sq * coef


# ── Switch Transformer balance loss (alternative, not used by default) ────

def switch_balance_loss(
    gate_probs: Tensor,
    topk_idx: Tensor,
    num_experts: int,
    coef: float,
) -> Tensor:
    """Fedus et al. (2022) Switch Transformer balance loss (alternative).

    Exact formula — Switch Transformers, Section 2.1:

        L_balance = α · E · Σ_{e=1}^{E}  f_e · P_e

    where ``f_e`` is the detached hard-selection frequency and ``P_e`` is the
    differentiable mean softmax probability for expert e.  See the module
    docstring for why this is NOT the default under a noisy gate: f_e is
    essentially fixed at 1/E by the noise, so the loss sits pinned at α and
    provides almost no gradient.

    Kept here so experiments can swap back to it without needing another
    file change; not called by any BEV MoE block by default.

    Args:
        gate_probs:  (B, E) pre-top-k full softmax probabilities.
        topk_idx:    (B, k) indices of selected experts per sample.
        num_experts: Total number of experts E.
        coef:        Scalar weight α applied to the loss.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    B, k = topk_idx.shape

    counts = torch.zeros(num_experts, device=gate_probs.device)
    counts.scatter_add_(
        0,
        topk_idx.reshape(-1),
        torch.ones(B * k, device=gate_probs.device),
    )
    f = (counts / (B * k)).detach()    # (E,) — no gradient

    P = gate_probs.mean(dim=0)          # (E,) — gradient flows here

    return num_experts * (f * P).sum() * coef


# ── Group balance loss (modality-specific variant only) ───────────────────

def group_balance_loss(
    gate_probs: Tensor,
    cam_expert_ids: List[int],
    lidar_expert_ids: List[int],
    coef: float,
    eps: float = 1e-8,
) -> Tensor:
    """Soft group-balance loss for modality-specific expert routing.

    Encourages equal total routing mass (summed softmax probability) to the
    camera-expert group vs. the LiDAR-expert group.  The loss is zero when
    both groups receive exactly half the total mass, and grows quadratically
    with deviation from that midpoint.

    This is a *soft* constraint — it does not hard-equalise group sizes,
    which would prevent the network from adapting routing to scene content.

    Args:
        gate_probs:       (B, E) full softmax routing probabilities.
        cam_expert_ids:   Indices of experts designated as camera experts.
        lidar_expert_ids: Indices of experts designated as LiDAR experts.
        coef:             Scalar weight applied to the loss.
        eps:              Small constant to avoid division by zero.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    cam_mass   = gate_probs[:, cam_expert_ids].sum()
    lidar_mass = gate_probs[:, lidar_expert_ids].sum()
    total      = cam_mass + lidar_mass + eps

    cam_frac   = cam_mass / total
    lidar_frac = lidar_mass / total

    return ((cam_frac - 0.5) ** 2 + (lidar_frac - 0.5) ** 2) * coef

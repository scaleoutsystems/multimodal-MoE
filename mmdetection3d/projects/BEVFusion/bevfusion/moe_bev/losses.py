"""Auxiliary MoE losses for expert balancing.

switch_balance_loss — Switch Transformer load-balance loss; combines detached
                      hard-selection frequency with differentiable router
                      probability so the gradient actually corrects imbalance.
                      Replaces importance_loss for hard top-k routing.
load_loss           — equal hard-selection counts across all experts (monitoring only,
                      no gradient).
group_balance_loss  — equal routing mass between two expert groups, e.g. camera vs
                      LiDAR experts in the modality-specific variant (differentiable).
importance_loss     — (superseded) CV² of soft probability mass; too weak for hard
                      top-k routing because dense probs are nearly uniform when
                      logits are small.  Kept for reference.

Variant usage:
    joint-modality experts (JointModalityMoEBlock) → switch_balance_loss + load_loss
    modality-specific (ModalitySpecificMoEBlock) → switch_balance_loss + load_loss
                                                   + group_balance_loss
    fusion-then-MoE (BEVMoEBlock on fused BEV) → switch_balance_loss + load_loss
"""
from typing import List

import torch
from torch import Tensor


def switch_balance_loss(
    gate_probs: Tensor,
    topk_idx: Tensor,
    num_experts: int,
    coef: float,
) -> Tensor:
    """Switch Transformer auxiliary load-balance loss (Fedus et al., 2022).

    Exact formula — Switch Transformers, Section 2.1:

        L_balance = α · E · Σ_{e=1}^{E}  f_e · P_e

    where:
        E   = num_experts
        α   = coef                         (scaling coefficient)

        f_e = (1 / (B · k)) · Σ_{b=1}^{B} Σ_{j=1}^{k} 1[topk_idx[b,j] = e]
            = fraction of all token-expert slots dispatched to expert e
            [computed from discrete top-k selections → detached, no gradient]

        P_e = (1 / B) · Σ_{b=1}^{B} gate_probs[b, e]
            = mean pre-top-k softmax probability assigned to expert e
            [differentiable — gradient flows back to the router]

    When expert e is overloaded (high f_e), the product f_e · P_e is large,
    so the gradient through P_e pushes the router to reduce its probability
    for that expert.  This directly counteracts the observed load imbalance,
    unlike importance_loss whose signal collapses to zero when logits are small.

    Scale: L_balance = α   when routing is perfectly uniform (f_e = P_e = 1/E).
           L_balance = α·E when all tokens are dispatched to a single expert.

    Args:
        gate_probs:  (B, E) pre-top-k full softmax probabilities.
        topk_idx:    (B, k) indices of selected experts per sample.
        num_experts: Total number of experts E.
        coef:        Scalar weight α applied to the loss.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    B, k = topk_idx.shape

    # f_e: detached dispatch frequency per expert.
    counts = torch.zeros(num_experts, device=gate_probs.device)
    counts.scatter_add_(
        0,
        topk_idx.reshape(-1),
        torch.ones(B * k, device=gate_probs.device),
    )
    f = (counts / (B * k)).detach()    # (E,) — no gradient

    # P_e: mean differentiable router probability per expert.
    P = gate_probs.mean(dim=0)          # (E,) — gradient flows here

    return num_experts * (f * P).sum() * coef


def importance_loss(gate_probs: Tensor, coef: float,
                    eps: float = 1e-8) -> Tensor:
    """(Superseded by switch_balance_loss for hard top-k routing.)

    Penalise uneven soft probability mass across experts via CV².
    Ineffective when gate logits are small because the dense softmax
    remains nearly uniform even when hard selections are skewed.

    Args:
        gate_probs: (B, E) softmax routing probabilities.
        coef: scalar weight applied to the loss.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    importance = gate_probs.sum(dim=0)
    mean = importance.mean()
    cv_sq = importance.var() / (mean ** 2 + eps)
    return cv_sq * coef


def load_loss(expert_counts: Tensor, coef: float,
              eps: float = 1e-8) -> Tensor:
    """Penalise uneven expert utilisation.

    Args:
        expert_counts: (E,) number of samples routed to each expert.
            Can be integer or float; detached from the graph (this loss
            provides a gradient-free monitoring signal and is typically
            combined with importance_loss which *is* differentiable).
        coef: scalar weight applied to the loss.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar loss tensor (not differentiable -- use switch_balance_loss
        for gradient-based balancing).
    """
    # Unlike switch_balance_loss, this operates on discrete counts (how many
    # samples were actually routed to each expert) which are not
    # differentiable.  .detach() ensures no gradient flows — this loss
    # serves purely as a logged monitoring signal alongside the
    # differentiable switch_balance_loss.
    counts = expert_counts.float()
    mean = counts.mean()
    cv_sq = counts.var() / (mean ** 2 + eps)
    return cv_sq.detach() * coef


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

    This is a *soft* constraint — it does not hard-equalise group sizes, which
    would prevent the network from adapting routing to scene content.  Starting
    coefficient ``moe_group_balance_loss_weight = 5e-3`` is intentionally small.

    Args:
        gate_probs:      (B, E) full softmax routing probabilities.
        cam_expert_ids:  Indices of experts designated as camera experts.
        lidar_expert_ids: Indices of experts designated as LiDAR experts.
        coef:            Scalar weight applied to the loss.
        eps:             Small constant to avoid division by zero.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    # Sum softmax probability mass assigned to each group across the batch.
    # gate_probs[:, cam_expert_ids] : (B, n_cam) — prob mass per cam expert per sample
    # .sum() : scalar — total cam routing mass over the batch
    cam_mass   = gate_probs[:, cam_expert_ids].sum()
    lidar_mass = gate_probs[:, lidar_expert_ids].sum()
    total      = cam_mass + lidar_mass + eps

    # Fraction of total mass going to each group (each in [0, 1], sum ≈ 1).
    cam_frac   = cam_mass / total
    lidar_frac = lidar_mass / total

    # Penalise deviation from the balanced midpoint (0.5 each).
    # Note: (cam_frac - 0.5)^2 + (lidar_frac - 0.5)^2 is minimised at 0.5/0.5
    # and equals 0.5 at the extreme (one group gets everything).
    return ((cam_frac - 0.5) ** 2 + (lidar_frac - 0.5) ** 2) * coef

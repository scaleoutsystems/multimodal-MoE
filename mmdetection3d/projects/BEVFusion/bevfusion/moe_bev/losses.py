"""Auxiliary MoE losses for expert balancing.

importance_loss     — equal gate-probability mass across all experts (differentiable).
load_loss           — equal hard-selection counts across all experts (monitoring only).
group_balance_loss  — equal routing mass between two expert groups, e.g. camera vs
                      LiDAR experts in the modality-specific variant (differentiable).

All imbalance measures use squared coefficient of variation (CV²) or a squared
fractional deviation, both of which are zero when balanced and grow smoothly
with imbalance — making them easy to tune via a scalar coefficient.

Variant usage:
    joint-modality experts (JointModalityMoEBlock) → importance_loss + load_loss
    modality-specific (ModalitySpecificMoEBlock) → importance_loss + load_loss
                                                   + group_balance_loss
    fusion-then-MoE (BEVMoEBlock on fused BEV) → importance_loss + load_loss
"""
from typing import List

from torch import Tensor


def importance_loss(gate_probs: Tensor, coef: float,
                    eps: float = 1e-8) -> Tensor:
    """Penalise uneven probability mass across experts.

    Args:
        gate_probs: (B, E) softmax routing probabilities.
        coef: scalar weight applied to the loss.
        eps: small constant to avoid division by zero.

    Returns:
        Scalar loss tensor (differentiable through gate_probs).
    """
    # Sum each expert's probability across the batch → "importance" of
    # each expert.  If the gate always picks one expert, that expert's
    # importance dominates and CV^2 is high → loss pushes the gate to
    # spread probability mass more evenly.
    importance = gate_probs.sum(dim=0)  # sum the probabilities column-wise across the batch (B, E) --> (E,)
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
        Scalar loss tensor (not differentiable -- use importance_loss
        for gradient-based balancing).
    """
    # Unlike importance_loss, this operates on discrete counts (how many
    # samples were actually routed to each expert) which are not
    # differentiable.  .detach() ensures no gradient flows — this loss
    # serves purely as a logged monitoring signal alongside the
    # differentiable importance_loss.
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

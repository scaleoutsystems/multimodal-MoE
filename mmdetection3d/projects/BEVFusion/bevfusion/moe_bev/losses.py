"""Auxiliary MoE losses for expert balancing.

importance_loss encourages equal probability mass across experts.
load_loss encourages equal utilisation (sample counts) across experts.
Both use the squared coefficient of variation (CV^2) which is zero when
all entries are equal and grows with imbalance.
"""
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
    importance = gate_probs.sum(dim=0)  # (E,)
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

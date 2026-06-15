"""
Build finite compute-ordered chains from per-exit scores, losses, and costs.

Adapters should end by producing these three arrays. The certification code then consumes only
the returned loss and cost matrices, independent of the model that produced them.
"""
from __future__ import annotations
import numpy as np


def build_chain(scores, exit_loss, exit_costs, thresholds):
    """
    Return loss_matrix and cost_matrix for a global-threshold early-exit chain.

    Parameters
    ----------
    scores:
        Array with shape (n, L). Higher scores make earlier exits more likely.
    exit_loss:
        Array with shape (n, L). Per-exit bounded loss, e.g. 1 - correctness.
    exit_costs:
        Array with shape (L,). Cumulative cost to reach each exit.
    thresholds:
        Array with shape (K,). Configuration k exits at the first l with
        scores[i, l] >= thresholds[k], otherwise at the final exit.
    """
    scores = np.asarray(scores, dtype=float)
    exit_loss = np.asarray(exit_loss, dtype=float)
    exit_costs = np.asarray(exit_costs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if scores.ndim != 2:
        raise ValueError("scores must have shape (n, L)")
    if exit_loss.shape != scores.shape:
        raise ValueError("exit_loss must have the same shape as scores")
    if exit_costs.shape != (scores.shape[1],):
        raise ValueError("exit_costs must have shape (L,)")
    if thresholds.ndim != 1:
        raise ValueError("thresholds must have shape (K,)")

    n, L = scores.shape
    K = thresholds.size
    loss_matrix = np.empty((n, K), dtype=float)
    cost_matrix = np.empty((n, K), dtype=float)
    rows = np.arange(n)

    for k, threshold in enumerate(thresholds):
        meets = scores >= threshold
        has_exit = meets.any(axis=1)
        first_exit = np.argmax(meets, axis=1)
        exit_idx = np.where(has_exit, first_exit, L - 1)
        loss_matrix[:, k] = exit_loss[rows, exit_idx]
        cost_matrix[:, k] = exit_costs[exit_idx]

    return loss_matrix, cost_matrix

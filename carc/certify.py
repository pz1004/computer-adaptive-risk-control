"""
FWER-controlling procedures that turn a vector of valid p-values into a *certified set*
  hat_Lambda  with   P( exists tau in hat_Lambda : R(tau) > alpha ) <= delta.

These are standard multiple-testing tools (we claim none as novel); the early-exit
contribution is matching the chain order to the compute objective (see selector.py).
"""
from __future__ import annotations
import numpy as np

__all__ = ["bonferroni", "fixed_sequence_chain", "holm"]


def bonferroni(pvals: np.ndarray, delta: float) -> set[int]:
    """Certify every config with p <= delta/K.  Assumption-free; loosest."""
    pvals = np.asarray(pvals, dtype=float)
    K = pvals.size
    return set(np.where(pvals <= delta / K)[0].astype(int).tolist())


def fixed_sequence_chain(pvals: np.ndarray, delta: float, order: list[int]) -> set[int]:
    """
    Fixed-sequence (fallback) test along a *pre-specified* order.
    `order` lists config indices from FIRST tested to LAST. We certify a prefix of `order`,
    stopping at the first p > delta. Per-test level is delta (no multiplicity correction).

    For the compute chain, pass order = configs sorted from most expensive to cheapest;
    the certified set is then a contiguous *cheap-ward* suffix of the compute ordering.
    """
    pvals = np.asarray(pvals, dtype=float)
    certified: set[int] = set()
    for idx in order:
        if pvals[idx] <= delta:
            certified.add(int(idx))
        else:
            break
    return certified


def holm(pvals: np.ndarray, delta: float) -> set[int]:
    """
    Holm step-down: a valid FWER procedure that needs no ordering or monotonicity and
    strictly dominates Bonferroni. Used as the robust fallback when the empirical
    monotonicity check fails. (Holm is a special case of the graphical family.)
    """
    pvals = np.asarray(pvals, dtype=float)
    K = pvals.size
    order = np.argsort(pvals)
    certified: set[int] = set()
    for i, idx in enumerate(order):
        if pvals[idx] <= delta / (K - i):
            certified.add(int(idx))
        else:
            break
    return certified

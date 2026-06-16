"""
FWER-controlling procedures that turn a vector of valid p-values into a *certified set*
  hat_Lambda  with   P( exists tau in hat_Lambda : R(tau) > alpha ) <= delta.

These are standard multiple-testing tools (we claim none as novel); the early-exit
contribution is matching the chain order to the compute objective (see selector.py).
"""
from __future__ import annotations
import numpy as np

__all__ = ["bonferroni", "fixed_sequence_chain", "holm", "multi_chain"]


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


def multi_chain(
    pvals: np.ndarray,
    delta: float,
    chains: list[list[int]],
    weights: np.ndarray | list[float] | None = None,
) -> set[int]:
    """
    Certify the union of several pre-specified fixed-sequence chains.

    Each chain receives level `delta * weight_m`, runs the standard fixed-sequence
    procedure, and contributes its certified prefix. If the weights are nonnegative
    and sum to at most one, the union controls FWER at level `delta` by a union bound
    over chains. Uniform weights give each chain level `delta / M`.
    """
    pvals = np.asarray(pvals, dtype=float)
    if not chains:
        raise ValueError("chains must contain at least one chain")

    K = pvals.size
    for chain in chains:
        if not chain:
            raise ValueError("chains must not contain empty chains")
        bad = [idx for idx in chain if idx < 0 or idx >= K]
        if bad:
            raise ValueError(f"chain contains out-of-range indices: {bad}")

    M = len(chains)
    if weights is None:
        weights_arr = np.full(M, 1.0 / M, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (M,):
            raise ValueError("weights must have shape (len(chains),)")
        if np.any(weights_arr < 0):
            raise ValueError("weights must be nonnegative")
        if weights_arr.sum() > 1.0 + 1e-12:
            raise ValueError("weights must sum to at most 1")

    certified: set[int] = set()
    for chain, weight in zip(chains, weights_arr):
        if weight <= 0:
            continue
        certified |= fixed_sequence_chain(pvals, delta * float(weight), chain)
    return certified

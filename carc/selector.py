"""
Compute-Adaptive Risk Control (CARC) selectors.

Convention: configurations are indexed 0..K-1 in order of *increasing compute*
(index 0 = cheapest). `loss_matrix[i, k]` is the loss of config k on calibration point i.

Returned dict fields:
    selected   : int or None      chosen config index (cheapest certified), None if infeasible
    certified  : set[int]         certified-safe set
    feasible   : bool
    rhat       : (K,) empirical risks
    pvals      : (K,) p-values
    extra      : method-specific diagnostics
"""
from __future__ import annotations
import numpy as np
from . import pvalues as pv
from . import certify as ct

__all__ = ["select_risk", "select_dual", "select_shift", "select_naive"]


def _pvals_from_losses(loss_matrix: np.ndarray, alpha: float, pvalue: str) -> np.ndarray:
    n, K = loss_matrix.shape
    rhat = loss_matrix.mean(axis=0)
    out = np.empty(K)
    for k in range(K):
        if pvalue == "hoeffding":
            out[k] = pv.hoeffding_simple(rhat[k], n, alpha)
        elif pvalue == "hb":
            out[k] = pv.hoeffding_bentkus(rhat[k], n, alpha)
        elif pvalue == "eb":
            out[k] = pv.empirical_bernstein(loss_matrix[:, k], alpha)
        else:
            raise ValueError(f"unknown pvalue '{pvalue}'")
    return out


def select_risk(
    loss_matrix,
    costs,
    alpha,
    delta,
    method="chain",
    pvalue="hb",
    chains=None,
    chain_weights=None,
):
    """Risk-only certificate (P1): cheapest config with R(tau) <= alpha at confidence 1-delta."""
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n, K = loss_matrix.shape
    rhat = loss_matrix.mean(axis=0)
    pvals = _pvals_from_losses(loss_matrix, alpha, pvalue)

    if method == "chain":
        order = list(range(K - 1, -1, -1))            # expensive -> cheap
        certified = ct.fixed_sequence_chain(pvals, delta, order)
    elif method == "multichain":
        if chains is None:
            raise ValueError("method='multichain' requires chains")
        certified = ct.multi_chain(pvals, delta, chains, weights=chain_weights)
    elif method == "bonferroni":
        certified = ct.bonferroni(pvals, delta)
    elif method == "holm":
        certified = ct.holm(pvals, delta)
    else:
        raise ValueError(f"unknown method '{method}'")

    if not certified:
        return dict(selected=None, certified=set(), feasible=False, rhat=rhat, pvals=pvals, extra={})
    selected = min(certified, key=lambda k: costs[k])   # cheapest certified
    extra = {}
    if method == "multichain":
        extra = {
            "num_chains": len(chains),
            "chain_weights": (
                np.full(len(chains), 1.0 / len(chains)).tolist()
                if chain_weights is None
                else np.asarray(chain_weights, dtype=float).tolist()
            ),
        }
    return dict(selected=int(selected), certified=certified, feasible=True,
                rhat=rhat, pvals=pvals, extra=extra)


def select_dual(loss_matrix, cost_matrix, alpha, delta, budget,
                method="chain", pvalue="hb", delta_split=(0.9, 0.1)):
    """
    Joint risk-compute certificate (P2): R(tau) <= alpha AND C(tau) <= budget,
    both with overall confidence 1-delta. Compute is bounded by a uniform (deterministic
    delta_C/K) Hoeffding upper-confidence bound; risk uses select_risk at level delta_R.
    """
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    n, K = loss_matrix.shape
    dR, dC = delta_split[0] * delta, delta_split[1] * delta

    costs = cost_matrix.mean(axis=0)                    # used to break ties (cheapest)
    res = select_risk(loss_matrix, costs, alpha, dR, method=method, pvalue=pvalue)
    certified = res["certified"]

    Cmax = cost_matrix.max()
    chat = cost_matrix.mean(axis=0)
    Uc = chat + Cmax * np.sqrt(np.log(K / dC) / (2.0 * n))   # uniform upper bound on C(tau)
    feasible_budget = {k for k in certified if Uc[k] <= budget}

    out = dict(rhat=res["rhat"], pvals=res["pvals"],
               certified=certified, chat=chat, Uc=Uc, extra={"dR": dR, "dC": dC})
    if not feasible_budget:
        out.update(selected=None, feasible=False)
        return out
    selected = min(feasible_budget, key=lambda k: chat[k])
    out.update(selected=int(selected), feasible=True, certified_budget=feasible_budget)
    return out


def select_shift(loss_test, what_test, costs, alpha, delta, eta,
                 method="chain", pvalue="eb"):
    """
    Shift-robust certificate under ESTIMATED weights (Theorem 4).
      loss_test  : (n2, K) losses on the held-out fold D2
      what_test  : (n2,)   estimated importance weights w_hat on D2 (fitted on D1)
      eta        : L1 weight-error budget; we test against alpha - eta
    Range W = max(what_test). Returns the cheapest config certified under Q.
    """
    loss_test = np.asarray(loss_test, dtype=float)
    what_test = np.asarray(what_test, dtype=float)
    costs = np.asarray(costs, dtype=float)
    n2, K = loss_test.shape
    alpha_def = alpha - eta
    if alpha_def <= 0:
        return dict(selected=None, certified=set(), feasible=False,
                    extra={"reason": "eta >= alpha; infeasible"})
    W = max(what_test.max(), 1e-12)
    wl = loss_test * what_test[:, None]                  # weighted losses in [0, W]

    pvals = np.empty(K)
    for k in range(K):
        if pvalue == "eb":
            pvals[k] = pv.weighted_empirical_bernstein(wl[:, k], alpha_def, W)
        elif pvalue == "hoeffding":
            pvals[k] = pv.weighted_hoeffding(wl[:, k], alpha_def, W)
        else:
            raise ValueError(f"unknown weighted pvalue '{pvalue}'")

    if method == "chain":
        order = list(range(K - 1, -1, -1))
        certified = ct.fixed_sequence_chain(pvals, delta, order)
    elif method == "holm":
        certified = ct.holm(pvals, delta)
    else:
        certified = ct.bonferroni(pvals, delta)

    rqhat = wl.mean(axis=0)
    if not certified:
        return dict(selected=None, certified=set(), feasible=False,
                    rqhat=rqhat, pvals=pvals, extra={"W": W})
    selected = min(certified, key=lambda k: costs[k])
    return dict(selected=int(selected), certified=certified, feasible=True,
                rqhat=rqhat, pvals=pvals, extra={"W": W})


def select_naive(loss_matrix, costs, alpha):
    """
    Baseline (NOT a valid certificate): pick the cheapest config whose *empirical* risk
    is <= alpha, with no finite-sample or multiplicity correction. Used to show that the
    naive recipe violates the target far above delta.
    """
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    costs = np.asarray(costs, dtype=float)
    rhat = loss_matrix.mean(axis=0)
    ok = np.where(rhat <= alpha)[0]
    if ok.size == 0:
        return dict(selected=None, feasible=False, rhat=rhat)
    selected = min(ok.tolist(), key=lambda k: costs[k])
    return dict(selected=int(selected), feasible=True, rhat=rhat)

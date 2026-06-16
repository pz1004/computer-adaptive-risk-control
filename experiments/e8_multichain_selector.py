"""
Experiment E8: multi-chain CARC selector redesign.

The current CARC chain searches one scalar-threshold path. This experiment builds
several pre-specified per-exit threshold chains, then compares:

  - scalar_chain: current single-chain CARC on the global-threshold chain only
  - multichain  : union of fixed-sequence chains with delta split across chains
  - holm        : Holm over the full multi-chain candidate family
  - bonferroni  : Bonferroni over the full multi-chain candidate family
  - naive       : uncorrected empirical-risk selection over the full family

True risk and true cost are computed by large Monte Carlo from the simulator.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carc import EarlyExitSim, select_naive, select_risk
from carc.chain import build_threshold_family
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json


def build_multichain_family(sim: EarlyExitSim, levels_per_chain: int = 30) -> tuple[np.ndarray, list[list[int]], list[str]]:
    """Create several monotone per-exit threshold chains."""
    base = np.linspace(0.38, 0.98, levels_per_chain)
    offsets = [
        ("global", [0.00, 0.00, 0.00, 0.00]),
        ("early_strict", [0.16, 0.10, -0.04, -0.08]),
        ("early_very_strict", [0.24, 0.14, -0.08, -0.12]),
        ("mid_strict", [0.10, 0.18, 0.02, -0.10]),
        ("late_permissive", [0.12, 0.08, -0.12, -0.14]),
        ("balanced_staggered", [0.06, 0.12, -0.02, -0.06]),
    ]
    if sim.L != 5:
        raise ValueError("E8's default threshold schedules expect L=5 exits")

    threshold_rows: list[np.ndarray] = []
    chains: list[list[int]] = []
    names: list[str] = []
    for name, offset in offsets:
        start = len(threshold_rows)
        offset_arr = np.asarray(offset, dtype=float)
        for level in base:
            row = np.clip(level + offset_arr, 0.0, np.nextafter(1.0, 0.0))
            threshold_rows.append(row)
        end = len(threshold_rows)
        chains.append(list(range(end - 1, start - 1, -1)))  # expensive -> cheap
        names.append(name)
    return np.vstack(threshold_rows), chains, names


def sample_family(sim: EarlyExitSim, n: int, rng: np.random.Generator, threshold_vectors: np.ndarray):
    """Sample latent exits and evaluate the full per-exit threshold family."""
    _d, correct, conf = sim._draw_latents(n, rng, lam=0.0)
    return build_threshold_family(conf, 1.0 - correct, sim.costs_exit, threshold_vectors)


def evaluate_method(
    name: str,
    loss: np.ndarray,
    costs_for_selection: np.ndarray,
    alpha: float,
    delta: float,
    Rtrue: np.ndarray,
    Ctrue: np.ndarray,
    chains: list[list[int]],
):
    """Run one selector and return the selected global index, if any."""
    if name == "scalar_chain":
        scalar_idx = np.array(sorted(chains[0]))
        res = select_risk(
            loss[:, scalar_idx],
            costs_for_selection[scalar_idx],
            alpha,
            delta,
            method="chain",
            pvalue="hb",
        )
        if not res["feasible"]:
            return None
        return int(scalar_idx[res["selected"]])
    if name == "multichain":
        res = select_risk(
            loss,
            costs_for_selection,
            alpha,
            delta,
            method="multichain",
            pvalue="hb",
            chains=chains,
        )
    elif name in {"holm", "bonferroni"}:
        res = select_risk(loss, costs_for_selection, alpha, delta, method=name, pvalue="hb")
    elif name == "naive":
        res = select_naive(loss, costs_for_selection, alpha)
    else:
        raise ValueError(f"unknown method {name}")
    if not res["feasible"]:
        return None
    return int(res["selected"])


def run(n: int = 1000, T: int = 1000, oracle_n: int = 250_000, levels_per_chain: int = 30) -> dict:
    sim = EarlyExitSim(seed=31)
    threshold_vectors, chains, chain_names = build_multichain_family(sim, levels_per_chain)
    K = threshold_vectors.shape[0]

    oracle_rng = np.random.default_rng(91001)
    oracle_loss, oracle_cost = sample_family(sim, oracle_n, oracle_rng, threshold_vectors)
    Rtrue = oracle_loss.mean(axis=0)
    Ctrue = oracle_cost.mean(axis=0)

    methods = ["scalar_chain", "multichain", "holm", "bonferroni", "naive"]
    alphas = [float(Rtrue.min() + margin) for margin in (0.04, 0.08, 0.15)]
    deltas = [0.05, 0.10]
    rows = []

    print(f"E8 multi-chain selector: n={n} T={T} oracle_n={oracle_n} K={K} chains={len(chains)}")
    print(f"risk_floor={Rtrue.min():.4f} cheapest_safe depends on alpha")
    print(f"{'alpha':>6} {'delta':>6} {'method':>13} | {'viol':>7} {'feas':>7} {'cost':>8} {'excess':>8}")

    for alpha in alphas:
        safe_all = np.where(Rtrue <= alpha)[0]
        safe_scalar = np.array([idx for idx in sorted(chains[0]) if Rtrue[idx] <= alpha], dtype=int)
        oracle_cost_all = float(Ctrue[safe_all].min()) if safe_all.size else None
        oracle_cost_scalar = float(Ctrue[safe_scalar].min()) if safe_scalar.size else None

        for delta in deltas:
            rng = np.random.default_rng(7200)
            accum = {
                method: {"viol": 0, "feas": 0, "costs": [], "risks": [], "excess": []}
                for method in methods
            }
            for _ in range(T):
                loss, cost = sample_family(sim, n, rng, threshold_vectors)
                costs_for_selection = cost.mean(axis=0)
                for method in methods:
                    selected = evaluate_method(
                        method,
                        loss,
                        costs_for_selection,
                        alpha,
                        delta,
                        Rtrue,
                        Ctrue,
                        chains,
                    )
                    if selected is None:
                        continue
                    oracle_ref = oracle_cost_scalar if method == "scalar_chain" else oracle_cost_all
                    accum[method]["feas"] += 1
                    accum[method]["viol"] += int(Rtrue[selected] > alpha)
                    accum[method]["costs"].append(float(Ctrue[selected]))
                    accum[method]["risks"].append(float(Rtrue[selected]))
                    if oracle_ref is not None:
                        accum[method]["excess"].append(float(Ctrue[selected] - oracle_ref))

            for method in methods:
                vals = accum[method]
                row = {
                    "alpha": float(alpha),
                    "delta": float(delta),
                    "n": int(n),
                    "T": int(T),
                    "method": method,
                    "pvalue": None if method == "naive" else "hb",
                    "candidate_count": int(levels_per_chain if method == "scalar_chain" else K),
                    "chain_count": int(1 if method == "scalar_chain" else len(chains)),
                    "oracle_cheapest_safe_cost": oracle_cost_scalar if method == "scalar_chain" else oracle_cost_all,
                    "feasible_count": int(vals["feas"]),
                    "feasible_rate": float(vals["feas"] / T),
                    "true_violation_count": int(vals["viol"]),
                    "true_violation_rate": float(vals["viol"] / T),
                    "true_violation_ci95": clopper_pearson(vals["viol"], T),
                    "mean_true_risk": mean_or_none(vals["risks"]),
                    "mean_true_cost": mean_or_none(vals["costs"]),
                    "mean_excess_cost": mean_or_none(vals["excess"]),
                }
                rows.append(row)
                print(
                    f"{alpha:6.3f} {delta:6.2f} {method:>13} | "
                    f"{row['true_violation_rate']:7.3f} {row['feasible_rate']:7.3f} "
                    f"{(row['mean_true_cost'] if row['mean_true_cost'] is not None else np.nan):8.3f} "
                    f"{(row['mean_excess_cost'] if row['mean_excess_cost'] is not None else np.nan):8.3f}"
                )

    return {
        "experiment": "e8_multichain_selector",
        "description": "Synthetic redesign evidence for multi-chain CARC",
        "git_sha": git_sha(),
        "simulator": {"seed": 31, "oracle_n": int(oracle_n), "L": int(sim.L)},
        "config": {
            "n": int(n),
            "T": int(T),
            "levels_per_chain": int(levels_per_chain),
            "chain_names": chain_names,
            "threshold_vectors_shape": list(threshold_vectors.shape),
            "deltas": deltas,
            "alphas": alphas,
        },
        "oracle": {
            "risk_floor": float(Rtrue.min()),
            "cost_at_risk_floor": float(Ctrue[int(np.argmin(Rtrue))]),
            "candidate_count": int(K),
            "chain_count": int(len(chains)),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/e8_multichain_selector.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--oracle-n", type=int, default=250_000)
    parser.add_argument("--levels-per-chain", type=int, default=30)
    args = parser.parse_args()
    payload = run(
        n=args.n,
        T=args.T,
        oracle_n=args.oracle_n,
        levels_per_chain=args.levels_per_chain,
    )
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

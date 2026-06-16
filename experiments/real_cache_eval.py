"""
Run E1/E2/E3/E4-style checks on a CARC per-exit cache.

This script uses empirical held-out split risk as the real-data diagnostic. It does not
convert a real-data split into an oracle true-risk measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from carc import select_naive, select_risk
from carc.chain import build_threshold_family
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json
from experiments.multichain_family import build_multichain_thresholds


def load_cache(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = ["loss_matrix", "cost_matrix", "thresholds", "meta_json"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"cache missing required arrays: {missing}")
    cache = {
        "loss_matrix": data["loss_matrix"].astype(float),
        "cost_matrix": data["cost_matrix"].astype(float),
        "thresholds": data["thresholds"].astype(float),
        "meta": json.loads(str(data["meta_json"])),
    }
    for optional in ("scores", "loss", "exit_cost"):
        if optional in data:
            cache[optional] = data[optional].astype(float)
    if cache["loss_matrix"].shape != cache["cost_matrix"].shape:
        raise ValueError("loss_matrix and cost_matrix shapes differ")
    return cache


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def monotonicity_diagnostic(loss_matrix: np.ndarray, cost_matrix: np.ndarray) -> dict:
    risks = loss_matrix.mean(axis=0)
    costs = cost_matrix.mean(axis=0)
    risk_diffs = np.diff(risks)
    cost_diffs = np.diff(costs)
    return {
        "full_pool_risk": risks.tolist(),
        "full_pool_cost": costs.tolist(),
        "risk_nonincreasing": bool(np.all(risk_diffs <= 1e-12)),
        "cost_nondecreasing": bool(np.all(cost_diffs >= -1e-9)),
        "num_adjacent_risk_increases": int(np.sum(risk_diffs > 1e-12)),
        "max_adjacent_risk_increase": float(np.max(risk_diffs)) if risk_diffs.size else 0.0,
        "min_adjacent_cost_diff": float(np.min(cost_diffs)) if cost_diffs.size else 0.0,
    }


def run(args) -> dict:
    cache = load_cache(args.cache)
    if args.family == "scalar":
        loss_matrix = cache["loss_matrix"]
        cost_matrix = cache["cost_matrix"]
        chains = None
        chain_names = None
        scalar_chain_idx = None
        methods = ["chain", "holm", "bonferroni"]
        thresholds_meta = {"family": "scalar", "thresholds": cache["thresholds"].tolist()}
    else:
        missing = [key for key in ("scores", "loss", "exit_cost") if key not in cache]
        if missing:
            raise ValueError(f"--family multichain requires cache arrays: {missing}")
        threshold_vectors, chains, chain_names = build_multichain_thresholds(
            num_exits=cache["scores"].shape[1],
            levels_per_chain=args.levels_per_chain,
            threshold_low=args.threshold_low,
            threshold_high=args.threshold_high,
        )
        loss_matrix, cost_matrix = build_threshold_family(
            cache["scores"],
            cache["loss"],
            cache["exit_cost"],
            threshold_vectors,
        )
        scalar_chain_idx = np.array(sorted(chains[0]), dtype=int)
        methods = ["scalar_chain", "multichain", "holm", "bonferroni"]
        thresholds_meta = {
            "family": "multichain",
            "levels_per_chain": int(args.levels_per_chain),
            "threshold_low": float(args.threshold_low),
            "threshold_high": float(args.threshold_high),
            "threshold_vectors_shape": list(threshold_vectors.shape),
            "chain_names": chain_names,
        }
    n_total, K = loss_matrix.shape
    alphas = parse_float_list(args.alphas)
    deltas = parse_float_list(args.deltas)
    calib_sizes = parse_int_list(args.calib_sizes)
    rng = np.random.default_rng(args.seed)
    rows = []

    print(f"cache={args.cache} n={n_total} K={K}")
    print(f"{'alpha':>6} {'delta':>6} {'ncal':>6} {'method':>12} | {'viol':>7} {'feas':>7} {'cost':>10}")
    for alpha in alphas:
        for delta in deltas:
            for calib_size in calib_sizes:
                if calib_size >= n_total:
                    raise ValueError(f"calib_size {calib_size} must be < cache size {n_total}")
                accum = {
                    name: {"viol": 0, "feas": 0, "costs": [], "risks": []}
                    for name in methods + ["naive"]
                }
                for _ in range(args.T):
                    perm = rng.permutation(n_total)
                    calib_idx = perm[:calib_size]
                    test_idx = perm[calib_size:]
                    calib_loss = loss_matrix[calib_idx]
                    calib_cost = cost_matrix[calib_idx]
                    test_loss = loss_matrix[test_idx]
                    test_cost = cost_matrix[test_idx]
                    calib_cost_mean = calib_cost.mean(axis=0)
                    for method in methods:
                        if method == "scalar_chain":
                            res = select_risk(
                                calib_loss[:, scalar_chain_idx],
                                calib_cost_mean[scalar_chain_idx],
                                alpha,
                                delta,
                                method="chain",
                                pvalue=args.pvalue,
                            )
                            selected = None if not res["feasible"] else int(scalar_chain_idx[res["selected"]])
                        elif method == "multichain":
                            res = select_risk(
                                calib_loss,
                                calib_cost_mean,
                                alpha,
                                delta,
                                method="multichain",
                                pvalue=args.pvalue,
                                chains=chains,
                            )
                            selected = None if not res["feasible"] else int(res["selected"])
                        else:
                            res = select_risk(calib_loss, calib_cost_mean, alpha, delta, method=method, pvalue=args.pvalue)
                            selected = None if not res["feasible"] else int(res["selected"])
                        if res["feasible"]:
                            k = selected
                            risk = float(test_loss[:, k].mean())
                            cost = float(test_cost[:, k].mean())
                            accum[method]["feas"] += 1
                            accum[method]["viol"] += int(risk > alpha)
                            accum[method]["risks"].append(risk)
                            accum[method]["costs"].append(cost)
                    naive = select_naive(calib_loss, calib_cost_mean, alpha)
                    if naive["feasible"]:
                        k = naive["selected"]
                        risk = float(test_loss[:, k].mean())
                        cost = float(test_cost[:, k].mean())
                        accum["naive"]["feas"] += 1
                        accum["naive"]["viol"] += int(risk > alpha)
                        accum["naive"]["risks"].append(risk)
                        accum["naive"]["costs"].append(cost)

                for method, vals in accum.items():
                    feasible = vals["feas"]
                    denom = max(feasible, 1) if args.violation_denominator == "feasible" else args.T
                    row = {
                        "alpha": float(alpha),
                        "delta": float(delta),
                        "calib_size": int(calib_size),
                        "test_size": int(n_total - calib_size),
                        "T": int(args.T),
                        "method": method,
                        "pvalue": None if method == "naive" else args.pvalue,
                        "feasible_count": int(feasible),
                        "feasible_rate": float(feasible / args.T),
                        "violation_count": int(vals["viol"]),
                        "violation_rate": float(vals["viol"] / denom),
                        "violation_denominator": args.violation_denominator,
                        "violation_ci95": clopper_pearson(vals["viol"], denom),
                        "mean_test_risk": mean_or_none(vals["risks"]),
                        "mean_test_cost": mean_or_none(vals["costs"]),
                    }
                    rows.append(row)
                    print(f"{alpha:6.3f} {delta:6.2f} {calib_size:6d} {method:>12} | "
                          f"{row['violation_rate']:7.3f} {row['feasible_rate']:7.3f} "
                          f"{(row['mean_test_cost'] if row['mean_test_cost'] is not None else np.nan):10.1f}")

    return {
        "experiment": "real_cache_eval",
        "description": "Cache-based E1/E2/E3/E4 diagnostics using empirical held-out risk",
        "git_sha": git_sha(),
        "cache": str(args.cache),
        "cache_meta": cache["meta"],
        "config": {
            "T": int(args.T),
            "seed": int(args.seed),
            "alphas": alphas,
            "deltas": deltas,
            "calib_sizes": calib_sizes,
            "pvalue": args.pvalue,
            "violation_denominator": args.violation_denominator,
            "threshold_family": thresholds_meta,
        },
        "monotonicity": monotonicity_diagnostic(loss_matrix, cost_matrix),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default="results/real_cache_eval.json")
    parser.add_argument("--alphas", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--deltas", default="0.05,0.10")
    parser.add_argument("--calib-sizes", default="500,2000")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pvalue", choices=["hb", "hoeffding", "eb"], default="hb")
    parser.add_argument("--violation-denominator", choices=["all", "feasible"], default="all")
    parser.add_argument("--family", choices=["scalar", "multichain"], default="scalar")
    parser.add_argument("--levels-per-chain", type=int, default=30)
    parser.add_argument("--threshold-low", type=float, default=0.0)
    parser.add_argument("--threshold-high", type=float, default=1.0)
    args = parser.parse_args()
    payload = run(args)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

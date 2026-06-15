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
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json


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
    loss_matrix = cache["loss_matrix"]
    cost_matrix = cache["cost_matrix"]
    n_total, K = loss_matrix.shape
    alphas = parse_float_list(args.alphas)
    deltas = parse_float_list(args.deltas)
    calib_sizes = parse_int_list(args.calib_sizes)
    methods = ["chain", "holm", "bonferroni"]
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
                        res = select_risk(calib_loss, calib_cost_mean, alpha, delta, method=method, pvalue=args.pvalue)
                        if res["feasible"]:
                            k = res["selected"]
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
    args = parser.parse_args()
    payload = run(args)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

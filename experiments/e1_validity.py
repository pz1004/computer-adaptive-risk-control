"""
Experiment E1 (validity) + E3 (naive baseline), self-contained on the synthetic simulator.

For each (alpha, delta), repeat over T calibration/test splits, select with the chain
certificate and with the naive uncorrected recipe, and measure how often the *true* risk
of the selected config exceeds alpha. The chain rate should be <= delta; naive should not.

Run:  python -m experiments.e1_validity
"""
import argparse
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carc import select_risk, select_naive, EarlyExitSim
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json


def run(n=1000, T=3000):
    sim = EarlyExitSim(seed=7)
    Rtrue, Ctrue = sim.oracle()
    rows = []
    print(f"risk floor (deepest config) = {Rtrue.min():.3f},  costliest config risk = {Rtrue.max():.3f}")
    print(f"{'alpha':>6} {'delta':>6} | {'chain_viol':>10} {'chain_feas':>10} {'chain_cost':>10} "
          f"| {'naive_viol':>10} {'naive_cost':>10}")
    for alpha in (Rtrue.min() + 0.04, Rtrue.min() + 0.08, Rtrue.min() + 0.15):
        for delta in (0.05, 0.10):
            rng = np.random.default_rng(2024)
            cv = cf = 0; cc = []
            nv = 0; nc = []; nf = 0
            for _ in range(T):
                loss, cost, _ = sim.sample(n, rng=rng)
                rc = select_risk(loss, Ctrue, alpha, delta, method="chain", pvalue="hb")
                if rc["feasible"]:
                    cf += 1; cc.append(Ctrue[rc["selected"]])
                    cv += int(Rtrue[rc["selected"]] > alpha)
                rn = select_naive(loss, Ctrue, alpha)
                if rn["feasible"]:
                    nf += 1; nc.append(Ctrue[rn["selected"]])
                    nv += int(Rtrue[rn["selected"]] > alpha)
            row = {
                "alpha": float(alpha),
                "delta": float(delta),
                "n": int(n),
                "T": int(T),
                "chain": {
                    "violation_count": int(cv),
                    "violation_rate": float(cv / T),
                    "violation_ci95": clopper_pearson(cv, T),
                    "feasible_count": int(cf),
                    "feasible_rate": float(cf / T),
                    "mean_cost": mean_or_none(cc),
                },
                "naive": {
                    "violation_count": int(nv),
                    "violation_rate": float(nv / T),
                    "violation_ci95": clopper_pearson(nv, T),
                    "feasible_count": int(nf),
                    "feasible_rate": float(nf / T),
                    "mean_cost": mean_or_none(nc),
                },
            }
            rows.append(row)
            print(f"{alpha:6.3f} {delta:6.2f} | {cv/T:10.4f} {cf/T:10.3f} "
                  f"{(np.mean(cc) if cc else np.nan):10.3f} | {nv/T:10.4f} "
                  f"{(np.mean(nc) if nc else np.nan):10.3f}")
    print("\nRead: chain_viol should sit at or below delta; naive_viol typically far exceeds it.")
    return {
        "experiment": "e1_validity",
        "description": "Synthetic E1 validity and E3 naive baseline",
        "git_sha": git_sha(),
        "simulator": {"seed": 7, "oracle_seed": 12345, "K": int(sim.K), "L": int(sim.L)},
        "risk_floor": float(Rtrue.min()),
        "costliest_config_risk": float(Rtrue.max()),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/e1_validity.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--T", type=int, default=3000)
    args = parser.parse_args()
    payload = run(n=args.n, T=args.T)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

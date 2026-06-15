"""
Experiment E2 (cost of the guarantee) + E7 ablation (certification method, p-value).

Compares the certified (selected) compute of the chain, Bonferroni, and Holm procedures
at the same (alpha, delta). The chain tests each hypothesis at level delta (no K penalty),
so it should certify cheaper configs than Bonferroni (level delta/K) at equal validity.

Run:  python -m experiments.e2_chain_vs_bonferroni
"""
import argparse
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carc import select_risk, EarlyExitSim
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json


def run(n=1000, T=2000):
    sim = EarlyExitSim(seed=7)
    Rtrue, Ctrue = sim.oracle()
    alpha, delta = Rtrue.min() + 0.08, 0.10
    rows = []
    print(f"alpha={alpha:.3f} delta={delta} n={n}  (K={sim.K} configs)\n")
    print(f"{'method':>12} {'pvalue':>8} | {'mean_cost':>10} {'feasible':>9} {'true_viol':>10}")
    for method in ("chain", "holm", "bonferroni"):
        for pvalue in ("hb", "hoeffding"):
            rng = np.random.default_rng(7)
            costs = []; feas = 0; viol = 0
            for _ in range(T):
                loss, cost, _ = sim.sample(n, rng=rng)
                r = select_risk(loss, Ctrue, alpha, delta, method=method, pvalue=pvalue)
                if r["feasible"]:
                    feas += 1; costs.append(Ctrue[r["selected"]])
                    viol += int(Rtrue[r["selected"]] > alpha)
            rows.append({
                "method": method,
                "pvalue": pvalue,
                "alpha": float(alpha),
                "delta": float(delta),
                "n": int(n),
                "T": int(T),
                "mean_cost": mean_or_none(costs),
                "feasible_count": int(feas),
                "feasible_rate": float(feas / T),
                "true_violation_count": int(viol),
                "true_violation_rate": float(viol / T),
                "true_violation_ci95": clopper_pearson(viol, T),
            })
            print(f"{method:>12} {pvalue:>8} | {(np.mean(costs) if costs else np.nan):10.3f} "
                  f"{feas/T:9.3f} {viol/T:10.4f}")
    print("\nRead: all methods keep true_viol <= delta; chain/Holm certify cheaper configs")
    print("than Bonferroni, and HB p-values beat plain Hoeffding (lower cost, more feasible).")
    return {
        "experiment": "e2_chain_vs_bonferroni",
        "description": "Synthetic E2 cost comparison and E7 p-value ablation",
        "git_sha": git_sha(),
        "simulator": {"seed": 7, "oracle_seed": 12345, "K": int(sim.K), "L": int(sim.L)},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/e2_chain_vs_bonferroni.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--T", type=int, default=2000)
    args = parser.parse_args()
    payload = run(n=args.n, T=args.T)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

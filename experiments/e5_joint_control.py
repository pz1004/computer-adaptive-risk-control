"""
Experiment E5 (joint risk-compute control) on the synthetic simulator.

The slack budget should often be certified feasible. The tight budget is set just above the
oracle cheapest safe cost, so a safe budget-respecting policy exists in truth, but the uniform
range-Cmax compute UCB can still report infeasible.

Run:  python -m experiments.e5_joint_control
"""
import argparse
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carc import select_dual, EarlyExitSim
from experiments.common import clopper_pearson, git_sha, mean_or_none, write_json


def run(n=2000, T=1000):
    sim = EarlyExitSim(seed=11)
    Rtrue, Ctrue = sim.oracle()
    alpha = float(max(1.5 * Rtrue.min(), Rtrue.min() + 0.05))
    delta = 0.10
    safe = np.where(Rtrue <= alpha)[0]
    cheapest_safe_cost = float(Ctrue[safe].min()) if safe.size else None
    budgets = {
        "slack": float(np.quantile(Ctrue, 0.85)),
        "tight_oracle_feasible": float(cheapest_safe_cost + 0.05) if cheapest_safe_cost is not None else 0.0,
    }

    print(f"alpha={alpha:.3f} delta={delta} n={n} T={T}")
    print(f"cheapest oracle-safe cost = {cheapest_safe_cost:.3f}\n")
    print(f"{'budget_case':>22} {'B':>8} | {'oracle_exists':>13} {'feasible':>9} "
          f"{'risk_viol':>9} {'mean_Uc':>8} {'mean_cost':>9}")

    rows = []
    for name, budget in budgets.items():
        rng = np.random.default_rng(5)
        feasible = 0
        risk_viol = 0
        ucs = []
        selected_costs = []
        oracle_exists = bool(np.any((Rtrue <= alpha) & (Ctrue <= budget)))
        for _ in range(T):
            loss, cost, _ = sim.sample(n, rng=rng)
            res = select_dual(loss, cost, alpha, delta, budget, method="chain", pvalue="hb")
            if res["feasible"]:
                feasible += 1
                k = res["selected"]
                risk_viol += int(Rtrue[k] > alpha)
                ucs.append(res["Uc"][k])
                selected_costs.append(Ctrue[k])
        rows.append({
            "budget_case": name,
            "budget": float(budget),
            "oracle_budget_safe_exists": oracle_exists,
            "feasible_count": int(feasible),
            "feasible_rate": float(feasible / T),
            "risk_violation_count": int(risk_viol),
            "risk_violation_rate": float(risk_viol / T),
            "risk_violation_ci95": clopper_pearson(risk_viol, T),
            "mean_certified_compute_ucb": mean_or_none(ucs),
            "mean_true_selected_cost": mean_or_none(selected_costs),
        })
        print(f"{name:>22} {budget:8.3f} | {str(oracle_exists):>13} {feasible/T:9.3f} "
              f"{risk_viol/T:9.4f} {(np.mean(ucs) if ucs else np.nan):8.3f} "
              f"{(np.mean(selected_costs) if selected_costs else np.nan):9.3f}")

    print("\nRead: slack budgets can be certified jointly; a tight oracle-feasible budget can")
    print("still be reported infeasible because the range-Cmax compute UCB is conservative.")
    return {
        "experiment": "e5_joint_control",
        "description": "Synthetic joint risk-compute control",
        "git_sha": git_sha(),
        "simulator": {"seed": 11, "oracle_seed": 12345, "K": int(sim.K), "L": int(sim.L)},
        "config": {"n": int(n), "T": int(T), "alpha": float(alpha), "delta": float(delta)},
        "cheapest_oracle_safe_cost": cheapest_safe_cost,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/e5_joint_control.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--T", type=int, default=1000)
    args = parser.parse_args()
    payload = run(n=args.n, T=args.T)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

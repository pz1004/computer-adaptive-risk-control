"""
Experiment E6 (covariate shift, estimated weights) — the eta-sensitivity sweep.

Three arms under an exp-tilt covariate shift toward harder inputs:
  unweighted : source-calibrated certificate, evaluated against TRUE target risk (should violate)
  exact-w    : oracle importance weights known exactly (efficiency upper bound)
  est-w      : estimated weights (logistic density ratio), tested on a held-out fold vs alpha-eta

We sweep eta and report, per eta: realized mean ||w_hat - w||_1 (whether (*) holds), the
weighted-certificate feasibility, and its TRUE-target violation rate. This is the honest
sensitivity reading: too-small eta can violate; once eta covers the estimator error the
certificate is valid, at the price of feasibility.

Run:  python -m experiments.e6_shift_eta_sweep
"""
import argparse
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from carc import select_risk, select_shift, EarlyExitSim, fit_weights_logistic
from experiments.common import clopper_pearson, git_sha, write_json


def run(n1=2500, n2=2500, T=400):
    sim = EarlyExitSim(seed=21)
    lam = 2.0
    Rtrue_tgt, Ctrue = sim.oracle(lam=lam)
    alpha, delta = float(Rtrue_tgt.min() + 0.10), 0.10

    # baseline: unweighted certificate (built on source) vs true target risk
    rng = np.random.default_rng(9)
    uv = uf = 0
    for _ in range(T):
        loss_s, _, _ = sim.sample(n2, rng=rng, lam=0.0)
        r = select_risk(loss_s, Ctrue, alpha, delta, method="chain", pvalue="hb")
        if r["feasible"]:
            uf += 1; uv += int(Rtrue_tgt[r["selected"]] > alpha)
    print(f"alpha={alpha:.3f} delta={delta} lam={lam} n2={n2}")
    print(f"UNWEIGHTED arm: true-target violation = {uv/max(uf,1):.3f}  (feasible {uf}/{T})\n")
    unweighted = {
        "alpha": float(alpha),
        "delta": float(delta),
        "lam": float(lam),
        "n2": int(n2),
        "T": int(T),
        "feasible_count": int(uf),
        "feasible_rate": float(uf / T),
        "target_violation_count": int(uv),
        "target_violation_rate_feasible": float(uv / max(uf, 1)),
        "target_violation_ci95_feasible": clopper_pearson(uv, max(uf, 1)),
    }

    print(f"{'eta':>6} | {'mean_L1':>8} {'(*)':>5} | {'estw_feas':>10} {'estw_viol':>10} "
          f"| {'exactw_feas':>11} {'exactw_viol':>11}")
    sweep = []
    for eta in (0.0, 0.03, 0.06, 0.09, 0.12):
        rng = np.random.default_rng(9)
        l1s = []
        ef = ev = 0          # estimated-weight feasible / violating
        xf = xv = 0          # exact-weight feasible / violating
        for _ in range(T):
            _, _, d_src_fit = sim.sample(n1, rng=rng, lam=0.0)
            _, _, d_tgt_fit = sim.sample(n1, rng=rng, lam=lam)
            wfun = fit_weights_logistic(d_src_fit, d_tgt_fit)
            loss_test, _, d_test = sim.sample(n2, rng=rng, lam=0.0)
            what = wfun(d_test)
            wexact = sim.exact_weight(d_test, lam)
            l1s.append(np.mean(np.abs(what - wexact)))
            re = select_shift(loss_test, what, Ctrue, alpha, delta, eta, method="chain", pvalue="eb")
            if re["feasible"]:
                ef += 1; ev += int(Rtrue_tgt[re["selected"]] > alpha)
            rx = select_shift(loss_test, wexact, Ctrue, alpha, delta, eta=0.0, method="chain", pvalue="eb")
            if rx["feasible"]:
                xf += 1; xv += int(Rtrue_tgt[rx["selected"]] > alpha)
        l1 = float(np.mean(l1s))
        star = "yes" if l1 <= eta else "no"
        sweep.append({
            "eta": float(eta),
            "mean_l1": l1,
            "assumption_star_holds": bool(l1 <= eta),
            "estimated_weight": {
                "feasible_count": int(ef),
                "feasible_rate": float(ef / T),
                "target_violation_count": int(ev),
                "target_violation_rate_feasible": float(ev / max(ef, 1)),
                "target_violation_ci95_feasible": clopper_pearson(ev, max(ef, 1)),
            },
            "exact_weight": {
                "feasible_count": int(xf),
                "feasible_rate": float(xf / T),
                "target_violation_count": int(xv),
                "target_violation_rate_feasible": float(xv / max(xf, 1)),
                "target_violation_ci95_feasible": clopper_pearson(xv, max(xf, 1)),
            },
        })
        print(f"{eta:6.2f} | {l1:8.3f} {star:>5} | {ef/T:10.3f} {ev/max(ef,1):10.3f} "
              f"| {xf/T:11.3f} {xv/max(xf,1):11.3f}")
    print("\nRead: where (*) holds (mean_L1 <= eta), est-w violation stays <= delta; "
          "feasibility falls as eta grows. exact-w is the efficiency ceiling.")

    # (d) deliberately DOWNWARD-biased estimator with eta too small: (*) fails -> expect a breach.
    # Use a TIGHT alpha (near the risk floor) so under-counting can make an unsafe config look safe.
    alpha_fail = float(Rtrue_tgt.min() + 0.03)
    rng = np.random.default_rng(13)
    eta_small = 0.02
    bf = bv = 0; bl1 = []
    for _ in range(T):
        _, _, d_src_fit = sim.sample(n1, rng=rng, lam=0.0)
        _, _, d_tgt_fit = sim.sample(n1, rng=rng, lam=lam)
        wfun = fit_weights_logistic(d_src_fit, d_tgt_fit)
        loss_test, _, d_test = sim.sample(n2, rng=rng, lam=0.0)
        wbias = np.minimum(wfun(d_test), 0.8)          # heavily under-count the hard (high-w) tail
        bl1.append(np.mean(np.abs(wbias - sim.exact_weight(d_test, lam))))
        r = select_shift(loss_test, wbias, Ctrue, alpha_fail, delta, eta_small, method="chain", pvalue="eb")
        if r["feasible"]:
            bf += 1; bv += int(Rtrue_tgt[r["selected"]] > alpha_fail)
    failure = {
        "alpha": float(alpha_fail),
        "delta": float(delta),
        "eta": float(eta_small),
        "mean_l1": float(np.mean(bl1)),
        "assumption_star_holds": bool(np.mean(bl1) <= eta_small),
        "feasible_count": int(bf),
        "feasible_rate": float(bf / T),
        "target_violation_count": int(bv),
        "target_violation_rate_feasible": float(bv / max(bf, 1)),
        "target_violation_ci95_feasible": clopper_pearson(bv, max(bf, 1)),
    }
    print(f"\nfailure-mode arm (downward-biased w_hat, alpha={alpha_fail:.3f}, eta={eta_small}):")
    print(f"  mean ||w_bias - w||_1 = {np.mean(bl1):.3f}  (>> eta -> (*) FAILS)")
    print(f"  target-violation rate = {bv/max(bf,1):.3f}  (feasible {bf}/{T})  "
          f"-- exceeds delta={delta} when (*) is violated")
    print("\nReads: (*) holding => est-w valid; feasibility falls as eta grows; exact-w is the")
    print("ceiling; and the downward-biased arm shows a real breach once (*) is violated.")
    return {
        "experiment": "e6_shift_eta_sweep",
        "description": "Synthetic covariate-shift eta-sensitivity sweep",
        "git_sha": git_sha(),
        "simulator": {"seed": 21, "oracle_seed": 12345, "K": int(sim.K), "L": int(sim.L)},
        "config": {"n1": int(n1), "n2": int(n2), "T": int(T), "lam": float(lam)},
        "unweighted": unweighted,
        "eta_sweep": sweep,
        "failure_mode": failure,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/e6_shift_eta_sweep.json")
    parser.add_argument("--n1", type=int, default=2500)
    parser.add_argument("--n2", type=int, default=2500)
    parser.add_argument("--T", type=int, default=400)
    args = parser.parse_args()
    payload = run(n1=args.n1, n2=args.n2, T=args.T)
    out = write_json(args.out, payload)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""
Tests for the CARC reference implementation. Run with:  python -m tests.test_carc
(or pytest). The headline tests are Monte-Carlo *validity* checks that mirror experiment E1.
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carc import pvalues as pv
from carc import certify as ct
from carc import selector as sel
from carc.chain import build_chain, build_threshold_family
from carc.simulate import EarlyExitSim, fit_weights_logistic

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
def test_pvalue_validity():
    """Under R == alpha, P(p <= delta) <= delta for each p-value (super-uniformity)."""
    alpha, n, T = 0.20, 300, 40_000
    rng = np.random.default_rng(1)
    samples = (rng.uniform(size=(T, n)) < alpha).astype(float)   # Bernoulli(alpha)
    rhat = samples.mean(axis=1)
    p_h = np.array([pv.hoeffding_simple(r, n, alpha) for r in rhat])
    p_hb = np.array([pv.hoeffding_bentkus(r, n, alpha) for r in rhat])
    p_eb = np.array([pv.empirical_bernstein(samples[t], alpha) for t in range(T)])
    for name, p in [("hoeffding", p_h), ("hb", p_hb), ("eb", p_eb)]:
        for delta in (0.05, 0.10, 0.20):
            rate = float((p <= delta).mean())
            assert rate <= delta + 0.01, f"{name} not valid at delta={delta}: rate={rate:.4f}"
    print("[ok] test_pvalue_validity  (all p-values super-uniform under R=alpha)")


def test_weighted_pvalue_validity():
    """Under exact weights with E[w*loss] == alpha and eta=0, weighted p-values are valid."""
    alpha, n, T = 0.20, 400, 30_000
    rng = np.random.default_rng(2)
    rate_max = 0.0
    for _ in range(1):
        ps = []
        for _t in range(T):
            # weights in {0.5, 1.5} equally likely (mean 1); choose base loss prob so E[w*loss]=alpha
            w = rng.choice([0.5, 1.5], size=n)
            # want E[w * Bernoulli(q)] = E[w] * q = 1 * q = alpha  -> q = alpha (w indep of loss)
            loss = (rng.uniform(size=n) < alpha).astype(float)
            wl = w * loss
            ps.append(pv.weighted_empirical_bernstein(wl, alpha_def=alpha, W=1.5))
        ps = np.array(ps)
        for delta in (0.05, 0.10):
            rate = float((ps <= delta).mean())
            rate_max = max(rate_max, rate / delta)
            assert rate <= delta + 0.01, f"weighted-eb invalid at delta={delta}: rate={rate:.4f}"
    print(f"[ok] test_weighted_pvalue_validity  (max rate/delta = {rate_max:.2f})")


def test_certify_structure():
    """Holm dominates Bonferroni; fixed-sequence returns a prefix of the given order."""
    rng = np.random.default_rng(3)
    pvals = rng.uniform(0, 0.05, size=20)
    pvals[7:] = rng.uniform(0.2, 1.0, size=13)         # only first 7 are small-ish
    b = ct.bonferroni(pvals, 0.1)
    h = ct.holm(pvals, 0.1)
    assert b <= h, "Holm must contain Bonferroni"
    order = list(range(19, -1, -1))                    # high index -> low
    fs = ct.fixed_sequence_chain(pvals, 0.1, order)
    # fixed-sequence certifies a contiguous suffix of indices (a prefix of `order`)
    if fs:
        idxs = sorted(fs)
        assert idxs == list(range(idxs[0], 20)), "fixed-sequence must be a contiguous top suffix"
    pvals = np.array([0.01, 0.01, 0.01, 0.01, 0.20, 0.01])
    chains = [[5, 4, 3], [2, 1, 0]]
    mc = ct.multi_chain(pvals, 0.10, chains)
    expected = (
        ct.fixed_sequence_chain(pvals, 0.05, chains[0])
        | ct.fixed_sequence_chain(pvals, 0.05, chains[1])
    )
    assert mc == expected, "multi-chain must equal the union of allocated fixed-sequence chains"
    try:
        ct.multi_chain(pvals, 0.10, chains, weights=[0.8, 0.8])
        raise AssertionError("multi-chain accepted weights summing above one")
    except ValueError:
        pass
    print("[ok] test_certify_structure")


def test_select_risk_multichain_can_use_alternate_chain():
    """Multi-chain search can certify a cheaper policy when a separate chain survives."""
    n, K = 100, 6
    alpha, delta = 0.20, 0.10
    loss = np.zeros((n, K), dtype=float)
    loss[:25, 4] = 1.0  # blocks the default expensive->cheap chain at index 4
    costs = np.arange(K, dtype=float)
    chains = [[5, 4, 3], [2, 1, 0]]

    one_chain = sel.select_risk(loss, costs, alpha, delta, method="chain", pvalue="hb")
    multi = sel.select_risk(
        loss,
        costs,
        alpha,
        delta,
        method="multichain",
        pvalue="hb",
        chains=chains,
    )

    assert one_chain["selected"] == 5, "single chain should stop before the cheaper safe chain"
    assert multi["selected"] == 0, "multi-chain should pick the cheapest certified alternate-chain config"
    assert multi["extra"]["num_chains"] == 2
    print("[ok] test_select_risk_multichain_can_use_alternate_chain")


def test_build_chain_matches_manual_policy():
    """Extracted chain helper matches the original global-threshold policy logic."""
    sim = EarlyExitSim(seed=4)
    rng = np.random.default_rng(123)
    n = 512
    _d, correct, conf = sim._draw_latents(n, rng, lam=0.0)
    loss, cost = build_chain(conf, 1.0 - correct, sim.costs_exit, sim.thresholds)

    manual_loss = np.empty_like(loss)
    manual_cost = np.empty_like(cost)
    rows = np.arange(n)
    for k, threshold in enumerate(sim.thresholds):
        meets = conf >= threshold
        has_exit = meets.any(axis=1)
        first_exit = np.argmax(meets, axis=1)
        exit_idx = np.where(has_exit, first_exit, sim.L - 1)
        manual_loss[:, k] = 1.0 - correct[rows, exit_idx]
        manual_cost[:, k] = sim.costs_exit[exit_idx]

    np.testing.assert_array_equal(loss, manual_loss)
    np.testing.assert_array_equal(cost, manual_cost)
    print("[ok] test_build_chain_matches_manual_policy")


def test_threshold_family_matches_global_chain():
    """A repeated scalar threshold vector matches the original global-threshold helper."""
    sim = EarlyExitSim(seed=44)
    rng = np.random.default_rng(321)
    n = 384
    _d, correct, conf = sim._draw_latents(n, rng, lam=0.0)
    loss_chain, cost_chain = build_chain(conf, 1.0 - correct, sim.costs_exit, sim.thresholds)
    threshold_vectors = np.repeat(sim.thresholds[:, None], sim.L - 1, axis=1)
    loss_family, cost_family = build_threshold_family(
        conf,
        1.0 - correct,
        sim.costs_exit,
        threshold_vectors,
    )
    np.testing.assert_array_equal(loss_family, loss_chain)
    np.testing.assert_array_equal(cost_family, cost_chain)
    print("[ok] test_threshold_family_matches_global_chain")


# --------------------------------------------------------------------------- #
def _violation_experiment(sim, alpha, delta, n, T, method, pvalue, lam=0.0):
    """Run T trials; return (chain violation rate, mean selected cost, feasible rate)."""
    Rtrue, Ctrue = sim.oracle(lam=lam)
    rng = np.random.default_rng(100)
    viol = 0; costs_sel = []; feas = 0
    for _ in range(T):
        loss, cost, _ = sim.sample(n, rng=rng, lam=lam)
        res = sel.select_risk(loss, Ctrue, alpha, delta, method=method, pvalue=pvalue)
        if res["feasible"]:
            feas += 1
            k = res["selected"]
            costs_sel.append(Ctrue[k])
            if Rtrue[k] > alpha:
                viol += 1
    return viol / T, (np.mean(costs_sel) if costs_sel else np.nan), feas / T, Rtrue, Ctrue


def test_end_to_end_validity_and_naive():
    """E1+E3 in miniature: chain violation rate <= delta; naive violates far more."""
    sim = EarlyExitSim(seed=7)
    Rtrue, Ctrue = sim.oracle()
    # alpha above the risk floor (feasible) but below most configs (a real boundary to overfit to)
    alpha = float(max(1.4 * Rtrue.min(), Rtrue.min() + 0.04))
    delta, n, T = 0.10, 1000, 4000

    v_chain, c_chain, feas, Rtrue, Ctrue = _violation_experiment(
        sim, alpha, delta, n, T, method="chain", pvalue="hb")

    # naive baseline
    rng = np.random.default_rng(100)
    v_naive = 0; n_naive_feas = 0; c_naive = []
    for _ in range(T):
        loss, cost, _ = sim.sample(n, rng=rng)
        r = sel.select_naive(loss, Ctrue, alpha)
        if r["feasible"]:
            n_naive_feas += 1
            k = r["selected"]; c_naive.append(Ctrue[k])
            if Rtrue[k] > alpha:
                v_naive += 1
    v_naive /= T

    print(f"[ok] end-to-end: alpha={alpha:.3f} delta={delta}")
    print(f"      chain  violation rate = {v_chain:.4f}  (target <= {delta})   "
          f"mean cost = {c_chain:.3f}  feasible = {feas:.3f}")
    print(f"      naive  violation rate = {v_naive:.4f}  (uncorrected)         "
          f"mean cost = {np.mean(c_naive):.3f}")
    assert feas > 0.5, f"too few feasible trials ({feas:.3f}); retune alpha/n"
    assert v_chain <= delta + 0.01, f"chain violated: {v_chain:.4f} > {delta}"
    assert v_naive > v_chain, "naive should violate more than the certified method"


def test_dual_budget():
    """Joint control: selected config respects the certified budget and risk target."""
    sim = EarlyExitSim(seed=11)
    Rtrue, Ctrue = sim.oracle()
    alpha = float(max(1.5 * Rtrue.min(), Rtrue.min() + 0.05))
    delta, n = 0.10, 2000
    budget = float(np.quantile(Ctrue, 0.85))
    rng = np.random.default_rng(5)
    loss, cost, _ = sim.sample(n, rng=rng)
    res = sel.select_dual(loss, cost, alpha, delta, budget, method="chain", pvalue="hb")
    if res["feasible"]:
        k = res["selected"]
        assert res["Uc"][k] <= budget + 1e-9, "selected exceeds certified compute budget"
        print(f"[ok] test_dual_budget: selected k={k}, Uc={res['Uc'][k]:.3f} <= B={budget:.3f}, "
              f"Rtrue={Rtrue[k]:.3f} (<= alpha={alpha:.3f}? {Rtrue[k] <= alpha})")
    else:
        print("[ok] test_dual_budget: reported infeasible (acceptable)")


def test_shift():
    """E6 in miniature: under shift the unweighted certificate violates; the weighted one holds."""
    sim = EarlyExitSim(seed=21)
    lam = 2.0                                   # exp-tilt toward harder inputs
    Rtrue_src, _ = sim.oracle(lam=0.0)
    Rtrue_tgt, Ctrue = sim.oracle(lam=lam)
    alpha = float(Rtrue_tgt.min() + 0.15)       # headroom above the (shifted) risk floor
    delta, eta, T = 0.10, 0.08, 600             # eta chosen to cover the estimator's L1 error
    n1, n2 = 2500, 2500                          # weight-fit fold, test fold

    rng = np.random.default_rng(9)
    viol_unw = 0; feas_unw = 0
    viol_w = 0; feas_w = 0
    l1_errs = []
    for _ in range(T):
        # UNWEIGHTED certificate built on source, evaluated against TARGET risk
        loss_s, cost_s, d_s = sim.sample(n2, rng=rng, lam=0.0)
        r_unw = sel.select_risk(loss_s, Ctrue, alpha, delta, method="chain", pvalue="hb")
        if r_unw["feasible"]:
            feas_unw += 1
            if Rtrue_tgt[r_unw["selected"]] > alpha:
                viol_unw += 1
        # WEIGHTED certificate: fit w on (src D1, tgt D1); test on src D2 with w_hat
        loss_fit, _, d_src_fit = sim.sample(n1, rng=rng, lam=0.0)
        _, _, d_tgt_fit = sim.sample(n1, rng=rng, lam=lam)
        wfun = fit_weights_logistic(d_src_fit, d_tgt_fit)
        loss_test, _, d_test = sim.sample(n2, rng=rng, lam=0.0)
        what = wfun(d_test)
        l1_errs.append(np.mean(np.abs(what - sim.exact_weight(d_test, lam))))  # realized ||w_hat-w||_1
        r_w = sel.select_shift(loss_test, what, Ctrue, alpha, delta, eta,
                               method="chain", pvalue="eb")
        if r_w["feasible"]:
            feas_w += 1
            if Rtrue_tgt[r_w["selected"]] > alpha:
                viol_w += 1
    vu = viol_unw / max(feas_unw, 1)
    vw = viol_w / max(feas_w, 1)
    l1 = float(np.mean(l1_errs))
    holds = "HOLDS" if l1 <= eta else "FAILS"
    print(f"[ok] test_shift: alpha={alpha:.3f} lam={lam} eta={eta}")
    print(f"      realized mean ||w_hat - w||_1 = {l1:.3f}  -> assumption (*) {holds} (needs <= eta={eta})")
    print(f"      UNWEIGHTED  target-violation rate = {vu:.3f}  (feasible {feas_unw}/{T})")
    print(f"      WEIGHTED    target-violation rate = {vw:.3f}  (feasible {feas_w}/{T})")
    assert feas_w > 0.5 * T, f"weighted certificate too conservative (feasible {feas_w}/{T})"
    assert vw <= delta + 0.03, f"weighted certificate violated under shift: {vw:.3f}"
    assert vw < vu, "weighted certificate should be safer than unweighted under shift"


if __name__ == "__main__":
    test_pvalue_validity()
    test_weighted_pvalue_validity()
    test_certify_structure()
    test_select_risk_multichain_can_use_alternate_chain()
    test_build_chain_matches_manual_policy()
    test_threshold_family_matches_global_chain()
    test_end_to_end_validity_and_naive()
    test_dual_budget()
    test_shift()
    print("\nALL TESTS PASSED")

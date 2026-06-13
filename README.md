# CARC — reference implementation

Reference code for *Compute-Adaptive Risk Control: Distribution-Free Joint Guarantees for
Early-Exit Networks*. It implements the certificates in the draft and reproduces the core
experiment-plan results on a synthetic early-exit simulator (no GPU / no datasets needed).

## Install & run

```bash
pip install numpy scipy            # numpy>=2, scipy>=1.13
cd carc_ref
python -m tests.test_carc                       # unit tests (validity + behavior)
python -m experiments.e1_validity               # E1 violation-frequency / E3 naive baseline
python -m experiments.e2_chain_vs_bonferroni    # E2 cost of the guarantee / E7 ablation
python -m experiments.e6_shift_eta_sweep        # E6 covariate-shift eta-sensitivity
```

## What maps to what

| Draft | Code |
|---|---|
| Hoeffding p-value Eq. (2) | `pvalues.hoeffding_simple` |
| Hoeffding–Bentkus (LTT default) | `pvalues.hoeffding_bentkus` |
| empirical-Bernstein (§6.1 note, experiments) | `pvalues.empirical_bernstein` |
| Bonferroni LTT (Thm 1) | `certify.bonferroni` |
| fixed-sequence chain (Thm 2) | `certify.fixed_sequence_chain` |
| Holm / graphical fallback (§4.2 remark) | `certify.holm` |
| risk certificate, P1 (§3–§4) | `selector.select_risk` |
| joint risk–compute, P2 (Thm 3) | `selector.select_dual` |
| shift-robust, estimated weights (Thm 4) | `selector.select_shift` |
| naive uncorrected recipe (§1, E3) | `selector.select_naive` |
| synthetic early-exit model + oracle risk/cost | `simulate.EarlyExitSim` |
| logistic density-ratio weights (E6) | `simulate.fit_weights_logistic` |

Configurations are indexed `0..K-1` by **increasing compute** (index 0 = cheapest). The chain
test walks expensive→cheap and certifies a contiguous cheap-ward suffix; the cheapest certified
config is returned.

## Representative output (seeded)

```
end-to-end : alpha=0.079 delta=0.10
  chain  violation = 0.0065  (<= delta)  feasible 0.906  mean cost 3.49
  naive  violation = 0.2333  (uncorrected)             mean cost 3.38
E2         : chain(hb) cost 3.26 < holm 3.31 < bonferroni 3.33 ; all true_viol <= delta
shift      : unweighted target-violation 0.58 ; weighted 0.000 at 99.5% feasibility,
             with realized ||w_hat - w||_1 = 0.060 <= eta=0.08  [assumption (*) holds]
shift (d)  : downward-biased w_hat, eta=0.02 too small (*) FAILS  ->  violation 0.90 >> delta
```

## Honest caveats (what the demo does and does not show)

- **Validity is demonstrated; tightness is not.** The Hoeffding/HB/EB p-values are conservative,
  so observed violation rates sit well below `delta`. That is correct behavior, not slack to be
  "fixed."
- **Cost-optimality is within the chain.** The scalar-threshold chain is a 1-D slice of the
  configuration space; "cheapest certified" means cheapest *along that chain* (draft §4.1).
- **Shift validity is conditional on (⋆).** Theorem 4 needs `E|w_hat - w| <= eta`. The unit test
  picks `eta` to cover the realized estimator error so (⋆) holds, and the certificate is valid.
  `e6_shift_eta_sweep` shows the full picture: (a) where (⋆) holds, the estimated-weight
  certificate is valid; (b) feasibility collapses as `eta` grows (the price of robustness);
  (c) exact weights are the efficiency ceiling; and (d) a **deliberately downward-biased**
  estimator with `eta` set too small (so (⋆) fails) **breaches** the target at ~0.90 >> delta —
  the honest failure mode.
- **Synthetic only.** `EarlyExitSim` is a controllable stand-in for a real multi-exit network so
  that ground-truth risk/cost are available for violation checking; it is not a model of any
  specific architecture.
```
carc_ref/
  carc/      pvalues.py  certify.py  selector.py  simulate.py
  tests/     test_carc.py
  experiments/  e1_validity.py  e2_chain_vs_bonferroni.py  e6_shift_eta_sweep.py
```

# CARC — reference implementation

Reference code for *Compute-Adaptive Risk Control: Distribution-Free Joint Guarantees for
Early-Exit Networks*. It implements the certificates in the draft, reproduces the synthetic
early-exit experiments (no GPU / no datasets needed), and provides the CIFAR-100 branchy-cache
diagnostic reported in the manuscript.

## Install & run

```bash
pip install numpy scipy            # numpy>=2, scipy>=1.13
python -m tests.test_carc                       # unit tests (validity + behavior)
python -m experiments.e1_validity               # E1 violation-frequency / E3 naive baseline
python -m experiments.e2_chain_vs_bonferroni    # E2 cost of the guarantee / E7 ablation
python -m experiments.e5_joint_control          # E5 joint risk-compute certificate
python -m experiments.e6_shift_eta_sweep        # E6 covariate-shift eta-sensitivity
```

## Real-data cache path

The CIFAR-100 branchy-cache diagnostic reported in the manuscript was generated with:

```bash
python -m adapters.cifar_branchy \
  --dataset cifar100 --download --epochs 80 \
  --checkpoint checkpoints/cifar_branchy_resnet56.pt \
  --out cache/cifar_branchy_resnet56.npz

python -m experiments.real_cache_eval \
  --cache cache/cifar_branchy_resnet56.npz \
  --out results/cifar_branchy_real_cache_eval.json \
  --alphas 0.05,0.10,0.20,0.30 --deltas 0.05,0.10 \
  --calib-sizes 500,2000 --T 1000

python -m experiments.real_cache_eval \
  --cache cache/cifar_branchy_resnet56.npz \
  --out results/cifar_branchy_real_cache_eval_exploratory.json \
  --alphas 0.35,0.40 --deltas 0.05,0.10 \
  --calib-sizes 500,2000 --T 1000
```

For a non-scientific smoke check that exercises the adapter and evaluator without downloading
CIFAR-100:

```bash
python -m adapters.cifar_branchy \
  --dataset fake --epochs 0 --allow-random-init \
  --max-train-samples 64 --max-cache-samples 64 \
  --batch-size 16 --threshold-count 8 --device cpu --num-workers 0 \
  --out cache/smoke_cifar_branchy.npz

python -m experiments.real_cache_eval \
  --cache cache/smoke_cifar_branchy.npz \
  --out results/smoke_real_cache_eval.json \
  --alphas 0.9 --deltas 0.1 --calib-sizes 16 --T 3
```

## What maps to what

| Draft | Code |
|---|---|
| Hoeffding p-value Eq. (2) | `pvalues.hoeffding_simple` |
| Hoeffding–Bentkus (LTT default) | `pvalues.hoeffding_bentkus` |
| empirical-Bernstein (§6.1 note, experiments) | `pvalues.empirical_bernstein` |
| Bonferroni LTT (Thm 1) | `certify.bonferroni` |
| fixed-sequence chain (Thm 2) | `certify.fixed_sequence_chain` |
| Holm fallback (§4.2 remark) | `certify.holm` |
| per-exit cache -> chain matrices | `chain.build_chain` |
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
E1/E3      : alpha=0.079 delta=0.10
  chain  violation = 0.0017  (<= delta)  feasible 0.911  mean cost 3.49
  naive  violation = 0.2313  (uncorrected)             mean cost 3.38
E2         : chain(hb) cost 3.260 < holm 3.311 < bonferroni 3.326 ; all true_viol <= delta
E5         : slack budget feasible 1.000 with 0 violations; tight oracle-feasible budget infeasible
E6         : unweighted target-violation 0.875 ; mean ||w_hat-w||_1 = 0.060
             eta=0.09 covers the mean error but feasibility collapses to 0.000
shift (d)  : downward-biased w_hat, eta=0.02 too small (*) FAILS  ->  violation 0.895 >> delta
CIFAR      : pre-registered alpha<=0.30 grid has no safe full-pool config (min risk 0.3055);
             exploratory alpha=0.35/0.40 chain violation 0.012-0.038 <= delta,
             naive violation 0.441-0.529
```

## Honest caveats (what the demo does and does not show)

- **Validity is demonstrated; tightness is not.** The Hoeffding/HB/EB p-values are conservative,
  so observed violation rates sit well below `delta`. That is correct behavior, not slack to be
  "fixed."
- **Cost-optimality is within the chain.** The scalar-threshold chain is a 1-D slice of the
  configuration space; "cheapest certified" means cheapest *along that chain* (draft §4.1).
- **Shift validity is conditional on (⋆).** Theorem 4 needs `E|w_hat - w| <= eta`.
  `e6_shift_eta_sweep` shows the full picture: (a) underspecified eta values may look good but
  lack the assumption needed for validity; (b) feasibility can collapse once eta honestly covers
  the estimator error; (c) exact weights are the efficiency ceiling; and (d) a **deliberately
  downward-biased** estimator with `eta` set too small (so (⋆) fails) **breaches** the target at
  ~0.895 >> delta — the honest failure mode.
- **Synthetic plus one CIFAR diagnostic.** `EarlyExitSim` is a controllable stand-in where
  ground-truth risk/cost are available for violation checking. The CIFAR-100 branchy run uses
  held-out test-pool risk as a finite-pool proxy, not population truth, and its pre-registered
  target grid is mostly infeasible because the final-exit risk is 0.3055.
```
adapters/     cifar_branchy.py
carc/         pvalues.py  certify.py  selector.py  simulate.py  chain.py
tests/        test_carc.py
experiments/  e1_validity.py  e2_chain_vs_bonferroni.py  e5_joint_control.py  e6_shift_eta_sweep.py
              real_cache_eval.py
```

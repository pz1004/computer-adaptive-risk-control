# CARC Real-Data Pre-Registration

Status: criteria frozen before the first real-data E1 run and executed in
`results/cifar_branchy_real_cache_eval.json`. Do not edit the criteria below to match observed
violations.

## E1 Pre-Registration: CIFAR-100 Branchy ResNet-56

- alpha grid: `{0.05, 0.10, 0.20, 0.30}`
- delta grid: `{0.05, 0.10}`
- splits T: `1000` fresh calibration/test partitions per cell
- calibration sizes: `{500, 2000}`
- p-value: Hoeffding-Bentkus (HB)
- method: fixed-sequence chain, tested expensive-to-cheap over configurations indexed by
  increasing compute
- pass criterion: empirical violation frequency must be at or below `delta` within a binomial
  confidence interval in every cell

## Integrity Rules

- Fix `(alpha, delta)`, chain, split count, calibration sizes, and pass criterion before
  inspecting calibration risks.
- Do not tune on test splits.
- Use fresh calibration/test splits per trial.
- If a cell violates the pass criterion after bug checks, report the violation rather than
  deleting or replacing the cell.

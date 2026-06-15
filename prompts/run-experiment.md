# Run Experiment Prompt

Run experiment `{experiment_name}` end to end.

Requirements:
- Load the relevant cache from `cache/` when the experiment uses real-model data.
- Use the pre-registered `(alpha, delta, T)` grid from `preregistration.md`.
- Call the matching `experiments/` script or create a narrowly scoped one if it does not exist.
- Save results to `results/` as JSON, including the full config and git SHA.
- Print headline numbers: violation frequency, binomial CI, feasibility rate, selected compute,
  and any comparison baseline.
- Do not change pass criteria after seeing results.
- Report violations and infeasible cells honestly.

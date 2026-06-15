# CARC — project memory for Codex

## What this repo is

Reference code and manuscript material for a TMLR paper on distribution-free risk control
for early-exit networks. The current manuscript source is `draft.md`. The current reference
implementation is the root `carc/` package, with synthetic experiments under `experiments/`.

## Hard rules

- Do not edit `carc/` to make a model fit. Models must be adapted to the selector interface,
  not the reverse. Planned core refactors are allowed only when they are test-guarded and do
  not change existing results.
- Configurations are indexed `0..K-1` by increasing compute. Index `0` is cheapest.
- Adapters must output `loss_matrix` with shape `(n, K)` and values in `[0, 1]`, plus
  `cost_matrix` with shape `(n, K)` and nonnegative costs.
- Every experiment must take an explicit seed and write the config, result, and git SHA to
  disk.
- Do not tune on test splits. Use fresh calibration/test splits per trial.
- Treat `preregistration.md` pass criteria as frozen after the first real-data E1 run.
- Do not turn planned real-data protocols into completed results unless result artifacts exist.
- Do not invent citations, BibTeX entries, experimental numbers, or dataset details.

## Codex workflow

- Before multi-step work, create or update `task_plan.md` and keep `notes.md` as the evidence log.
- Before editing the manuscript, identify whether each claim is backed by proof, local results,
  verified venue guidance, or a planned protocol.
- For TMLR, preserve double-blind anonymization in manuscript text, supplements, code links,
  and reproducibility statements.
- Use `prompts/run-experiment.md` and `prompts/paper-sync.md` as reusable task prompts; they are
  not executable shell commands.

## Useful commands

- Run tests: `python -m pytest tests/`
- Run tests without pytest: `python -m tests.test_carc`
- Reproduce E1/E3 synthetic results: `python -m experiments.e1_validity`
- Reproduce E2 synthetic comparison: `python -m experiments.e2_chain_vs_bonferroni`
- Reproduce E6 synthetic shift sweep: `python -m experiments.e6_shift_eta_sweep`
- Build a CIFAR branchy cache: `python -m adapters.cifar_branchy --dataset cifar100 --download --epochs 80 --checkpoint checkpoints/cifar_branchy_resnet56.pt --out cache/cifar_branchy_resnet56.npz`
- Evaluate a real cache: `python -m experiments.real_cache_eval --cache cache/cifar_branchy_resnet56.npz --out results/cifar_branchy_real_cache_eval.json`

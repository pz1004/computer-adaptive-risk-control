# Completing the CARC Paper with Codex — a step-by-step guide

This guide takes you from the current artifacts (`main.tex`, `draft.md`, and the root `carc/`
reference implementation) to a submittable IEEE Access Research Article, using Codex in this repository.
It is written for *this* project: every phase ends with an **acceptance gate** tied to a
specific claim in the draft, and the work is sequenced so the single most important
experiment (CIFAR validity) lands first.

This Codex version assumes work happens from the repository root. Use `AGENTS.md` for
project memory, persistent planning files (`task_plan.md`, `notes.md`) for multi-step
work, and reusable prompt files in `prompts/` instead of tool-specific slash commands.

---

## 0. Orientation — what is done and what gates acceptance

**Done and frozen (do not re-derive):**
- Theory and proofs (draft §§3–6, Appendix A, A.4) — complete and self-consistent.
- Reference implementation `carc/` — p-values, certification, selectors, synthetic
  simulator; all unit tests pass; controlled-simulation results are in draft §10.1.

**Original evidence gap:** the draft §10.2 real-data study was planned but not run. The first
Codex completion pass closed the minimum CIFAR gap by adding a real branchy-CNN cache diagnostic
(`results/cifar_branchy_real_cache_eval*.json`). The IEEE Access retargeting pass added a
preregistered digits tabular-cascade confirmation
(`results/digits_tabular_real_cache_eval.json`). Remaining ImageNet, language, and real-shift
studies still strengthen the claim-evidence match. Everything below exists to keep that work
sequenced **without disturbing the proven core**.

**The governing design principle for all new code:**

> Never modify `carc/` to fit a model. The certified selectors already consume a loss matrix
> and a cost matrix. The entire real-data effort is writing *adapters* that turn a trained
> model into those two matrices. The proof-bearing code stays byte-for-byte identical between
> simulation and real data — which is itself a reproducibility argument you can state in the paper.

**Two scientific-integrity rules to honor from the start** (they protect the result, and they
are cheap to follow if you start now):
1. **Pre-register before you run.** Write the target grid `(α, δ)`, split count `T`, and the
   pass criterion into a file *before* executing E1. Do not adjust them after seeing violations.
2. **Never tune on the test split, and never reuse one calibration draw across an α-sweep and
   then keep the prettiest cell.** The guarantee is per single use (draft §11). Fresh splits per
   trial; fixed `(α, δ)` and chain chosen before looking at calibration risk.

If results come back modest (small chain-vs-Bonferroni gap; conservative shift certificate),
**do not chase numbers** — the manuscript is already written to accept that outcome. Reducing a
claim is a valid IEEE Access remedy; fabricating a stronger one is not.

---

## 1. Phase 0 — Repository and Codex setup

**Objective:** a clean repo, an `AGENTS.md` that encodes the rules above, and a few reusable
Codex prompts so you are not re-typing context.

### 1.1 Lay out the repo

```
draft.md                   # current manuscript source
carc/                      # frozen proof-bearing core
  pvalues.py certify.py selector.py simulate.py chain.py   # see 2.1 for chain.py
adapters/                  # NEW: model -> per-exit cache  (the only model-specific code)
  cifar_branchy.py imagenet_msdnet.py llm_cascade.py tabular_mlp.py
cache/                     # NEW: cached per-exit tensors (.npz), git-ignored
experiments/               # extend the existing scripts to consume real caches
  e1_validity.py ... e6_shift.py runner.py
paper/                     # the draft + LaTeX build
  CARC.md  CARC.tex  refs.bib  figures/
tests/                     # existing unit tests + new adapter tests
preregistration.md         # the frozen (α, δ, T, pass-criterion) grid  <-- write first
prompts/                   # reusable Codex task prompts
  run-experiment.md paper-sync.md
AGENTS.md                  # project memory for Codex
```

### 1.2 Create `AGENTS.md` (project memory)

Create this file at the repository root. It is the single most leveraged artifact in the
workflow because Codex reads repo instructions before acting.

```markdown
# CARC — project memory for Codex

## What this repo is
Reference code + experiments for an IEEE Access manuscript on distribution-free risk control for
early-exit networks. Theory is frozen; real-cache evidence now includes the digits confirmation
and CIFAR diagnostic in the manuscript experiments section.

## Hard rules
- NEVER edit files under carc/ to make a model fit. carc/ is the proven, unit-tested core.
  Models are adapted TO carc/, never the reverse. If carc/ seems wrong, stop and ask.
- Configs are indexed 0..K-1 by INCREASING compute (index 0 = cheapest). Preserve this everywhere.
- Adapters output two arrays only: loss_matrix (n,K) in [0,1], cost_matrix (n,K) >= 0.
  Everything downstream reuses carc.selector unchanged.
- Reproducibility: every experiment takes an explicit seed and writes results + config to disk.
  No tuning on test splits. Fresh calibration/test split per trial.
- The pass criteria in preregistration.md are frozen. Do not edit them to match results.

## Conventions
- numpy + scipy for carc/ (no heavy deps). PyTorch only inside adapters/.
- Run unit tests with `python -m pytest tests/` before committing changes that touch carc/.
- Validity is the headline; compute savings are modest and that is fine (see paper §4.2, §10.1).

## Useful commands
- Reproduce simulation results: `python -m experiments.e1_validity`
- Build a real cache: `python -m adapters.cifar_branchy --out cache/cifar_r56.npz`
```

### 1.3 Add reusable Codex prompts

Create `prompts/` files. To use one, paste or reference the prompt file in a Codex turn and
replace the argument placeholder explicitly.

`prompts/run-experiment.md`:
```markdown
Run experiment {experiment_name} end to end: load the relevant cache from cache/, call the
matching experiments/ script with the pre-registered (alpha, delta, T) grid from
preregistration.md, save results to results/ as JSON with the config and git SHA, and
print the headline numbers. Do NOT change pass criteria. Report violations honestly.
```

`prompts/paper-sync.md`:
```markdown
Read results/{result_file}.json and update the corresponding subsection of draft.md
with the actual numbers, replacing any placeholder. Flag in your reply every place where
the realized number is WORSE than the draft currently implies, so I can decide whether to
soften the claim. Never overwrite a claim to look better than the data.
```

**Acceptance gate 0:** `python -m pytest tests/` passes; `AGENTS.md` and the two prompts
exist; `preregistration.md` is stubbed (you will fill its grid in Phase 2).

---

## 2. Phase 1 — The adapter abstraction (the keystone)

**Objective:** factor the chain-building logic out of the simulator so real models and the
synthetic model share one code path, then define the adapter contract.

### 2.1 Refactor: extract `carc/chain.py`

Right now `simulate.EarlyExitSim.sample` does two things: (a) generate a synthetic model, and
(b) turn per-exit (score, loss, cost) into the threshold-chain loss/cost matrices. Step (b)
is exactly what real adapters need. Ask Codex to create a short persistent plan, then:

> Extract the policy-simulation logic from `simulate.EarlyExitSim.sample` into a new pure
> function `carc/chain.py::build_chain(scores, exit_loss, exit_costs, thresholds)` returning
> `(loss_matrix, cost_matrix)`, with configs ordered by increasing threshold = increasing
> compute. Refactor `EarlyExitSim.sample` to call it. Add a unit test asserting the refactor
> is behavior-preserving (same outputs on a fixed seed). Do not change any numbers.

This is a safe, test-guarded refactor. Review the diff before accepting it.

### 2.2 Define the adapter contract

Every adapter produces a **per-exit cache** (the only model-specific artifact):

```python
# adapters/_contract.py  (documentation, not enforced)
# An adapter writes an .npz with, for n held-out calibration points and L exits:
#   scores   : (n, L) float in [0,1]   confidence at each exit (e.g. top softmax prob)
#   correct  : (n, L) bool/float       per-exit correctness of the prediction
#   loss     : (n, L) float in [0,1]   per-exit loss (use 1-correct for 0/1; or selective risk, FNR)
#   exit_cost: (L,)   float            cumulative FLOPs (or chosen cost proxy) to reach exit l
#   meta     : dict                    model name, dataset, split sizes, seed, git SHA
# Then carc/chain.build_chain(scores, loss, exit_cost, thresholds) -> (loss_matrix, cost_matrix)
# feeds carc.selector.* UNCHANGED.
```

Note `loss` is carried per-exit (not just `1-correct`) so the same cache supports the three
loss functionals the paper promises (0–1 error, selective risk, class-conditional FNR).

**Acceptance gate 1:** `build_chain` unit test passes; feeding a *synthetic* cache through
`build_chain → selector` reproduces the §10.1 numbers bit-for-bit. This proves the real-data
path is the same code as the verified path.

---

## 3. Phase 2 — CIFAR branchy ResNet (the load-bearing experiment)

This is the minimum experiment that lets the paper pass the claim-evidence audit. Do it carefully and in
order; it carries E1 (validity), E2 (cost), E3 (naive baseline), and E4 (monotonicity diagnostic).

### 3.1 Train / obtain the model and build the cache

Ask Codex to write `adapters/cifar_branchy.py`:

> Implement a branchy ResNet-56 on CIFAR-100 with 3–4 intermediate exits (early-exit heads
> after selected residual stages). Train it (or load a checkpoint), then run the held-out set
> through all exits and write the per-exit cache per `adapters/_contract.py`: per-exit top
> softmax prob as `scores`, `correct`, `loss = 1-correct`, and `exit_cost` = cumulative FLOPs
> per exit measured with a profiler (e.g. fvcore/ptflops). Use a fixed seed; record it in meta.

Keep training modest — this paper is about the certificate, not SOTA accuracy. A standard
recipe is fine. Verify the cache: monotone `exit_cost`, `scores` in `[0,1]`, sane accuracy.
In the current Codex workflow, the implemented cache/evaluation commands are:

```bash
python -m adapters.cifar_branchy --dataset cifar100 --download --epochs 80 \
  --checkpoint checkpoints/cifar_branchy_resnet56.pt \
  --out cache/cifar_branchy_resnet56.npz

python -m experiments.real_cache_eval --cache cache/cifar_branchy_resnet56.npz \
  --out results/cifar_branchy_real_cache_eval.json \
  --alphas 0.05,0.10,0.20,0.30 --deltas 0.05,0.10 \
  --calib-sizes 500,2000 --T 1000

python -m experiments.real_cache_eval --cache cache/cifar_branchy_resnet56.npz \
  --out results/cifar_branchy_real_cache_eval_exploratory.json \
  --alphas 0.35,0.40 --deltas 0.05,0.10 \
  --calib-sizes 500,2000 --T 1000
```

### 3.2 Pre-register, then run E1 (validity — the central test)

Fill `preregistration.md` **now**, before running:

```markdown
## E1 pre-registration (CIFAR-100 branchy ResNet-56)
alpha grid : {0.05, 0.10, 0.20, 0.30}     # frozen before knowing feasibility
delta grid : {0.05, 0.10}
splits T   : 1000 fresh calibration/test partitions per cell
calib size : {500, 2000} (also feeds E4)
p-value    : Hoeffding-Bentkus (HB)
method     : chain (fixed-sequence), expensive->cheap
PASS       : empirical violation freq <= delta within a binomial CI in EVERY cell
```

Then run using `prompts/run-experiment.md` with `cifar_e1`. The script should, per trial,
split the cache into calibration/test, call `carc.selector.select_risk`, and record
`1{R_test(τ̂) > α}`. Report the violation frequency per cell with a Clopper–Pearson CI.

**Acceptance gate E1:** every cell at or below δ within CI. If a cell is *above* δ, that is a
real finding — first rule out bugs (label leakage between splits, a non-i.i.d. cache, an
off-by-one in the chain order), and only if the method is genuinely at fault do you report it
and investigate. Do not delete the cell.

### 3.3 E3 (naive baseline) and E2 (cost)

Run `select_naive` and `select_risk` on the same splits:
- **E3:** naive violation frequency (expected ≫ δ) vs. chain (≤ δ), at matched α. This is the
  paper's motivating contrast (§10.1 reports ~0.2313–0.3513 naive in simulation; confirm a real,
  non-trivial naive violation here — if real naive violation is small, *say so* and soften the
  motivation, per §1).
- **E2:** certified expected FLOPs for chain vs. Holm vs. Bonferroni at matched (α, δ), plus the
  exit-index histogram. Expect a small chain-vs-Bonferroni gap; report it honestly.

### 3.4 E4 (calibration size and the monotonicity diagnostic)

Sweep calib size `n`; show certified compute approaching the cheapest in-chain safe policy with
the gap tracking `Δ_n = O(√(ln(K/δ)/n))`. Report empirical monotonicity of `R̂` along the chain
**as a diagnostic** (the draft is careful that this is necessary, not sufficient, for Assumption M);
attach simultaneous CIs on the `R_k` differences. If monotonicity is violated on the real model,
run the Holm fallback there and report it.

**Acceptance gate Phase 2:** E1 passes pre-registration on a real model; E2/E3/E4 produce the
plots and the honest numbers. In the first CIFAR run, the pre-registered grid revealed that
`alpha <= 0.30` was not achievable because the full-pool minimum risk was `0.3055`; the
exploratory feasible-target run at `alpha in {0.35, 0.40}` supplies the positive real-cache
diagnostic. *At this point the paper has a minimum real-model check* — the rest strengthens it.

---

## 4. Phase 3 — Scale out (optional but strengthens engineering validation)

Each is a new adapter feeding the identical pipeline. Prioritize by marginal value:

1. **Tabular multi-exit MLP** — cheapest to run, cleanest i.i.d. assumption, easy to audit.
   Good second data point for E1/E2.
2. **ImageNet MSDNet / early-exit ResNet-50** — the recognizable vision result; reuse the CIFAR
   adapter structure with a larger backbone and cached logits.
3. **LLM size-cascade** — small→medium→large as a 3-exit policy on a text-classification set.
   This stresses the chain with very uneven `exit_cost`; it is the most novel setting and the
   most useful for broad IEEE Access validation, but also the most engineering. Treat as a stretch goal.

For each: write the adapter, build the cache, run `prompts/run-experiment.md` with
`<name>_e1`, and then run the E2/E3 suite. Reuse everything else.

**Acceptance gate Phase 3:** E1 holds on every dataset you include; any dataset where it does
not is either fixed (bug) or dropped from the claim (not hidden).

---

## 5. Phase 4 — Shift (E6), joint control (E5), ablations (E7)

### 5.1 E6 — covariate shift (the part to handle most carefully)

The simulation already exercises all four arms (unweighted violates; exact-weight ceiling;
estimated-weight valid when (⋆) holds; biased-estimator breach when (⋆) fails). On real data:
- **Vision:** ImageNet-C corruptions as the target distribution.
- **Tabular:** a label-frequency-shifted test set.
- Fit `ŵ` with a source-vs-target classifier (the `fit_weights_logistic` pattern generalizes).
- **Report the realized `‖ŵ−w‖₁` where you can estimate it**, the η-sensitivity curve (validity
  *and* feasibility vs η), and the split cost (`n₁/n₂`). Keep the honest message from §6.2/§10.1:
  validity holds only when η covers the estimator error, and an honestly-sized η may make the
  certificate infeasible. If on real data the estimated-weight certificate is never both valid
  and feasible, that is the finding — Theorem 4 stays a sensitivity tool, exactly as drafted.

### 5.2 E5 — joint risk–compute

Verify the joint event at rate ≥ 1−δ. Implement and compare the **empirical-Bernstein compute
bound** against the range-`C_max` Hoeffding one; the draft (§5) predicts EB is far less often
infeasible. This is a small, high-value addition that fixes the "loose dual bound" limitation.

### 5.3 E7 — ablations

p-value (Hoeffding vs HB vs EB); chain granularity `K`; scalar vs per-exit chain; chain vs Holm.
These mostly reuse existing code with different arguments.

**Acceptance gate Phase 4:** each table/figure exists with real numbers and a one-line honest
reading; negative results are reported, not buried.

---

## 6. Phase 5 — Backfill the paper and make figures

**Objective:** replace every §10.1/§10.2 number and "planned" hedge with measured results, and
generate publication figures.

- Use `prompts/paper-sync.md` for each result file. The prompt is written to **flag every place
  the real number is worse than the draft implies** — review those flags and soften claims where
  needed. The abstract and §1 already lead with "validity at near-naive cost," so modest compute
  gaps need no rewrite; only update the specific figures.
- Move the §10.2 "planned" framing to past tense for whatever you actually ran; keep a clearly
  labeled "planned" note only for anything you did not run.
- Figures: ask Codex to write a `figures/` script producing (i) violation-frequency vs δ
  with CIs, (ii) risk–compute Pareto (chain/Holm/Bonferroni/naive), (iii) the E4 band-vs-n plot,
  (iv) the E6 η-sensitivity curve. Save vector PDFs.
- Fill **Appendix B** with the real backbone configs, exit placements, FLOPs accounting, splits,
  and the weight-estimation procedure (the draft already points Appendix B at these).

**Acceptance gate Phase 5:** no "planned"/placeholder text remains for executed experiments;
every figure is generated by a committed script from a committed result file.

---

## 7. Phase 6 — LaTeX, citations, IEEE Access formatting

- Convert `draft.md` or the current manuscript source to the **official IEEE Access LaTeX template**. Ask
  Codex to do the mechanical conversion section by section, preserving the math verbatim,
  then proofread the compiled PDF against the markdown for dropped equations.
- Build `refs.bib`: every borrowed result must cite the real source. The draft names them —
  LTT (Angelopoulos, Bates, Candès, Jordan, Lei), conformal risk control, Hoeffding–Bentkus
  (Bates et al.), Maurer–Pontil empirical Bernstein, weighted conformal (Tibshirani et al.),
  graphical tests (Bretz–Maurer–Brannath), Holm, BranchyNet/MSDNet, Jin & Candès sensitivity.
  **Verify each citation resolves to a real paper** — do not let auto-generated bib entries
  invent venues/years. Cross-check titles on the web before trusting them.
- IEEE Access specifics: use the official IEEE Access template, submit matching source and PDF,
  include named authors in source/PDF, add 3--10 keywords, author biographies, acknowledgments
  and funding/disclosure text where applicable, define acronyms at first use in the abstract and
  body, and keep the main article under 20 pages when possible. Confirm current requirements on
  the IEEE Access site before submitting.

**Acceptance gate Phase 6:** the PDF compiles, every `\cite` resolves to a verified reference,
and the math matches the markdown.

---

## 8. Phase 7 — Final pre-submission audit (re-run the gate)

Re-apply the IEEE Access claim-evidence and presentation audit, now against the finished paper:

**Claims ↔ evidence.** Walk every claim sentence in the abstract, contributions,
and theorem statements, and confirm each is backed by either a proof (Appendix A) or a measured
result (now real, not simulated). Specifically check the four items the audit flagged:
- the compute claim is stated as "near-naive cost," matching the realized (likely small) gap;
- Theorem 4 is billed as a conservative sufficient condition, matching the real shift results;
- every p-value used in a result has a proof (A.7 covers EB) — confirm none crept in unproven;
- the implemented fallback is named Holm; "graphical" appears only as an unimplemented generalization.

**Engineering usefulness.** Confirm the practical takeaway leads: *naive confidence-threshold
tuning silently violates target risk; a distribution-free certificate fixes the validity problem
when feasible, with compute premium reported*, now demonstrated on a feasible real-cache study.

**Reproducibility.** Fresh clone → `pip install` → `pytest` → one `make`/script reproduces a
headline table and figure from cache. Seeds and `(α, δ, T)` are in `preregistration.md`.

A good use of Codex here: open the compiled PDF text and the result JSONs, and ask it to
**list every numeric claim in the paper alongside the result file that backs it, flagging any
unbacked number.** That is the final claim-evidence reconciliation in one pass.

---

## 9. Working effectively with Codex on this project

- **Use persistent planning for anything touching `carc/` or multi-file work.** Start with
  `task_plan.md`, confirm the plan does not edit the frozen core, and update the plan after
  each acceptance gate.
- **Use read-only scans before edits** ("how does the chain order flow from simulate to
  selector?") so the implementation path is evidence-based. If multi-agent tools are available,
  keep delegated scans read-only and let the main Codex run make edits.
- **Name exact file paths and line ranges** (for example, `carc/selector.py` around
  `select_risk`) rather than describing them indirectly.
- **One goal or conversation per phase** so context does not bleed between the CIFAR run and
  the LaTeX conversion. Re-read `task_plan.md` and `notes.md` before major decisions.
- **Commit per acceptance gate.** Each gate is a natural, reviewable checkpoint; small diffs make
  Codex edits easy to audit.
- **Make Codex argue against you.** After it backfills a result, ask: "as a skeptical IEEE Access
  reviewer, where does this number undercut a claim in the paper?" — this reproduces the audit
  loop continuously instead of once at the end.

---

## 10. Critical path, in one line each

1. Phase 0 — repo, `AGENTS.md`, reusable prompts, `preregistration.md` stub.
2. Phase 1 — `carc/chain.py` refactor + adapter contract (synthetic reproduces §10.1).
3. **Phase 2 — CIFAR E1/E2/E3/E4. ← minimum real-model gate now landed.**
4. Phase 5+6 (paper backfill + LaTeX) — after Phase 2, these are the remaining submission blockers.
5. Phases 3–4 — more datasets, shift, joint, ablations — strengthen engineering validation and robustness.
6. Phase 7 — final claim-evidence reconciliation, then submit.

The discipline that matters most is the cheapest: pre-register before you run, never tune on
test, and when a number comes back smaller than hoped, reduce the claim rather than the standard.
The draft was deliberately written so honest results require almost no rewriting — only the
figures change.

# Notes: CARC Manuscript Venue Retargeting

## Sources To Inspect
- `completion_guide.md`
- `draft.md`
- Official TMLR author/submission guidance

## Findings
- `completion_guide.md` is written for Claude Code. It names `CLAUDE.md`, `.claude/commands/`, Claude Code plan mode, `@` file mentions, `/context`, and `/usage`.
- The current checkout uses root `draft.md` and root `carc/`; `CARC_TMLR_draft.md`, `carc_ref/`, `paper/CARC.md`, result JSONs, real-data adapters, caches, `refs.bib`, and TMLR LaTeX files were not present in the root file scan.
- Current official TMLR guidance says submissions are double blind and must be anonymized, and the submission PDF must use the official TMLR LaTeX style/template. The guide's single-blind statement is stale or wrong.
- TMLR acceptance criteria focus on whether claims are supported by accurate, convincing, clear evidence and whether some TMLR readers would be interested.
- `draft.md` already marks the real-data study as planned. That boundary should stay unless real result artifacts are added.
- `draft.md` section 7 duplicates the input line and lists a graphical branch operationally, while the draft and guide say the implemented fallback is Holm and graphical procedures are only a generalization.
- Current Python environment has `torch 2.12.0+cu130`, `torchvision 0.27.0+cu130`, CUDA available, `sklearn`, `numpy`, and `scipy`; `fvcore` and `ptflops` are missing.
- Real-data adapters, CIFAR cache, checkpoints, and real-cache result JSONs are still missing.

## Draft Edit Ledger
- `draft.md`: removed the duplicated algorithm input line.
- `draft.md`: replaced the operational graphical branch in the algorithm with the implemented Holm fallback and tightened related-work/limitations/proof wording so graphical procedures are only an unimplemented generalization.
- `draft.md`: added a TMLR-facing broader-impact statement.
- `draft.md`: added a conclusion that keeps the real-data study as the remaining empirical burden.
- `draft.md`: revised the reproducibility checklist to mention anonymized TMLR supplementary material and the current reference-implementation boundary.
- `completion_guide.md`: converted Claude Code-specific workflow language to Codex language, including `AGENTS.md`, persistent planning, and `prompts/`.
- `completion_guide.md`: corrected TMLR from single-blind/optional anonymous to double-blind/anonymized.
- `AGENTS.md`: added project-specific Codex memory.
- `prompts/run-experiment.md` and `prompts/paper-sync.md`: added reusable task prompts replacing Claude slash commands.
- `preregistration.md`: added the real-data E1 pre-registration stub.
- `README.md`: removed stale `cd carc_ref`, changed the fallback label to Holm, and updated the root file tree.
- `carc/chain.py`: added `build_chain(scores, exit_loss, exit_costs, thresholds)`.
- `carc/simulate.py`: refactored `EarlyExitSim.sample` to call `build_chain`.
- `tests/test_carc.py`: added `test_build_chain_matches_manual_policy`.
- `completion_guide.md` and `README.md`: updated the chain helper signature and mapping.
- `experiments/common.py`: added JSON writing, git SHA capture, Clopper-Pearson CIs, and JSON-friendly means.
- `experiments/e1_validity.py`: now writes `results/e1_validity.json`.
- `experiments/e2_chain_vs_bonferroni.py`: now writes `results/e2_chain_vs_bonferroni.json`.
- `experiments/e5_joint_control.py`: added joint-control artifact generation, writing `results/e5_joint_control.json`.
- `experiments/e6_shift_eta_sweep.py`: now writes `results/e6_shift_eta_sweep.json`.
- `draft.md`: synchronized §10.1 numbers to generated result JSONs, including weaker E6 feasibility wording.
- `.gitignore`: added Python cache, pytest cache, and local `cache/` ignores while keeping `results/` trackable.
- Next implementation target: add a CIFAR branchy adapter that writes the per-exit cache contract and a cache-based real-data evaluator. Smoke verification will use `torchvision.datasets.FakeData`; it will not be reported as real evidence.
- `adapters/cifar_branchy.py`: added BranchyResNet-56-style CIFAR adapter with four exits, optional training/checkpoint loading, analytical exit-MAC estimates, per-exit cache writing, and `build_chain` integration.
- `experiments/real_cache_eval.py`: added cache-based E1/E2/E3/E4 diagnostic runner for chain/Holm/Bonferroni/naive across `(alpha, delta, calibration size)` grids.
- `README.md`, `completion_guide.md`, and `AGENTS.md`: added exact CIFAR cache and real-cache evaluation commands.
- `.gitignore`: added `checkpoints/` and `data/` for local real-data artifacts.
- `adapters/cifar_branchy.py`: fixed checkpoint reuse so an epoch-80 checkpoint is not trained for another 80 epochs when the documented command is rerun; added `checkpoint_epoch`, `num_exits`, and `exit_macs` to cache metadata.
- Real CIFAR run: trained `checkpoints/cifar_branchy_resnet56.pt` for 80 epochs, then wrote `cache/cifar_branchy_resnet56.npz` with 10,000 CIFAR-100 test examples, 4 exits, 100 thresholds, per-exit accuracies `[0.0646, 0.3040, 0.6147, 0.6945]`, and exit MACs `[443968, 42911296, 84331648, 125753600]`.
- `results/cifar_branchy_real_cache_eval.json`: pre-registered grid used `alpha={0.05,0.10,0.20,0.30}`, `delta={0.05,0.10}`, calibration sizes `{500,2000}`, `T=1000`, HB p-values. Full-pool minimum chain risk was `0.3055`, so the frozen grid had no truly safe configuration. Chain violation rate over all split attempts was `0` for `alpha<=0.20` and `0.000-0.013` at `alpha=0.30`; naive at `alpha=0.30` violated `0.281-0.457`.
- `results/cifar_branchy_real_cache_eval_exploratory.json`: post hoc feasible-target diagnostic at `alpha={0.35,0.40}`. Chain violation rates were `0.012-0.038`, below the matching `delta`; naive violation rates were `0.441-0.529`.
- `draft.md`, `README.md`, and `completion_guide.md`: synchronized to report the CIFAR diagnostic honestly, with the pre-registered infeasibility result separated from exploratory feasible-target evidence.

## Verification
- Official TMLR pages checked: author guide, submission instructions, acceptance criteria, and reviewer guide.
- Text audit: no active Claude-specific paths or stale `carc_ref` / `CARC_TMLR_draft` references remain in edited public files.
- Text audit: no operational `Graphical:` branch remains in `draft.md`.
- Ran `python -m tests.test_carc` before and after the chain refactor; all tests passed both times.
- Smoke-ran all JSON-output experiment scripts with tiny `T`; all wrote valid JSON.
- Ran full default scripts for E1/E3, E2/E7, E5, and E6; generated four `results/*.json` artifacts.
- Validated all four JSON artifacts with `python -m json.tool`.
- Re-ran `python -m tests.test_carc` after experiment-script changes; all tests passed.
- Re-ran stale-number/stale-Claude wording checks and `git diff --check`; checks passed after updating `completion_guide.md`'s naive simulation range.
- Syntax-checked the new adapter and real-cache evaluator with `python -m py_compile`.
- Smoke-built `cache/smoke_cifar_branchy.npz` using FakeData; verified expected arrays and shapes.
- Smoke-ran `experiments.real_cache_eval` on the FakeData cache; validated then removed `results/smoke_real_cache_eval.json` to avoid confusing smoke output with real evidence.
- Re-ran `python -m tests.test_carc` after adding the adapter/evaluator; all tests passed.
- Remaining completion gates beyond this pass: broader real-data backbones, citation/BibTeX verification, TMLR LaTeX conversion, and final claim-evidence audit.

## TMLR LaTeX Conversion Ledger
- `main.tex`: replaced the stock TMLR template with the CARC manuscript converted from `draft.md`.
- `main.tex`: kept anonymous TMLR review mode with `\usepackage{tmlr}` and no author metadata in the source.
- `main.tex`: converted Markdown headings, emphasis, lists, code-style paths, display equations, theorem/proof text, and the algorithm summary to LaTeX.
- `main.tex`: placed `\bibliographystyle{tmlr}` and `\bibliography{main}` before `\appendix`, so Appendix A/B content appears after references.
- `main.tex`: added `hyperref` with `hypertexnames=false` to avoid duplicate anchors from manual equation tags.
- `main.bib`: added 19 verified entries used by `main.tex`; every cited key resolves and no unused entries remain.
- `main.pdf`: generated successfully with `tectonic --keep-logs --keep-intermediates main.tex`, then TeX intermediates were removed after inspection.

## Citation Verification Ledger
- DOI content negotiation: LTT/AOAS, JACM distribution-free risk-controlling prediction sets, Hoeffding, Bentkus, Bretz-Maurer-Brannath, BranchyNet, SkipNet, Jin-Ren-Candes, Wiens, Lei et al., and Vovk et al.
- arXiv BibTeX: Maurer-Pontil empirical Bernstein, MSDNet, patience exiting, weighted conformal under shift, selective classification, and FrugalGPT.
- OpenReview: ICLR 2024 Conformal Risk Control.
- JSTOR stable metadata and corroborating bibliographic pages: Holm 1979.
- Official TMLR pages rechecked for the LaTeX conversion: current author guidance requires double-blind anonymized submissions, the official TMLR LaTeX style/template, optional appendices after references, anonymized supplementary material, and a broader-impact statement when significant harm risk exists.

## TMLR Conversion Verification
- Ran `tectonic --keep-logs --keep-intermediates main.tex`; it produced `main.pdf`. Non-fatal warnings remained: `algorithm.sty` invalid UTF-8 byte in the downloaded package, underfull boxes, and Tectonic's `main.bbl` rerun bookkeeping warning.
- Inspected `main.blg` and `main.log`; no missing bibliography entries, undefined citations, LaTeX errors, or BibTeX errors were present.
- Ran template-leftover check: no `Formatting Instructions`, `Kyunghyun Cho`, `Sample figure`, `goodfellow2016deep`, or stock anonymous-author text appears in `main.tex`.
- Ran placeholder check: no `[CITATION NEEDED]`, `PLACEHOLDER`, `TODO`, or `undefined` markers appear in `main.tex` or `main.bib`.
- Ran anonymity source check: no author names, affiliations, acknowledgments, funding text, GitHub links, OpenReview links, or HTTP links appear in `main.tex`.
- Ran citation-key check: `main.bib` has 19 entries, `main.tex` cites 19 keys, with no missing or unused keys.
- Ran section-count check: `draft.md` has 14 main sections and `main.tex` has 14 main sections; bibliography precedes `\appendix`; appendix sections in TeX are `Proofs` and `Extended experimental detail`, with the full Theorem 2b proof retained as a subsection under Appendix A.
- Ran theorem-tag check for `(P1)`, `(P2)`, `(1)`, `(2)`, `(3)`, `($\star$)`, and `(MP)`.
- Ran `pdftotext main.pdf -` spot check: the PDF begins with `Anonymous authors / Paper under double-blind review`, `References` appears before `Proofs`, and `Extended experimental detail` appears after the proof appendix.
- Ran whitespace and `git diff --check`; no trailing whitespace or diff-check errors.

## IEEE Access Retargeting Ledger
- Current target changed from TMLR to IEEE Access Research Article.
- Official IEEE Access submission guidance checked on 2026-06-15: double-column single-spaced IEEE Access template, source plus matching PDF, source/PDF author list, all-author biographies below references, 3--10 keywords, acronym definitions, grammar review, accurate non-retracted references, and a strong recommendation to keep the article under 20 pages unless justified.
- Current `main.tex` is still TMLR-specific at the start of this pass: `\usepackage{tmlr}`, anonymous/no-author PDF, TMLR reproducibility checklist, planned-protocol framing, and a 24-page TMLR-layout PDF.
- IEEE Access evidence-first retargeting needs one feasible real-cache confirmation study beyond the current CIFAR diagnostic, because the CIFAR pre-registered grid is mostly infeasible and the feasible CIFAR cells are exploratory.
- Implementation decision for this pass: add a tabular real-cache benchmark using a real sklearn dataset and a staged classifier cascade, writing cache/result artifacts with the existing loss/cost matrix interface. Do not edit `carc/` for this.
- `preregistration.md`: added the IEEE Access confirmatory digits tabular cascade block before running the new result artifact.
- `adapters/tabular_cascade.py`: added a three-exit sklearn cascade cache builder using `load_digits`; exits are GaussianNB, standardized logistic regression, and standardized RBF SVC with relative cost proxy `[1, 4, 12]`.
- Corrected the tabular adapter to clip exact confidence scores below `1.0` so threshold `1` is the always-final policy expected by the chain convention.
- `cache/digits_tabular_cascade.npz`: built with seed `20260615`, 898 train examples, 899 cache examples, 64 features, 10 classes, exit accuracies `[0.8631813125695217, 0.967741935483871, 0.9810901001112347]`, and final/full-pool minimum risk `0.018909899888765295`.
- `results/digits_tabular_real_cache_eval.json`: pre-registered grid used `alpha={0.05,0.10,0.15}`, `delta={0.05,0.10}`, calibration sizes `{100,300}`, `T=1000`, HB p-values, and violation denominator `all`. Chain violation range was `0.000-0.051`, never exceeding its matching `delta`; naive violation peaked at `0.571`. Chain feasibility ranged `0.139-1.000`.

## IEEE Manuscript Conversion Ledger
- `main.tex`: converted from `\usepackage{tmlr}` anonymous review source to `\documentclass{ieeeaccess}` with IEEE Access metadata fields, keywords, author/affiliation/ORCID placeholders, correspondence placeholder, acknowledgment/funding/AI-disclosure placeholder, author biography placeholder, and `IEEEtran` bibliography style.
- `main.tex`: reframed the paper as an IEEE Access Research Article about certified threshold selection for adaptive inference systems under finite calibration data.
- `main.tex`: added the digits tabular-cascade real-cache confirmation as Section 10.2 and Table 1; retained CIFAR-100 as Section 10.3 diagnostic negative/feasibility evidence.
- `main.tex`: removed TMLR checklist/broader-impact wording and replaced it with `Engineering and deployment considerations` and `Data and code availability`.
- `main.tex`: converted `\citet`/`\citep` to IEEE-compatible `\cite` commands and verified 19 cited keys match 19 BibTeX entries with no missing or unused keys.
- `main.tex`: fixed the P0 guarantee wording: if no configuration certifies, the procedure reports infeasibility and no risk guarantee is attached to any operational fallback unless that fallback separately certifies.
- `main.tex`: added guarded compatibility definitions for `\xfigwd`, `biography`, and `\if@biographyTOCentrynotmade` because the available IEEE Access mirror used for a temporary syntax build references those symbols before defining them under Tectonic/XeTeX.
- `README.md`: added exact digits cache/evaluator reproduction commands and representative output for the feasible real-cache confirmation.
- `completion_guide.md`: retargeted submission gates from TMLR to IEEE Access requirements.
- `main.pdf`: replaced the old 24-page TMLR-layout PDF with an 18-page IEEE-style PDF generated in a temporary build directory. The build used a downloaded IEEE Access mirror plus temporary XeTeX compatibility edits outside the repo because official/CTAN shell downloads were blocked and the local repo lacks `ieeeaccess.cls`.

## IEEE Retargeting Verification
- Re-ran `python -m experiments.real_cache_eval --cache cache/digits_tabular_cascade.npz --out results/digits_tabular_real_cache_eval.json --alphas 0.05,0.10,0.15 --deltas 0.05,0.10 --calib-sizes 100,300 --T 1000 --seed 20260615 --pvalue hb --violation-denominator all`; it reproduced the 48-row artifact.
- Validated `results/digits_tabular_real_cache_eval.json` with `python -m json.tool`; chain max violation is `0.051` and no chain row exceeds its matching `delta`.
- Ran `python -m tests.test_carc`; all tests passed.
- Ran `python -m py_compile adapters/tabular_cascade.py experiments/real_cache_eval.py`; no syntax errors.
- Ran `git diff --check`; no whitespace errors.
- Ran source and PDF venue-marker checks: no `TMLR`, `tmlr`, `anonymous`, `double-blind`, `\citet`, `\citep`, `Reproducibility checklist`, or `Broader impact` markers remain in `main.tex`, `README.md`, `completion_guide.md`, or extracted `main.pdf` text.
- Ran `pdfinfo main.pdf`; rebuilt artifact is 18 pages.
- Ran `tectonic --keep-logs --keep-intermediates main.tex` in the repo; it fails at `main.tex:2` because `ieeeaccess.cls` is not installed locally. This is the remaining template dependency, not a manuscript syntax failure.

## Editorial Rewrite Ledger
- `main.tex`: rewrote the title, abstract, introduction, contributions, relation-to-LTT discussion, related work, implementation overview, experiment narrative, limitations, engineering considerations, data/code statement, and conclusion for a more precise IEEE Access Research Article framing.
- `main.tex`: sharpened the problem statement around certified threshold selection under finite calibration data, separated validity from feasibility and efficiency, and kept the empirical claims tied to the controlled simulation, digits real-cache confirmation, and CIFAR-100 diagnostic.
- `main.tex`: down-modulated broad claims by stating that the work does not claim state-of-the-art backbone accuracy, does not validate larger ImageNet or language deployments, and treats shift certification as a sensitivity result requiring a supplied weight-error budget.
- `main.tex`: preserved mathematical notation, displayed equations, theorem/proof structure, citation keys, labels, algorithmic blocks, tables, and reported numeric evidence while editing the surrounding prose.
- `main.tex`: added `\vol{14}` and `\Year{2026}` after verifying that IEEE Access volume 14 corresponds to 2026; the temporary build mirror also needed a non-repo footer patch because its non-JTEHM branch hard-coded `2016`.
- `README.md`: updated the descriptive manuscript title to `Compute-Adaptive Risk Control: Finite-Sample Certificates for Early-Exit and Cascade Inference`.

## Editorial Rewrite Verification
- Ran citation-key validation after the rewrite: `main.tex` cites 19 keys, `main.bib` contains 19 entries, with no missing or unused keys.
- Rebuilt `main.pdf` from the edited source in `/tmp/carc-ieee-build`; the build succeeded and the copied PDF is 18 pages.
- Extracted `main.pdf` text with `pdftotext`; the PDF begins with the revised title and uses `VOLUME 14, 2026`.
- Ran source/PDF stale-marker checks: no old title, `TMLR`, `tmlr`, `anonymous`, `double-blind`, `\citet`, `\citep`, `[CITATION NEEDED]`, `planned protocol`, `remaining protocol`, or `VOLUME 4, 2016` markers remain in `main.tex`, `README.md`, `completion_guide.md`, or extracted PDF text.
- Ran style-marker checks: no `groundbreaking`, `revolutionary`, `silently`, `honest`, `turnkey`, `obvious`, or standalone `very` markers remain in `main.tex` or extracted PDF text.
- Ran `python -m tests.test_carc`; all tests passed.
- Ran `git diff --check`; no whitespace errors.

# Task Plan: Retarget CARC to IEEE Access

## Goal
Retarget the CARC manuscript from anonymous TMLR submission form to an IEEE Access Research Article, adding one feasible confirmatory real-cache experiment and keeping all claims tied to local artifacts.

## Phases
- [x] Phase 1: Initialize plan and scope
- [x] Phase 2: Read guide, draft, and venue requirements
- [x] Phase 3: Identify required manuscript edits and Codex wording updates
- [x] Phase 4: Patch files with supported, conservative changes
- [x] Phase 5: Verify consistency and report changed/unchanged files
- [x] Phase 6: Build real-data adapter and cache-evaluation path
- [x] Phase 7: Run CIFAR cache diagnostics and synchronize `draft.md`
- [x] Phase 8: Convert `draft.md` to anonymous TMLR `main.tex`
- [x] Phase 9: Add verified `main.bib`, compile, and run submission checks
- [x] Phase 10: Update venue plan and inspect current IEEE Access requirements
- [x] Phase 11: Add a feasible confirmatory real-cache benchmark without changing `carc/`
- [x] Phase 12: Convert manuscript source from TMLR framing/template to IEEE Access framing/source
- [x] Phase 13: Rebuild/check PDF, tests, result JSONs, citations, and venue leftovers
- [x] Phase 14: Perform elite technical editorial rewrite of the active IEEE Access manuscript

## Priorities
- P0: Complete required sections or placeholders in `draft.md` using only evidence available in the repo or verified venue guidance.
- P0: Replace Claude Code-specific instructions in `completion_guide.md` with Codex-appropriate wording where needed.
- P1: Improve TMLR fit, clarity, claim boundaries, and section consistency.
- P2: Defer new experiments, new citations, or unsupported claims unless the guide explicitly requires them and evidence exists.
- P0: For the real-data gate, first create an executable CIFAR cache adapter and cache-based E1/E2/E3/E4 runner; only revise `draft.md` to real-data past tense after a real cache result exists.
- P0: For IEEE Access, remove anonymous/TMLR-specific source text and add IEEE-facing metadata, keywords, data/code availability, acknowledgments, and author-biography placeholders.
- P0: Add one feasible real-cache confirmation study and explicitly separate it from the CIFAR infeasibility diagnostic.
- P0: Fix infeasible fallback wording so reported infeasibility is not presented as a risk guarantee.
- P1: Keep long proofs and row-level tables available but move the submission-facing narrative toward engineering validation and concise IEEE readability.

## Key Questions
1. What concrete completion steps does `completion_guide.md` require?
2. Which claims in `draft.md` are already supported by local evidence?
3. Which Claude-specific terms need to become Codex-specific?
4. What TMLR-specific requirements matter for the Markdown draft?
5. Which IEEE Access requirements require source-level changes?
6. Which feasible real-cache result can be added in this pass without changing the certified selector core?
7. Which manuscript claims need sharper title, introduction, related-work, evidence, limitation, or conclusion wording without changing the underlying theory or results?

## Decisions Made
- Treat TMLR as the target venue.
- Keep changes small, evidence-based, and reversible.
- Do not invent citations, results, or experimental details.
- Use `draft.md` as the manuscript source and root `carc/` as the reference implementation, because `CARC_TMLR_draft.md` and `carc_ref/` are not present in this checkout.
- Treat the real-data study as unrun unless actual adapters, caches, or result artifacts are found.
- Use `AGENTS.md` and reusable prompt files instead of `CLAUDE.md` and `.claude/commands/`.
- Add a TMLR broader-impact statement because the current TMLR author guide requires one when work carries meaningful harm risk; the statement frames over-trust and assumption misuse as the relevant risks.
- Add a conclusion that preserves the evidence boundary: current evidence is proof plus controlled simulation; real-model evidence remains future work.
- Implement Phase 1's dataset-free refactor by extracting `carc.chain.build_chain`; leave real-data phases pending because no adapters, caches, or datasets are present.
- Preserve existing synthetic evidence as JSON artifacts before any future TMLR paper sync.
- Torch/TorchVision are installed and CUDA is available; fvcore/ptflops are not installed, so CIFAR exit cost uses an explicit analytical MAC estimate unless a profiler is later added.
- The first CIFAR-100 cache diagnostic is a real-model check, not a full real-data suite. The pre-registered `alpha <= 0.30` grid is mostly infeasible because the full-pool minimum chain risk is `0.3055`; the post hoc feasible-target check at `alpha in {0.35, 0.40}` is labeled exploratory in `draft.md`.
- Replace the stock TMLR `main.tex` with the manuscript converted from `draft.md`; keep `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, and `math_commands.tex` unchanged.
- Keep the TMLR submission anonymous by using `\usepackage{tmlr}` review mode and no author metadata in `main.tex`.
- Add only verified bibliography entries to `main.bib`; DOI, arXiv, OpenReview, and JSTOR/stable-page metadata were used, and all citation keys in `main.tex` resolve.
- Keep appendices after `\bibliographystyle{tmlr}` and `\bibliography{main}` to match current TMLR author guidance.
- Retargeting strategy is evidence-first IEEE Access Research Article.
- Current official IEEE Access guidance requires the IEEE Access template, matching source/PDF, author names in both source and PDF, author biographies, 3--10 keywords, acronym definitions, grammar review, accurate non-retracted references, and recommends keeping the main article under 20 pages.
- Use a tabular real-cache benchmark for the added confirmatory evidence in this pass, because it is feasible locally, keeps calibration/test splits clean, and does not require changing `carc/`.
- Keep `main.tex` as the active manuscript source but convert it to IEEE-facing source; retain TMLR files as untouched legacy template files unless cleanup is explicitly requested.
- Keep author names, affiliations, ORCID values, funding, acknowledgments, AI-use disclosure, repository DOI, and manuscript DOI as explicit placeholders because the required metadata was not provided.
- Do not vendor a mirrored or locally patched IEEE Access class into the repo in this pass; use the official IEEE Access template files for final submission.
- For the editorial rewrite, keep the venue-ready IEEE Access structure and change prose only: preserve equations, labels, citations, algorithms, proofs, tables, and numerical evidence unless a local inconsistency requires a narrow correction.
- Use the title `Compute-Adaptive Risk Control: Finite-Sample Certificates for Early-Exit and Cascade Inference` because it matches the evidence scope better than the earlier broader title.

## Errors Encountered
- Goal creation was not needed because this exact goal is already active in the goal tracker.
- One `rg` numeric-claim search used an invalid regex escape (`\g`); reran equivalent targeted checks after removing that pattern.
- The first Markdown-to-LaTeX converter run had a Python quote-escaping syntax error before writing `main.tex`; fixed the converter and reran.
- The first Tectonic compile failed because `math_commands.tex` already defines `\argmin`; removed the duplicate declaration from `main.tex`.
- The next compile failed on `\tag{\star}`; changed the tag to the compile-safe `\tag{$\star$}` form.
- Tectonic continues to report non-fatal `algorithm.sty` encoding, underfull-box, and `main.bbl` rerun bookkeeping warnings, but it writes `main.pdf`; `main.blg` and `main.log` showed no undefined citations or BibTeX errors during verification.
- Initial tabular cascade cache used exact probability values of 1.0, which made threshold 1 not always force the final exit. Fixed by clipping confidence scores to the next float below 1.0, rebuilt `cache/digits_tabular_cascade.npz`, and reran `results/digits_tabular_real_cache_eval.json`.
- Local `tectonic --keep-logs --keep-intermediates main.tex` in the repo fails immediately because `ieeeaccess.cls` is not installed locally.
- CTAN/official shell downloads for `ieeeaccess` returned HTTP 403, and the environment has no `tlmgr`, `kpsewhich`, `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `bibtex` executable in PATH. A temporary syntax/PDF build used a downloaded IEEE Access mirror plus local XeTeX-only compatibility patches outside the repo; the official template should replace this for submission.

## Status
**Completed with template caveat** - `main.tex`, `main.pdf`, the digits confirmatory artifact, README, completion guide, preregistration, notes, and plan now target IEEE Access, and `main.tex` has received a venue-facing technical editorial rewrite. A clean source rebuild from the repo still requires installing the official IEEE Access template files (`ieeeaccess.cls` and dependencies).

---

# Task Plan: IEEE Access Pre-Submission Critical Review

## Goal
Produce a senior-editor-style pre-submission critique of the current IEEE Access manuscript, limited to weaknesses, current acceptance probability, high-yield non-experimental revisions, and post-revision acceptance probability.

## Phases
- [x] Phase 1: Confirm active manuscript and review constraints
- [x] Phase 2: Read manuscript structure, claims, evidence artifacts, and IEEE Access requirements
- [x] Phase 3: Audit weaknesses and desk-reject/major-revision risks
- [x] Phase 4: Deliver constrained final review

## Priorities
- P0: Do not recommend additional physical or computational experiments.
- P0: Separate administrative IEEE Access readiness issues from scientific-review issues.
- P0: Ground criticisms in `main.tex`, `main.pdf`, local result artifacts, and official IEEE Access guidance.
- P1: Prefer concrete textual, structural, or theoretical revisions that can be done before submission.

## Status
**Completed** - constrained IEEE Access readiness review prepared from `main.tex`, `main.pdf`, local result artifacts, and current official IEEE Access guidance.

---

# Task Plan: Multi-Chain CARC Redesign and Broad Evidence

## Goal
Redesign CARC with a clearly stronger certified selector, prove its validity, and build broad synthetic and real-cache evidence for an IEEE Access-ready revision.

## Phases
- [x] Phase 1: Inspect current selector, tests, and experiment architecture
- [x] Phase 2: Implement the first redesigned selector with unit tests
- [x] Phase 3: Add fast synthetic evidence comparing redesigned selector to chain, Holm, Bonferroni, and naive
- [x] Phase 4: Extend cache construction/evaluation so richer real-cache policy families can exercise the redesigned selector
- [x] Phase 5: Run broad evidence on at least one feasible neural early-exit/cascade cache beyond digits and the current CIFAR infeasibility diagnostic
- [x] Phase 6: Add theorem/proof and evidence summaries to `main.tex` only after result artifacts exist
- [x] Phase 7: Rebuild/check package and reassess IEEE Access readiness
- [x] Phase 8: Replace or cross-check the mirrored IEEE Access class/assets against the official IEEE Access ZIP before final upload
- [x] Phase 9: Final scientific completion audit and acceptance-readiness estimate

## Redesign Candidate
- Multi-chain CARC: pre-specify `M` compute-ordered chains inside a larger candidate family, run fixed-sequence testing within each chain at level `delta / M`, union the certified sets, and choose the cheapest certified configuration.

## Why This Is Worth Implementing First
- Validity proof is direct: each chain has FWER at `delta / M`; a union bound over `M` chains gives total FWER at `delta`.
- It keeps the existing adapter contract (`loss_matrix`, `cost_matrix`) and adds only chain metadata.
- It can search a richer threshold family than the current scalar chain while avoiding the full `delta / K` penalty of Bonferroni over all configurations when `M << K`.

## Current Decision
- Use the newly trained CIFAR-10 branchy cache as the feasible neural-cache benchmark for the multi-chain redesign. Keep CIFAR-100 as the infeasibility stress test because its preregistered target grid remains below the finite-pool risk frontier.
- Treat the local IEEE Access PDF rebuild as a diagnostic package gate, not final template provenance, because the class/assets in the repo came from a public mirror after official shell downloads were blocked.
- Replace the mirrored template with the official IEEE Access 2026-05-13 LaTeX bundle. Keep only a minimal non-pdfTeX `spotcolor` guard in `ieeeaccess.cls` so Tectonic can build locally; pdfTeX still follows the official spot-color path.
- Treat DOI, repository/archive DOI, and acknowledgment/funding/AI-disclosure text as author-supplied upload metadata. Do not invent them or count them as scientific-redesign gaps.

## Deferred User-Supplied Upload Metadata
- Manuscript DOI or publisher placeholder policy.
- Public source repository URL and archival DOI, if the authors choose to provide them before submission.
- Final acknowledgments, funding, and AI-disclosure wording.

## Status
**Completed for the scientific redesign objective** - Multi-chain certification, tests, E8 synthetic evidence, multi-chain real-cache evaluation support, manuscript proof/evidence summaries, a newly trained CIFAR-10 feasible neural-cache benchmark, official IEEE Access template files, a successful local IEEE Access PDF rebuild, and final acceptance-readiness audit now exist. Remaining actions are author-supplied upload metadata, not redesign/evidence work.

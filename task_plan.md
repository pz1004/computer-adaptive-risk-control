# Task Plan: Complete TMLR Draft

## Goal
Complete `draft.md` for a TMLR submission by following `completion_guide.md`, while adapting Claude Code-specific wording in the guide or related instructions to Codex.

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

## Priorities
- P0: Complete required sections or placeholders in `draft.md` using only evidence available in the repo or verified venue guidance.
- P0: Replace Claude Code-specific instructions in `completion_guide.md` with Codex-appropriate wording where needed.
- P1: Improve TMLR fit, clarity, claim boundaries, and section consistency.
- P2: Defer new experiments, new citations, or unsupported claims unless the guide explicitly requires them and evidence exists.
- P0: For the real-data gate, first create an executable CIFAR cache adapter and cache-based E1/E2/E3/E4 runner; only revise `draft.md` to real-data past tense after a real cache result exists.

## Key Questions
1. What concrete completion steps does `completion_guide.md` require?
2. Which claims in `draft.md` are already supported by local evidence?
3. Which Claude-specific terms need to become Codex-specific?
4. What TMLR-specific requirements matter for the Markdown draft?

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

## Errors Encountered
- Goal creation was not needed because this exact goal is already active in the goal tracker.
- One `rg` numeric-claim search used an invalid regex escape (`\g`); reran equivalent targeted checks after removing that pattern.
- The first Markdown-to-LaTeX converter run had a Python quote-escaping syntax error before writing `main.tex`; fixed the converter and reran.
- The first Tectonic compile failed because `math_commands.tex` already defines `\argmin`; removed the duplicate declaration from `main.tex`.
- The next compile failed on `\tag{\star}`; changed the tag to the compile-safe `\tag{$\star$}` form.
- Tectonic continues to report non-fatal `algorithm.sty` encoding, underfull-box, and `main.bbl` rerun bookkeeping warnings, but it writes `main.pdf`; `main.blg` and `main.log` showed no undefined citations or BibTeX errors during verification.

## Status
**Completed current Codex pass** - `draft.md` has been converted to anonymous TMLR `main.tex`, `main.bib` has verified bibliography entries with all citations resolved, and `main.pdf` compiles. Remaining work for submission is broader dataset coverage and a final claim-evidence audit.

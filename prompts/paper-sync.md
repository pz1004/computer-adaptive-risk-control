# Paper Sync Prompt

Read `results/{result_file}.json` and update the corresponding subsection of `draft.md`.

Requirements:
- Replace placeholders only with numbers directly backed by the result file.
- Flag every place where the realized number is worse than the current draft implies.
- Soften claims when the result is weaker than expected.
- Keep unexecuted experiments explicitly labeled as planned.
- Do not overwrite a claim to look better than the data.
- After editing, list changed sections and run a text check for remaining placeholders or stale
  planned/completed wording.

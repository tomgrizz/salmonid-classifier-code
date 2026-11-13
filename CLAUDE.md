# CLAUDE.md

Guidance for Claude Code sessions (including every subagent of a workflow) working in this
repository. Human-facing documentation is `README.md` and `docs/`; this file is only what a
fresh session needs in order not to break things.

## What this repository is

The public **code and data deposit** for a manuscript submitted to the Canadian Journal of
Fisheries and Aquatic Sciences (CJFAS): a deep learning model that identifies and counts five
salmon and trout species in fishway video at two Lake Ontario tributaries (Ganaraska River,
Credit River), validated against technician counts, 2020 to 2025.

- The manuscript lives in a sibling repo, `../salmonid-classifier-paper` (`manuscript.md`).
  **Never edit that repo from here.** Read it to check numbers.
- The private research repo this code was extracted from is `../lake-ontario-fish-cv`.
  **Read-only.** Its v2.7 code is split across two unrelated branches (`main` and
  `lamprey-chinook-wounding`); read `main` with `git -C ../lake-ontario-fish-cv show main:<path>`.
  Never check out, stash or commit there.
- Readers are fisheries scientists and journal reviewers, not ML engineers.

The published model is **v2.7** (DEIMv2-X, checkpoint epoch 17; held-out macro-F1 0.961 on the
366-track test set). v2.6 (0.966) is superseded: 62 of its test fish had been trained on.
Nothing in the manuscript uses v2.6. Any text that calls v2.6 "published" is a bug.

## The one check that must stay green

```bash
"C:/Users/lomu-/AppData/Local/Programs/Python/Python311/python.exe" run_tier1.py
```

It recomputes every result that can be recomputed from `data/`, compares each regenerated
table with the committed one value by value, and re-derives the manuscript's headline numbers.
Expected last lines: `RESULT: N passed, 0 failed` and `Every regenerated result matches the
published one.` Run it before and after any change to `salmonid/`, `data/` or `run_tier1.py`.
About one minute, no GPU, no external data.

Interpreter: the `python` on PATH is 3.14 and is the wrong one. Use the 3.11 path above. The
DEIMv2 (GPU) environment is `../lake-ontario-fish-cv/.venv_deim/Scripts/python.exe`.

## Hard rules

1. **Never write into `data/`.** It holds the published tables exactly as the manuscript
   quotes them. Regenerated output goes to `output/` (git-ignored); figures to `figures/`
   (git-ignored). The only legitimate change to `data/` is replacing a table with a
   regenerated one that a human has decided to publish, recorded in
   `docs/REPRODUCIBILITY_REVIEW.md`.
2. **Do not change a computation that feeds a manuscript number.** This code produced published
   results. If the code and the manuscript disagree, report it; do not tune either. Known
   places where the implemented method differs from a simpler description (daily-correlation
   day set, passage dating, the handling of "Small Salmonid" tracks, and so on) are documented
   in the module docstrings and in `docs/REPRODUCIBILITY_REVIEW.md`: leave them as they are.
3. **External data is read-only**: `G:/Projects`, `G:/RiverWatcher`, `F:/RiverWatcher`,
   `C:/ssl_cache`, `../DEIMv2`, `../lomu_interview`, `../model_pro`. Never launch training or
   full-corpus inference (days of GPU time) without the author asking for it.
4. **Paths come from `salmonid/config.py` only.** No drive letters, no user-profile paths, no
   corpus folder names typed into other modules. Missing Tier-2 inputs must fail through
   `config.require(...)` with one clear sentence.
5. **No personal names** in code, comments or docs ("the lead author", "a reviewer"). No
   references to files that are not in this repository, except as documented Tier-2 inputs.
6. **Commit only when the author asks.** Never push. Never rewrite history.
7. **The licence is the author's decision** (pending sign-off from the Ontario Ministry of
   Natural Resources). Do not replace `LICENSE` or add licence headers.

## Layout

```
run_tier1.py        the one command: run every Tier-1 step in order, then verify
salmonid/           the code, one sub-package per pipeline stage (each has a README.md)
  config.py           every path and constant; MODEL_VERSION, CORPUS_TAG, PUBLISHED_PREDS
  holdout.py          keeps the held-out test set out of training
  data/  train/       build the training set; launch training        (Tier 2)
  detect/             run the model on video; track fish             (Tier 2)
  eval/               held-out species score (Tier 1) + development gates
  count/              daily counts, the upstream/downstream rule, its validation
  results/            canonical tables, agreement statistics, intervals
  paper/              the remaining manuscript tables and checks
  figures/            every manuscript figure
  provenance/         records what a training run used
data/               published tables (read-only; data/README.md is the data dictionary)
configs/            the DEIMv2 configuration chain, as run
provenance/         the committed record of the published training run
docs/               REPRODUCE, PAPER_TO_CODE, DATA_AVAILABILITY, REPRODUCIBILITY_REVIEW, ...
environment/        exact package versions
```

Tier 1 = runs from files in `data/`. Tier 2 = needs external downloads (weights, video,
stored detections, technician records).

## Writing style for this repository

Every module starts with a docstring whose first paragraph begins `In plain terms:` and tells a
fisheries scientist, in three to six sentences without jargon, what the script does and why it
exists. Then `Inputs`, `Outputs`, `Feeds` (which manuscript table, figure or section),
`Run` (the exact `python -m salmonid.…` command), `Tier`. Inline comments explain *why*, not
*what*. State limitations plainly. No marketing words ("honest", "deploy-grade", "robust")
unless they name a defined quantity. Docs follow the manuscript's conventions: Canadian
spelling (-ize, -our), "count" not "enumeration", no em dashes, AFS common names (Chinook
Salmon, Coho Salmon, Atlantic Salmon, Rainbow Trout, Brown Trout).

Manuscript item numbers (as of 2026-11-10): Tables 1 to 5, S1 to S4; Figures 1 to 7, S1 to S3.
Check `docs/PAPER_TO_CODE.md` and the manuscript itself before writing a table or figure
number: the numbering has changed several times.

## Where the open work is

`docs/REPRODUCIBILITY_REVIEW.md` is the register of reproducibility findings: what was fixed,
what is open, and what needs an author decision. Each open item has an identifier, the files
involved and an acceptance check. Start there. Items marked **AUTHOR** need a human decision
(licence, deposit DOIs, data-sharing terms, manuscript wording); do not resolve those yourself.
A saved multi-agent audit lives in `.claude/workflows/` (see `.claude/workflows/README.md`).

# `salmonid/eval/`: the held-out species score and the development checks

This folder holds two different kinds of script.

1. **The held-out species score.** How the 366-track test set was built, how the published
   model (v2.7) is given one species per test track, and how that is scored. This is where
   manuscript Table 1 and Figure 2 come from.
2. **Development checks ("gates").** Quick pass/fail checks used while models were being
   developed. None of them produces a number in the manuscript. The published model does
   **not** pass all of them; what each one returned is stated below and in the module's
   docstring.

Tier 1 means the script runs from the files committed in `data/`. Tier 2 means it needs
external files that are not in git (model weights, stored detections, the private labelling
database).

## Held-out species score (in the order the steps are run)

| File | What it does | Feeds | Tier | Status |
|---|---|---|---|---|
| `build_holdout_testset_queue.py` | Chose 500 never-labelled, fully visible tracks from an earlier run for a reviewer to label; the 215 that received a species became the "pristine" tier of the test set. | Defines part of the test set behind Table 1, Figure 2, Section 3.1 | 2; cannot be re-run without the private labelling database, and a re-run gives a different draw | Published path (record of how the set was drawn) |
| `finalize_testset_v6.py` | Assembled the frozen test set: the 215 reviewer-labelled tracks plus 151 Coho Salmon, Atlantic Salmon and Brown Trout tracks drawn from the previously labelled pool ("heldout" tier), and wrote the "do not train on these" list. | Defines the 366-track test set (`data/test_set_v6_gt.json`, `data/test_set_v6_holdout.json`) | 2; same limits as above | Published path (record of how the set was assembled) |
| `predict_testset_v6_from_store.py` | Gives each test track one species by averaging the model's per-frame scores inside the track's boxes. | The predictions file `data/preds_v27_ckpt0017.csv` behind Table 1 and Figure 2 | 2 | Published path |
| `score_testset_v6.py` | Compares the predicted species with the reviewers' species: precision, recall and F1 per species, macro-F1, accuracy, confusion matrix. | Table 1 point estimates (macro-F1 0.961; 352 of 366 correct), Figure 2, Section 3.1 | 1 | Published path |

The 95% intervals of Table 1 are computed in `salmonid/results/classification_ci.py`, and
Cohen's kappa for the held-out set in `salmonid/paper/`.

## Development checks

| File | What it does | Feeds | Tier | Status | Result for the published model (v2.7) |
|---|---|---|---|---|---|
| `deimv2_bbox_gate.py` | Checks that the model's boxes overlap hand-drawn boxes closely (median IoU at least 0.80 per river) and are not shifted vertically, on 150 hand-boxed frames from 2025. Does not count false-positive boxes. | Development check; no manuscript number | 2 | Supporting check | **FAIL.** Credit River median IoU 0.737 (42 boxes; below 0.80); Ganaraska River 0.805 (126 boxes; pass); no vertical shift at either river (ratios 1.010 and 1.014). Re-run 2026-11-10. |
| `count_r_gate.py` | Checks that daily counts follow the technicians' counts: correlation at least 0.90 and season total within 15%, per river and species. | Development check; no manuscript number | 1 on the committed daily tables (`data/net_rule_sweep/<year>_up50.csv`, `data/activity_sweep/<year>_activity.csv`); 2 otherwise | Supporting check | **FAIL** on the 2025 net counts: 9 of 15 rows flagged. Chinook Salmon and Rainbow Trout pass at both rivers and pooled; Coho Salmon, Atlantic Salmon and Brown Trout are flagged. The other seasons' net counts fail as well (7 to 14 of 15 rows flagged). Run 2026-11-10. |
| `score_tracked.py` | Re-scores stored tracks against the technicians' per-video species under different track settings (minimum length, edge margin, voting rule). | Development check; no manuscript number | 2 | Supporting check (development tool) | Not run. No track file with technician species was built for v2.7; the published checkpoint was chosen on the held-out score. |
| `bbox_regression_gate.py` | The same box check for detectors of the retired two-stage pipeline. It cannot load the published model. | Development check; no manuscript number | 2 | Legacy two-stage code kept for provenance | Not applicable. |
| `corpus_year_gate.py` | GO / NO-GO check between seasons of the retired two-stage runs: daily fish passages against the technician record. Reads a track database that the published model does not produce. | Development check; no manuscript number | 2 | Legacy two-stage code kept for provenance | Not applicable. |

A check with nothing to evaluate (missing files, empty input) now fails and exits with a
non-zero code; in the original research scripts several of them reported a pass in that case.

## Facts about the test set that the scripts establish

* Two tiers. "pristine": 215 tracks never labelled before (121 Chinook Salmon, 65 Rainbow
  Trout, 21 Coho Salmon, 7 Atlantic Salmon, 1 Brown Trout). "heldout": 151 tracks from the
  previously labelled pool (59 Brown Trout, 53 Atlantic Salmon, 39 Coho Salmon), withheld
  from training once chosen.
* Labels were assigned by reviewers in the project's labelling tool. The heldout-tier labels
  are the ones already in the labelling database; they were not reviewed again for the test set.
* A test track is the frozen sequence of boxes from an earlier detection run. The published
  model is graded on the fish inside those boxes; its own tracker is not involved.
* Test tracks had to be confidently detected by an earlier model and show the whole fish in
  at least three frames, and half of the pristine queue was chosen to be hard for that
  earlier model.
* The construction cannot be repeated exactly. The committed JSON files are the test set.

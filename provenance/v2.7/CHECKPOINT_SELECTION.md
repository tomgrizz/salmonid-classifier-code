# Why checkpoint 0017 is the published model

**In plain terms.** Training does not produce one model. It saves a copy of the model
(a "checkpoint") every two passes through the training pictures, and the copies differ
slightly. One of them has to be chosen. This page gives the scores of every candidate, says
which one was chosen and why, and states plainly what the choice does and does not show.

## The short answer

- Seven candidates were scored on the held-out test set (366 fish).
- Six of them score between 0.959 and 0.963 (macro-averaged F1). The differences are one or
  two fish out of 366 and are far smaller than the uncertainty of the score itself
  (95% interval for checkpoint 0017: 0.939 to 0.980, `data/classification_ci.csv`).
- Checkpoint 0017 was chosen. It is tied with the best candidates on the species score, has
  the tightest boxes of any candidate at the Credit River, and is the checkpoint that was then
  used for all the multi-year counts.
- **The choice was made by looking at the held-out test set.** The manuscript working copy of
  2026-11-10 does not say this; it should (see `docs/REPRODUCIBILITY_REVIEW.md`, item L4). The
  0.961 is therefore the score of the chosen candidate on the set used to choose it, not a
  score on fish that played no part in any decision. Because the candidates are so close, the
  size of this effect is bounded by the spread in the table below (about 0.003).

## The candidates

DEIMv2 saves a checkpoint every 2 epochs until epoch 25, then keeps only the one with the best
validation score (`best_stg2`). Epochs are numbered from 0, so `checkpoint0017` is the model
after the 18th pass. The run had 30 epochs.

Species score on the held-out test set, all 366 fish, one prediction per fish:

| Checkpoint | Fish scored | Correct | Accuracy | Macro F1 | Chinook | Coho | Atlantic | Rainbow | Brown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0013 | 366 | 348 | 0.951 | 0.9491 | 0.959 | 0.932 | 0.939 | 0.949 | 0.966 |
| 0015 | 366 | 352 | 0.962 | 0.9592 | 0.975 | 0.949 | 0.957 | 0.949 | 0.966 |
| **0017 (published)** | 366 | 352 | 0.962 | **0.9605** | 0.967 | 0.949 | 0.957 | 0.963 | 0.966 |
| 0019 | 366 | 352 | 0.962 | 0.9606 | 0.967 | 0.941 | 0.966 | 0.963 | 0.966 |
| 0021 | 366 | 352 | 0.962 | 0.9602 | 0.967 | 0.949 | 0.948 | 0.970 | 0.966 |
| 0023 | 366 | 352 | 0.962 | 0.9602 | 0.967 | 0.941 | 0.957 | 0.977 | 0.958 |
| best_stg2 | 365 | 352 | 0.964 | 0.9626 | 0.971 | 0.957 | 0.949 | 0.977 | 0.958 |

Things to notice:

- Five checkpoints classify exactly 352 of 366 fish correctly. They differ only in which
  species the 14 errors fall on.
- Checkpoint 0019 is 0.0001 above 0017. To three decimals the two are tied at 0.961.
- `best_stg2` has the highest number in the table, but it was scored on 365 fish: one Coho
  Salmon had no detection matched to it and was left out of the score. Counted as the miss it
  is, `best_stg2` scores 0.9610 with the same 352 of 366 correct.

Every row can be recomputed from files in this repository:

```
python -m salmonid.eval.score_testset_v6 --preds provenance/v2.7/candidate_preds/preds_0019.csv
```

`candidate_preds/preds_0017.csv` is identical to `data/preds_v27_ckpt0017.csv`, the file every
held-out number and figure in the manuscript is computed from.

## Box placement check

Each candidate was also compared with hand-drawn boxes on a separate set of frames (42 at the
Credit River, 126 at the Ganaraska River). The check asks for a median overlap (IoU) of at
least 0.80 at each river and a box centre within 10% of the true height.

| Checkpoint | Credit median IoU | Ganaraska median IoU | Height ratio (both) | Result |
|---|---:|---:|---|---|
| 0013 | 0.729 | 0.787 | 1.012 / 1.014 | fails both on overlap |
| 0015 | 0.712 | 0.801 | 1.011 / 1.014 | Ganaraska passes |
| **0017** | **0.737** | **0.805** | 1.010 / 1.014 | Ganaraska passes, Credit fails |
| 0019 | 0.727 | 0.804 | 1.011 / 1.013 | Ganaraska passes |
| 0021 | 0.715 | 0.799 | 1.009 / 1.013 | fails both on overlap |
| 0023 | 0.717 | 0.805 | 1.010 / 1.013 | Ganaraska passes |
| best_stg2 | 0.710 | 0.782 | 1.010 / 1.014 | fails both on overlap |

**No v2.7 checkpoint passes this check at the Credit River.** The boxes are in the right place
(the height ratio is within 1.5% everywhere) but are looser than the hand-drawn ones.
Checkpoint 0017 has the tightest boxes at the Credit of any candidate and passes at the
Ganaraska. Looser boxes do not change which species a fish is called or which way it is
counted as moving; they would matter for measuring fish size, which this paper does not do.

## Why not the checkpoint with the best validation score

`best_stg2` is chosen by DEIMv2 on the validation pictures. In this project those are split
from the training pictures frame by frame, not fish by fish: about 98% of the validation fish
also have frames in training (`salmonid/data/prepare_dataset.py`, "Known limitations"). The
validation score therefore rewards memory of the training fish and keeps rising to the last
epoch (average precision 0.715 at epoch 17, 0.720 at 19, 0.724 at 23, 0.760 at the final
epoch 29), while the held-out score is flat from epoch 15 on. It was not used to choose the
model. `best_stg2` also has the loosest boxes at both rivers.

## What was recorded at the time

The lead author's note of 2026-08-04, written when the multi-year counts were re-run with
v2.7, gives the reasoning: 0017 is tied with 0019 on the held-out score, 0021 and 0023 are
0.960, `best_stg2` is nominally highest but is selected on the validation score, and 0017 is
the checkpoint the counts were run with.

## Where these numbers come from

| Number | Source | In this repository? |
|---|---|---|
| Species scores of all seven candidates | re-scored on 2026-11-10 with `salmonid.eval.score_testset_v6` from `provenance/v2.7/candidate_preds/preds_*.csv` and `data/test_set_v6_gt.json` | yes |
| The same scores, as printed by the original run | `results_366.txt` in the run's scoring folder on the project data drive (`config.EVAL_DIR`), written 2026-07-26 by `runners/original/run_v27_eval_topup.ps1`. They agree with the re-scored values to every printed digit. | no (external) |
| Box placement check | `results.txt` in the same folder, written 2026-07-25/26 by `runners/original/run_deimv2_fish_x_v27.ps1` | no (external) |
| Validation average precision per epoch | `log.txt` in the training output folder (`config.DEIM_OUTPUTS / deimv2_fish_x_v27`) | no (external) |
| 95% interval of the macro F1 | `data/classification_ci.csv` | yes |

A note on the first scoring pass: it read footage from a mirror disk that lacked 8 of the
364 test clips, so checkpoints 0013 and 0015 were first scored on 358 fish (0.955 and 0.965).
The second pass restored the 8 clips and re-scored every candidate on all 366; the table
above is the second pass. On the complete set, 0015 scores 0.959.

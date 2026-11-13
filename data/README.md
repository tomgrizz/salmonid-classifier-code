# Data dictionary

Every file in this folder is a small, published table. Together they are enough to recompute
the paper's classification score, its agreement statistics, and every figure that is drawn from
counts (`python run_tier1.py`). There is no imagery and no personal data here: the rows are
per-fish species labels and per-day counts.

**These files are never overwritten by the code.** Everything a run regenerates goes to
`output/` and is compared against the copies here.

Words used below:

- **track**: one fish followed through one video clip by the tracking software.
- **net count**: fish that moved upstream minus fish that moved downstream.
- **activity count**: fish that moved upstream plus fish that moved downstream.
- **model**: the published model, DEIMv2-X "v2.7" (training epoch 17), with BoT-SORT tracking.
- **technician**: the count recorded by technicians reviewing the same RiverWatcher video.
- **river / site**: `Ganaraska` (Corbett's Dam fishway, Port Hope) or `Credit` (Streetsville
  fishway, Mississauga).
- **species**: `Chinook`, `Coho`, `Atlantic`, `Rainbow Trout`, `Brown Trout` (short forms of the
  AFS common names Chinook Salmon, Coho Salmon, Atlantic Salmon, Rainbow Trout, Brown Trout).

Integrity: `CHECKSUMS.sha256` lists the SHA-256 of every file in this folder
(`sha256sum -c CHECKSUMS.sha256` from inside `data/`).

---

## 1. Held-out test set (species classification)

### `test_set_v6_gt.json`: the reviewer's answer for each of the 366 test fish

| Field | Meaning |
|---|---|
| `name`, `note` | Label for the set. |
| `tracks[].video_id` | The video clip, as its path inside the footage archive (`<river>/<season folder>/<batch folder>/<clip>.mp4`). |
| `tracks[].track_id` | The track number of the fish within the detection run the set was drawn from. With `video_id` it identifies one test fish. |
| `tracks[].species` | Species assigned by the reviewer (the reference answer). |
| `tracks[].tier` | `pristine` = the track had never been labelled before the test set was drawn (215 tracks, almost all Chinook and Rainbow Trout). `heldout` = a rare-species track taken from the previously labelled pool and withheld from training (151 tracks: most of the Atlantic, Brown Trout and Coho). |
| `tracks[].site` | River. |

366 tracks in 364 videos: Chinook 121, Rainbow Trout 65, Coho 60, Atlantic 60, Brown Trout 60.

### `test_set_v6_holdout.json`: the exclusion list that keeps the test set out of training

| Field | Meaning |
|---|---|
| `video_ids` | The 364 videos that contain a test fish. The published model's training set was built to exclude every image from these videos; the audit that checks this, and the one known exception (25 frames from one video), are in `docs/REPRODUCIBILITY_REVIEW.md` (item L2). |
| `tracks`, `items` | The same 366 `(video_id, track_id)` pairs in two layouts, kept for the earlier track-level exclusion. |

Read by `salmonid/holdout.py`.

### `preds_v27_ckpt0017.csv`: the published model's answer for each test fish

| Column | Meaning |
|---|---|
| `video_id`, `track_id` | The test fish (matches `test_set_v6_gt.json`). |
| `pred_species` | Species the model assigned to the track: the model's per-frame species scores are averaged over the frames matched to the track and the highest average wins. |

Scoring these 366 rows against the reference gives macro-F1 0.961 and 352 correct (manuscript
Table 1, Figure 2). Produced by `salmonid/eval/predict_testset_v6_from_store.py`.

### `preds_checkpoint0007.csv`: the superseded v2.6 model (comparison only)

Same columns. v2.6 scores 0.966 on this set, but 62 of the 366 test fish had been in its
training data, so that score is not a clean test and **no manuscript number uses this file**.
It is kept so the comparison can be re-run.

### `classification_ci.csv`: Table 1 with its confidence intervals

One row per species plus a `MACRO` row. `n_true` = test fish of that species; `n_pred` = fish
the model called that species; `tp`/`fp`/`fn` = correct calls, wrong calls of this species,
missed fish of this species. `precision` = tp / n_pred, `recall` = tp / n_true, `f1` = their
harmonic mean. `prec_lo/hi`, `rec_lo/hi` = Wilson 95% interval. `f1_lo/hi` = bootstrap 95%
interval over tracks (10,000 resamples, seed 12345). Written by
`salmonid/results/classification_ci.py`.

---

## 2. Daily and seasonal counts (model against technician), 2020 to 2025

### `canonical_daily.csv`: one row per river, species and day (3,775 rows)

| Column | Meaning |
|---|---|
| `year`, `river`, `species`, `date` | The day (`date` is ISO `YYYY-MM-DD`). |
| `model_net`, `tech_net` | Net count that day, model and technician. Can be negative (more fish moved down than up). |
| `model_act`, `tech_act` | Activity count that day (never negative). |

A day appears for a river and species when either the model or the technician recorded at
least one passage of that species there. Days on which neither did are absent, not zero.
The model's passage date is the date the video file was last modified (the counter writes one
file per passage event); see `docs/COUNTING_METHOD.md`.

### `canonical_summary.csv`: one row per river, species and season (60 rows)

| Column | Meaning |
|---|---|
| `model_net`, `tech_net`, `model_act`, `tech_act` | Season totals of the daily columns. |
| `net_pct_err`, `act_pct_err` | 100 × (model − technician) / technician. Blank if the technician total is zero. |
| `net_r`, `act_r` | Pearson correlation between the model's and the technician's daily counts over that season's rows in `canonical_daily.csv`. Blank if undefined. |
| `active_days` | Days with a non-zero technician net count (or, if none, with a non-zero model net count). |
| `recovered` | `True` for the 2020 Ganaraska season, whose video was recovered from an archive that stores each passage in several shorter clips. The model over-counts that season, so it is shown in the figures but **excluded from every agreement statistic**. |

Both files are assembled by `salmonid/results/build_canonical_results.py` from the per-season
count tables below (`net_rule_sweep/<year>_up50.csv`, `activity_sweep/`, `recovered_2020/`).
`run_tier1.py` step 3 rebuilds them and checks the result against these copies.

### `activity_sweep/<year>_activity.csv` (6 files) and `recovered_2020/` (2 files)

The per-season daily tables the canonical tables are assembled from, with the same columns as
the rule tables in section 4 (`date`, `site`, `species`, `model`, `tech`).
`activity_sweep/` holds the activity counts for each season. `recovered_2020/` holds the net
and activity counts for the recovered 2020 Ganaraska footage, which was processed separately;
for that season and river these rows replace those of the regular tables. All were written by
`salmonid/count/corpus_countr_from_store.py` (Tier 2: stored tracks and technician records).

---

## 3. Agreement statistics (all recomputed by `python run_tier1.py`)

| File | What it holds | Feeds |
|---|---|---|
| `concordance_ccc.csv` | Lin's concordance correlation coefficient (`ccc`) between model and technician counts, at `level` = `daily` or `annual`, per species and river: `n` (river-days or seasons), `pearson_r`, bootstrap 95% interval `ci_lo`/`ci_hi` (independent days, 2,000 resamples), accuracy factor `cb` (CCC = r × Cb), McBride `band`. | Table 5, Figure 6 |
| `agreement_bland_altman.csv` | Bland-Altman statistics for daily Chinook net counts, per river and `pooled`: mean difference `bias`, constant limits `loa_lo`/`loa_hi`; ratio (log-scale) limits `ratio_bias`/`ratio_lo`/`ratio_hi` with the number of zero days dropped; and how the spread of the difference grows with the count (`spread_*`, `prop_*`). | Section 3.5, Figure S3 |
| `agreement_ba_by_count_band.csv` | The same difference summarized within bands of the day's mean count: `days`, `mean_diff`, `sd_diff`, `median_ratio` (model ÷ technician). | Section 3.5 |
| `agreement_ccc_block.csv` | Daily CCC with three bootstrap intervals: `iid_*` (independent days), `strat_*` (days resampled within season), `block_*` (moving blocks of `block_len` consecutive days, 4,000 resamples). `spearman_rho` is the rank correlation. `autocorr_widening` = block width ÷ stratified width. | Section 3.5, ρ column of Table 5 |
| `agreement_interannual.csv` | Season totals, model against technician, per species and river: `r2`, `spearman_rho`, `ccc` with bootstrap interval, `cb`. | Section 3.3 |

---

## 4. The upstream/downstream rule

A track counts as upstream if it moves at least 50 pixels in the upstream direction and as
downstream if it moves at least 300 pixels the other way; otherwise it is not counted in the
net. These files hold the evidence for that choice.

| File | What it holds |
|---|---|
| `net_rule_sweep/<year>_<rule>.csv` (48 files) | Daily net counts under eight candidate rules for each season 2020 to 2025. Columns `date`, `site`, `species`, `model`, `tech`. Rule suffix `up<U>` = upstream threshold U px with the downstream threshold at 300; `u<U>_d<D>` = upstream U, downstream D. `up50` is the published rule. |
| `net_rule_sweep/net_rule_sensitivity.json` | Season percentage errors under each rule, as plotted in the sensitivity figure. |
| `net_rule_loyo.csv` | Leave-one-year-out result: for each held-out season, the rule that the other seasons would have chosen and its error on the held-out season. `view` names the error definition (`pooled_all5` is the one the manuscript reports). |

Feeds Section 3.4. Recomputed by `salmonid/count/net_rule_loyo.py` and
`salmonid/figures/render_sensitivity_figure.py`.

---

## 5. Other

| File | What it holds |
|---|---|
| `figure_checksums.json` | SHA-256 of each data-driven figure as built for the manuscript, with the library versions used. `run_tier1.py` reports whether your regenerated figures are byte-identical. |
| `paper/` | Small per-video and per-fish tables behind the remaining manuscript tables; see `paper/README.md`. `paper/expected/` holds reference copies of what the `salmonid/paper/` scripts write, for the automated comparison. |
| `CHECKSUMS.sha256` | SHA-256 of every file in this folder. |
| `sample/` | Reserved for a small demonstration bundle of stored detections; empty for now (see `sample/README.md`). |

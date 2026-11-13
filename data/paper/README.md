# data/paper: small tables behind the manuscript's non-count statistics

The files in `data/` hold the daily and annual counts. The files here hold everything else the
manuscript's statistics rest on: which species the model and a person gave the same video or
the same fish, what the model did with each technician-reviewed video, and the per-track
answers of the three classifiers compared in Table S4. Each file is the smallest table from
which the published number can be recomputed without video, model weights or the
technicians' raw records. The scripts that read them are in `salmonid/paper/`, and
`docs/PAPER_TO_CODE.md` maps every manuscript number to its command.

Two kinds of file are here:

* **Input tables** (read by the scripts). Rebuilding one needs external (Tier-2) files; the
  command is given under "Rebuilt by".
* **Published outputs** (written by the scripts). They are committed so that a re-run can be
  compared with the values the manuscript quotes.

Species are written with the short keys used throughout the repository: `Chinook`, `Coho`,
`Atlantic`, `Rainbow Trout`, `Brown Trout` (Chinook Salmon, Coho Salmon, Atlantic Salmon,
Rainbow Trout, Brown Trout). Dates are ISO `YYYY-MM-DD`. All files are UTF-8 CSV.

## How videos and tracks are identified

No file here contains a person's name, a user name, an e-mail address or a machine path.

* `key` (per-video tables): the last two parts of the video's path, lower-cased: the
  recording-session folder and the clip number, e.g. `25  09  15  14  42/244.mp4`. The folder
  name is the session start, `YY  MM  DD  hh  mm`.
* `video_id` (blind-review tables): `River/2025/<download batch>/<session folder>/<clip>.mp4`.
* `video_hash` (held-out and Table S4 tables): the first 8 hexadecimal digits of the MD5 of
  the video's full path. A hash is used because some 2020 archive folder names in the full
  path contain staff first names. To link a row to `data/test_set_v6_gt.json`, apply the same
  hash to that file's `video_id` (`hashlib.md5(video_id.encode()).hexdigest()[:8]`) and match
  on `(video_hash, track_id)`.

## Input tables

### `pervideo_2025_credit.csv` (5,330 rows), `pervideo_2025_ganaraska.csv` (8,662 rows)

One row per 2025 video in which the technician logged exactly one record naming one of the
five species and the model returned at least one counted track.

| Column | Meaning |
|---|---|
| `river` | `Credit` or `Ganaraska` |
| `key` | video identifier (see above) |
| `date` | date of the video (file timestamp) |
| `tech_species` | species the technician recorded |
| `tech_count` | number of fish on the technician's record (fish). `1.0` = a single fish; rows with a larger count are kept so that the older "one record" denominator can be reproduced, and are dropped for the manuscript's single-fish denominator |
| `n_tracks` | number of tracks the model formed in the video (overlap-based tracker; Methods 2.4) |
| `dom_species` | species of the model's dominant track, the track with the most detections |
| `dom_conf_top` | the dominant track's mean score for that species (0 to 1) |
| `dom_conf2_species` | the dominant track's second-choice species (used only by the confidence-floor sweep) |

* Supports: Section 3.2 (per-video agreement, kappa, one-track and multi-track subsets, Coho
  Salmon and Atlantic Salmon paragraphs) and Table 2 (Videos, Recall, Precision).
* Read by: `python -m salmonid.paper.pervideo_agreement`.
* Rebuilt by: `python -m salmonid.paper.pervideo_agreement --from-external --write-tables`
  from the 2025 technician review exports and the published model's 2025 per-track
  diagnostics. A rebuild on the original workstation is byte-identical.
* Note: the technician's species and count in these tables are derived from the OMNR review
  exports, which are not themselves redistributed.

### `stepB_credit.csv` (9,237 rows), `stepB_ganaraska.csv` (17,160 rows)

One row per 2025 video that appears in the technician's video-linked record.

| Column | Meaning |
|---|---|
| `river` | `Credit` or `Ganaraska` |
| `key` | video identifier (see above) |
| `tech_fish` | `1` if any technician record for the video names one of the five species; `0` if the technician logged no fish (a false trigger) |
| `state` | the model's outcome in the counting pass: `counted track` (at least one track of three or more processed frames), `detections, no track` (something detected, no track of that length), `no detection` |
| `n_tracks_any` | tracks of any length (count) |
| `n_tracks_counted` | tracks of three or more processed frames (count) |

* Supports: Section 3.2, "Few technician-logged fish went uncounted" (96.7% and 96.6%; 2.4%
  and 1.8%; 0.9% and 1.7%; 6 of 195 and 22 of 250).
* Read by: `python -m salmonid.paper.detection_outcomes`.
* Rebuilt by: `python -m salmonid.paper.detection_outcomes --from-external --write-tables`
  (technician exports, the 2025 video list, the counting-pass tracks). Byte-identical on the
  original workstation.

### `blind_v27_clips.csv` (155 rows)

One row per clip of the 2025 blind review.

| Column | Meaning |
|---|---|
| `video_id` | clip identifier (see above) |
| `site` | `Credit` or `Ganaraska` |
| `n_truth` | fish the reviewer identified in the clip (count) |
| `n_pred` | tracks the published model returned for the clip (count) |
| `npair` | fish that can be paired, the smaller of the two counts |
| `match` | paired fish whose species agree, pairing like with like first |
| `truth` | the reviewer's species, separated by `;` (`unsure` = could not identify) |
| `pred` | the model's track species, separated by `;` |
| `dom` | species of the model's dominant track |

* 150 of the 155 clips hold at least one paired fish; in the other five the reviewer saw no
  fish. The manuscript's "150 clips" are those 150.
* Supports: Section 3.2, blind-review paragraph (196 of 208; 94.2%; kappa 0.90; by species).
* Read by: `python -m salmonid.paper.blind_review`.
* Rebuilt by: `python -m salmonid.paper.blind_review --from-external` (the reviewer's record
  and the published model's 2025 tracks). Byte-identical on the original workstation.

### `backbone_track_predictions.csv` (2,184 rows = 546 tracks x 4 classifiers)

One row per test set, classifier and track.

| Column | Meaning |
|---|---|
| `set` | `bench` = the manuscript's 366-track held-out set; `clips` = an earlier 180-track test set (not used in the manuscript) |
| `model` | `resnet50`, `dinov2_vitb14`, `dinov3_vitb16`; `check_twostage_v6` rows are a check of the inference harness only |
| `video_hash`, `track_id` | track identifier (see above) |
| `site` | `Credit` or `Ganaraska` |
| `year` | year of the video |
| `session` | recording-session folder, `YY  MM  DD  hh  mm` (gives the month) |
| `gt_species` | species given by the reviewers |
| `tier` | held-out tier (`pristine` = sampled from never-labelled tracks; `heldout` = drawn from the labelled pool); blank for `clips` |
| `n_frames` | frames of the track the classifier was shown (up to 10) |
| `pred_species` | the classifier's species (highest mean class probability) |
| `p_Chinook` ... `p_Brown_Trout` | mean class probability over those frames (0 to 1) |
| `in_classifier_training` | `1` if the track's images were in the training or validation list of these classifiers. The 127 `bench` tracks flagged `1` are excluded from Table S4 |

* Supports: Table S4; Section 3.1 (last paragraph); Sections 2.3, 3.2 and 4.1.
* Read by: `python -m salmonid.paper.backbone_comparison`.
* Rebuilt by: `python -m salmonid.paper.backbone_comparison --build-table --predictions <file>`
  from the inference run's output and the classifiers' training lists. The inference run
  itself (three classifier checkpoints) is not part of this repository.

### `heldout_adjacent_clip_tracks.csv` (366 rows)

One row per held-out track.

| Column | Meaning |
|---|---|
| `video_hash`, `track_id` | track identifier (see above) |
| `species`, `tier`, `site` | as in `data/test_set_v6_gt.json` |
| `adjacent_training_images` | training images from the clip numbered one before or one after the track's clip, same recording folder (count) |
| `adjacent_same_species_images` | of those, images carrying a box of this track's species (count) |
| `adjacent_same_species` | `1` if that count is above zero (76 tracks) |

* Supports: Methods 2.4 (76 tracks); Sections 3.1, 4.1 and 4.5 (macro-F1 0.961 to 0.950;
  Brown Trout 0.966 to 0.935; Atlantic Salmon 0.957 to 0.953).
* Read by: `python -m salmonid.paper.heldout_adjacent_clips`.
* Rebuilt by: `python -m salmonid.paper.heldout_adjacent_clips --build-table` from the
  published model's training list.

### `table_s1_ganaraska_annual_counts.csv`, `table_s2_credit_annual_counts.csv`

The machine-readable form of manuscript Tables S1 and S2: annual upstream totals by species,
as published by the Lake Ontario Management Unit in its annual report to the Great Lakes
Fishery Commission. Rows are `All Salmon & Trout`, the five species, `Season start`,
`Season end`, `Days monitored` and `Notes`; columns are the years 2017 to 2025 and `Total`.
Counts are fish. `All Salmon & Trout` includes fish not identified to species, so it normally
exceeds the sum of the five species rows.

* Supports: Tables S1 and S2; Section 2.1 (annual ranges); Section 2.2 (unidentified
  salmonids as a share of the Chinook Salmon count; 9,748 against 10,533).
* Read by: `python -m salmonid.paper.annual_report_checks`, `python -m salmonid.paper.count_tables`.
* Two cells are flagged in the Table S2 caption for confirmation against the annual report
  (the 2021 Atlantic Salmon value and the 2024 `All Salmon & Trout` value).

## Published outputs (for comparison with a re-run)

| File | Written by | Columns |
|---|---|---|
| `pervideo_summary_v27.csv` | `pervideo_agreement` | `name` (`<River>_fish` = single-fish denominator, the one the manuscript reports; `<River>_row` = one record of any count), `n`, `correct`, `acc`, `kappa`, `k_lo`, `k_hi` (95% interval, 10-day moving-block bootstrap), `pe` (chance agreement), `n_1to1`, `acc_1to1`, `kappa_1to1` (videos resolved into one track), `n_multi`, `share_multi`, `acc_multi` (two or more tracks), `n_disagree`, `n_chin_coho`, `share_dis_sept`, `share_ref_chinook` |
| `pervideo_per_species_v27.csv` | `pervideo_agreement` | `name`, `species`, `n_tech` (technician videos), `n_called` (videos the model called that species), `tp` (both), `recall`, `precision` |
| `blind_v27_pairs.csv` | `blind_review` | `ref` (reviewer's species), `pred` (model's species), `clip` |

All proportions are 0 to 1.

## Map layers for Figure 1 (`spatial/`)

| Layer | Committed | Source and licence |
|---|---|---|
| `spatial/ganaraska_full.json` | yes | Ganaraska River and North Ganaraska River centre lines. OpenStreetMap data, Overpass API export of 2026-06-10. (c) OpenStreetMap contributors, Open Database License (ODbL) 1.0, https://www.openstreetmap.org/copyright. The ways carry the tag `source = CanVec 6.0 - NRCan` |
| `credit_river.geojson` | **no** | Credit River centre line (425 segments with a stream-order attribute). Its source and licence were not recorded with the file, so it is not redistributed. An equivalent line can be exported from OpenStreetMap (ODbL) or the Ontario Hydro Network watercourse layer (Open Government Licence - Ontario) |
| `lake_ontario.gpkg` | **no** | Lake Ontario shoreline polygon. Source and licence not recorded; not redistributed. An equivalent polygon is in Natural Earth 1:10m lakes (public domain) or the Ontario Hydro Network waterbody layer |
| `ne_cache/` | no | Natural Earth 1:50m countries and lakes for the locator inset, public domain. Downloaded by `salmonid.paper.fig1_study_map` on first run |

To draw Figure 1, place the two missing layers beside `ganaraska_full.json` (or point
`SALMONID_SPATIAL_DIR` at a folder holding all three) and run
`python -m salmonid.paper.fig1_study_map`. With the original layers the figure is
byte-identical to the published one; with substitute layers the shoreline and river detail
will differ slightly.

## What is deliberately not here

* The technicians' 2025 review exports and counter exports (third-party OMNR records). Only
  the derived tables above are included. Table S3 is computed from the exports by
  `python -m salmonid.paper.technician_record_2025` (Tier 2).
* The reviewer's fish-by-fish blind-review record (`reference_recount.csv`); the per-clip
  table above is derived from it.
* The inference output of the backbone comparison in its original form, whose video paths
  contain staff first names; the hashed table above replaces it.

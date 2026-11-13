# Paper-to-code map

This page answers one question for every number, table and figure in the manuscript: **which
command in this repository produces it, from which file, and does a re-run give the printed
value?** It follows the manuscript from the abstract to the supplement, using the manuscript's
current numbering (Tables 1 to 5 and S1 to S4; Figures 1 to 7 and S1 to S3).

It was checked on 2026-11-10 against the working copy of the manuscript of that date. Every
command marked Tier 1 below was run on that day, on a clean copy of the committed files, and
the value it gave is written in the Status column.

## How to read the tables

**Tier.** Tier 1 means the command runs from the small files committed in this repository
(`data/`, `data/paper/`, `provenance/`): no video, no model weights, no graphics card. Tier 2
means it also needs something that cannot be stored here: the trained model, the fishway
video, the stored detections, or the technicians' 2025 review records (see
`docs/DATA_AVAILABILITY.md`).

**Status.**

| Status | Meaning |
|---|---|
| `REPRODUCES` | The command was run and gives the printed value. The regenerated value is stated. |
| `MISMATCH` | The command was run and gives a different value, or gives the printed value only under a reading the text does not state. Both are stated. |
| `EXTERNAL` | The code is in this repository but needs Tier-2 inputs. Where it was run on the original workstation, the result is stated. |
| `NO CODE` | Nothing in this repository computes it. Where the number came from is stated. |
| `NOT COMPUTED` | A value taken from the literature, a site description, or a setting; no calculation. |

**Commands.** Each command has a short label used in the tables. All are run from the
repository root. `python run_tier1.py` runs every Tier-1 command below (C1 to C8 and the P
commands that need no external files) in order and checks each result against the committed
copy under `data/`.

| Label | Command | Reads | Writes |
|---|---|---|---|
| C1 | `python -m salmonid.eval.score_testset_v6` | `data/test_set_v6_gt.json`, `data/preds_v27_ckpt0017.csv` | printout (per-species table, confusion matrix) |
| C2 | `python -m salmonid.results.classification_ci` (add `--tier heldout` or `--tier pristine` for one part of the set) | same | `output/tier1/classification_ci.csv` |
| C3 | `python -m salmonid.results.compute_concordance` | `data/canonical_daily.csv`, `data/canonical_summary.csv` | `output/tier1/concordance_ccc.csv`, `figures/fig_concordance.*`, `figures/fig_bland_altman.*` |
| C4 | `python -m salmonid.results.agreement_stats` | same | `output/tier1/agreement_{bland_altman,ba_by_count_band,ccc_block,interannual}.csv` |
| C5 | `python -m salmonid.count.net_rule_loyo` | `data/net_rule_sweep/*.csv` | `output/tier1/net_rule_loyo.csv` |
| C6 | `python -m salmonid.figures.render_sensitivity_figure` | `data/net_rule_sweep/*.csv` | `figures/fig_net_rule_sensitivity.png` |
| C7 | `python -m salmonid.figures.render_manuscript_figures` | `data/test_set_v6_gt.json`, `data/preds_v27_ckpt0017.csv`, `data/canonical_daily.csv` | `figures/fig_confusion_heldout.*`, `figures/fig_daily_net_2025_*.*` |
| C8 | `python -m salmonid.figures.render_hero_regulation` | `data/canonical_daily.csv`, `data/canonical_summary.csv` | `figures/fig_hero_performance.*`, `fig_species_phenology.*`, `fig_ganaraska_sanctuary.*` |
| P1 | `python -m salmonid.paper.heldout_kappa` | `data/test_set_v6_gt.json`, `data/preds_v27_ckpt0017.csv` | `output/paper/heldout_kappa.csv` |
| P2 | `python -m salmonid.paper.heldout_adjacent_clips` | `data/paper/heldout_adjacent_clip_tracks.csv` + the two files above | `output/paper/heldout_adjacent_clip_sensitivity.csv` |
| P3 | `python -m salmonid.paper.backbone_comparison` | `data/paper/backbone_track_predictions.csv` | `output/paper/table_s4.csv`, `backbone_mcnemar.csv` |
| P4 | `python -m salmonid.paper.pervideo_agreement` | `data/paper/pervideo_2025_{credit,ganaraska}.csv` | `output/paper/pervideo_summary_v27.csv`, `pervideo_per_species_v27.csv` |
| P5 | `python -m salmonid.paper.detection_outcomes` | `data/paper/stepB_{credit,ganaraska}.csv` | `output/paper/detection_outcomes_2025.csv` |
| P6 | `python -m salmonid.paper.blind_review` | `data/paper/blind_v27_clips.csv` | `output/paper/blind_v27_pairs.csv`, `blind_v27_summary.csv` |
| P7 | `python -m salmonid.paper.count_tables` | `data/canonical_daily.csv`, `data/canonical_summary.csv`, `data/paper/table_s2_credit_annual_counts.csv` | `output/paper/table2_count_columns.csv`, `table3_chinook_multiyear.csv`, `table4_three_species.csv`, `fig4_panel_annotations.csv`, `chinook_daily_difference_by_year.csv`, `quiet_day_sums.csv`, `sanctuary_window_share.csv` |
| P8 | `python -m salmonid.paper.daily_totals_and_peak_days` | `data/canonical_daily.csv` | `output/paper/daily_totals_and_peak_days.csv` |
| P9 | `python -m salmonid.paper.annual_report_checks` | `data/paper/table_s1_*.csv`, `table_s2_*.csv` | `output/paper/annual_report_checks.csv` |
| T1 | `python -m salmonid.paper.technician_record_2025` (Tier 2) | 2025 technician review exports | `output/paper/table_s3_review_record.csv` |
| T2 | `python -m salmonid.paper.fig1_study_map` (Tier 2) | map layers (`data/paper/README.md`) | `output/paper/fig1_study_map.{png,pdf}` |
| T3 | `python -m salmonid.figures.fig_species_morphology` (Tier 2) | fish cutout library | `figures/fig_species_morphology.{png,tiff}` |
| T4 | `python -m salmonid.figures.fig_bbox_grid` (Tier 2) | fish cutout library, video | `figures/bbox_sample_grid.{png,tiff}` |
| T5 | `python -m salmonid.data.audit_coco_holdout_leak --coco <training list> --report-adjacent` (Tier 2) | training list of the published model | printout |
| T6 | `python -m salmonid.count.run_corpus` (Tier 2) | stored detections and tracks, technician files | `output/canonical_daily.csv`, `output/canonical_summary.csv`, the rule sweeps |
| T7 | `python -m salmonid.detect.track_from_store --tracker bytetrack ...` (Tier 2) | stored 2025 Ganaraska River detections | a track file for the tracker comparison |

P4, P5 and P6 accept `--from-external`, and P2 and P3 accept `--build-table`, to rebuild
their committed input table from the Tier-2 files. P4, P5 and P6 were rebuilt that way on the
original workstation and the rebuilt tables were byte-identical to the committed ones.

**Where the daily counts come from.** Every count statistic below starts from
`data/canonical_daily.csv` and `data/canonical_summary.csv`. Those two files are themselves
the output of the Tier-2 chain (model, video, tracking, technician files; command T6). The
Tier-1 statuses below therefore say "given the committed daily counts, the statistic
reproduces"; regenerating the daily counts is `EXTERNAL` and was not re-run for this map.

---

## Abstract

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| Abstract | 240,000 videos, two rivers, six years | none | the 13 video lists of the Tier-2 runs | 2 | `NO CODE`. The manuscript repository's record gives 246,070 videos across the stored runs; no script here counts them. |
| Abstract | macro F1 0.96 | C1 | printout, `MACRO F1` | 1 | `REPRODUCES` 0.961 |
| Abstract | daily Chinook Salmon r = 0.94 to 0.99 | P7 | `table3_chinook_multiyear.csv` | 1 | `REPRODUCES` 0.941 to 0.993 (2021 to 2025, both rivers) |
| Abstract | concordance 0.94 to 0.97 (three abundant species) | C3 | `concordance_ccc.csv`, `ccc`, daily rows | 1 | `REPRODUCES` 0.943 to 0.967 |
| Abstract | daily totals across species differ by a median of three fish | P8 | `daily_totals_and_peak_days.csv`, `median_abs_diff` | 1 | `REPRODUCES` 3.0 (1,702 days) |
| Abstract | Chinook Salmon annual totals differ by 6% (Ganaraska River) and 10% (Credit River), 2022 to 2025 | P7 | `table4_three_species.csv`, `mean_abs_net_error_pct` | 1 | `REPRODUCES` 5.50 and 10.36. Note the first is exactly on the rounding boundary (5.50), printed as 6. |
| Abstract | year-to-year r-squared at least 0.98 | C4 | `agreement_interannual.csv`, `r2` | 1 | `REPRODUCES` 0.998 and 0.989 |
| Abstract | nearly a year of manual review; processed in four days | none | none | n/a | `NO CODE`. 240,000 videos / 1,000 per technician day = 240 workdays; the four days come from the run logs (see Section 4.7). |

## 2. Methods

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 2.1 | drainage areas, coordinates, distances, installation dates | none | none | n/a | `NOT COMPUTED` (site description) |
| 2.1 | Ganaraska River: Chinook Salmon 5,000 to 28,000 a year; Coho Salmon 1,300 to 4,100; Brown Trout 76 to 399; Atlantic Salmon 0 to 29 | P9 | `annual_report_checks.csv` | 1 | `REPRODUCES` 5,039 to 28,294; 1,325 to 4,090; 76 to 399; 0 to 29 |
| 2.1 | Ganaraska River: Rainbow Trout 5,500 to 9,800 a year | P9 | same | 1 | `MISMATCH`. Table S1 gives 1,172 (2020) to 9,751. The printed lower end, 5,520 (2023), is the minimum only if 2020 (deployment delayed past the spring run) is left out; the text does not say so. |
| 2.1 | passage can exceed 1,500 fish per day | P7 | printout, "Busiest single day" | 1 | `REPRODUCES` (largest technician net daily total 2,143, all species) |
| 2.1 | Credit River: 810 to 9,748 Chinook Salmon a year; Atlantic Salmon typically 15 to 40 | P9 | same | 1 | `REPRODUCES` 810 to 9,748; Atlantic Salmon 5, 18, 15, 49, 26, 40, 23, 35 by year |
| Figure 1 | study map | T2 | `fig1_study_map.png`, `.pdf` | 2 | `EXTERNAL`. Two of three map layers are not redistributed (source and licence not recorded; see `data/paper/README.md`). With the original layers the PNG and PDF are byte-identical to the published figure. |
| 2.2 | video 1280 x 960 px, about 10 frames per second, 10 to 20 s | none | none | n/a | `NOT COMPUTED` (instrument description) |
| 2.2 | unidentified salmonids equal to 27% to 36% of the Ganaraska River Chinook Salmon count in 2020 to 2022, 3% or less from 2023 | P9 | `annual_report_checks.csv`, `pct_of_chinook` | 1 | `REPRODUCES` 35.6%, 26.6%, 26.6%; then 2.2%, 0.0%, 2.5% |
| 2.2 | 2025 Credit River event-based reference 10,533 Chinook Salmon, 8% more than the published 9,748 | P7 | printout, "two references" | 1 | `REPRODUCES` 10,533; +8.1% |
| 2.2, Table S3 | review periods, records and holding totals | T1 | `table_s3_review_record.csv` | 2 | `EXTERNAL`. Run on the original workstation: every cell of Table S3 reproduces (13,376; 2,611 with 638 holding; 14,248; 26,406; 7,380 with 2,917 holding). |
| 2.3 | 960 x 960 px input, optimiser, loss, augmentation | none | `configs/deimv2/deimv2_fish_x_v27.yml`, `provenance/v2.7/config_snapshot/` | n/a | `NOT COMPUTED` (settings; the files are the record) |
| 2.3 | training set: 45,249 images, 52,444 annotated fish; Rainbow Trout 18,341, Coho Salmon 12,954, Chinook Salmon 12,177, Brown Trout 6,327, Atlantic Salmon 2,067; 578 juvenile | `python -m salmonid.provenance.record_run_manifest` | `provenance/v2.7/run_manifest.json`, `data.train` | 2 | `EXTERNAL`. The committed record of the training run gives exactly these counts. |
| 2.3 | 11,644 synthetic boxes; 3,906 model-labelled companion fish | none | none | 2 | `NO CODE`. A direct count of the training list gives 11,644 boxes on composite images and 3,906 images of the companion-fish source; no script here prints them. |
| 2.3 | backbone comparison: three classifiers trained on the same 40,638 images | P3 `--build-table` | printout | 2 | `EXTERNAL`. Run on the original workstation: 40,638 training images. |
| 2.3 | every third frame; 40 px header strip masked | none | `salmonid/detect/detect_store.py` (`--frame-skip 3`), `config.HEADER_MASK_PX` | n/a | `NOT COMPUTED` (settings) |
| 2.3 | track criteria: detections kept at 0.30; a track starts above 0.50 and needs confirmation; at least three processed frames | none | `salmonid/detect/track_from_store.py`, `config.MIN_FRAMES` | n/a | `NOT COMPUTED` (settings) |
| 2.3 | about 900 videos per GPU-hour | none | none | n/a | `NO CODE`. From the timing of the stored runs; no script. |
| 2.4 | 366 tracks, 60 to 121 per species; 215 never-labelled tracks; 151 from the labelled pool (53 Atlantic Salmon, 59 Brown Trout, 39 Coho Salmon) | C1; C2 `--tier heldout` | printouts | 1 | `REPRODUCES` 366; 121, 60, 60, 65, 60; 215 and 151; 53, 59, 39 |
| 2.4 | 25 hand-annotated frames from the video of one held-out Atlantic Salmon track entered training | T5 | printout, "UNKEYED" | 2 | `EXTERNAL`. Run on the original workstation: 25 pictures from 1 held-out clip. |
| 2.4 | for 76 tracks the adjacent clip contributed same-species training images | P2 (flag built by T5's rule) | `data/paper/heldout_adjacent_clip_tracks.csv` | 1 | `REPRODUCES` 76 of 366 (T5 run on the original workstation also gives 76) |
| 2.4 | kappa intervals: 2,000 resamples; tracks, clips, or 10-day blocks | P1, P6, P4 | see Section 3 | 1 | `REPRODUCES` (the three scripts use 2,000 resamples and those units) |
| 2.5 | upstream at 50 px or more, downstream at 300 px or more, first three against last three frames | none | `config.NET_UP_THRESH`, `config.NET_DOWN_THRESH`, `salmonid/count/` | n/a | `NOT COMPUTED` (the rule; its validation is Section 3.4) |
| 2.6 | CCC bootstrap, 2,000 resamples | C3 | `concordance_ccc.csv` | 1 | `REPRODUCES` (2,000). The moving-block repeat in C4 uses 4,000 resamples, which the Methods do not state. |
| 2.6 | Python 3.11.9, SciPy, pandas, NumPy, Matplotlib | none | `requirements.txt`, `environment/` | n/a | `NOT COMPUTED` |

## 3.1 Classification accuracy

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.1, Table 1 | macro F1 0.961 (0.939 to 0.980) | C2 | `classification_ci.csv`, `MACRO` row | 1 | `REPRODUCES` 0.961 [0.939, 0.980] |
| Table 1 | per species n, precision, recall, F1 with intervals; macro precision 0.964, recall 0.958 | C2 | `classification_ci.csv` | 1 | `REPRODUCES`, every cell (the file is identical to `data/classification_ci.csv`). The macro precision and recall are the means of the five rows: 0.964 and 0.958. |
| 3.1 | per-species F1 from 0.949 (Coho Salmon) to 0.967 (Chinook Salmon); Atlantic Salmon 0.957 | C1 | printout | 1 | `REPRODUCES` 0.949, 0.967, 0.957 |
| 3.1 | accuracy 0.962, 352 of 366 | C1 | printout, `acc` | 1 | `REPRODUCES` 0.962; 352 on the diagonal |
| 3.1 | kappa 0.951 (0.926 to 0.975); chance agreement 0.22 | P1 | `heldout_kappa.csv`, `kappa`, `kappa_boot_lo/hi`, `p_e` | 1 | `REPRODUCES` 0.951 [0.926, 0.975]; 0.223 |
| 3.1, Figure 2 | 6 of 14 errors are the Chinook-Coho pair; four Coho Salmon called Chinook Salmon; four Atlantic Salmon called Rainbow or Brown Trout; three Brown Trout; one Chinook Salmon called Atlantic Salmon | C1 | printout, confusion matrix | 1 | `REPRODUCES` (2 + 4; 4; 3 + 1; 1 + 2; 1) |
| Figure 2 | confusion matrix | C7 | `figures/fig_confusion_heldout.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical to the published figure) |
| 3.1 | accuracy 97.7% on the 215 never-labelled tracks, 94.0% on the 151 others | C1; C2 `--tier heldout` | printouts | 1 | `REPRODUCES` 0.977; 0.940 |
| 3.1, 4.1, 4.5 | without the 76 tracks: macro F1 0.961 to 0.950; Brown Trout 0.966 to 0.935; Atlantic Salmon 0.957 to 0.953 | P2 | `heldout_adjacent_clip_sensitivity.csv` | 1 | `REPRODUCES` 0.9605 to 0.9501; 0.966 to 0.935; 0.957 to 0.953 |
| Figure 3 | body-shape plate | T3 | `figures/fig_species_morphology.*` | 2 | `EXTERNAL`. Run on the original workstation: the PNG is byte-identical to the published one. |
| 3.1, Table S4 | 239 tracks; 46, 22, 11 misidentified; by river 40, 16, 6 of 109 and 6, 6, 5 of 130; false Atlantic Salmon 32, 3, 1 of 232 (all Credit River, 103); 127 excluded; 7 Atlantic Salmon | P3 | `table_s4.csv` | 1 | `REPRODUCES`, every cell |
| 3.1 | discordant tracks 39 and 4 (DINOv3), 28 and 4 (DINOv2); exact McNemar p < 0.001 | P3 | `backbone_mcnemar.csv` | 1 | `REPRODUCES` 39 against 4, p = 3.1e-08; 28 against 4, p = 1.9e-05 |
| 3.1 | 24 of the 32 were Chinook Salmon from September and October | P3 | printout, "by true species and month" | 1 | `REPRODUCES` 10 in September + 14 in October |
| 3.1 | the classifier predictions themselves | none | `data/paper/backbone_track_predictions.csv` | 2 | `NO CODE`. The three classifiers and the inference harness are from the earlier two-stage system and are not in this repository; the committed table is their output. |
| Figure S1 | representative detections | T4 | `figures/bbox_sample_grid.*` | 2 | `EXTERNAL`. Run on the original workstation: PNG and TIFF byte-identical to the published ones. |

## 3.2 Operational agreement with technician counts (2025)

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.2, 4.4, 5 | same species on 98.6% of single-fish videos at the Credit River (5,131 of 5,203) and 90.6% at the Ganaraska River (7,396 of 8,160) | P4 | `pervideo_summary_v27.csv`, rows `*_fish`, `acc`, `correct`, `n` | 1 | `REPRODUCES` 5,131 of 5,203 (0.9862); 7,396 of 8,160 (0.9064) |
| 3.2 | Chinook Salmon 88% and 81% of these videos; chance agreement 79% and 68% | P4 | `share_ref_chinook`, `pe` | 1 | `REPRODUCES` 0.884, 0.807; 0.789, 0.680 |
| 3.2 | kappa 0.93 (0.84 to 0.97) and 0.71 (0.54 to 0.86) | P4 | `kappa`, `k_lo`, `k_hi` | 1 | `REPRODUCES` 0.935 [0.84, 0.97]; 0.707 [0.54, 0.86] |
| 3.2 | 612 of the 764 Ganaraska River disagreements are the Chinook-Coho pair; 86% in September | P4 | `n_chin_coho`, `n_disagree`, `share_dis_sept` | 1 | `REPRODUCES` 612 of 764; 0.86 |
| 3.2, 4.4 | one-track videos: 99.1% and 94.7% (kappa 0.96 and 0.82); about a fifth split; on those 96.7% and 75.2% | P4 | `acc_1to1`, `kappa_1to1`, `share_multi`, `acc_multi` | 1 | `REPRODUCES` 0.9914, 0.9472 (0.957, 0.817); 0.216, 0.209; 0.9670, 0.7519 |
| 3.2 | counted track in 96.7% of 9,042 Credit River and 96.6% of 16,910 Ganaraska River videos; nothing detected 2.4% and 1.8%; short of three frames 0.9% and 1.7% | P5 | `detection_outcomes_2025.csv` | 1 | `REPRODUCES` 96.7% of 9,042; 96.6% of 16,910; 2.4%, 1.8%; 0.9%, 1.7% |
| 3.2 | false triggers: a track counted in 6 of 195 and 22 of 250 | P5 | same | 1 | `REPRODUCES` 6 of 195; 22 of 250 |
| 3.2 | over 1,000 fish on the busiest Ganaraska River days | P7 | printout, "Busiest single day" | 1 | `REPRODUCES` (maximum 2,143) |
| 3.2, 4.4 | blind review: 150 clips; 196 of 208 paired fish (94.2%; 90.2% to 96.7%) | P6 | `blind_v27_summary.csv` | 1 | `REPRODUCES` 196 of 208; 94.2% [90.2, 96.7]; 150 clips with a paired fish |
| 3.2 | blind review kappa 0.90 (0.84 to 0.95) | P6 | same | 1 | `REPRODUCES` 0.899 [0.84, 0.95]. The script as saved in the manuscript repository gives a zero-width interval because of a programming slip (it passed the table's built-in `clip` method instead of the column named `clip`); the version here uses the column. `--as-saved` shows the original behaviour. |
| 3.2 | reviewer identified 213 fish; two could not be identified | P6 | printout | 1 | `REPRODUCES` 213; 2 |
| 3.2 | blind review by species: Chinook Salmon 95% (125), Rainbow Trout 98% (54), Coho Salmon 77% (13), Atlantic Salmon 100% (6), Brown Trout 100% (8); Atlantic Salmon interval 61% to 100% | P6 | `blind_v27_summary.csv` | 1 | `REPRODUCES` 119/125, 53/54, 10/13, 6/6, 8/8; [61, 100] |
| 3.2, 4.4 | review of 30 disagreements: 12 technician errors, 15 model errors, 3 unresolved | none | none | n/a | `NO CODE`. A manual review made during development against an earlier model version; the record is a note in the research repository, not a data file. |
| 3.2, Table 2 | 2025 daily correlations 0.94 to 0.99 (Chinook Salmon, Rainbow Trout, Ganaraska River Coho Salmon) | P7 | `table2_count_columns.csv`, `daily_r` | 1 | `REPRODUCES` 0.941 to 0.990 |
| 3.2 | Ganaraska River 2025 totals within about 4% (Chinook Salmon) and 3% (Rainbow Trout); Credit River Chinook Salmon 13% below the event-based reference, 6% below the published total | P7 | `table2_count_columns.csv`; printout "two references" | 1 | `REPRODUCES` +3.6%, +3.3%; -12.6%, -5.6% |
| 3.2 | Atlantic Salmon held-out precision 0.982; operational precision 84% on 19 Credit River videos | C2; P4 | `classification_ci.csv`; `pervideo_per_species_v27.csv` | 1 | `REPRODUCES` 0.982; 16 of 19 (0.842) |
| 3.2 | Ganaraska River Coho Salmon recall 54%, precision 72%; 439 missed, 414 called Chinook Salmon, 409 in September, 269 split into two or more tracks | P4 | `pervideo_per_species_v27.csv`; printout "Coho misses (single-fish denominator)" | 1 | `REPRODUCES` 0.537, 0.716; 439; 414; 409; 269 |
| 3.2 | Ganaraska River Coho Salmon daily r 0.981; 2025 total 20% below | P7 | `table2_count_columns.csv` | 1 | `REPRODUCES` 0.981; -20.0% |
| 3.2 | Credit River: 16 single-fish Coho Salmon videos, 11 identified, four Chinook Salmon videos also called Coho Salmon (recall 69%, precision 73%) | P4 | `pervideo_per_species_v27.csv`; floor-sweep printout | 1 | `REPRODUCES` 16; 11; 4; 0.688; 0.733 |
| 2.3 | "No per-species confidence floor is applied" | P4 | printout, "Coho floor sweep" | 1 | `REPRODUCES` the evidence: at a floor of 0.60 Credit River precision rises to 1.00 and recall falls to 0.44; no floor raises Ganaraska River precision |
| 3.2 | crowding: 0.98 and 1.00 tracks per technician Chinook Salmon in single-fish videos; 0.67 and 0.63 in videos of four or more fish ("about a third fewer") | none | none | 2 | `NO CODE`. These ratios were carried over from the text written for an earlier model version and were not re-derived on the published model; no script that computes them was found in this repository, the manuscript repository or the research repository. |
| Table 2 | Videos, Recall, Precision (ten rows) | P4 | `pervideo_per_species_v27.csv`, rows `*_fish`: `n_tech`, `recall`, `precision` | 1 | `REPRODUCES`, every cell (6,585 / 95.7% / 93.5%; 4,597 / 99.1% / 99.6%; 948 / 53.7% / 71.6%; 16 / 68.8% / 73.3%; 1; 23 / 69.6% / 84.2%; 363 / 96.1% / 86.2%; 539 / 97.4% / 92.9%; 263 / 89.0% / 77.5%; 28 / 78.6% / 81.5%) |
| Table 2 | Daily r, Days, 95% CI, % error (ten rows) | P7 | `table2_count_columns.csv` | 1 | `REPRODUCES`, every cell (Days 76, 70, 63, 20, 3, 21, 154, 109, 87, 26; all p < 0.001). The older script `python -m salmonid.paper.correlation_stats` uses days with a technician count (74, 68, 60, ...) and gives intervals that differ by up to 0.03; it is kept for the record and is not the source of the printed table. |
| Table 2 caption | Ganaraska River Atlantic Salmon: model 6, technician 3; one video; three days | P7; P4 | same | 1 | `REPRODUCES` 6 and 3; 1; 3 |
| Figure S2 | cumulative 2025 counts by species, both rivers | C7 | `figures/fig_daily_net_2025_{credit,ganaraska}.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical to the published figures) |

## 3.3 Multi-year agreement

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.3, Table 3 | Chinook Salmon % error and daily r by river and year (twelve cells); 0.94 to 0.99 from 2021; 0.81 in 2020 at the Ganaraska River | P7 | `table3_chinook_multiyear.csv` | 1 | `REPRODUCES`, every cell (+58.6 / 0.810; +33.9 / 0.961; +10.9 / 0.989; -4.9 / 0.961; -2.6 / 0.984; +3.6 / 0.978; Credit +4.9 / 0.993; -2.2 / 0.951; -16.7 / 0.966; -6.8 / 0.993; -5.3 / 0.964; -12.6 / 0.941) |
| 3, 3.3, 4, 4.2, 5, Table 4 | mean absolute net error and median daily r, 2022 to 2025 (six rows): 6% / 0.98; 10% / 0.96; 14% / 0.98; 41% / 0.77; 11% / 0.99; 9% / 0.96 | P7 | `table4_three_species.csv` | 1 | `REPRODUCES` 5.50 / 0.981; 10.36 / 0.965; 13.70 / 0.979; 41.09 / 0.773; 10.71 / 0.987; 8.62 / 0.958 |
| 3.3 | each of the three most recent Ganaraska River years within about 5% | P7 | `table3_chinook_multiyear.csv` | 1 | `REPRODUCES` -4.9, -2.6, +3.6 |
| Table 4 caption | Credit River Coho Salmon 31 to 65 fish a year since 2023, against 431 in 2022 | P9 | `annual_report_checks.csv` (Table S2) | 1 | `REPRODUCES` 65, 60, 31; 431 |
| 3.3, Table 2 | Ganaraska River Brown Trout over-counted by 41%; precision 78% | P7; P4 | `table2_count_columns.csv`; `pervideo_per_species_v27.csv` | 1 | `REPRODUCES` +41.1%; 0.775 |
| 3.3 | Chinook Salmon inter-annual r-squared 0.998 (Ganaraska River, 2021 to 2025) and 0.989 (Credit River, 2020 to 2025); Spearman's rho 1.00; inter-annual CCC 0.990 and 0.976 | C4 | `agreement_interannual.csv`, `r2`, `spearman_rho`, `ccc` | 1 | `REPRODUCES` 0.998, 0.989; 1.0, 1.0; 0.990, 0.976 |
| 3.3 | pooled rank correlation rho = 0.91, n = 33 annual counts, p < 0.001 | C4 | printout, section 4 | 1 | `REPRODUCES` 0.91; 33; p = 1e-13 |
| 3.3 | Rainbow Trout inter-annual r-squared 0.67 and 0.74 | C4 | `agreement_interannual.csv` | 1 | `REPRODUCES` 0.669, 0.739 |
| Figure 4 | annual counts, model against technician | C8 | `figures/fig_hero_performance.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical) |
| Figure 4 caption | Ganaraska River Chinook Salmon r-squared 0.96 over six years | P7 | `fig4_panel_annotations.csv` | 1 | `REPRODUCES` 0.964 |
| Figure 5 | passage timing by species | C8 | `figures/fig_species_phenology.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical) |

## 3.4 Directional rule and tracker

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.4 | 50 / 300 px rule: mean absolute error 3.6% over 2022 to 2025, all species and both rivers pooled | C5 | printout; `net_rule_loyo.csv` | 1 | `REPRODUCES` 3.56% |
| 3.4 | "Symmetric or lower-threshold alternatives gave 10% to 26%" | C5 | printout, first table | 1 | `MISMATCH`. The seven alternatives in the sweep give 10.5% to 33.6%. The four that lower the downstream threshold or use equal thresholds of 150 px or less give 10.5% to 25.8%; the three that raise the upstream threshold give 18.1% (150 / 300 px), 28.9% (250 / 300 px) and 33.6% (300 / 300 px, a symmetric rule). |
| 3.4 | the same rule chosen in every leave-one-year-out fold; held-back error 1.0% to 7.1% | C5 | `net_rule_loyo.csv`, `heldout_err_pct` | 1 | `REPRODUCES` 4 of 4 folds; 1.04% to 7.08% |
| 3.4 | BoT-SORT Chinook Salmon 2025 Ganaraska River error +3.6% | P7 | `table2_count_columns.csv` | 1 | `REPRODUCES` +3.6% |
| 3.4 | ByteTrack +3.5%; overlap-only linker +30% | T7, then the counting step of T6; `salmonid.count.corpus_countr_from_store` for the overlap-only linker | none committed | 2 | `EXTERNAL`. Not re-run for this map. The manuscript repository's record of the original run gives 18,247 (+3.5%) and 22,976 (+30.3%) against 17,638. No committed table holds these two counts. |

## 3.5 Count agreement beyond correlation

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.5, 5, Table 5 | daily CCC 0.94 to 0.97; accuracy factor at least 0.988 | C3 | `concordance_ccc.csv`, `ccc`, `cb` | 1 | `REPRODUCES` 0.943 to 0.967; smallest 0.988 |
| Table 5 | Days, Pearson r, CCC, 95% CI, accuracy factor, band (six rows) | C3 | `concordance_ccc.csv`, daily rows | 1 | `REPRODUCES`, every cell (identical to `data/concordance_ccc.csv`) |
| Table 5 | Spearman's rho (six rows) | C4 | `agreement_ccc_block.csv`, `spearman_rho` | 1 | `REPRODUCES` 0.970, 0.971, 0.953, 0.523, 0.800, 0.673 |
| 3.5 | lower limit 0.90 or above for five of six series; Credit River Coho Salmon down to 0.83 | C3 | `concordance_ccc.csv`, `ci_lo` | 1 | `REPRODUCES` 0.951, 0.923, 0.904, 0.831, 0.919, 0.900 |
| 3.5 | moving-block bootstrap widened five of six intervals by factors of about 1.1 to 2.4 | C4 | `agreement_ccc_block.csv`, `vs_published` | 1 | `REPRODUCES` 1.07, 1.40, 0.82, 1.09, 1.90, 2.37 |
| 3.5 | Rainbow Trout lower bounds from 0.919 to 0.881 (Ganaraska River) and 0.900 to about 0.79 (Credit River) | C3; C4 | `ci_lo`; `block_lo` | 1 | `REPRODUCES` 0.919 to 0.881; 0.900 to 0.791 |
| 3.5 | Spearman's rho 0.970 and 0.971 (Chinook Salmon); 0.523 (Credit River Coho Salmon); 0.800 and 0.673 (Rainbow Trout) | C4 | `agreement_ccc_block.csv` | 1 | `REPRODUCES` |
| 3.5, Figure S3 | Chinook Salmon mean difference -1.5 fish per day; limits -122 to +119; 857 days | C3; C4 | printout; `agreement_bland_altman.csv`, pooled row | 1 | `REPRODUCES` -1.54; -122.1 to +119.0; 857 |
| 3.5 | 6.4 fewer per day at the Credit River, 2.9 more at the Ganaraska River | C4 | `agreement_bland_altman.csv`, `bias` | 1 | `REPRODUCES` -6.40; +2.94 |
| 3.5 | Ganaraska River +22.8 fish per day in 2021; -1.1 over 2022 to 2025 | P7 | `chinook_daily_difference_by_year.csv` | 1 | `REPRODUCES` +22.80; -1.07 |
| 3.5, abstract | daily total differs by a median of three fish (1,702 days) | P8 | `daily_totals_and_peak_days.csv` | 1 | `REPRODUCES` 3.0; 1,702 |
| 3.5 | median ratio 1.00 on quiet days (up to about 20 fish at the Ganaraska River, 50 at the Credit River) | C4 | `agreement_ba_by_count_band.csv`, `median_ratio` | 1 | `REPRODUCES` 1.000 in the 1-4 and 5-19 bands (Ganaraska River) and the 1-4, 5-19 and 20-49 bands (Credit River) |
| 3.5 | summed over the same days, the model was 27% above at the Ganaraska River (one to two fish per day) and equal at the Credit River | P7 | `quiet_day_sums.csv` | 1 | `MISMATCH`. The text does not say how a day's size is measured. Measured by the mean of the two counts (as the count bands are): Ganaraska River +26.6% (1.4 fish per day), Credit River +2.8%. Measured by the technician's count: +27.5% (1.6 per day) and +0.4%. No single reading gives both "27%" and "equal". |
| 3.5 | Ganaraska River median ratio 1.11 between 20 and 199 fish, 0.99 on the busiest days; Credit River 0.97 (50 to 199) and 0.95 (above 200) | C4 | `agreement_ba_by_count_band.csv` | 1 | `REPRODUCES` 1.113 and 1.117; 0.987; 0.968; 0.949 |
| 3.5, 5 | the 127 days (15%) carrying two thirds of the Chinook Salmon run: model 7% below; within river 7% (Ganaraska River) and 11% (Credit River) | P8 | `daily_totals_and_peak_days.csv`, `model_over_tech` | 1 | `REPRODUCES` 127 of 857 (14.8%), 0.933; 0.931; 0.888 |
| Figure 6 | concordance plots | C3 | `figures/fig_concordance.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical) |
| Figure S3 | Bland-Altman plot; caption values -1.5, -122 to +119, n = 857, +2.9, -6.4 | C3; C4 | `figures/fig_bland_altman.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical; values as above) |

## 3.6 The Ganaraska Chinook Salmon sanctuary

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 3.6 | 85% to 98% of the run inside 1 September to 14 October; about 90% in a typical year | P7 | `sanctuary_window_share.csv`, `model_pct` | 1 | `REPRODUCES` 85.0% to 97.9%; median 89% |
| 3.6 | within 4 percentage points of the technician's in every year | P7 | `model_minus_technician_points` | 1 | `REPRODUCES` largest gap 4.0 points (2020) |
| 3.6 | 44-day window; closure dates | none | none | n/a | `NOT COMPUTED` (regulation) |
| Figure 7 | cumulative run and window share | C8 | `figures/fig_ganaraska_sanctuary.*` | 1 | `REPRODUCES` (PNG and PDF byte-identical). C8 draws the percentages but does not print them; P7 prints them with the same formula. |

## 4. Discussion and 5. Conclusions

Every statistic in the Discussion and Conclusions repeats one from the Results and is mapped
there (macro F1 0.961; 6% and 10%; 98.6% and 90.6%; 96.2%; 75.2%; 94.2%; 12 and 15 of 30;
0.957, 0.966, 0.953, 0.935; concordance 0.94 to 0.97; the 15% of days and 7%). The rows below
are the ones that appear only in the Discussion.

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| 4.1 | figures quoted from other studies (mean average precision above 90% and 80%; 91%; 3.3%) | none | none | n/a | `NOT COMPUTED` (literature) |
| 4.5 | technicians logged 24% of Credit River events and 40% of Ganaraska River events as holding | T1 | `table_s3_review_record.csv`, `holding_pct` | 2 | `EXTERNAL`. Run on the original workstation: 24.4% and 39.5%. |
| 4.5 | every species scored on at least 60 held-out tracks | C1 | printout | 1 | `REPRODUCES` (minimum 60) |
| 4.7 | 240 technician workdays at about 1,000 videos per day; four days of processing; about 260 hours summed across parallel processes | none | none | n/a | `NO CODE`. Arithmetic on the video total and the program's review rate; the processing time is from run logs. |
| 4.7 | stocking totals and recruitment shares | none | none | n/a | `NOT COMPUTED` (literature) |

## Supplementary tables

| Where | Value as printed | Command | Output (file / column) | Tier | Status |
|---|---|---|---|---|---|
| Table S1 | Ganaraska River annual upstream totals, 2017 to 2025 | P9 (checks the column totals) | `data/paper/table_s1_ganaraska_annual_counts.csv` | 1 | `NOT COMPUTED` (published annual-report values). The five species column totals in the file equal the printed totals. |
| Table S2 | Credit River annual upstream totals, 2018 to 2025 | P9 | `data/paper/table_s2_credit_annual_counts.csv` | 1 | `NOT COMPUTED` (published values). Column totals agree. The 2024 All S & T value is 25 below the sum of its species columns, as the caption notes. |
| Table S3 | 2025 review record | T1 | `table_s3_review_record.csv` | 2 | `EXTERNAL` (see Section 2.2 above) |
| Table S4 | backbone comparison | P3 | `table_s4.csv` | 1 | `REPRODUCES` (see Section 3.1 above) |

---

## Summary

The map has 118 rows. A row is one statement, table block or figure; a table whose cells all come from one command is one row.

| Status | Items |
|---|---:|
| `REPRODUCES` | 85 |
| `MISMATCH` | 3 |
| `EXTERNAL` | 10 |
| `NO CODE` | 8 |
| `NOT COMPUTED` | 12 |
| **Total** | **118** |

Of the 106 rows that carry a computed result (all but `NOT COMPUTED`), 85 reproduce from
committed files. Of the 10 `EXTERNAL` rows, 9 were also run or checked on the original workstation,
where the Tier-2 inputs exist, and gave the printed values; the one not re-run is the tracker
comparison (T7). Regenerating the daily counts from video (T6) was not re-run either. It has
no row of its own because every count row is stated "given the committed daily counts".

Figures: all eleven figure files (Figures 1 to 7, S1, S2a, S2b, S3) were regenerated and match
the published files. Eight of them (Figures 2, 4, 5, 6, 7, S2a, S2b, S3) regenerate at Tier 1
and are byte-identical; Figures 1, 3 and S1 need Tier-2 inputs.

The three `MISMATCH` rows are wording, not arithmetic: Section 2.1 (Rainbow Trout range
leaves out 2020 without saying so), Section 3.4 ("10% to 26%" leaves out two rules of the
sweep) and Section 3.5 (the quiet-day sums of 27% and "equal" come from two different ways of
sizing a day).

## Gaps, ranked

Ranked by how much a reader checking the paper against the journal's code-and-data policy
would be affected.

1. **Crowding ratios in Section 3.2 (0.98, 1.00, 0.67, 0.63; "about a third fewer fish").**
   No code anywhere, and the numbers predate the published model. They support a stated
   finding. Either re-derive them on the published model (it needs the late-2025
   web-application records and the counting-pass tracks) and add the script, or remove the
   sentence.
2. **The daily counts themselves are not regenerated at Tier 1.** Tables 2 to 5 and Figures
   4 to 7, S2 and S3 all start from `data/canonical_daily.csv`. The code that builds it is
   here (T6) but needs the stored detections, the video for tracking, and the technician
   files for every year. This is inherent in the size of the inputs; the Data Availability
   statement should say where the stored detections and the technician daily records can be
   obtained.
3. **Tracker comparison in Section 3.4 (+3.5% and +30%).** The code is here (T7 and the
   counting step) but the two resulting counts are in no committed table, so the sentence
   cannot be checked without the stored detections. Committing the two small daily-count
   files of those runs would make it Tier 1.
4. **Backbone-comparison inference (Table S4 inputs).** Table S4 recomputes from the
   committed predictions, but the three classifiers that made the predictions, and the
   harness that ran them, are not in this repository.
5. **Three wording mismatches** (Section 2.1 Rainbow Trout range; Section 3.4 "10% to 26%";
   Section 3.5 quiet-day sums). Each is a sentence that the data support only under a
   reading the sentence does not state. They need a change to the text, not to the code.
6. **The 30-disagreement review (12 / 15 / 3).** A manual review against an earlier model
   version, with no data file. It is described as such in the text; a reader cannot check it.
7. **Counts with no script:** the video total (240,000), the synthetic and companion-fish
   box counts (11,644 and 3,906), the processing rate (900 videos per GPU-hour) and the
   processing time (four days, 260 hours). Each is a simple count or a log reading.
8. **Figure 1 map layers.** Two of the three layers cannot be redistributed because their
   source was not recorded. Redrawing the map from open layers (Natural Earth, OpenStreetMap)
   would remove the dependency.
9. **The technicians' 2025 review records** (Table S3, the 24% and 40% holding shares, and
   the rebuild of three committed tables) are third-party records and are not included.

## Notes for maintainers

* Modules marked "WRITTEN FOR THE DEPOSIT" in their docstring (`count_tables`,
  `heldout_adjacent_clips`, `annual_report_checks`, `technician_record_2025`) implement a
  calculation that had been done by hand or in a scratch session that was not kept. The
  others are ports of saved scripts with the calculations unchanged.
* `salmonid.paper.correlation_stats` is the original correlation script, kept unchanged. It
  does not reproduce the Days and interval columns of Table 2 as now printed; P7 does.
* P1 prints a macro-F1 interval of 0.940 to 0.979 beside kappa. Table 1's interval (0.939 to
  0.980) comes from C2, which resamples separately; both are correct for their own resampling.
* The committed tables in `data/paper/` identify 2020 videos by a hash because some archive
  folder names contain staff first names. The same full paths are present in
  `data/test_set_v6_gt.json`, `data/test_set_v6_holdout.json` and the two `data/preds_*.csv`
  files.

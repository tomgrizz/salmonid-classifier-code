# `salmonid/results/`: the count tables and the agreement statistics

These scripts turn the per-season count files into the two master tables the manuscript
reads from, and compute the statistics that say how well the model's counts agree with the
technicians'. None of them runs the model or opens a video.

Tier 1 means the script runs from the files committed in `data/`. Tier 2 means it needs
external files that are not in git.

## Count tables and agreement statistics (in the order they are run)

| File | What it does | Feeds | Tier | Status |
|---|---|---|---|---|
| `build_canonical_results.py` | Stacks the per-season count files into one daily table (`canonical_daily.csv`) and one season-summary table (`canonical_summary.csv`). | Tables 2 (count columns), 3, 4 and 5; Figures 4 to 7; Sections 3.2, 3.3, 3.5, 3.6 | 2 | Published path |
| `compute_concordance.py` | Computes Lin's concordance (a 0-to-1 agreement score) between model and technician counts, daily and annual, and draws the concordance and Bland-Altman figures. | Table 5; Figure 6; Figure S3 | 1 | Published path |
| `agreement_stats.py` | Recomputes agreement in ways suited to daily counts: Bland-Altman limits per river and as a ratio, intervals that keep neighbouring days together, and agreement between years. | Sections 3.3 and 3.5; the Spearman column of Table 5; the Figure S3 caption | 1 | Published path |
| `classification_ci.py` | Puts 95% intervals on the held-out precision, recall and F1 of each species and on the macro-F1. | Table 1 intervals (macro-F1 0.961, 0.939 to 0.980) | 1 | Published path |

## Blind review

A reviewer re-identified the fish in a sample of 2025 clips without being shown the model's
species. The two scripts here drew that sample and first scored it, both on the tracks of
the earlier model (v2.6). The figures in the manuscript (Section 3.2: 196 of 208 paired
fish, 94.2%, kappa 0.90) are for the published model (v2.7) and are computed from the same
reviewer record by the blind-review recomputation in `salmonid/paper/`.

| File | What it does | Feeds | Tier | Status |
|---|---|---|---|---|
| `build_reference_sample.py` | Drew the sample of clips: grouped by river, time of season and crowding, drawn from every group, with extra clips for rare species. 206 clips were drawn; the reviewer completed 155. | The clip sample behind the blind review (Sections 2.4, 3.2, 4.4) | 2 | Supporting (record of how the sample was drawn) |
| `score_reference_sample.py` | Scores the reviewer's record against the earlier model's tracks: species and direction agreement, missed and duplicate fish, net-count comparison. | No manuscript number (it reports 180 of 190 tracks, 94.7%, for the earlier model) | 2 | Supporting check; superseded for the published figures |

Points a reader should know about the blind-review sample:

* The sample contains only clips in which the model counted at least one fish, and it
  over-represents clips with rare species on purpose. Its agreement rate is not an estimate
  of the season-wide rate.
* Clips added for a rare species were chosen because the earlier model reported that
  species in them. All six fish the reviewer identified as Atlantic Salmon are in such clips, so the
  agreement on them shows how often a reported Atlantic Salmon was confirmed, not how many
  Atlantic Salmon the model finds.
* Of the 51 drawn clips that were not reviewed, 48 are late-phase Ganaraska River clips (62
  drawn there, 14 reviewed).
* The `weight` stored with each clip is not a valid sampling weight (the rare-species
  top-up breaks it) and nothing uses it. The "reweighted" line printed by the scorer
  corrects only for sampled clips that were not reviewed.
* The draw is seeded and was reproduced exactly on the authors' machine, but it also depends
  on the order in which files are listed and on the footage files being present, so the
  same result is not guaranteed elsewhere. The reviewed clips are listed in
  `data/paper/blind_v27_clips.csv`.

## Removed

`compile_manuscript_results.py` generated a results document from fixed text written for the
superseded model and wrote it to a path outside this repository. It was removed; nothing
imported it, and the git history keeps it. `docs/PAPER_TO_CODE.md` is the current map from
each manuscript number to the command that reproduces it.

# `salmonid/paper/`: the remaining manuscript tables and in-text statistics (stage F)

`salmonid/results/` holds the main agreement statistics. This folder holds everything else the
manuscript quotes: the 2025 per-video comparisons, the blind review, the kappa statistics, the
count tables, the backbone comparison, and a few checks on numbers in the text.

All but two run from committed files, and all of those are run and checked by
`python run_tier1.py` (steps 10 to 18). They read the small tables in `data/paper/`
(documented in `data/paper/README.md`) and write to `output/paper/`.

| File | What it computes | Manuscript item | Tier |
|---|---|---|---|
| `pervideo_agreement.py` | For each 2025 video with one technician-logged fish: did the model name the same species? Agreement, Cohen's kappa, the breakdown of disagreements. | Section 3.2; Table 2 per-video columns | 1 |
| `detection_outcomes.py` | What the model did with every technician-reviewed 2025 video: counted a track, detected but did not track, or detected nothing; and how often a no-fish video produced a count. | Section 3.2 | 1 |
| `blind_review.py` | Agreement between the model and a reviewer who named the fish in 150 clips without seeing the model's answer. | Section 3.2 | 1 |
| `heldout_kappa.py` | Cohen's kappa for the held-out test set. | Section 3.1 | 1 |
| `heldout_adjacent_clips.py` | The held-out score with the 76 test fish removed whose neighbouring clip supplied a same-species training image. | Sections 3.1, 4.1, 4.5 | 1 |
| `backbone_comparison.py` | Three classifiers that differ only in their backbone, compared on the same tracks, with exact McNemar tests. | Table S4 | 1 |
| `count_tables.py` | The count columns of Table 2, Tables 3 and 4, the annotations printed on Figure 4, the by-year daily differences, the sanctuary-window shares. | Tables 2, 3, 4; Figures 4, 7; Section 3.5 | 1 |
| `daily_totals_and_peak_days.py` | The median daily difference across species, and agreement on the busiest days. | Abstract; Section 3.5; Conclusions | 1 |
| `annual_report_checks.py` | Ranges and shares quoted from the published annual technician totals. | Sections 2.1, 2.2; Tables S1, S2 | 1 |
| `correlation_stats.py` | An earlier calculation of correlation intervals, kept for the record; `count_tables.py` is the source for Table 2. | (none) | 1 |
| `technician_record_2025.py` | Record counts by river and review period, and the share logged as holding. | Table S3; Section 4.5 | 2 |
| `fig1_study_map.py` | The study-area map. | Figure 1 | 2 |
| `greyscale_proofs.py` | Greyscale copies of the figures, to check they read without colour. | (none) | 1 |
| `_common.py` | Shared arithmetic: kappa, Wilson interval, bootstrap. | | |

Several modules can also rebuild their input table from the external (Tier-2) files with a
`--from-external` or `--build-table` option; each module's header says what that needs.

Four modules (`count_tables`, `heldout_adjacent_clips`, `annual_report_checks`,
`technician_record_2025`) were written when the deposit was prepared, to implement
calculations the manuscript describes but for which no saved script existed. Each says so in
its header. They reproduce the printed values.

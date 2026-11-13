# Expected outputs of the `salmonid/paper/` scripts

Each file here is what one of the manuscript-analysis scripts in `salmonid/paper/` writes when
run on the committed input tables in `data/paper/` and `data/`. They are kept so that
`python run_tier1.py` can compare a fresh run against them value by value. Every random step in
those scripts is seeded, so a re-run is expected to match exactly.

They are outputs, not inputs: nothing reads these files except the comparison in
`run_tier1.py`. `docs/PAPER_TO_CODE.md` says which manuscript item each one supports, and
`data/paper/README.md` documents the input tables.

| File | Written by (`python -m salmonid.paper.<name>`) |
|---|---|
| `blind_v27_summary.csv` | `blind_review` |
| `detection_outcomes_2025.csv` | `detection_outcomes` |
| `heldout_kappa.csv`, `heldout_kappa_per_species.csv` | `heldout_kappa` |
| `heldout_adjacent_clip_sensitivity.csv` | `heldout_adjacent_clips` |
| `table_s4.csv`, `backbone_mcnemar.csv` | `backbone_comparison` |
| `table2_count_columns.csv`, `table3_chinook_multiyear.csv`, `table4_three_species.csv`, `fig4_panel_annotations.csv`, `chinook_daily_difference_by_year.csv`, `sanctuary_window_share.csv`, `quiet_day_sums.csv` | `count_tables` |
| `daily_totals_and_peak_days.csv` | `daily_totals_and_peak_days` |
| `annual_report_checks.csv` | `annual_report_checks` |

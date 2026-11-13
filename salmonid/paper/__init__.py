"""paper sub-package.

The analyses behind manuscript numbers, tables and figures that are not produced by the
pipeline packages (``eval``, ``count``, ``results``, ``figures``). Each module is one analysis
and runs as ``python -m salmonid.paper.<name>``. ``docs/PAPER_TO_CODE.md`` lists, for every
statistic, table and figure in the manuscript, the command that reproduces it.

Run from committed files (Tier 1):
  heldout_kappa              Cohen's kappa on the 366 held-out tracks (Section 3.1)
  heldout_adjacent_clips     held-out score without tracks next to a training clip (Section 3.1)
  backbone_comparison        Table S4 and its paired tests (Sections 2.3, 3.1)
  pervideo_agreement         2025 per-video species agreement, Table 2 per-video columns (Section 3.2)
  detection_outcomes         technician-fish videos by model outcome; false triggers (Section 3.2)
  blind_review               blind-review agreement on the published model's tracks (Section 3.2)
  count_tables               Tables 2 (count columns), 3 and 4; text statistics of Sections 3.3 to 3.6
  correlation_stats          the original daily-correlation and inter-annual script (superseded day count)
  daily_totals_and_peak_days median daily difference; the busiest days (Sections 3.5, 5)
  annual_report_checks       ranges and shares quoted from Tables S1 and S2 (Sections 2.1, 2.2)

Need external files (Tier 2):
  technician_record_2025     Table S3 (needs the technician review exports)
  fig1_study_map             Figure 1 (needs two map layers that are not redistributed here)
  greyscale_proofs           greyscale copies of the figures for the journal's legibility check

Several Tier-1 modules have a ``--from-external`` or ``--build-table`` option that rebuilds
their committed input table from the Tier-2 files; ``data/paper/README.md`` describes each table.
"""

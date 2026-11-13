"""results sub-package: the master count tables and the agreement statistics.

``README.md`` in this folder has one line per file. In the order they are run:

    build_canonical_results    stacks the per-season count files into the two master tables,
                               canonical_daily.csv and canonical_summary.csv (Tier 2)
    compute_concordance        Lin's concordance between model and technician counts; the
                               concordance and Bland-Altman figures (Tier 1)
    agreement_stats            per-river Bland-Altman limits, intervals that respect
                               day-to-day dependence, inter-annual agreement (Tier 1)
    classification_ci          95% intervals for the held-out precision, recall and F1 (Tier 1)

Blind review (a reviewer re-identified fish in a sample of 2025 clips without seeing the
model's species). These two modules drew and first scored that sample on the earlier model's
tracks; the figures in the manuscript were recomputed for the published model in
``salmonid/paper/``:

    build_reference_sample     drew the stratified sample of clips (Tier 2)
    score_reference_sample     scores the reviewer's record against the earlier model (Tier 2)
"""

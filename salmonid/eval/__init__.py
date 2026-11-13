"""eval sub-package: scoring species identification, and the development checks.

Two groups of modules; ``README.md`` in this folder has one line per file.

Held-out species score (the manuscript's Table 1 and Figure 2), in the order they are run:

    build_holdout_testset_queue      chose the never-labelled tracks a reviewer then labelled
    finalize_testset_v6              assembled the frozen 366-track test set (two tiers)
    predict_testset_v6_from_store    one species per test track from the model's detections
    score_testset_v6                 precision, recall, F1 and the confusion matrix (Tier 1)

The first two record how the test set was made and cannot be re-run without the private
labelling database; the last one runs from the committed files.

Development checks ("gates"). These gave a pass or fail during model development. None of
them produces a manuscript number, and the published model (v2.7) does not pass all of them
(each module's docstring says what it returned):

    deimv2_bbox_gate        are the model's boxes in the right place? (applies to v2.7)
    count_r_gate            do the daily counts follow the technicians' counts?
    score_tracked           per-video species presence under different track settings
    bbox_regression_gate    box check for the retired two-stage detectors (legacy)
    corpus_year_gate        season-by-season check of the retired two-stage runs (legacy)
"""

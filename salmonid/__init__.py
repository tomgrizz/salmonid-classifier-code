"""salmonid: the code behind the manuscript's fish-identification and counting results.

The published model is **v2.7**: a single DEIMv2-X network (checkpoint of training epoch 17)
that finds each fish in a video frame and scores its species in one pass. On the frozen
366-track held-out test set it reaches a macro-averaged F1 of 0.961. An earlier model, v2.6,
is superseded and is not used for any number in the manuscript; where it is mentioned in this
package it is for comparison or because a file was produced with it.

Sub-packages, in the order the work is done:

    data/        build the training pictures, keeping the held-out test fish out   (Tier 2)
    train/       launch the training run of the published model                    (Tier 2)
    detect/      run the model on video and join detections into one track per fish (Tier 2)
    eval/        score species identification on the held-out test set (Tier 1);
                 development checks ("gates"); how the test set was built
    count/       turn tracks into daily counts and compare them with the technicians
    results/     the master count tables, the agreement statistics and their intervals
    paper/       the remaining manuscript tables and checks, one module per analysis
    figures/     the manuscript figures
    provenance/  the record of what a training run used
    util/        two small helpers (a file-naming rule, optional notifications)

Two modules sit at the top level: ``config`` (every file location and fixed constant) and
``holdout`` (the guard that keeps the held-out test set out of training).

"Tier 1" means a step runs from the small files committed in ``data/``; "Tier 2" means it
needs large external files (model weights, footage, stored detections) that are not in git.
See the top-level ``README.md``, and ``docs/PAPER_TO_CODE.md`` for the command behind each
manuscript number.
"""

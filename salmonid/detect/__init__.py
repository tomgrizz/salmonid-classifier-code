"""detect sub-package: running the model on video and following each fish (Tier 2).

In the order they are run:

    deimv2_infer            loads the published model and returns, per frame, a box and six
                            class scores for every fish (used by the next step, not run directly)
    detect_store            "Pass A": runs the model over every video once and saves the
                            per-frame detections, one file per video
    audit_store_integrity   checks that no saved file is damaged and no video is missing
    track_from_store        "Pass B": joins the saved detections into one track per fish

The tracks are counted by ``salmonid.count``. Needs the model weights, the external DEIMv2
code, the footage and a GPU.
"""

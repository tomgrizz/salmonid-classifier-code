"""data sub-package: building the pictures the model was trained on (Tier 2).

Every step asks the hold-out guard (``salmonid.holdout``) before using a clip, so that no
fish of the held-out test set reaches training. In the order they are run:

    prepare_dataset           labelled fish -> training and validation pictures with boxes
    build_extra_coco          adds hand-picked extra examples (rare species, look-alikes, debris)
    sam2_composite            makes composite pictures (partly visible fish, overlapping fish)
    merge_synth_into_coco     merges the composites into the picture list
    audit_coco_holdout_leak   checks a finished picture list for held-out test clips, and can
                              write the cleaned list the published model (v2.7) was trained on

All of it needs the private labelling database and the footage, so it documents how the
training set was made; an outside reader cannot re-run it.
"""

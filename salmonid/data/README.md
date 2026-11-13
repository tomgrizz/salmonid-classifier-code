# `salmonid/data/`: building and checking the training set (stage B)

The model learns from a list of pictures, each with a box around a fish and the fish's species.
The scripts here assembled that list and checked it. All are **Tier 2**: they need the video,
the project's labelling database and the image archive, none of which is in this repository.

**What you can verify without rebuilding anything:** the audit script, run on the published
training list, confirms that no picture comes from a test video; and it rebuilds the published
list byte for byte from the earlier one. See `docs/REPRODUCE.md`, stage B.

| File | What it does | Role for the published model |
|---|---|---|
| `audit_coco_holdout_leak.py` | Checks a training list for pictures of test fish, at four levels (same track, same video, a composite made from a test video, the same fish in the same frame). With `--out` it writes a cleaned copy. | **The step that produced the published training list**, and the gate the training launcher runs before every training run. |
| `prepare_dataset.py` | Gathers the human-labelled fish and writes them as a picture list. | Built the human-labelled core of the list (as part of the earlier list the published one was cleaned from). |
| `build_extra_coco.py` | Adds hand-picked extra examples: rare species, look-alike species, and confusing debris as negatives. | Supplied 14,013 pictures of fish and 397 without. |
| `sam2_composite.py` | Makes synthetic practice pictures by pasting cut-out fish onto empty backgrounds, partly hidden or side by side. | Its composites are 7,391 of the 45,249 training pictures. |
| `merge_synth_into_coco.py` | Folds the composites into the list and applies the test-set exclusion. | Wrote the earlier list the published one was cleaned from. |

Run order, when building from scratch: `prepare_dataset` → `build_extra_coco` →
`sam2_composite` → `merge_synth_into_coco` → `audit_coco_holdout_leak`.

Every builder reads the test-set exclusion from `salmonid/holdout.py`, which excludes by
**video**: a picture is barred if it comes from any video that contains a test fish.

Known limits of the published training data are documented in each module and collected in
`docs/REPRODUCIBILITY_REVIEW.md` (items L1 to L6). They were not "fixed" here, because the
published model was trained on the data as it was.

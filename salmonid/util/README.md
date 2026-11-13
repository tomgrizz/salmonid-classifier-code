# `salmonid/util/`: two small helpers

Neither file computes a result or appears in the manuscript.

| File | What it does | Feeds | Tier | Status |
|---|---|---|---|---|
| `schema.py` | Holds the one rule that names the per-video detection tables of the earlier run (the tables that carry the boxes of the 366 held-out test tracks), and describes in plain words which class number is which species. | No manuscript number; used by the scripts that read those tables (`eval/predict_testset_v6_from_store.py`, `eval/build_holdout_testset_queue.py`, `data/prepare_dataset.py`, `data/audit_coco_holdout_leak.py`) | Not applicable (no data of its own) | Published path (helper) |
| `ntfy.py` | Sends an optional "job finished" notification at the end of a long run. Off unless a topic is configured. | Nothing | Not applicable | Supporting (convenience only) |

Class numbers of the published model: 0 Chinook Salmon, 1 Coho Salmon, 2 Atlantic Salmon,
3 Rainbow Trout, 4 Brown Trout, 5 Small Salmonid. Class 5 is detected by the model but is
excluded from the reported counts.

`schema.py` used to contain the output layout of the retired two-stage pipeline (which
labelled class 5 "Background"). That part was unused and has been removed; the git history
keeps it.

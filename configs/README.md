# configs/ — the training and model settings files

**In plain terms.** These files are the "recipe cards" the detector software (DEIMv2) reads:
which network to build, how big the pictures are, how long to train, and which picture lists
to train on. They are kept exactly as they were when the models were trained, because a recipe
that has been tidied afterwards is no longer evidence of what was done.

## Which file matters

| File | What it is |
|---|---|
| `deimv2/deimv2_fish_x_v27.yml` | **The recipe of the published model (v2.7).** Used by `python -m salmonid.train.train_v27`. |
| `deimv2/deimv2_fish_x_v25.yml` | The network definition loaded at detection time (`config.INFER_CONFIG`). The network is the same in v2.5, v2.6 and v2.7; v2.7 differs from v2.5 only in its picture lists and output folder. |
| `deimv2/deimv2_dinov3_x_coco.yml`, `base/`, `dataset/`, `runtime.yml` | Files the two above include. Unchanged from the DEIMv2 project. |
| `deimv2/deimv2_fish_x_v26.yml` | The previous model. Superseded: 62 of the 366 held-out test fish were in its training pictures, so its score is not reported. Kept for the documented comparison only. |
| every other `deimv2/deimv2_fish_*.yml` | Earlier experiments (smaller networks, earlier data). Not used for any number in the manuscript. Kept as the development record. |

## Checked on 2026-11-10

- `deimv2/deimv2_fish_x_v27.yml` is byte-identical to the copy the trainer read and to the
  copy saved with the run (`provenance/v2.7/config_snapshot/`).
- The six included files are identical, apart from line endings, to the copies saved with the
  run and to the DEIMv2 project at the pinned commit.
- Every other file is byte-identical to the research repository's copy.

## Read these files with two cautions

1. **They contain the original workstation's absolute paths** (for example
   `ann_file: C:/ssl_cache/...`). That is deliberate: it is the record of which files were
   read. The launcher does not rely on them; it passes the locations from
   `salmonid/config.py` to the trainer explicitly (see `salmonid/train/train_v27.py`).
2. **Their comments were written at the time and some are now known to be wrong.** In
   particular `deimv2_fish_x_v26.yml` says its training list was "test_set_v6 leak-clean
   verified"; it was not (see `provenance/v2.7/README.md`). The comments are left as written.
   Where a comment and `provenance/v2.7/` disagree, `provenance/v2.7/` is right.

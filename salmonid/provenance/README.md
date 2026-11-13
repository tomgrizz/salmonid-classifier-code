# `salmonid/provenance/`: recording what a training run used

| File | What it does |
|---|---|
| `record_run_manifest.py` | Writes a "receipt" for a training run before it starts: the exact commit of this code and of the DEIMv2 code (and whether either had uncommitted changes), the patch applied to DEIMv2, a copy of every configuration file, the checksum and species composition of the training and validation lists, the seed, the command, and the software and graphics card. |

The training launcher (`salmonid/train/train_v27.py`) calls it automatically.

The receipt of the **actual published run** is committed in `provenance/v2.7/` at the root of
the repository, with the checksum of the published weights and the account of how the
checkpoint was chosen. That folder's `README.md` explains what each file shows, including two
things the receipt records candidly: the research code had uncommitted changes when training
started, and an earlier version of this script dropped the first character of two file names.

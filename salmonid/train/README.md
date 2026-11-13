# `salmonid/train/`: training the model (stage C)

| File | What it does |
|---|---|
| `train_v27.py` | Launches a training run of the published model the way it was run for the paper: it checks that the training list is the published one, runs the leak audit, writes a record of the run, starts the DEIMv2 trainer with the published settings and seed, and can then score the candidate checkpoints on the test set. |

**Tier 2.** Needs a graphics card with 32 GB of memory, the DEIMv2 environment
(`requirements-deim.txt`), the DEIMv2 code at the pinned commit (`docs/DEIMV2_PIN.md`), the
training set and two starting-weight files. About 28 hours.

```bash
python -m salmonid.train.train_v27 --dry-run   # run every check, print every command, train nothing
python -m salmonid.train.train_v27 --eval      # train, then score the candidate checkpoints
```

The settings are in `configs/deimv2/deimv2_fish_x_v27.yml`. The record of the actual published
run, the checksum of its weights and the reasoning behind the choice of checkpoint are in
`provenance/v2.7/`.

A re-run will not give the identical model: training is seeded, but the graphics-card
routines it uses are not deterministic.

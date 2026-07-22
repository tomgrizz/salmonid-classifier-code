# Reproducing the single-stage DEIMv2-X v2.6 model

The published model is **one DEIMv2-X model** that does detection + species in a single pass. This
is the canonical, end-to-end runbook — from raw data to the reported score. The pipeline spans two
repositories:

- **this repo** — data builders, inference, evaluation, counting, figures, and the run-manifest
  recorder (all under `salmonid/`), plus the fish config chain (`configs/`).
- **DEIMv2** (external, `Intellindust-AI-Lab/DEIMv2`, Apache-2.0) — `train.py` + the model engine.
  It lives outside this repo and carries a working-tree patch, so it is pinned: see
  `docs/DEIMV2_PIN.md` (+ `DEIMV2_SHA.txt`, `deimv2_worktree.patch`).

> **Just want the numbers/figures?** You do not need any of this. The held-out score, the
> concordance statistics, and the figures reproduce from committed data — see the top-level README
> "Tier 1" (`make score` / `make concordance` / `make figures`).

## Environment (pinned)

| component | value |
|---|---|
| DEIMv2 repo SHA | see `docs/DEIMV2_SHA.txt` (+ `deimv2_worktree.patch`: registers `ColorJitter`, torchvision `_transform`→`transform`) |
| training env `.venv_deim` | Python 3.11, torch 2.7.0+cu128, torchvision 0.22.0+cu128, CUDA 12.8 (`requirements-deim.txt`) |
| base env (eval/count/figures + tracking) | Python 3.11, numpy < 2, boxmot 10.0.52 (`requirements.txt`) |
| GPU | RTX 5090 (sm_120), 32 GB |
| backbone init | `ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |
| detector init (v2.5) | `ckpts/deimv2_x_coco.pth` |
| **v2.6 warm-start init** | `outputs/deimv2_fish_x_v25/checkpoint0019.pth` |

Reconstruct the DEIMv2 repo with `make setup-deimv2` (prints the clone + checkout + `git apply` +
config-copy steps), then set `SALMONID_DEIM_DIR`.

## The leak guard (never train on eval data)

`salmonid/holdout.py::load_holdout_pairs()` is imported by **every** data builder and drops the
frozen `test_set_v6` tracks unconditionally (no flag to forget). The committed
`data/test_set_v6_holdout.json` is always present, so the publishable test set is always excluded.
This is why the reported score is honest — do not remove it.

## Step 1 — build the training COCO (masked_v26)

v2.6's COCO = the v2.5 masked base + **targeted Coho/Rainbow** SAM2 synthetic fish. The compositor
applies `review.csv` (human reject/relabel) by default and `--mask-header-px 40` so synthetic fish
are born header-masked, matching the base.

```bash
PY=".../python.exe"   # base Python 3.11
# synthetic fish (CPU; applies review.csv + the holdout filter automatically)
$PY -m salmonid.data.sam2_composite partial   --n 2000 --sweep --mask-header-px 40 --gallery
$PY -m salmonid.data.sam2_composite proximity --n 1200 --pair coho_chinook --mask-header-px 40 --gallery
# fold into the base, drop holdout images, cap synth -> deimv2_fish_6class_masked_v26/train.json
$PY -m salmonid.data.merge_synth_into_coco \
    --base  $SALMONID_COCO_CACHE/deimv2_fish_6class_masked_v25/train.json \
    --synth $SALMONID_PROJECTS_ROOT/sam2_synth_partial/synth_coco.json \
    --synth $SALMONID_PROJECTS_ROOT/sam2_synth_proximity/synth_coco.json \
    --out   $SALMONID_COCO_CACHE/deimv2_fish_6class_masked_v26/train.json
```

> **Reproducibility caveat.** The compositor is not seeded, so exact synth pixels differ per build.
> We therefore treat the built COCO as a **frozen artifact**: the run manifest records its sha256,
> and re-training uses that exact file. To reproduce the number *exactly*, use the deposited
> `train.json`/`val_clean.json` (see `docs/DATA_AVAILABILITY.md`); to rebuild the *distribution*,
> re-run the commands above.

## Step 2 — train (warm-start, seeded + manifested)

```powershell
powershell -File runners/run_deimv2_fish_x_v26.ps1
```

The runner (a) refuses to start if the COCO or the warm-start checkpoint is missing, (b) writes
`run_manifest.json` + `config_snapshot/` + `deimv2_worktree.patch` into the output dir via
`salmonid.provenance.record_run_manifest`, then (c) runs DEIMv2's
`train.py -c configs/deimv2/deimv2_fish_x_v26.yml -t outputs/deimv2_fish_x_v25/checkpoint0019.pth
-d cuda --use-amp --seed 0`. The v26 config trains **10 epochs at lr 1e-4** (warm-start
continuation), batch 8, 960 px.

> **Determinism.** The seed is fixed (`--seed 0`) but AMP + non-deterministic cudnn mean runs are
> **seed-pinned, not bitwise reproducible**. Report as "seeded," not "deterministic."

## Step 3 — select + evaluate (select on operational F1, not lab AP)

Operational F1 peaks early; lab mAP does not predict the operational ranking. Select the deploy
checkpoint on the tracked-operational sweep over the **leak-clean op_dev** split, confirm **once**
on op_test and the held-out `test_set_v6`, then pass the gates. The published checkpoint is
**epoch 7** (`checkpoint0007.pth`).

```bash
# .venv_deim: Pass A (full-distribution detect-store)
$VPY -m salmonid.detect.detect_store --full-dist --config configs/deimv2/deimv2_fish_x_v25.yml \
     --ckpt $SALMONID_DEIM_DIR/outputs/deimv2_fish_x_v26/checkpoint0007.pth --mask-header-px 40 ...
# base env: Pass B (BoT-SORT) then score
$PY -m salmonid.detect.track_from_store ...
$PY -m salmonid.eval.score_tracked --sweep --keep-keys op_dev_keys.txt --drop-keys op_cropleak_keys.txt  # tune
$PY -m salmonid.eval.score_tracked        --keep-keys op_test_keys.txt --drop-keys op_cropleak_keys.txt  # confirm ONCE

# held-out species F1 (the headline 0.966):
$PY -m salmonid.eval.predict_testset_v6_from_store --store-dir <v26 test store> --out data/preds_checkpoint0007.csv
$PY -m salmonid.eval.score_testset_v6 --preds data/preds_checkpoint0007.csv

# mandatory gates:
$PY -m salmonid.eval.deimv2_bbox_gate --checkpoint <ckpt>   # Ganaraska IoU>=0.80 / cy +-0.10
$PY -m salmonid.eval.count_r_gate --daily-csv <daily csv> --min-days 8 --r-gate 0.90
```

The `--config …v25.yml` is intentional: the X architecture is identical across v2.5/v2.6, so the
snapshotted v25 config is the inference architecture definition, loaded together with the v26
weights.

## Step 4 — corpus counts + results + figures

Multi-year counts (2019–2025, both rivers) run in two passes via the runners, then aggregate:

```powershell
powershell -File runners/run_corpus_v26_detstore_by_year.ps1   # Pass A: detect-stores
powershell -File runners/run_corpus_v26_botsort_by_year.ps1    # Pass B: BoT-SORT track aggregates
```
```bash
$PY -m salmonid.count.corpus_countr_from_store --agg ... --out-csv <year daily csv>   # daily model-vs-tech
$PY -m salmonid.results.build_canonical_results        # -> data/canonical_daily.csv + canonical_summary.csv
$PY -m salmonid.results.compute_concordance            # -> concordance_ccc.csv + Fig 7 / S4
$PY -m salmonid.figures.render_hero_regulation         # hero / phenology / sanctuary
$PY -m salmonid.figures.render_manuscript_figures      # confusion + daily-net panels
$PY -m salmonid.figures.render_sensitivity_figure      # net-rule sensitivity
```

See `docs/counting_and_movement.md` for the counting method and `docs/MANUSCRIPT_RESULTS.md` for
the full numbers, tables, and figure index.

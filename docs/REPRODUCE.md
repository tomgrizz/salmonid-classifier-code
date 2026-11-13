# Reproducing the work, step by step

This is the runbook for the whole pipeline, from raw video to the manuscript's tables and
figures, in the order the steps are run.

- If you only want to check the paper's numbers, you need **Part 1** and nothing else.
- **Part 2** documents the steps that need large external files. Their code is in this
  repository; their inputs are described in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

Each step below says what it does, what it reads and writes, how long it took on the original
workstation, and how far it was re-verified when this deposit was prepared (2026-11-10).

All commands are run from the repository root. `<…>` marks a location that comes from
`salmonid/config.py`; you set it with an environment variable or a `.env` file
(see `.env.example`), or you place the files under `external/`.

---

## Part 1. From the files in this repository (Tier 1)

**Environment.** Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate                       # macOS / Linux: source .venv/bin/activate
pip install -r environment/tier1.lock.txt    # exact published versions (15 packages)
```

**Run and verify everything.**

```bash
python run_tier1.py
```

| Step | Command (`python -m …`) | Reads | Writes | Feeds |
|---|---|---|---|---|
| 1 | `salmonid.eval.score_testset_v6` | `data/test_set_v6_gt.json`, `data/preds_v27_ckpt0017.csv` | printout | Table 1, Figure 2 |
| 2 | `salmonid.results.classification_ci` | same | `output/tier1/classification_ci.csv` | Table 1 intervals |
| 3 | `salmonid.results.build_canonical_results --sweep-root data --recovered-dir data/recovered_2020 --out-dir output/tier1` | `data/net_rule_sweep/*_up50.csv`, `data/activity_sweep/`, `data/recovered_2020/` | `canonical_daily.csv`, `canonical_summary.csv` | every count result |
| 4 | `salmonid.results.compute_concordance` | `data/canonical_*.csv` | `concordance_ccc.csv`, two figures | Table 5, Figures 6, S3 |
| 5 | `salmonid.results.agreement_stats` | same | four `agreement_*.csv` | Sections 3.3, 3.5 |
| 6 | `salmonid.count.net_rule_loyo` | `data/net_rule_sweep/` | `net_rule_loyo.csv` | Section 3.4 |
| 7 | `salmonid.figures.render_sensitivity_figure` | same | figure, `net_rule_sensitivity.json` | Section 3.4 |
| 8 | `salmonid.figures.render_manuscript_figures` | test set, `canonical_daily.csv` | figures | Figures 2, S2 |
| 9 | `salmonid.figures.render_hero_regulation` | `data/canonical_*.csv` | figures | Figures 4, 5, 7 |
| 10 to 18 | `salmonid.paper.*` | `data/paper/`, `data/` | `output/paper/*.csv` | Tables 2, 3, 4, S4; Sections 3.1, 3.2 |

The whole run takes about one minute. It compares each regenerated table with the committed
copy and re-derives the headline numbers; the expected result is `0 failed`.

**Why a re-run matches exactly.** Every random step (the bootstrap resampling behind the
confidence intervals) uses a fixed seed and a fixed number of resamples, recorded in the
module that uses it: 10,000 for the classification intervals, 2,000 for the independent-day
concordance intervals, 4,000 for the moving-block and inter-annual intervals, 2,000 for the
kappa intervals.

**Figures.** With the lock file installed, the regenerated PNG and PDF files are byte-identical
to those in the manuscript (`data/figure_checksums.json`). With other package versions the
numbers are the same and the image files may differ slightly; `run_tier1.py` reports that as a
note, not a failure.

Verified for this deposit: in a new virtual environment and a clean copy of the committed
files, with every external location pointed at a non-existent folder, all checks passed.

---

## Part 2. The full pipeline (Tier 2)

**Environments.** Two more environments are used (see [`../environment/README.md`](../environment/README.md)):

- *base* (`requirements-track.txt`, exact versions `environment/track.lock.txt`): building
  the training set, tracking, counting, the photo figures. Note the instruction there about
  installing `boxmot==10.0.52`.
- *DEIMv2* (`requirements-deim.txt`, exact versions `environment/deim.lock.txt`): training and
  detection, with a graphics card. It hosts the external DEIMv2 code at the commit and patch
  given in [`DEIMV2_PIN.md`](DEIMV2_PIN.md).

The original hardware was one NVIDIA RTX 5090 (32 GB), Windows 11.

### Stage A. Set aside the test set

| | |
|---|---|
| What | 366 fish are chosen and labelled by species, and the 364 videos that contain them are barred from training. |
| Code | `salmonid/eval/build_holdout_testset_queue.py`, `salmonid/eval/finalize_testset_v6.py` |
| Needs | The project's labelling database and the earlier detection run the tracks were drawn from. Neither is deposited. |
| Result (committed) | `data/test_set_v6_gt.json`, `data/test_set_v6_holdout.json` |
| Re-verified | Not re-run: it cannot be, without the labelling database. The committed files are the frozen result. |

### Stage B. Build the training set and check it for leaks

| | |
|---|---|
| What | Labelled frames and synthetic composites are assembled into the training list; the list is then audited so that no image comes from a test video. |
| Code | `salmonid/data/` (builders), `salmonid/data/audit_coco_holdout_leak.py` (the audit), `salmonid/holdout.py` (the exclusion rule) |
| Needs | Training images and lists (`<COCO_CACHE>`), the SAM2 cutout library, the labelling database |
| Result | `<COCO_CACHE>/deimv2_fish_6class_masked_v26_clean/{train,val_clean}.json`: 45,249 training images with 52,444 labelled fish; 2,160 validation images |

```bash
# is the published training list free of test videos? (exit code 1 if anything leaks)
python -m salmonid.data.audit_coco_holdout_leak --coco <COCO_V27>/train.json --strict

# the same, plus two report-only sections (images with no video key; neighbouring clips)
python -m salmonid.data.audit_coco_holdout_leak --coco <COCO_V27>/train.json --report-adjacent
```

How the published list was actually made: the earlier model's list (`masked_v26`) was passed
through this audit with `--out`, which removes every image from a test video. Re-running that
command reproduces the published `train.json` and `val_clean.json` **byte for byte**; their
SHA-256 values equal those recorded at training time (`../provenance/v2.7/run_manifest.json`).

What cannot be rebuilt: the earlier list itself. No record survives of the exact commands
that built it, several of its builders were never part of this repository, and the synthetic
images depend on external inputs. It is therefore treated as a frozen input, identified by
checksum. Details and the known limitations of the training data are in
[`REPRODUCIBILITY_REVIEW.md`](REPRODUCIBILITY_REVIEW.md).

Re-verified: the audit was run on the published list (clean at every level) and on the earlier
list (finds the 62 test fish that were in it), and the byte-for-byte rebuild was confirmed.

### Stage C. Train the model

| | |
|---|---|
| What | DEIMv2-X is trained from the public COCO-pretrained starting weights for 30 epochs (image size 960, batch size 8, learning rate 0.00025, seed 0). |
| Code | `salmonid/train/train_v27.py`; configuration `configs/deimv2/deimv2_fish_x_v27.yml` |
| Needs | The training set, the DEIMv2 code, the two starting-weight files, a 32 GB graphics card |
| Time | About 28 hours (24 July 14:45 to 25 July 18:30, 2026) |
| Result | `<DEIM_DIR>/outputs/deimv2_fish_x_v27/checkpoint0017.pth` (819 MB), the published model |

```bash
python -m salmonid.train.train_v27 --dry-run     # run every check, print every command, train nothing
python -m salmonid.train.train_v27 --eval        # checks -> leak audit -> record -> train -> score
```

The launcher refuses to start unless the training list matches its published checksum, the
leak audit passes, and the DEIMv2 code is at the pinned commit with the patch applied. It
writes a record of the run before training begins.

Training is **seeded but not bit-for-bit repeatable** (the graphics-card routines it uses are
not deterministic), so a re-run gives a slightly different model. The published model is
identified by the checksum of its weights file in `../provenance/v2.7/CHECKSUMS.md`.
`../provenance/v2.7/CHECKPOINT_SELECTION.md` gives the score of every candidate checkpoint and
explains why epoch 17 was chosen.

Re-verified: the dry run (all 18 checks pass on the original workstation). Training was not
re-run.

### Stage D. Detect and track

| | |
|---|---|
| What | Pass A: the model marks and names fish in every third frame of each video and stores the result per video. Pass B: BoT-SORT links the marks into one track per fish. |
| Code | `salmonid/detect/detect_store.py` (A), `salmonid/detect/audit_store_integrity.py`, `salmonid/detect/track_from_store.py` (B) |
| Needs | The video (246,070 files for 2020 to 2025), the model weights, the re-identification weights |
| Time | Several days for the six seasons, two processes side by side |
| Result | `<PROJECTS_ROOT>/deimv2_x_v27_corpus<year>/store/` and `agg_*.pkl` |

```bash
python -m salmonid.count.run_corpus --steps all --dry-run   # print every command for every season
python -m salmonid.count.run_corpus --steps all             # run them (days of GPU time)
```

Re-verified: the stored results were audited (246,070 store files, none corrupt, no video
missing from the track files). The tracking module was re-run in its motion-only mode on the
2025 Ganaraska detections and reproduced the stored tracks exactly. **Pass A and the BoT-SORT
mode of Pass B were not re-run for this deposit**; see the review document.

### Stage E. Count, and compare with the technicians

| | |
|---|---|
| What | Each track gets a species, a date and a direction; daily totals are written beside the technicians' totals. See [`COUNTING_METHOD.md`](COUNTING_METHOD.md). |
| Code | `salmonid/count/corpus_countr_from_store.py`, run for every season by `salmonid/count/run_corpus.py` |
| Needs | The track files from stage D, the technician records, and either the video (for file dates) or saved date maps |
| Time | About two minutes |
| Result (committed) | `data/net_rule_sweep/`, `data/activity_sweep/`, `data/recovered_2020/`, and from them `data/canonical_daily.csv`, `data/canonical_summary.csv` |

```bash
python -m salmonid.count.run_corpus                 # audit -> count -> assemble -> compare with data/
```

Re-verified: run from the stored tracks on the original workstation, this regenerated all 48
rule-comparison tables and both canonical tables **identically** to the committed copies
(50 of 50 files).

### The held-out predictions

```bash
python -m salmonid.eval.predict_testset_v6_from_store     # -> output/preds_v27_ckpt0017.csv
python -m salmonid.eval.score_testset_v6 --preds output/preds_v27_ckpt0017.csv
```

Each test fish is given a species by averaging the model's species scores over the frames in
which a detection overlaps the fish's reference box. Re-verified: all 366 predictions were
regenerated from the stored detections and are identical to `data/preds_v27_ckpt0017.csv`.
Each prediction rests on a median of 29 frames (minimum 6, maximum 88).

### Stage F extras that need external files

| Item | Command | Needs |
|---|---|---|
| Figure 1 (study map) | `python -m salmonid.paper.fig1_study_map` | map layers (see `data/paper/README.md`) |
| Figure 3 (body shapes) | `python -m salmonid.figures.fig_species_morphology` | the SAM2 cutout library |
| Figure S1 (example detections) | `python -m salmonid.figures.fig_bbox_grid` | the cutout library and the video |
| Table S3 (2025 technician record) | `python -m salmonid.paper.technician_record_2025` | the 2025 technician exports |
| Tracker comparison, Section 3.4 | `python -m salmonid.detect.track_from_store --tracker bytetrack …` then the count step | stored detections for 2025 |
| Rebuild the tables in `data/paper/` | each `salmonid.paper` module's `--from-external` option | stored tracks, technician exports |

---

## What was run originally

`../runners/original/` holds the PowerShell scripts that launched the original training and
processing runs, unchanged apart from a banner. They are a record, not a way to run the
pipeline: they contain the original workstation's paths and call scripts by their names in
the private research repository. The Python entry points above do the same steps from
`salmonid/config.py`.

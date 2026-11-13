# Counting salmon and trout in fishway video: code and data

This repository holds the code and the small data files needed to reproduce the results of

> Gristey, T., Fisher, Q., Collier, S., Ford, M., Yuille, M., Leos Barajas, V., and deKerckhove, D.
> *Deep learning identification and counting of migratory salmon and trout from fishway video:
> validation against six years of counts at two Lake Ontario tributaries.*
> Submitted to the Canadian Journal of Fisheries and Aquatic Sciences.

A computer-vision model watches RiverWatcher fishway video from the Ganaraska River and the
Credit River (Ontario, Canada), identifies five species (Chinook Salmon, Coho Salmon, Atlantic
Salmon, Rainbow Trout, Brown Trout), and counts how many moved upstream and downstream each
day. The paper compares those counts with the counts technicians made from the same video,
2020 to 2025.

You do not need a background in machine learning to check the paper's numbers. Start below.

---

## Start here: recompute the paper's results in about a minute

You need Python 3.11 or newer and four common packages. No video, no trained model, no
graphics card.

```bash
git clone https://github.com/tomgrizz/salmonid-classifier-code
cd salmonid-classifier-code

python -m venv .venv
.venv\Scripts\activate                 # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r environment/tier1.lock.txt

python run_tier1.py
```

`run_tier1.py` runs 18 steps in the order the paper presents its results. It then checks every
regenerated table against the published copy in `data/`, value by value, and re-derives the
headline numbers quoted in the manuscript. The last lines should read:

```
RESULT: 90 passed, 0 failed, 0 notes, 0 skipped
Every regenerated result matches the published one.
```

Other ways to run it:

```bash
python run_tier1.py --list        # show the 18 steps and what each one feeds, run nothing
python run_tier1.py --only 1 2    # run chosen steps
python run_tier1.py --no-figures  # statistics only
```

Each step prints the exact command it runs, so any step can be repeated on its own.
What you get: tables in `output/` and figures in `figures/`. Nothing in `data/` is ever
overwritten.

---

## What can be reproduced, and from what

The work has two tiers.

**Tier 1: from the files in this repository.** Everything computed from counts and labels.

**Tier 2: needs large external files.** Anything that touches video, the trained model, or the
technicians' own records: training, running the model on video, tracking, and the three
figures that show fish images or a map. The code for all of it is here and documented; the
inputs are described in [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).

| Manuscript item | Tier | Reproduced by |
|---|---|---|
| Table 1, Figure 2: species classification on the 366-fish test set (macro-F1 0.961) | 1 | steps 1, 2, 8 |
| Table 2: 2025 agreement by species and river | 1 | steps 10, 16 |
| Tables 3 and 4: multi-year count agreement | 1 | step 16 |
| Table 5, Figures 6 and S3: concordance and Bland-Altman | 1 | steps 4, 5 |
| Figures 4, 5, 7, S2: annual counts, run timing, sanctuary window, 2025 cumulative counts | 1 | steps 8, 9 |
| Section 3.2: per-video agreement, detection outcomes, blind review | 1 | steps 10, 11, 12 |
| Section 3.4: the upstream/downstream rule and its leave-one-year-out check | 1 | steps 6, 7 |
| Table S4: comparison of three classifier backbones | 1 | step 15 |
| Tables S1, S2: published annual technician totals | 1 | step 18 |
| The daily and seasonal count tables themselves, from the per-season count tables | 1 | step 3 |
| The per-season count tables, from stored tracks and technician records | 2 | `python -m salmonid.count.run_corpus` |
| The model's answers for the 366 test fish, from stored detections | 2 | `python -m salmonid.eval.predict_testset_v6_from_store` |
| Stored detections and tracks, from video | 2 | `python -m salmonid.count.run_corpus --steps all` |
| The trained model | 2 | `python -m salmonid.train.train_v27` |
| Figure 1 (map), Figure 3 (body shapes), Figure S1 (example detections), Table S3 | 2 | see the map below |

[`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) goes through the manuscript line by line: every
number, table and figure, the command that produces it, the file it reads, and whether a
re-run gives the printed value.

---

## The order of the whole pipeline

The steps below are the full path from raw video to the paper. Steps A to E are Tier 2 and
were run once on a workstation with a graphics card; their outputs are the tables in `data/`.
Step F is what `run_tier1.py` runs. [`docs/REPRODUCE.md`](docs/REPRODUCE.md) gives every
command with its inputs, outputs and running time.

| | Stage | What happens | Code | Tier |
|---|---|---|---|---|
| A | Set aside the test set | 366 fish are chosen and labelled, and every video containing one is barred from training. | `salmonid/eval/finalize_testset_v6.py`, `salmonid/holdout.py` | 2 |
| B | Build the training set | Labelled video frames and synthetic composite images are assembled into a list of 45,249 training images, then checked for test-set leaks. | `salmonid/data/` | 2 |
| C | Train the model | The detector is trained for 30 passes over the training set; the copy saved after pass 18 ("checkpoint 0017") is the published model. | `salmonid/train/train_v27.py` | 2 |
| D | Detect, then track | The model marks and names every fish in every third video frame ("Pass A"); a tracker links those marks into one track per fish ("Pass B"). | `salmonid/detect/` | 2 |
| E | Count | Each track is given a species, a date and a direction, and the daily totals are set beside the technicians' totals. | `salmonid/count/` | 2 |
| F | Statistics and figures | Classification score, agreement statistics, tables and figures. | `salmonid/results/`, `salmonid/paper/`, `salmonid/figures/` | **1** |

The 18 steps of stage F:

| Step | What it computes | Manuscript item |
|---|---|---|
| 1 | Held-out species score | Table 1, Figure 2, Section 3.1 |
| 2 | Confidence intervals for the species score | Table 1 |
| 3 | Rebuild the daily and seasonal count tables | the source of every count result below |
| 4 | Concordance of daily counts, Bland-Altman | Table 5, Figures 6 and S3 |
| 5 | Agreement statistics for autocorrelated daily counts | Sections 3.3 and 3.5 |
| 6 | Leave-one-year-out check of the direction rule | Section 3.4 |
| 7 | Sensitivity of the count to the rule thresholds | Section 3.4 |
| 8 | Figures: confusion matrix, 2025 cumulative counts | Figures 2 and S2 |
| 9 | Figures: annual counts, run timing, sanctuary window | Figures 4, 5 and 7 |
| 10 | 2025 per-video species agreement and Cohen's kappa | Section 3.2, Table 2 |
| 11 | 2025 detection outcomes on reviewed videos | Section 3.2 |
| 12 | Blind review of 150 clips | Section 3.2 |
| 13 | Cohen's kappa on the held-out test set | Section 3.1 |
| 14 | Held-out score without tracks next to a training clip | Sections 3.1, 4.1, 4.5 |
| 15 | Backbone comparison of three classifiers | Table S4 |
| 16 | Count tables | Tables 2, 3 and 4 |
| 17 | Daily totals and the busiest days | Abstract, Conclusions |
| 18 | Checks against the published annual technician totals | Sections 2.1 and 2.2, Tables S1 and S2 |

---

## What is in this repository

```
run_tier1.py        the one command: runs the 18 Tier-1 steps in order, then verifies
salmonid/           the code, one folder per stage (each folder has its own README)
  config.py           every file location and study constant, in one place
  holdout.py          keeps the test set out of training
  data/               stage B: build and audit the training set
  train/              stage C: launch training
  detect/             stage D: run the model on video; track fish
  eval/               stage A and the held-out score, plus development checks
  count/              stage E: daily counts and the upstream/downstream rule
  results/            stage F: count tables, agreement statistics, intervals
  paper/              stage F: the remaining manuscript tables and checks
  figures/            stage F: every figure
  provenance/         records exactly what a training run used
data/               the published tables (data/README.md describes every file and column)
configs/            the model configuration files, exactly as run
provenance/v2.7/    the record of the published training run: settings, file fingerprints,
                    and why checkpoint 0017 was chosen
environment/        exact package versions for each of the three environments
runners/original/   the PowerShell scripts that actually launched the original runs (record only)
docs/               the documents listed below
```

Every script begins with a paragraph headed "In plain terms" that says what it does and why,
followed by its inputs, outputs, the manuscript item it feeds, and the command to run it.

| Document | Read it for |
|---|---|
| [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md) | Where every number, table and figure in the manuscript comes from |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | The full pipeline, command by command, including the Tier-2 steps |
| [`docs/COUNTING_METHOD.md`](docs/COUNTING_METHOD.md) | How a track becomes a count, stated exactly as implemented |
| [`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md) | Every data file, committed or external, and how to obtain it |
| [`docs/REPRODUCIBILITY_REVIEW.md`](docs/REPRODUCIBILITY_REVIEW.md) | What was checked, what reproduces, and the known limitations and open items |
| [`docs/SOFTWARE_CITATIONS.md`](docs/SOFTWARE_CITATIONS.md) | Every library, model, dataset and statistical method used, with versions, licences and citations |
| [`docs/CODE_MAP.md`](docs/CODE_MAP.md) | How the modules connect |
| [`docs/DEIMV2_PIN.md`](docs/DEIMV2_PIN.md) | The exact version of the external detector code |
| [`data/README.md`](data/README.md) | The data dictionary |

---

## The model, briefly

One detector, DEIMv2-X (Huang et al. 2025) on a self-supervised DINOv3 ViT-S/16+ backbone
(Siméoni et al. 2025), finds each fish and names its species in the same pass. BoT-SORT
(Aharon et al. 2022), as implemented in BoxMOT 10.0.52, links detections into tracks. A track's
species is the highest of its frame-averaged species scores; its direction comes from how far
it travelled across the frame. Internally this model is called "v2.7"; the manuscript does not
use version labels. An earlier model, "v2.6", appears in a few file names for comparison only:
62 of its 366 test fish turned out to have been in its training data, so it was retrained and
is not used for any published number.

---

## Limits of this deposit

These are stated in full, with evidence, in
[`docs/REPRODUCIBILITY_REVIEW.md`](docs/REPRODUCIBILITY_REVIEW.md). The main ones:

- **The Tier-2 steps cannot be re-run from this repository alone.** They need the trained
  model, the video and the technician records. See `docs/DATA_AVAILABILITY.md` for the status
  of each.
- **Training is seeded but not bit-for-bit repeatable.** A re-run would give a slightly
  different model. The published model is identified by the fingerprint of its weights file
  in `provenance/v2.7/CHECKSUMS.md`.
- **The test set informed the choice of checkpoint.** Seven saved copies of the model were
  scored on the test set and one was chosen; they differ by about 0.003 in macro-F1
  (`provenance/v2.7/CHECKPOINT_SELECTION.md`).
- **The exclusion of test fish from training is by video, with known residual overlap**: 25
  training frames from one test video entered through a separate annotation source, and 76
  test fish have a same-species training image in the neighbouring clip. Step 14 reports the
  score without those 76 (0.950).

---

## Licence

The licence for this repository has not yet been finalized; see [`LICENSE`](LICENSE). Until it
is, no reuse rights are granted beyond reading and running the code for peer review.
Third-party components keep their own licences, summarized in
[`docs/SOFTWARE_CITATIONS.md`](docs/SOFTWARE_CITATIONS.md).

## How to cite

Cite the article above. Machine-readable citation metadata for this repository, and for every
library, model and dataset it depends on, is in [`CITATION.cff`](CITATION.cff) and
[`docs/software.bib`](docs/software.bib).

## Glossary

| Term | Meaning here |
|---|---|
| Track | One fish followed through one video clip |
| Net count | Fish that moved upstream minus fish that moved downstream |
| Activity count | Upstream plus downstream passages |
| Held-out test set | 366 fish kept out of training and used only to score the model |
| Macro-F1 | The average over the five species of a score that combines how many fish of a species were found and how many calls of that species were right; 1 is perfect |
| Checkpoint | A copy of the model saved partway through training |
| Detect-store / stored detections | A file per video holding everything the model marked in it, so that tracking and counting can be redone without running the model again |
| Lin's CCC | A 0-to-1 measure of how closely two sets of counts agree, including agreement in level, not only in pattern |
| Tier 1 / Tier 2 | Runs from files in this repository / needs external files |

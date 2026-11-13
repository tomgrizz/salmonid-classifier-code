# Reproducibility review

**Date:** 2026-11-10. **Scope:** this repository, checked against the manuscript working copy of
the same date and against the private research repository the code was extracted from.

This is the register of what was checked, what reproduces, what was fixed, and what is still
open. It is written to be read by the authors, by reviewers, and by anyone continuing the work.
Each item has an identifier so it can be referred to.

How to read the status column:

- **FIXED**: was a defect, corrected in this repository, with the check that shows it.
- **OPEN**: can be closed by further work on this repository, without an author decision.
- **AUTHOR**: needs a decision or action that only the authors (or the Ontario Ministry of
  Natural Resources) can take.
- **LIMIT**: a documented property of the published model or data. Deliberately not changed,
  because changing it would change published results. Should be disclosed, not fixed.

---

## 1. Summary

| Question | Answer |
|---|---|
| Do the paper's statistics and data-driven figures recompute from this repository alone? | **Yes.** `python run_tier1.py`: 18 steps, 90 checks, 0 failures, in a new environment and a clean copy of the files with no external data. All 17 data-driven figure files are byte-identical to the manuscript's. |
| Does the committed code regenerate the committed data from the stored model output? | **Yes.** All 48 rule-comparison tables, both canonical count tables and all 366 held-out predictions were regenerated on the original workstation and are identical. |
| Can the published training set be verified? | **Yes, its last step.** The leak audit rebuilds the published training lists byte for byte from the earlier lists, and confirms no picture comes from a test video. The earlier lists themselves cannot be rebuilt (L1). |
| Can an outside reader re-run training, detection and tracking? | **Not yet.** The weights, training images, stored detections, video and technician records are not deposited (A1 to A5). |
| How much of the manuscript is mapped to code? | 118 items in `docs/PAPER_TO_CODE.md`: 85 reproduce at Tier 1, 10 need external files, 3 are wording mismatches (A12), 8 have no computation in the repository (A13), 12 need none. |
| Is the deposit ready for CJFAS? | **Not until the AUTHOR items in section 4 are settled**, chiefly the licence (A9), the DOI deposit (A1, A10) and the data-access route (A4). |

The most important thing this review found is structural. Before it, **no single commit of any
repository contained the code that produced the paper.** The public deposit held the code of
the superseded model; the fixes behind the published model were split across two unrelated
branches of the private research repository, some of them uncommitted when the model was
trained. This repository now holds that code in one place, verified against the published
outputs (section 2).

---

## 2. What was fixed

### Reproduction path

| ID | Was | Now | Check |
|---|---|---|---|
| F1 | The documented first command (`make score`) scored the superseded model and printed 0.966; the README promised 0.961. | Every entry point defaults to the published predictions (`config.PUBLISHED_PREDS`). | `run_tier1.py` step 1 |
| F2 | The documented interval command defaulted to the superseded model and **overwrote the published Table 1 file** in `data/`. | Defaults to the published model; writes to `output/`. | step 2 |
| F3 | Tier-1 commands wrote into `data/`, so a reproduction run silently modified the published tables. | Everything regenerated goes to `output/` (git-ignored). `data/` is read-only by design. | `git status` after a run shows no change under `data/` |
| F4 | The committed moving-block and inter-annual tables did not regenerate with the documented settings. | Established that they were produced with 4,000 resamples, not the documented 2,000; that is now the stated default, and they regenerate exactly. See A14 for the manuscript wording. | step 5 |
| F5 | `data/net_rule_sweep/net_rule_sensitivity.json` held values from the superseded model (4.39% where the published tables give 3.56%). | Replaced with the file regenerated from the committed tables. | step 7 |
| F6 | `data/2024_botsort_net.csv` was output of the superseded model, used only by a preview figure not in the manuscript. | Removed, with the preview figures that read it (F12). | — |
| F7 | No way to run everything and no check that results matched. `make` is not available on most Windows machines. | `run_tier1.py`: one cross-platform command that runs the steps in order, compares every table value by value, and re-derives 26 headline numbers against the values printed in the manuscript. | — |
| F8 | The canonical count tables could only be rebuilt with the stored tracks and technician records. | The per-season activity tables and the recovered-season tables (8 files, 190 KB) were added to `data/`; the canonical tables now rebuild at Tier 1. | step 3 |
| F9 | Nine manuscript analyses lived only in the manuscript repository or had no saved script (per-video agreement, blind review, kappa, Tables 2 to 4 and S4, and others). | Ported or written as `salmonid/paper/`, with their small input tables in `data/paper/`. | steps 10 to 18 |

### The published model's code

| ID | Was | Now | Check |
|---|---|---|---|
| F10 | The deposited exclusion rule (`holdout.py`) and data builders were the track-level versions that let 62 test fish into the superseded model's training data. The video-level fix, its audit, the training configuration and the launcher were not in the deposit. | Ported from the research repository: video-level `holdout.py`, the four builders, `audit_coco_holdout_leak.py`, `configs/deimv2/deimv2_fish_x_v27.yml`, `salmonid/train/train_v27.py`. | Audit on the published list: clean at every level. Audit on the earlier list: 334 leaked images, 62 test fish. Audit with `--out`: reproduces the published lists byte for byte. |
| F11 | The deposited detection, tracking, counting and assembly code pointed at the superseded model's folders and lacked later fixes (three silent data-loss paths, store integrity checks, the comparison tracker). The two research branches had diverged on the same files. | Three-way merge of the deposit and both branches, with the lineage of each file recorded in its header. New: `audit_store_integrity.py`, `run_corpus.py`. | 50 of 50 count tables and 366 of 366 predictions regenerate identically. Comparison-tracker output reproduced exactly. Store audit: 246,070 files, none corrupt, none missing. |
| F12 | Figure scripts drew three preview figures not in the manuscript, one with hard-coded numbers from an earlier pipeline. Two figure modules did work at import time. | Preview figures removed; every script has a `main()`; figure numbers in the headers match the manuscript. | Figures regenerate byte-identically before and after |
| F13 | No record of the published training run in any repository. | `provenance/v2.7/`: the run record, configuration snapshot, patch, checksums of the weights and inputs, and the species calls of all seven candidate checkpoints with the account of the choice. | — |
| F14 | Four development checks ("gates") could report a pass with no data, and one could never fail. Two scripts that build the test set and the blind-review sample wrote into the original data folders by default, so a re-run would have overwritten the frozen originals. | Checks with nothing to evaluate now fail; failures return a non-zero exit code; the builders write to `output/` and refuse to overwrite. No threshold or calculation changed. | Each was run with missing inputs and stops with one sentence; outputs compared with the original scripts where they could be run |
| F25 | `salmonid/util/schema.py` described the label space of an earlier pipeline (class 5 as "background") and was mostly unused. A results compiler held hard-coded superseded numbers. | Reduced to the one helper in use, with the correct class list; the compiler was removed. | — |

### Environment and packaging

| ID | Was | Now |
|---|---|---|
| F15 | One requirements file forced a deep-learning stack and a video library (no wheels for Apple Silicon) on readers who only wanted the statistics. Nothing was pinned. | `requirements.txt` holds the four Tier-1 packages. `environment/` holds exact lock files for all three environments. |
| F16 | Unpinned indirect dependencies. A fresh install on 2026-11-10 pulled new major versions of two of them and one freshly downloaded binary was blocked by the test machine's security policy. | `environment/tier1.lock.txt` pins all 15. Verified in a new environment. |
| F17 | The requirements stated that the tracking library needs numpy below 2. | Corrected. The published counts were tracked with boxmot 10.0.52 **and numpy 2.4.4** (established from the run logs). The library's own metadata asks for an older numpy, so it must be installed with `--no-deps`; this is documented. Version 21 cannot run this code. |
| F18 | Default locations in `config.py` were the original workstation's drives and user folder. | Defaults are `external/` inside the clone. The original workstation uses a local, git-ignored `.env`. Missing inputs stop with one sentence naming the input and the variable (`config.require`). |
| F19 | A public notification channel name was built into the code; anyone running it posted to it. | Notifications are off unless `SALMONID_NTFY_TOPIC` is set. |
| F20 | Importing the detection module changed the working directory and failed with a traceback when DEIMv2 was absent. | Done inside the model-loading function, with a clear message. |

### Documentation

| ID | Was | Now |
|---|---|---|
| F21 | README, runbook, code map, data page and reviewer guide described the superseded model, with figure and table numbers from an earlier draft. Three documents quoted superseded results. | README, `REPRODUCE.md`, `DATA_AVAILABILITY.md`, `DEIMV2_PIN.md`, `CODE_MAP.md` rewritten. `PAPER_TO_CODE.md` replaces the reviewer guide. Superseded result documents removed. The earlier counting notes moved to `docs/history/` under a banner; `COUNTING_METHOD.md` states the published method. |
| F22 | No data dictionary. Eight committed files were undocumented, including the published predictions. | `data/README.md`, `data/paper/README.md`, `data/CHECKSUMS.sha256`. |
| F23 | Citation file named one of seven authors, the wrong journal, a placeholder URL and an old version; no dependency was cited. | `CITATION.cff` has all seven authors, the CJFAS article, and 68 verified references. `docs/software.bib` and `docs/SOFTWARE_CITATIONS.md` give versions, licences and uses. ORCIDs, licence, DOI and release date are marked TODO (A10). |
| F24 | Seven counting modules of an earlier pipeline were presented as the method. | Each carries a STATUS note; `salmonid/count/README.md` separates the published path from the record. |

---

## 3. Limits of the published model and data (disclose; do not change)

| ID | Limit | Evidence | Where the manuscript stands |
|---|---|---|---|
| L1 | **The base training list cannot be rebuilt.** No repository records the commands that built the list the published one was cleaned from. Several of its builders (the co-occurrence miner that supplied 3,906 pictures, the extra-example lists, the header-mask step) were never in this repository, and the synthetic pictures depend on external inputs. Only the final cleaning step is reproducible. | `salmonid/data/` module headers | The list is identified by checksum. State that the training set is deposited as built, or not at all (A2). |
| L2 | **25 training frames come from one test video** (an Atlantic Salmon), through an annotation source that carried no video identifier and so bypassed the exclusion. Two validation frames likewise. | `audit_coco_holdout_leak --report-adjacent`, UNKEYED section | **Not in the current manuscript** (an earlier draft disclosed it). Methods 2.4 says the videos were "excluded from training" (A15). |
| L3 | **76 of the 366 test fish have a same-species training picture from the neighbouring clip**, which may show the same individual. Without them macro-F1 is 0.950 (Brown Trout 0.935). | `run_tier1.py` step 14 | Disclosed in Sections 3.1, 4.1, 4.5. |
| L4 | **The checkpoint was chosen using the test set.** Candidates differ by about 0.003; one is 0.0001 higher than the published one. | `provenance/v2.7/CHECKPOINT_SELECTION.md` | **Not in the current manuscript**, which does not mention how the model copy was chosen (A15). |
| L5 | **No candidate checkpoint meets the project's own box-placement threshold at the Credit River** (median overlap with hand-drawn boxes 0.737 against 0.80; Ganaraska 0.805). | same file | **Check whether the manuscript mentions this** (A15). |
| L6 | Other properties of the training data: the training/validation split is by picture, so 97.6% of validation tracks also appear in training (validation scores are not a measure of generalization and were not used to select the model); 184 frames carry conflicting species labels on the same box; real frames carry one box each; negatives were not verified empty; 560 training frames come from clips of an older evaluation set that no manuscript number uses. | `salmonid/data/` module headers | For the authors to judge. |
| L7 | **Training is seeded, not repeatable bit for bit.** | `provenance/v2.7/run_manifest.json` | The published model is identified by the checksum of its weights. |
| L8 | **The training-run record shows uncommitted code.** The research repository had uncommitted changes to the exclusion code when training started, so the record's commit does not pin that code. The audit's byte-for-byte rebuild is the evidence that the deposited version is the one used. | `provenance/v2.7/README.md` | — |
| L9 | **Frames are squeezed, not letterboxed**: 1280 × 960 is resized to 960 × 960. | `configs/`, `salmonid/detect/deimv2_infer.py` | The current manuscript does not describe the resize; no conflict. |
| L10 | **The counting rules are in frames, not seconds, and the frame rate changed.** 2020 to 2022 video is 25 to 30 frames per second; 2023 to 2025 about 10. "Every third frame" and "at least three positions" are therefore about 2.5 times less strict in time for 2020 to 2022, the seasons with the larger over-counts. | `docs/COUNTING_METHOD.md` | **New finding.** The manuscript gives "about 10 frames per second" and "three processed frames (seven raw frames)" for all seasons; both hold only for 2023 to 2025 (A15). |
| L11 | Other counting properties: passage date is the video file's modification time in local time; juvenile ("Small Salmonid") tracks are assigned to an adult species; a three-position track can never register a direction; days without video are scored as model zero; technician records without a species are excluded; daily correlations use days on which either series is non-zero. | `docs/COUNTING_METHOD.md` | Each is stated there. Check Methods 2.5 and 2.6 against it. |
| L12 | **The tracker comparison did not hold one setting equal.** The comparison tracker kept lost tracks four times longer. With the setting matched, the net count is unchanged (+3.4% against +3.5%) but it produces 2.5% more tracks. | `salmonid/detect/track_from_store.py` header | The net-count conclusion stands. Methods 2.3 says the comparison used "the same settings"; one setting differed (A15). |
| L13 | **A repair to the 2025 Credit River data has no script or log.** 679 videos were re-processed by hand on 5 August 2026 after a damaged file stopped a run. The evidence is file dates and the clean audit. | `salmonid/detect/track_from_store.py` header | — |

---

## 4. Decisions and actions for the authors

### Deposit and access (CJFAS conditions)

| ID | Item | What is needed |
|---|---|---|
| A1 | **Model weights are not deposited.** | Deposit `checkpoint0017.pth` with its checksum. Because it derives from DINOv3 it must carry the DINOv3 License; Dryad (CC0 only) cannot host it on those terms, Zenodo can. Reviewers need access at submission. |
| A2 | **Training set is not deposited.** | Decide what can be released (lists, synthetic pictures, frames derived from OMNR video). If frames cannot be released, say so in the statement. The lists must be rewritten with relative paths. |
| A3 | **Stored detections and tracks are not deposited.** | Depositing the track files (about 370 MB), the video lists and the video-to-date tables (about 21 MB) lets a reader rebuild every count without the video. Recommended. |
| A4 | **Raw video and technician records.** | The manuscript must name an institutional contact, not an author's address, and describe the restriction. The current statement says a portion of the video will be deposited: confirm or remove. |
| A5 | **Second exclusion list** (`ls_test_set_v5.json`) is external, so the leak audit stops on a fresh clone unless told to continue, and the byte-for-byte rebuild needs it. It contains a reviewer's first name and the labels of an older evaluation set. | Commit a keys-only version (video and track identifiers), or keep it external. |
| A6 | **Staff first names in video paths.** Twenty-three test videos from 2020 sit in archive folders named after staff. These paths are in four committed files and have been public since the first commit. | Accept, or pseudonymize consistently (the exclusion rule matches on these strings). |
| A7 | **Tables derived from OMNR review exports** (`data/paper/pervideo_2025_*.csv`, `stepB_*.csv`): technician species, count and fish/no-fish flag per video. | Confirm they may be published. |
| A8 | `data/sample/` is empty. | Add a small demonstration bundle, or leave the folder as documented. |
| A9 | **No licence.** The repository is public with a placeholder `LICENSE`. | Choose, with OMNR sign-off. Considerations in `SOFTWARE_CITATIONS.md`: the tracking library is AGPL-3.0 and is imported by one module; the weights fall under the DINOv3 License; DEIMv2 configuration files and a patch are redistributed and need the Apache notices; `data/` needs a data licence. |
| A10 | **No DOI.** | Tag a release, archive it, and complete the TODO lines in `CITATION.cff` (ORCIDs, licence, DOI, release date; two authors have no affiliation in the manuscript). |
| A11 | **Verbatim records contain the original workstation's paths and user name** (`provenance/v2.7/run_manifest.json`, `runners/original/`, some configuration comments). | Keep as evidence, or redact. Commit metadata also carries a personal e-mail address. |

### Manuscript text

| ID | Item | Evidence |
|---|---|---|
| A12 | **Three statements do not match the computation as worded.** (1) Section 2.1, Ganaraska Rainbow Trout "5,500 to 9,800 a year": the table gives 1,172 to 9,751 unless 2020 is excluded. (2) Section 3.4, alternatives "10% to 26%": the seven alternatives give 10.5% to 33.6%. (3) Section 3.5, quiet-day sums "27% above ... equal": no single definition of day size gives both figures. | `PAPER_TO_CODE.md`, MISMATCH rows |
| A13 | **Eight statements have no computation in any repository**: the crowding ratios in Section 3.2 (carried from the superseded model); "240,000 videos" (the record says 246,070); the workday and processing-time figures; "about 900 videos per GPU-hour"; the two training-composition counts (confirmed by direct count, but no script); the review of 30 disagreements; the classifier predictions behind Table S4. | `PAPER_TO_CODE.md`, NO CODE rows |
| A14 | **Methods 2.6 says 2,000 bootstrap resamples.** The moving-block and inter-annual intervals used 4,000. | F4 |
| A15 | **Statements to add or correct, from section 3**: L2 (25 training frames from a test video are not mentioned); L4 (the test set was used to choose the model copy; not mentioned); L5 (box placement at the Credit River; check whether to mention); L10 (frame rate and the frame-based rules differ by season); L11 (check the dating and day-set wording against `COUNTING_METHOD.md`); L12 (the tracker comparison's settings were not all the same). | section 3 |
| A16 | The blind-review interval in the manuscript came from a corrected calculation; the saved script in the manuscript repository has a one-word bug that gives a zero-width interval. | `salmonid/paper/blind_review.py` header (`--as-saved` shows it) |
| A17 | Software used but not cited in the manuscript: the tracking library (BoxMOT) above all; NumPy, pandas, Matplotlib; several statistical methods. One existing reference has the wrong DOI (SciPy). Figure 1 needs the OpenStreetMap attribution, and the source of two of its map layers is unrecorded. | `SOFTWARE_CITATIONS.md` |
| A18 | Four analysis modules were written for this deposit from the Methods text because no saved script existed. They reproduce the printed values; a co-author should read them. | `salmonid/paper/README.md` |
| A19 | **Blind review (Section 3.2): how the sample is described.** 206 clips were drawn and 155 reviewed; 150 is the number with at least one paired fish. Of the 51 clips not reviewed, 48 are late-season Ganaraska clips, so the reviewed set is not an even subset of the stratified draw. Twelve clips were added because the earlier model reported a rare species in them, and all 6 Atlantic Salmon the reviewer identified are in those clips, so "100% for Atlantic Salmon (n = 6)" measures confirmation of reported fish, not detection. The sample was drawn from the earlier model's tracks. | `salmonid/results/build_reference_sample.py`, `score_reference_sample.py` headers; `data/paper/blind_v27_clips.csv` |
| A20 | **Held-out test set (Section 2.4): details the text omits.** Candidate tracks also had to be confidently detected by an earlier model, were drawn 100 per species as that model predicted them, and half were its least certain cases. Of 500 queued, 293 were reviewed and 78 set aside. The never-labelled tier was labelled by one reviewer; the queue showed the earlier model's species, and the database records the labels as "confirm" or "specify" actions, so the labels may not have been made blind to that model (to confirm). The 151 rare-species labels were not reviewed again for the test set. Test tracks are box sequences from an earlier detection run. | `salmonid/eval/README.md`; `finalize_testset_v6.py`, `build_holdout_testset_queue.py` headers |

---

## 5. Open work on this repository

| ID | Item | Acceptance check |
|---|---|---|
| O1 | **Detection (Pass A) and the BoT-SORT mode of tracking (Pass B) were not re-run.** The ported modules are verified by reading and by the comparison-tracker run, not by regenerating a stored result. | Run one video through `detect_store` and compare with its existing stored file; run one river-season shard through `track_from_store` and compare with the stored track file. |
| O2 | **The DEIMv2 environment on the original workstation loaded intermittently on 2026-11-10.** In one session the machine's Application Control policy blocked one of its compiled libraries; in another the same day the detector loaded and the box check ran. Fresh downloads of one Tier-1 library were blocked the same way (F16). | Establish why on the workstation (not by bypassing the policy), then do O1. |
| O3 | The training launcher only warns if the wrong PyTorch version is active. | Make the mismatch stop the run. |
| O4 | `sam2_composite --no-holdout-filter` still disables the exclusion. | Remove the option, or require an explicit confirmation. |
| O5 | No unit tests. The statistics are verified only by regenerating published outputs. | A test per estimator against a hand-computed case (Lin's CCC, Wilson interval, moving-block resample, Bland-Altman limits, Cohen's kappa). |
| O6 | The sanctuary-window percentages are computed in the figure script and again in `count_tables`. | One function, used by both. |
| O7 | No continuous-integration check. | A workflow that runs `python run_tier1.py` on Linux, macOS and Windows. |
| O8 | Tier 1 has been verified on Windows only. | Run on Linux and macOS (O7 covers this). |

---

## 6. Development checks ("gates")

The research project used several scripts as go/no-go checks before trusting a model. They
are not manuscript results, and the published model does not pass all of them. Each was
re-run on 2026-11-10; details are in `salmonid/eval/README.md` and the module headers.

| Check | Threshold | Result for the published model |
|---|---|---|
| Box placement against hand-drawn boxes (`deimv2_bbox_gate`, 150 frames from 2025) | median overlap (IoU) at least 0.80 per river | **Fails at the Credit River**: 0.737 (42 boxes). Ganaraska River 0.805 (126 boxes), passes. Vertical placement is correct at both. |
| Daily counts against technician counts (`count_r_gate`, on the committed 2025 tables) | daily correlation at least 0.90 and season total within 15%, per river and species | **Fails**: 9 of 15 rows flagged. Chinook Salmon and Rainbow Trout pass at both rivers; the flagged rows are Coho Salmon, Atlantic Salmon and Brown Trout. Other seasons also fail on the rarer species. |
| `score_tracked`, `corpus_year_gate`, `bbox_regression_gate` | — | Not applicable to the published model (they read outputs of the earlier pipeline). Kept for the record. |

These thresholds were development targets, not claims of the manuscript, and the manuscript
reports the underlying numbers for each species. What is not in the manuscript is the box-
placement result at the Credit River (L5). The checks also have known blind spots, stated in
their headers: the box check ignores boxes drawn where there is no fish and has no minimum
sample size, and the count check's interval treats days as independent.

---

## 7. How this review was done

- Every Tier-1 command was run on a clean copy of the committed files in a new environment,
  with all external locations pointed at a non-existent folder.
- The committed count tables and predictions were regenerated from the stored model output on
  the original workstation (read-only) and compared.
- The published training lists were audited and rebuilt from the earlier lists.
- Each ported module was compared line by line with its counterpart on both branches of the
  research repository.
- Every manuscript number was traced to a command and the command run.
- Every dependency's citation and licence was checked at its source.
- An earlier internal code review (2026-09-29) was used as a checklist; each of its findings
  in scope was re-checked against the current files.

Not done: training, detection and BoT-SORT tracking were not re-run (O1, O2). Nothing was
committed; the changes are in the working tree for the authors to review.

# Data and model availability

This page lists every data file the work depends on: what is in this repository, what is not,
why, and what a reader needs to do to obtain it. It also records how the deposit stands
against the data and code policy of the Canadian Journal of Fisheries and Aquatic Sciences
(CJFAS).

Items marked **TO DO (authors)** are not yet in place. They are decisions or actions only the
authors and the Ontario Ministry of Natural Resources (OMNR) can take.

---

## 1. In this repository (Tier 1)

Everything needed to recompute the classification score, the agreement statistics, the count
tables and the count figures. No imagery. `data/README.md` and `data/paper/README.md` describe
every file and column; `data/CHECKSUMS.sha256` fingerprints every file.

| Files | What they are | Size |
|---|---|---|
| `data/test_set_v6_gt.json`, `test_set_v6_holdout.json` | The 366-fish held-out test set: reviewer labels and the list of excluded videos | 0.2 MB |
| `data/preds_v27_ckpt0017.csv` | The published model's species call for each test fish | 30 KB |
| `data/net_rule_sweep/` (48), `activity_sweep/` (6), `recovered_2020/` (2) | Daily model and technician counts per season, under each candidate direction rule | 1.2 MB |
| `data/canonical_daily.csv`, `canonical_summary.csv` | The daily and seasonal counts every count result is computed from | 0.2 MB |
| `data/concordance_ccc.csv`, `agreement_*.csv`, `classification_ci.csv`, `net_rule_loyo.csv` | The published statistics tables | under 0.1 MB |
| `data/paper/` | Per-video and per-fish comparison tables behind Section 3.2, Tables 2 to 4 and S4, and the published annual technician totals (Tables S1, S2) | 1 MB |
| `provenance/v2.7/` | The record of the published training run and the species calls of each candidate checkpoint | 0.3 MB |
| `configs/` | The model configuration, as run | under 0.1 MB |

Two points the authors should confirm before the deposit is made public
(**TO DO (authors)**; see the review document, items A6 and A7):

- Video identifiers in several files are paths inside the footage archive. Twenty-three test
  videos from 2020 sit in archive folders named after staff members (first names only).
- `data/paper/pervideo_2025_*.csv` and `stepB_*.csv` are derived from OMNR's 2025 technician
  review exports (technician species, count and fish/no-fish flag per video; no names).

## 2. Not in this repository (Tier 2)

| Item | Size | Needed for | Status |
|---|---|---|---|
| **Published model weights** `checkpoint0017.pth` (SHA-256 `5cd9bda8…`, `provenance/v2.7/CHECKSUMS.md`) | 819 MB | Running the model on video | **TO DO (authors): deposit.** See the licence note below. |
| **Training set**: image lists `train.json`, `val_clean.json` (checksums in `provenance/v2.7/`) and the images they name | 20 MB + about 4.9 GB | Retraining; auditing the training data | **TO DO (authors): deposit or state the restriction.** The lists name images by absolute paths on the original workstation and must be rewritten to relative paths for a deposit. |
| **Stored detections and tracks** for 2020 to 2025 | track files about 370 MB; per-video detection files much larger | Rebuilding the count tables from tracks | **TO DO (authors).** Depositing the track files, the video lists and the video-to-date tables (about 21 MB) would let a reader rebuild every count without the video. |
| **Stored detections for the test videos** and the reference tracks they are matched to | small | Rebuilding `preds_v27_ckpt0017.csv` | **TO DO (authors)** |
| **Raw fishway video**, both rivers, 2020 to 2025 (246,070 files) | several terabytes | Detection and tracking from scratch; file dates | Held by OMNR, Lake Ontario Management Unit. **TO DO (authors): name the access route.** |
| **Technician records**: official daily counts 2020 to 2023, the 2024 Credit River review export, the 2025 per-fish review exports | small | Rebuilding the technician columns of the count tables; Table S3 | Third-party (OMNR) records. **TO DO (authors): name the institutional contact.** The annual totals are published (Tables S1, S2). |
| **Labelling database** and the earlier detection run the test tracks were drawn from | — | Rebuilding the test set | Not deposited; the frozen test set is the result. |
| **SAM2 cutout library** (fish cut out of frames) | about 3 GB | Figures 3 and S1; rebuilding synthetic training images | **TO DO (authors)** |
| **Three classifiers of the backbone comparison** (Table S4) and their inference code | — | Regenerating `data/paper/backbone_track_predictions.csv` | Not deposited. The per-fish predictions are; Table S4 recomputes from them. |
| **Starting weights**: COCO-pretrained DEIMv2-X; DINOv3 ViT-S+/16 backbone | 206 MB, 115 MB | Retraining | Public, from their authors. Obtain the DINOv3 weights from Meta's release under the DINOv3 License. Checksums in `provenance/v2.7/CHECKSUMS.md`. |
| **Re-identification weights** `osnet_x0_25_msmt17.pt` (SHA-256 in `salmonid/config.py`) | 3 MB | Tracking | Public, distributed with BoxMOT. Not redistributed here. |
| **DEIMv2 code** at commit `0fff8d4` plus `docs/deimv2_worktree.patch` | — | Training and detection | Public; see `DEIMV2_PIN.md`. |
| **Map layers** for Figure 1 | small | Figure 1 | OpenStreetMap layer included (`data/paper/spatial/`, ODbL). The lake and Credit River layers are not: their source is not recorded. **TO DO (authors)** |

### Licence constraints that affect where things can be deposited

Set out with sources in `SOFTWARE_CITATIONS.md` ("Licence notes"). In brief:

- **The fine-tuned model weights are a derivative of DINOv3**, whose licence allows
  redistribution of derivatives only under the same DINOv3 License, with a copy attached. A
  repository that accepts only CC0 data (Dryad) cannot host them under those terms; Zenodo
  accepts custom licences.
- **BoxMOT, the tracking library, is AGPL-3.0.** This repository imports it and does not
  redistribute it. The licence chosen for this repository needs to take that into account.
- **DEIMv2 was Apache-2.0 at the pinned commit**; the upstream project later changed its
  licence. The pinned commit is the one that applies.
- **OpenStreetMap data** in Figure 1 requires the attribution "© OpenStreetMap contributors".

## 3. How the deposit stands against the CJFAS policy

| CJFAS requirement | Status |
|---|---|
| Data and code needed to reproduce the results are public at publication, in a repository that issues a DOI, with a reuse licence | Code and Tier-1 data are in a public GitHub repository. **TO DO (authors):** choose the licence; archive a tagged release in a DOI-issuing repository (Zenodo, or Dryad for the data); add the DOI to `CITATION.cff` and the manuscript. |
| Code to reproduce statistical results and figures | Present and verified: `python run_tier1.py` regenerates every statistics table and every data-driven figure and checks them. Figures 1, 3 and S1 need Tier-2 inputs. |
| Available to editors and reviewers at submission | The public repository serves. Reviewers who want to run the model need the weights: **TO DO (authors)**, for example a restricted Zenodo record with a reviewer link. |
| Exemptions are described in the data availability statement (third-party data, large data, only the portion used) | The raw video (large; held by OMNR) and the technician records (third-party) fall under these. The statement must say so and give an institutional contact. |
| The authors' own e-mail may not be the only access route | **TO DO (authors):** name an OMNR Lake Ontario Management Unit contact point, not an individual author. |
| No "data not shown" | Eight manuscript statements have no computation behind them in this repository; they are listed in `PAPER_TO_CODE.md` ("Gaps, ranked") for the authors to support or remove. |
| Datasets used are cited (FORCE11 format) | `CITATION.cff` and `software.bib` cite the software, models and datasets. The deposit itself needs a citation in the manuscript once it has a DOI. |
| Species distribution records in GBIF or similar (encouraged) | Not applicable to video counts as they stand; the annual totals are published in the Lake Ontario Management Unit annual report. |

## 4. Wording for the manuscript's Data Availability statement

A draft that matches what is actually deposited. Replace the bracketed items once they exist.

> The code used for the analysis, the model configuration, the record of the training run, and
> the derived data underlying the reported results (the held-out test set labels and model
> predictions, the daily and seasonal model and technician counts, and the per-video comparison
> tables) are available at https://github.com/tomgrizz/salmonid-classifier-code and are archived
> in [repository] at https://doi.org/[DOI]. The trained model weights are archived in
> [repository] at https://doi.org/[DOI] under the DINOv3 License. The technician-reviewed annual
> totals are published by the Lake Ontario Management Unit [@omnrf2026lomu]. The raw fishway
> video and the technicians' passage-level review records are held by the Ontario Ministry of
> Natural Resources and were used under [permission / agreement]; because of their size and
> ownership they are not deposited, and can be requested from [institutional contact, Lake
> Ontario Management Unit] by researchers who meet the Ministry's criteria for access.

Check this draft against the current manuscript statement, which also names a second
repository for the review applications and says a portion of the video will be deposited.

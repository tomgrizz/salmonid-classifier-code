# Data & weights availability

This repository ships the **small analysis artifacts** needed to reproduce the paper's numbers
and figures (Tier 1). The **large artifacts** — model weights, raw footage, the training COCO,
and the SAM2 cutout library — cannot live in git and are provided as external downloads (Tier 2).
The prose in the "Committed / External" tables below is also the basis for the manuscript's
Data & Code Availability statement.

## Committed in this repo (`data/`) — Tier 1, no download

| File | Bytes | What it is | Consumed by |
|---|---|---|---|
| `test_set_v6_gt.json` | 77 KB | Frozen per-track species ground truth (366 tracks, pristine + rare-class heldout tiers) | `eval/score_testset_v6.py`, confusion figure |
| `test_set_v6_holdout.json` | 112 KB | The `(video_id, track_id)` exclusion keys the leak guard drops from all training | `salmonid/holdout.py` |
| `preds_checkpoint0007.csv` | 30 KB | v2.6 per-track species predictions on the held-out set | `eval/score_testset_v6.py`, confusion figure |
| `canonical_daily.csv` | 188 KB | Per-day model-vs-technician counts, both rivers, 2019–2025 | hero/daily figures, concordance |
| `canonical_summary.csv` | 4.7 KB | Per-(year, river, species) seasonal totals + error/r metrics | scorecard, concordance, results compiler |
| `concordance_ccc.csv` | 0.9 KB | Lin's CCC agreement table (committed output; `make concordance` regenerates it) | reference / figures |
| `net_rule_sweep/*.csv` | ~1 MB (48) | Net-threshold sweep across `|Δx|` rules | `figures/render_sensitivity_figure.py` |
| `2024_botsort_net.csv` | 26 KB | 2024 held-out daily net (BoT-SORT) | 2024 daily-net panel |

None of these contain imagery or personal data — they are track-level labels and daily counts.

## External downloads — Tier 2 (set the matching `SALMONID_*` env var; see `.env.example`)

| Artifact | Approx size | Env var / config | Needed for | Where to obtain |
|---|---|---|---|---|
| Published checkpoint `deimv2_fish_x_v26/checkpoint0007.pth` | 819 MB | `SALMONID_DEIM_DIR` → `outputs/…` (`config.CHECKPOINT_V26`) | inference, corpus counts | **[deposit — Zenodo/OMNRF, add DOI]** |
| Warm-start `deimv2_fish_x_v25/checkpoint0019.pth` | 819 MB | `SALMONID_DEIM_DIR` (`config.WARMSTART_V25`) | retraining v2.6 | **[deposit]** |
| DEIMv2 init ckpts (`deimv2_x_coco.pth`, `dinov3_vits16plus_pretrain_*.pth`) | ~205 MB + ~115 MB | `SALMONID_DEIM_DIR/ckpts` (`config.DEIM_CKPTS`) | training from the COCO init | Intellindust-AI-Lab / HF `xysun917/dinov3-vits16plus` |
| Frozen training COCO `deimv2_fish_6class_masked_v26/{train.json,val_clean.json}` | ~20 MB | `SALMONID_COCO_CACHE` (`config.COCO_V26`) | retraining v2.6 (frozen artifact — see note) | **[deposit]** |
| Raw tunnel-camera footage (Ganaraska + Credit, 2019–2025) | multi-TB | `SALMONID_FOOTAGE_ROOT[_FAST]` | detection, full-corpus counts | OMNRF data-sharing (**[policy/contact]**) |
| SAM2 cutout library + `manifest.csv` | ~3 GB | `SALMONID_PROJECTS_ROOT/sam2_cutouts` (`config.SAM2_CUTOUTS`) | morphology / bbox-grid photo figures | **[deposit]** |
| Corpus detect-stores (`deimv2_x_v26_corpus<year>/store`) | large | `SALMONID_PROJECTS_ROOT` | reproducing daily counts from stored detections | **[deposit / regenerate via runners]** |
| Pre-2024 technician NET counts (`Ganny_Total.csv`, `FINAL_CREDIT_*.csv`) | small | `SALMONID_LOMU_INTERVIEW_DIR`, `SALMONID_MODEL_PRO_DIR` | historical (2018–2023) validation | OMNRF |

> **Frozen-COCO note.** The SAM2 compositor is not seeded, so a rebuilt COCO differs pixel-wise.
> To reproduce the v2.6 numbers *exactly*, use the deposited `train.json`/`val_clean.json`
> (their sha256 is pinned in the run manifest). To rebuild the *distribution*, follow
> `docs/REPRODUCE.md` step 1.

## Model weights & training repo

The single-stage model trains in the external **DEIMv2** repository
(`Intellindust-AI-Lab/DEIMv2`, Apache-2.0), pinned to the commit in `docs/DEIMV2_SHA.txt` with
the required working-tree patch `docs/deimv2_worktree.patch`. See `docs/DEIMV2_PIN.md` and
`make setup-deimv2`.

## Suggested manuscript Data & Code Availability statement (fill the brackets)

> Code to reproduce the model evaluation, counting analysis, statistics, and figures is available
> at `https://github.com/<org>/salmonid-classifier-code` (archived at Zenodo, DOI **[…]**). The
> trained model weights and derived classification outputs are deposited at **[Zenodo/Ontario Data
> Catalogue, DOI …]**. Raw tunnel-camera video is held by the Ontario Ministry of Natural
> Resources (Lake Ontario Management Unit) and is available from the authors on reasonable request,
> subject to the Ministry's data-sharing policy.

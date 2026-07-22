# salmonid-classifier-code

Reproduction code for the manuscript **"Against the Current: deep-learning enumeration of
migratory salmonids at Lake Ontario fishways, validated on technician counts"**
(Ontario Ministry of Natural Resources — Lake Ontario Management Unit; target journal
*Fisheries Research*).

This repository contains the code and the small analysis data needed to reproduce the paper's
model evaluation, counting/movement analysis, statistics, and figures for the **published
single-stage model (DEIMv2-X, "v2.6")**. It is a curated, annotated extract of a larger working
research repository — every module here has been reviewed and commented for external readers.

> **Reviewing the paper?** Start with **`docs/REVIEW_GUIDE.md`** 

## What the model is

One **DEIMv2-X** object detector (DETR-family, on a self-supervised **DINOv3 ViT-S/16+**
backbone) that performs **detection + species classification in a single pass**, coupled to a
multi-object tracker for counting. It classifies five salmonid species —
`0 Chinook · 1 Coho · 2 Atlantic · 3 Rainbow Trout · 4 Brown Trout` (a 6th detection class,
"Small Salmonid", is detected but excluded from the official report). On the frozen held-out
test set (`test_set_v6`, 366 tracks) it reaches **macro-F1 0.966**.

## Reproducibility — two tiers

**Tier 1 — turnkey from committed data (no downloads).** The small published artifacts (the
frozen held-out test set + its predictions, the canonical daily/summary results, the net-rule
sweeps) live in `data/`, so the headline score, the concordance statistics, and every figure
that does not need raw imagery reproduce on a fresh clone:

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -r requirements.txt && pip install -e .

make score          # -> held-out species macro-F1 0.966 (test_set_v6)
make concordance    # -> Lin's CCC table + concordance / Bland-Altman figures
make figures        # -> manuscript figures regenerated from data/canonical_*.csv
```

(No `make`? Run the underlying `python -m salmonid.…` commands the Makefile prints.)

**Tier 2 — full pipeline (needs external downloads).** Training the model, running full-corpus
inference, or regenerating the photo figures (species morphology, detection grid) needs the
819 MB checkpoint, the raw tunnel-camera footage, the frozen training COCO, and the SAM2 cutout
library — none of which can live in git. Point at them with the `SALMONID_*` environment
variables (copy `.env.example` to `.env`); see **`docs/DATA_AVAILABILITY.md`**. Training also
needs the external DEIMv2 repo reconstructed at a pinned commit — see **`docs/DEIMV2_PIN.md`**
and `make setup-deimv2`.

## Repository layout

```
salmonid/            the annotated code, by pipeline stage
  config.py            central path + constant resolver (the one place paths live)
  holdout.py           the leak guard — excludes eval tracks from ALL training
  data/                training-data prep: masked COCO build + SAM2 synthetic composites
  detect/              single-stage inference: DEIMv2 decode, detect-store, BoT-SORT tracking
  eval/                held-out evaluation (the 0.966 score) + mandatory pre-deploy gates
  count/               counting & net movement; model-vs-technician daily aggregation
  results/             canonical results assembly + concordance (Lin's CCC) statistics
  figures/             manuscript figure generators (shared CJFAS style helper)
  provenance/          per-run reproducibility manifest
  util/                shared helpers (parquet schema, notifications)
configs/             the DEIMv2 fish config chain (snapshotted) incl. the v26 training recipe
runners/             PowerShell launchers (train v26; multi-year corpus detect + track)
data/                committed small artifacts (Tier 1) + the data/sample/ demo bundle (Tier 2)
docs/                REPRODUCE.md · DATA_AVAILABILITY.md · CODE_MAP.md · DEIMV2_PIN.md ·
                     counting_and_movement.md · MANUSCRIPT_RESULTS.md
```

Start with **`docs/CODE_MAP.md`** for how the modules chain together and which figure/number
each produces, and **`docs/REPRODUCE.md`** for the end-to-end training runbook.

## Environment

Python 3.11. Two virtual environments are used because of a hard numpy conflict:
- **base env** (`requirements.txt`, numpy < 2) — evaluation, counting, results, figures, and
  BoT-SORT tracking (boxmot 10.0.52).
- **`.venv_deim`** (`requirements-deim.txt`, numpy 2.x, torch 2.7 + cu128) — DEIMv2 training and
  Pass-A detection, hosting the external DEIMv2 engine. Developed on an RTX 5090 (CUDA 12.8).

## External dependency

Model training uses **DEIMv2** (`Intellindust-AI-Lab/DEIMv2`, Apache-2.0), pinned to a specific
commit plus a required working-tree patch — both captured in `docs/DEIMV2_PIN.md`,
`docs/DEIMV2_SHA.txt`, and `docs/deimv2_worktree.patch`. The fish config chain is snapshotted
here under `configs/`.

## License & citation



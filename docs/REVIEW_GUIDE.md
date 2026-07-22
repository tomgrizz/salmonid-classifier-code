# Reviewer's guide — checking the paper against the code

**Who this is for:** a co-author (e.g. Quinn) who wants to go through the paper's technical
details and confirm that every figure, table, and headline number actually comes out of the code
and data — in plain language, no machine-learning background assumed.

**The short version.** Most of the paper's *numbers and figures* can be re-generated on your own
laptop in a couple of minutes from data already included in this repo — you do **not** need the
video footage or the trained model for those. A few things (training the model, the photo figures,
the per-clip accuracy tables) need large files that are too big to include; those are marked
clearly below. And there are a few places where the **paper text and the shipped data disagree** and
need a human decision — those are listed at the end under "Please reconcile."

---

## 1. One-time setup (about 2 minutes)

```bash
cd salmonid-classifier-code
python -m venv .venv && .venv/Scripts/activate     # Windows; on Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Then, to regenerate everything that can be regenerated from the included data:

```bash
make score          # the held-out species accuracy (the 0.966 headline)
make concordance    # the agreement statistics (Lin's CCC, Bland–Altman)
make figures        # the manuscript figures that don't need photos
```

Every command prints where it read its inputs and where it wrote its outputs, so you can trace
each number back to a file.

## 2. What the tags mean

| Tag | Meaning |
|---|---|
| ✅ **included** | Re-generates now from data in this repo. Run the command and compare to the paper. |
| ⬇ **needs big files** | The logic/code is here and readable, but re-running needs a large download (model weights, raw video, or the 3 GB image-cutout library). See `docs/DATA_AVAILABILITY.md`. |
| 📄 **lives in the paper repo** | Not produced by this code — it's made in the manuscript repo or hand-entered. |
| ⚠ **reconcile** | The code/data and the paper text disagree; a human needs to decide. See Section 6. |

---

## 3. Figures

| Fig | What it shows | How to check it | Status |
|---|---|---|---|
| **1** | Study-area map | Made in the paper repo (`scripts/make_fig1_map.py`) | 📄 paper repo |
| **2** | Species confusion matrix (v2.6, acc 0.967) | `make figures` → `figures/fig_confusion_v26.pdf`; numbers also from `make score` | ✅ included |
| **3** | 2025 operational agreement (accuracy + daily-r bars) | Caption says "to be regenerated from v2.6"; no finished generator ships yet. Underlying numbers = Table 4 | ⚠ figure pending |
| **4** | Multi-year Chinook net + 2024 single-stage stars | `make figures` → `figures/fig_multiyear_chinook_net.png` | ✅ included, but ⚠ see §6 (model) |
| **5** | 2025 Ganaraska run-timing | `make figures` → `figures/fig_seasonal_timing.png` | ✅ included |
| **6** | Body-shape comparison (depth-to-length) | `python -m salmonid.figures.fig_species_morphology` (needs the cutout library) | ⬇ needs big files |
| **7** | Concordance (CCC) scatter, 3 species | `make concordance` → `figures/fig_concordance.png` | ✅ included |
| **S1** | Detection sample grid | `python -m salmonid.figures.fig_bbox_grid` (needs cutouts + footage) | ⬇ needs big files |
| **S2** | 2025 cumulative net, both rivers | `make figures` → `figures/fig_daily_net_2025_*.pdf` | ✅ included |
| **S3** | 2024 held-out cumulative net | `make figures` → `figures/fig_daily_net_2024_*.pdf` | ✅ included |
| **S4** | Bland–Altman, daily Chinook net | `make concordance` → `figures/fig_bland_altman.png` | ✅ included |

## 4. Tables

| Table | What it shows | How to check it | Status |
|---|---|---|---|
| **1, 2** | Historical technician upstream totals (Ganaraska / Credit) | Hand-entered technician records (paper repo `tables/`) | 📄 reference data |
| **3** | Held-out per-species precision/recall/F1, macro 0.966 | `make score` reproduces the **v2.6** numbers (macro 0.966) | ✅ included, but ⚠ see §6 (v2.5 vs v2.6) |
| **4a** | 2025 per-video accuracy by model & site | Needs the 2025 per-clip model output + technician per-clip export | ⬇ needs big files |
| **4b** | 2025 per-species agreement (recall/precision, daily-r, % error) | Daily-r and % error come from the included `canonical_daily.csv`; recall/precision need the per-clip export | ⬇ partly (r & %err reproducible) |
| **5** | Multi-year Chinook net (% error, daily-r) 2020–2025 | Values live in the included `canonical_summary.csv` | ✅ data included, but ⚠ see §6 (model) |
| **6** | Concordance (CCC) for the 3 abundant species | `make concordance` reproduces this table **exactly** | ✅ included |
| **S1** | Backbone comparison (ResNet vs DINOv2 vs DINOv3) | A model-development experiment; those older models aren't shipped | ⬇ not reproducible here |

## 5. Key numbers (abstract & results)

| Claim in the paper | How to check it | Status |
|---|---|---|
| **Macro-F1 0.966** on the held-out set | `make score` prints exactly 0.966 | ✅ included |
| Confusion confined to Chinook↔Coho + a few Atlantic | `make score` prints the matrix | ✅ included |
| Daily upstream-count *r* = 0.96–0.99 (main species) | In `canonical_daily.csv`; summarised in `canonical_summary.csv` | ✅ included |
| Concordance CCC 0.94–0.96, *C~b~* ≥ 0.99 | `make concordance` reproduces the table | ✅ included |
| Bland–Altman bias −2.0 fish/day, LoA [−122, +118], n=851 | `make concordance` prints exactly this | ✅ included |
| Ganaraska over-counts (Chinook +27%, Brown +68%); 1.48 vs 1.02 tracks/fish | % error in `canonical_summary.csv`; tracks-per-fish needs the corpus stores | ⬇ partly |
| 2025 per-video accuracy 98.5% / 98.9% / 91.1% | Needs the 2025 per-clip export | ⬇ needs big files |
| "≈30% of disagreements were technician errors"; "94.7% blind" | From the blind reference-sample review (`salmonid/results/score_reference_sample.py`); the review data isn't included | ⬇ needs big files |
| Training set 41,738 images / 45,969 fish (per-species counts) | From the frozen training dataset (a deposited download); the exact recipe is in `configs/` + `docs/REPRODUCE.md` | ⬇ needs big files |
| Direction rule = max(0.234 × width, 300 px); why | Readable in `salmonid/count/` + `config.py`; the sensitivity figure (`make figures` → `figures/fig_net_rule_sensitivity.png`) shows it is the best of the family | ✅ included |

## 6. Please reconcile (paper text vs shipped data) — the important part

These are not code bugs; they are places where a human (you) needs to decide what the paper should
say, because the code/data has moved on since the text was written.

1. **Multi-year model: the shipped data is single-stage v2.6, but the paper says two-stage.**
   The paper (Table 5, Fig 4, §3.5, §2.6) states the 2020–2024 multi-year results are from the older
   **two-stage** pipeline, with the single-stage reprocessing "still underway." But the data included
   here (`canonical_daily.csv`, and therefore Table 6 / Fig 7 concordance, the hero and cumulative-net
   figures) is **single-stage v2.6 for all years 2019–2025** (see the header of
   `salmonid/results/build_canonical_results.py`). In other words the single-stage multi-year run has
   since finished. Consequences to decide: (a) the Table 5 / Fig 4 numbers will change if regenerated
   from the shipped single-stage data; (b) the manuscript's "still underway / two-stage" wording is now
   out of date; (c) note that Table 6 (concordance) already *is* single-stage, so the paper is
   internally inconsistent (Table 5 two-stage vs Table 6 single-stage). **Decision needed:** update the
   multi-year text/figures to single-stage, or deliberately keep the two-stage version for those.

2. **Table 3 per-species numbers are the v2.5 model; `make score` gives v2.6.**
   The paper says so explicitly (Table 3 caption), but it means the included predictions reproduce the
   v2.6 per-species values (Atlantic F1 0.966, Rainbow 0.977) which differ slightly from the printed
   v2.5 values (Atlantic 0.947, Rainbow 0.963). The macro number (0.966) is v2.6 and matches. The exact
   v2.5 per-species table is **not** reproducible from the included data. **Decision needed:** either
   refresh Table 3 to the v2.6 per-species breakdown, or keep the note as is.

3. **The two-stage comparison is being removed from the code but is still in the paper.**
   Per the project decision, the two-stage baseline was intentionally left out of this code repo. The
   paper still reports it as the comparison baseline (§2.3, Table 4a rows, Fig 3, several sentences).
   **Decision needed:** trim or keep those comparisons in the manuscript.

4. **Fig 3 is not finished.** Its caption says "to be regenerated from the single-stage v2.6 outputs,"
   and no generator ships for it. **Decision needed:** produce Fig 3 (the underlying numbers are in
   Table 4), or drop it.

5. **Data/Code Availability is still a placeholder** in the manuscript (§ Data Availability). Ready-to-
   paste text is in `docs/DATA_AVAILABILITY.md` (last section). The model weights, training data, and
   footage still need to be deposited (Zenodo / Ontario Data Catalogue) and their DOIs filled in.

## 7. Where the included data comes from (provenance)

Everything under `data/` is a small, published artifact — no video, no imagery:

- `test_set_v6_gt.json` + `preds_checkpoint0007.csv` → the held-out score & confusion (Fig 2, Table 3).
- `canonical_daily.csv` / `canonical_summary.csv` → all the daily-count figures & tables (single-stage
  v2.6, 2019–2025, both rivers; technician = the deduplicated official counts).
- `concordance_ccc.csv` → Table 6 / Fig 7 (regenerated exactly by `make concordance`).
- `net_rule_sweep/` → the direction-rule sensitivity figure.
- `2024_botsort_net.csv` → the 2024 held-out figures (Fig S3).

For how the modules fit together, see `docs/CODE_MAP.md`; for the end-to-end training recipe, see
`docs/REPRODUCE.md`; for the big downloads, see `docs/DATA_AVAILABILITY.md`.

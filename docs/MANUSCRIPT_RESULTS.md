# Manuscript results — single-stage v2.6 counter (compiled 2026-07-14)

Single-stage DEIMv2-X detector+classifier (**v2.6**, held-out `test_set_v6` macro-F1 0.966) + BoT-SORT, evaluated as an automated fish counter for two Lake Ontario tributaries (Credit, Ganaraska), **2019–2025**. All numbers below come from one canonical dataset (`docs/manuscript_results/canonical_daily.csv`); the model is compared against the **deduplicated official technician counts** (the fish-truth). `~` marks the two footage-recovery cells (2019 Credit, 2020 Ganaraska) — timing-reliable but magnitude-inflated by clipping (see Caveats).

## Performance summary — net count accuracy, 2022–2025

Automated Counter vs technician, main species, corpus years (recovered/anomaly cells excluded).

| Species | River | Mean \|Error\| | r² |
|---|---|---:|---:|
| Chinook | Ganaraska | 5% | 0.96 |
| Chinook | Credit | 10% | 0.94 |
| Rainbow Trout | Ganaraska | 11% | 0.90 |
| Rainbow Trout | Credit | 14% | 0.92 |
| Coho | Ganaraska | 13% | 0.94 |
| Coho | Credit | 32% | 0.23 |

**Abundance-index validation (the monitoring test):** as an index of relative inter-annual abundance, the automated counter reproduces the technician's year-to-year peaks and troughs — annual r² **0.96 (Ganaraska) / 0.98 (Credit) for Chinook**, mean main-species annual r² **0.88** (mean of the per-panel Chinook/Coho/Rainbow r² across both rivers, 2020–2025; computed in `_render_hero_regulation.py`), and **Chinook year-ranking identical to the technician on both rivers (Spearman ρ = 1.00)**. So the counter detects whether a run is up or down — the core monitoring requirement — even where absolute calibration varies.

Plus: run-timing daily-r 0.95–0.97 (all main species); species discrimination 0.966 held-out / 94.7% blind human; **its independent count confirms ~89% of Chinook are protected by the new Ganaraska sanctuary**.

## 1. Species discrimination

- **Held-out test set** (`test_set_v6`, never trained): macro-F1 **0.966** (Chinook 0.971, Coho 0.950, Atlantic 0.966, Rainbow 0.977, Brown 0.967). Figure `fig_confusion_v26.png`.
- **Blind reference sample** (human adjudication on real footage, model call hidden): **94.7%** species agreement (Atlantic/Brown 100%, Rainbow 98%, Chinook 95%, Coho 90%); the residual is the known kyped-Chinook→Coho confusion (Coho precision floor 0.55 fixes it: 42→80%).

## 2. Seasonal counts — model vs technician NET (% error, daily-r)

The headline count agreement, all years and species. `~` = recovered footage (magnitude-caveated).

### Ganaraska

| species | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Chinook | — | ~+57% (0.813) | +32% (0.962) | +13% (0.991) | -3% (0.959) | -3% (0.985) | -1% (0.975) |
| Coho | — | ~+15% (0.984) | -2% (0.994) | -6% (0.917) | -3% (0.984) | -25% (0.983) | -19% (0.985) |
| Atlantic | — | ~+0% (0.247) | -25% (0.426) | +62% (0.394) | +100% (0.616) | +125% (-0.779) | +33% (0.982) |
| Rainbow Trout | — | ~+48% (0.508) | +28% (0.988) | -11% (0.982) | -25% (0.856) | +2% (0.991) | -6% (0.966) |
| Brown Trout | — | ~+73% (0.832) | +192% (0.632) | +196% (0.854) | +96% (0.693) | +46% (0.819) | +42% (0.956) |

### Credit

| species | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Chinook | ~+17% (0.968) | +5% (0.994) | -3% (0.961) | -15% (0.968) | -4% (0.993) | -6% (0.969) | -15% (0.943) |
| Coho | ~-35% (0.879) | -38% (0.905) | -16% (0.318) | +22% (0.979) | -22% (0.626) | -78% (-0.337) | -6% (0.67) |
| Atlantic | ~+128% (0.278) | -7% (0.584) | -14% (0.68) | +85% (0.741) | +35% (0.252) | +22% (0.287) | -14% (0.808) |
| Rainbow Trout | ~+53% (0.996) | +44% (0.831) | +64% (0.966) | +7% (0.99) | -10% (0.969) | +25% (0.92) | -16% (0.949) |
| Brown Trout | ~+133% (0.141) | -24% (0.0) | +650% (0.409) | +67% (0.128) | +77% (0.472) | -14% (0.857) | -13% (0.748) |

**Chinook (the dominant species) is deploy-grade on the corpus years** — mean |net error| **7%** across 9 river-years (excl. recovered + the 2021 low-volume anomaly), daily-r 0.94–0.99. Ganaraska 2023–2025: −3%, −3%, −1%.

## 3. Daily count agreement (run reproduction)

Daily model-vs-tech net Pearson r for the main species (corpus years). The model reproduces the daily run pattern, not just seasonal totals. Figure `fig_daily_countr_chinook.png`.

| species | Ganaraska (median daily-r) | Credit (median daily-r) |
|---|---|---|
| Chinook | 0.98 | 0.97 |
| Coho | 0.98 | 0.67 |
| Rainbow Trout | 0.98 | 0.97 |
| Brown Trout | 0.85 | 0.75 |

## 3b. Count agreement — concordance (Lin's CCC)

Correlation (Pearson r) measures whether the daily pattern tracks; it does **not** measure whether the
counts *agree in level* — a systematically biased counter can still correlate near-perfectly. Lin's
concordance correlation coefficient (CCC = r × Cb, where Cb is an accuracy factor penalizing departure
from the 1:1 line) tests agreement directly. Computed on daily net, pooled corpus years, per species×river
(`docs/manuscript_results/compute_concordance.py` → `concordance_ccc.csv`; bootstrap 95% CIs; McBride 2005
bands). **Provenance = single-stage v2.6 canonical daily counts** (same source as the rest of this doc).

| species | river | river-days | Pearson r | CCC | 95% CI | Cb | band |
|---|---|---:|---:|---:|---|---:|---|
| Chinook | Ganaraska | 439 | 0.967 | **0.965** | [0.950, 0.978] | 0.997 | substantial |
| Chinook | Credit | 412 | 0.966 | **0.958** | [0.925, 0.984] | 0.991 | substantial |
| Coho | Ganaraska | 316 | 0.966 | **0.956** | [0.923, 0.978] | 0.990 | substantial |
| Coho | Credit | 188 | 0.960 | **0.960** | [0.804, 0.985] | 1.000 | substantial |
| Rainbow Trout | Ganaraska | 797 | 0.948 | **0.942** | [0.904, 0.968] | 0.994 | moderate |
| Rainbow Trout | Credit | 483 | 0.955 | **0.954** | [0.884, 0.975] | 0.999 | substantial |

**Read:** daily CCC 0.94–0.96 with **Cb ≥ 0.99 everywhere** → the daily agreement is genuine concordance,
limited by scatter not systematic bias. CI lower bound clears the 0.90 validity gate for Chinook (both
rivers) and the higher-count river of each other species; Coho-Credit and Rainbow-Credit fall just under
(smaller n). **Bland–Altman, daily Chinook net:** bias **−2.0 fish/day**, 95% LoA [−122, +118] (n=851).

- **Annual (abundance-index) CCC** is *not* robust: n=5–6 seasons → CIs span weak→almost-perfect (e.g. Coho
  Ganaraska CCC 0.81 [0.35, 0.98]). Report inter-annual agreement as **rank** (Chinook Spearman ρ=1.00) +
  r², not annual CCC. Chinook annual CCC is nonetheless strong (Gan 0.99 / Credit 0.97).
- Figures: `fig_concordance.png` (3-panel daily Lin plots) + `fig_bland_altman.png`, rendered into the paper
  repo `salmonid-classifier-paper/figures/` (Figure 7 + Figure S4 there); analysis mirrors the CCC method of
  Godeaux et al. 2026 (*Ecol. Inform.*, acoustic-camera fish counts vs manual).

## 4. Run timing & species composition

- **Run timing** (`fig_run_timing.png`): cumulative-net phenology curves reproduce *when* each species runs on both rivers (spring Rainbow → fall Chinook/Coho), model vs technician.
- **Species composition** (`fig_species_breakdown.png`): the model resolves the shifting daily species mix across the season.

## 5. Supporting results

- **Net-rule sensitivity** (`fig_net_rule_sensitivity.png`): the deployed asymmetric rule (up≥50/down≥300 px) is the family optimum (mean |%err| 4.4% vs 11–26% for any alternative) — it encodes the technicians' own convention.
- **Error decomposition** (reference sample): where the model diverges from the technician it is usually the **technician over-counting** (raw event logs re-trigger ~1.5–3.5× per fish); the model sits within ~9% of the human truth on per-clip activity. Model errors decompose into milling up-over-call (~21%) and crowd under-count (~0.14/clip); over-segmentation is 0.9% (BoT-SORT).

## 6. Methods note — technician data tiers

Two tiers of technician data, treated distinctly: (1) **deduplicated official daily counts** (`daily_Ganaraska.csv`, `Ganny_Total`, `FINAL_CREDIT`) = the **fish-count ground truth** the model is scored against (net); (2) **raw per-event Vaki logs** = passage events (re-trigger-inflated, variable) used only for the up/down split and run timing, never as a fish count. Per-event net is validated to match the deduplicated net where the exports are clean.

## 7. Caveats (all characterized)

- **Recovered years** (2019 Credit +17%, 2020 Ganaraska +57% Chinook, `~`): the archived footage is clipped finely, so a fish spans multiple clips → BoT-SORT (which links within a clip) over-counts. Timing is intact (daily-r 0.81–0.97); magnitude is inflated by a consistent ~1.4–1.6× factor → report as timing-recovered / magnitude-caveated (or apply a cross-clip correction).
- **2021**: smallest run in the record; both the model and technician over-count at low volume (+32% Ganaraska Chinook) with an unclassified-Salmonid bucket; daily-r stays 0.96.
- **Rainbow Trout** under-reads magnitude in high-holding years (steelhead mill/hold — documented, not a model miss); timing reliable.
- **Activity magnitude** is not a unique-fish count for either method (model per-clip tracks; tech beam re-triggers) — report net for magnitude, activity only with the caveat.
- **2019 Ganaraska**: no footage in the archive → model gap (technician data exists).

## Figure index

| figure | shows |
|---|---|
| **`fig_hero_performance.png`** | **HERO** — Automated Counter (line) vs Technician (bars), net, species × river, 2020–2025 (per-panel %err + r²) |
| `fig_hero_activity.png` | same grid, total activity counts |
| **`fig_ganaraska_sanctuary.png`** | **APPLIED** — the counter confirms ~89% of Chinook protected by the new Sept 1–Oct 14 sanctuary |
| `fig_confusion_v26.png` | held-out species confusion (macro-F1 0.966) |
| `fig_multiyear_scorecard.png` | net %err × main species × year, both rivers |
| `fig_daily_countr_chinook.png` | daily Automated-vs-Technician agreement (Chinook, 2024) |
| `fig_run_timing.png` | cumulative-net phenology (2024, 3 species) |
| **`fig_species_phenology.png`** | **average run timing across all years** — spring Rainbow → fall Chinook/Coho, reproduced |
| `fig_species_breakdown.png` | daily species composition (2024) |
| `fig_net_rule_sensitivity.png` | net-rule is the family optimum |
| **`fig_concordance.png`** | **AGREEMENT** — daily Lin-CCC 1:1 plots (Chinook/Coho/Rainbow, both rivers), CCC per river |
| `fig_bland_altman.png` | daily Chinook net Bland–Altman (bias −2.0/day, LoA [−122,+118]) |

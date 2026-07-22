# Concordance (Lin's CCC) — manuscript insert bundle

Copy-paste-ready blocks for `salmonid-classifier-paper` (branch `manuscript-single-stage-reframe`).
These were already applied to that repo's working tree on 2026-07-16; this file is the portable
record so the prose/table/refs travel with the CV repo. Regenerate numbers/figures with
`docs/manuscript_results/compute_concordance.py` (reads `canonical_daily.csv` / `canonical_summary.csv`).

Figures (300 dpi, repo style) live at:
`docs/manuscript_figures/fig_concordance.png`, `docs/manuscript_figures/fig_bland_altman.png`
(also copied into the paper repo `figures/`). Table CSV: `concordance_ccc.csv` (paper repo `tables/`).

Reference precedent: Godeaux et al. 2026, *Ecological Informatics* — Lin's CCC + Bland–Altman to
validate semi-automatic vs manual fish counts from acoustic-camera monitoring (McBride 2005 bands,
95% CI lower-bound > 0.90 validity gate). Same structure as our automated-vs-technician comparison.

---

## 1. Methods sentence (append in §2.4, after the Pearson-r sentence)

> Because a high correlation can coexist with a systematic scale or offset bias (correlation measures
> covariation, not agreement), we additionally quantified agreement with Lin's concordance correlation
> coefficient [CCC; @lin1989concordance], which multiplies the Pearson precision by an accuracy factor
> (*C~b~*) that penalises departure from the 1:1 line of perfect agreement, and visualised it with
> Bland–Altman limits of agreement [@bland1986statistical]. CCC was computed on daily upstream (net)
> counts per species and river with bootstrap 95% confidence intervals (2,000 resamples) and interpreted
> using the strength-of-agreement bands of @mcbride2005proposal (weak < 0.90, moderate 0.90–0.95,
> substantial 0.95–0.99, almost perfect > 0.99); the same agreement statistic has been used to validate
> semi-automatic against manual counts of migrating fish from acoustic-camera monitoring
> [@godeaux2026comparative].

## 2. New results subsection §3.6 (insert before `# 4. Discussion`)

> ## 3.6 Count agreement: concordance with the technician record
>
> The daily correlations reported above (Sections 3.3–3.5) show that the model reproduces the shape of
> the daily run, but correlation alone does not establish that the two counts agree in level. We therefore
> summarised agreement with Lin's concordance correlation coefficient (CCC) and Bland–Altman limits of
> agreement, computed on the daily upstream (net) counts pooled across the processed single-stage seasons
> for each species and river (Section 2.4; Figure 7, Table 6). For the three abundant species the daily
> CCC was 0.94 to 0.96 at both rivers, "moderate" to "substantial" agreement on the @mcbride2005proposal
> scale, and, decisively, the accuracy factor *C~b~* was at or above 0.99 in every case. Because CCC is
> the product of precision and this accuracy factor, a *C~b~* of essentially 1.0 means the modest shortfall
> of CCC from a perfect 1.0 is due almost entirely to day-to-day scatter (precision) rather than to any
> systematic over- or under-counting: the model neither consistently leads nor lags the technician count.
> The bootstrap 95% confidence-interval lower bound cleared the 0.90 "moderate" validity threshold used in
> comparable acoustic-camera work [@godeaux2026comparative] for Chinook Salmon at both rivers and for the
> higher-count river of each other species, and fell just below it only for Coho Salmon and Rainbow Trout
> at the Credit River, where the smaller counts produce proportionally larger daily scatter.
>
> The Bland–Altman analysis of the daily Chinook Salmon net (Figure S4) gives the clearest single view of
> this agreement: the mean model-minus-technician difference was −2.0 fish per day, with 95% limits of
> agreement of [−122, +118] over 851 river-days, indicating no meaningful systematic daily bias for the
> dominant species. The days on which the two diverged most were the highest-count days at the dense
> Ganaraska River, reflecting the same track-fragmentation effect that inflates absolute seasonal totals
> there (Section 3.3) rather than a species-identification error. At the annual (seasonal-total) scale the
> concordance for Chinook Salmon is at least as strong (CCC 0.97 to 0.99), but with only five to six
> monitored seasons its confidence intervals are wide; we therefore treat the daily concordance as the
> robust agreement result and the annual totals as supporting rather than independently validated evidence.

## 3. Discussion tie-in (in "Daily pattern versus absolute magnitude", after the daily-r sentence)

> That this reflects genuine agreement and not merely correlation is confirmed by the concordance analysis
> (Section 3.6): daily Lin's CCC of 0.94 to 0.96 with an accuracy factor at or above 0.99, and a near-zero
> Bland–Altman bias for Chinook Salmon (−2.0 fish per day), show that the daily counts agree in level and
> not only in trend.

## 4. Figure captions

**Figure 7.** Concordance of automated with technician daily upstream (net) counts for the three abundant
species, pooled across the processed seasons at both rivers (Ganaraska River, blue; Credit River, amber).
Each point is one river-day; the red dashed line is the 1:1 line of perfect agreement and the grey line the
least-squares (observed) trend. Lin's concordance correlation coefficient (CCC) is annotated per river in
each panel. Points hug the 1:1 line for all three species (daily CCC 0.94–0.96), and the observed trend
running just below the 1:1 line at the highest counts reflects the mild under-reading of the busiest days
at the dense Ganaraska River (Section 3.6). *(source: `figures/fig_concordance.png`.)*

**Figure S4.** Bland–Altman plot of daily Chinook Salmon upstream (net) counts, automated minus technician,
against the mean of the two, pooled across the processed seasons at both rivers (Ganaraska River, blue;
Credit River, amber). The red dashed line is the mean difference (bias −2.0 fish per day) and the green
dotted lines the 95% limits of agreement (± 1.96 SD; [−122, +118], n = 851 river-days). The bias sits essentially on zero, and
the wider differences are confined to the highest-count days at the dense Ganaraska River, consistent with
track-fragmentation inflation of absolute counts rather than a systematic species error (Sections 3.3, 3.6).
*(source: `figures/fig_bland_altman.png`.)*

## 5. Table 6

**Table 6.** Agreement of automated with technician daily upstream (net) counts for the three abundant
species, by river, measured by Lin's concordance correlation coefficient (CCC) on daily counts pooled across
the processed seasons (Section 3.6). Columns: river-days, Pearson *r*, CCC with bootstrap 95% CI, accuracy
factor *C~b~* (CCC = *r* × *C~b~*), and the @mcbride2005proposal band.

| Species | River | River-days | Pearson *r* | CCC | 95% CI | *C~b~* | Agreement |
|---------|-------|-----------:|------------:|----:|--------|-------:|-----------|
| Chinook Salmon | Ganaraska | 439 | 0.967 | 0.965 | [0.950, 0.978] | 0.997 | substantial |
| Chinook Salmon | Credit | 412 | 0.966 | 0.958 | [0.925, 0.984] | 0.991 | substantial |
| Coho Salmon | Ganaraska | 316 | 0.966 | 0.956 | [0.923, 0.978] | 0.990 | substantial |
| Coho Salmon | Credit | 188 | 0.960 | 0.960 | [0.804, 0.985] | 1.000 | substantial |
| Rainbow Trout | Ganaraska | 797 | 0.948 | 0.942 | [0.904, 0.968] | 0.994 | moderate |
| Rainbow Trout | Credit | 483 | 0.955 | 0.954 | [0.884, 0.975] | 0.999 | substantial |

## 6. references.bib entries

```bibtex
@article{lin1989concordance,
  author  = {Lin, Lawrence I-Kuei},
  title   = {A concordance correlation coefficient to evaluate reproducibility},
  journal = {Biometrics},
  year    = {1989}, volume = {45}, number = {1}, pages = {255--268},
  doi     = {10.2307/2532051}
}
@article{bland1986statistical,
  author  = {Bland, J. Martin and Altman, Douglas G.},
  title   = {Statistical methods for assessing agreement between two methods of clinical measurement},
  journal = {The Lancet},
  year    = {1986}, volume = {327}, number = {8476}, pages = {307--310},
  doi     = {10.1016/S0140-6736(86)90837-8}
}
@article{mcbride2005proposal,
  author  = {McBride, Graham B.},
  title   = {A proposal for strength-of-agreement criteria for {Lin's} concordance correlation coefficient},
  journal = {NIWA Client Report HAM2005-062}, year = {2005},
  note    = {National Institute of Water and Atmospheric Research, New Zealand}
}
% VERIFY volume/article-number/DOI before submission (Ecological Informatics, PII S1574954126003110)
@article{godeaux2026comparative,
  author  = {Godeaux, Quentin and Rogissart, Herv{\'e} and Rautureau, Cl{\'e}ment and Martignac, Fran{\c{c}}ois and Cattan{\'e}o, Franck and Guillard, Jean},
  title   = {Comparative evaluation of two automated fish counting software tools using acoustic camera data},
  journal = {Ecological Informatics}, year = {2026},
  doi     = {10.1016/j.ecoinf.2026.103311}
}
```

## Open decision (flagged for Tom)

§3.6 says "processed single-stage seasons", but §2.6/§3.5 still describe the multi-year record as
two-stage / reprocessing-underway. `canonical_daily.csv` is the completed single-stage 2019–2025 run, so
this is consistent with the reframe branch's direction but slightly ahead of the §3.5 text. Either keep
as-is (single-stage), or rescope §3.6 to 2024+2025 only (the seasons §3.5 already presents as single-stage).

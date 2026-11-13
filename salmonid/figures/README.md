# `salmonid/figures/`: the manuscript's figures (stage F)

| Figure | Script | Reads | Tier |
|---|---|---|---|
| 1 Study map | `salmonid/paper/fig1_study_map.py` | map layers | 2 |
| 2 Confusion matrix | `render_manuscript_figures.py` | test-set labels and predictions | 1 |
| 3 Body-shape comparison | `fig_species_morphology.py` | the SAM2 cutout library | 2 |
| 4 Annual counts, model and technician | `render_hero_regulation.py` | canonical count tables | 1 |
| 5 Run timing by species | `render_hero_regulation.py` | canonical count tables | 1 |
| 6 Concordance | `salmonid/results/compute_concordance.py` | canonical count tables | 1 |
| 7 Sanctuary window | `render_hero_regulation.py` | canonical count tables | 1 |
| S1 Example detections | `fig_bbox_grid.py` | cutout library and video | 2 |
| S2 2025 cumulative counts | `render_manuscript_figures.py` | canonical daily table | 1 |
| S3 Bland-Altman | `salmonid/results/compute_concordance.py` | canonical count tables | 1 |
| (supporting, not placed) rule sensitivity | `render_sensitivity_figure.py` | `data/net_rule_sweep/` | 1 |

`manuscript_style.py` is the shared style: journal column widths, text of at least 7 points,
colours and line styles that stay distinguishable in greyscale, 600 dpi PNG plus vector PDF
(flattened 600 dpi TIFF for the photo figures), and PDF files without a creation date so
that a rebuild is byte-identical.

All Tier-1 figures are drawn by `python run_tier1.py` (steps 4, 7, 8, 9) into `figures/`.
With the package versions in `environment/tier1.lock.txt` the files are byte-identical to
the ones in the manuscript; `data/figure_checksums.json` holds the fingerprints.

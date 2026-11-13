"""figures sub-package: the scripts that draw the manuscript figures.

Each script reads finished tables (or stored images) and draws; none runs the model.

    manuscript_style              shared fonts, colours and sizes (used by the others, not run)
    render_manuscript_figures     Figure 2 (confusion matrix) and Figure S2 (2025 running
                                  totals at each river) (Tier 1)
    render_hero_regulation        Figures 4, 5 and 7 (Tier 1)
    render_sensitivity_figure     the counting-rule comparison of Section 3.4; supporting
                                  material, not placed in the manuscript (Tier 1)
    fig_species_morphology        Figure 3; needs the library of fish cut-outs (Tier 2)
    fig_bbox_grid                 Figure S1; needs video frames (Tier 2)

Figures 6 and S3 are drawn by ``salmonid.results.compute_concordance`` and Figure 1 (the
study map) by ``salmonid.paper``. ``docs/PAPER_TO_CODE.md`` lists every figure with its command.
"""

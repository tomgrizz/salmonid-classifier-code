"""In plain terms: this file draws two of the charts that appear in the paper. It does not run
the computer-vision model. It reads results tables that are already saved in this repository and
turns them into print-ready figures: the species "confusion matrix" (a grid showing how often each
species was named correctly and which species it was mistaken for) and the 2025 running-total
charts that compare the model's count against the technicians' count through the season.

Pipeline stage: **figures** (the final step). No detection, tracking or counting happens here.

Figures produced
----------------
* ``fig_confusion_heldout`` -> manuscript **Figure 2**. The published model's species confusion
  matrix on the 366-track held-out test set (352 correct). Inputs: ``config.PUBLISHED_PREDS``
  (per-track predictions) + ``config.TESTSET_V6_GT`` (per-track reviewer labels).
* ``fig_daily_net_2025_credit`` and ``fig_daily_net_2025_ganaraska`` -> manuscript **Figure S2**
  panels (a) and (b). Cumulative net count through 2025, model against technician, one panel per
  species, with each panel's daily correlation and season totals in its title. Input:
  ``config.CANONICAL_DAILY``.

Inputs : committed files in data/ only.
Outputs: PNG (600 dpi) + vector PDF for each figure, in ``config.FIGURES_DIR`` (figures/).
Tier   : 1 (runs from committed data; no model, no GPU; a few seconds).
Run    : python -m salmonid.figures.render_manuscript_figures

Where the other manuscript figures are drawn: Figures 4, 5 and 7 in ``render_hero_regulation``;
Figures 6 and S3 in ``salmonid.results.compute_concordance``; Figure 3 in
``fig_species_morphology`` and Figure S1 in ``fig_bbox_grid`` (both Tier 2: they need fish
images); Figure 1 (the map) in ``salmonid.paper``. docs/PAPER_TO_CODE.md has the full list.

Note on the daily correlation printed in Figure S2: it is computed over every day in the
canonical table for that river and species (days on which either the model or the technician
recorded a passage). Manuscript Table 2 uses the same day set.
"""
import csv, json, collections
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from salmonid import config
from salmonid.figures.manuscript_style import (set_style, save_fig, COL1_IN, MODEL_C, AFS,
                                               MODEL_LABEL, TECH_LABEL)

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
LAB = [AFS[s] for s in SPECIES]   # full AFS common names on the artwork (CJFAS house style)
# Fixed panel order for the cumulative-net figure, in class order (matching Figure 2, Figure 4 and
# Tables 1 and 2). Atlantic Salmon is omitted: it is presence-level, with near-zero counts.
PANEL_SPECIES = ["Chinook", "Coho", "Rainbow Trout", "Brown Trout"]
GT = config.TESTSET_V6_GT   # committed frozen per-track species ground truth (test_set_v6)


def load_gt():
    """Load the held-out ground truth as {(video_id, track_id): species}.

    Reads ``GT`` (config.TESTSET_V6_GT); keeps only tracks whose species is one of the five
    reported salmonids (drops "Small Salmonid" / any other label).
    """
    return {(t["video_id"], int(t["track_id"])): t["species"]
            for t in json.load(open(GT))["tracks"] if t["species"] in SPECIES}


def load_preds(p):
    """Load per-track model predictions from CSV ``p`` as {(video_id, track_id): pred_species}."""
    return {(r["video_id"], int(r["track_id"])): r["pred_species"] for r in csv.DictReader(open(p))}


def confusion(gt, preds):
    """Build the 5x5 species confusion matrix M[true, pred] (row=true, col=predicted).

    Only tracks present in ``gt`` whose prediction is one of the five species are tallied;
    tracks missing a prediction, or predicted outside the five species, are skipped.
    """
    M = np.zeros((5, 5), int)
    for k, t in gt.items():
        p = preds.get(k)
        if p in SPECIES:
            M[SPECIES.index(t), SPECIES.index(p)] += 1
    return M


def fig_confusion():
    """Single-stage confusion matrix on the held-out test set (for the manuscript).

    Cells are shaded on one sequential hue by their within-row share (the diagonal is recall); every
    non-empty cell prints the track count and the within-row (recall) %.

    Reads the PUBLISHED model's predictions (config.PUBLISHED_PREDS), scored on all 366 tracks.
    """
    gt = load_gt()
    pf = config.PUBLISHED_PREDS
    M = confusion(gt, load_preds(pf))
    row = M.sum(1, keepdims=True)
    Mn = M / np.clip(row, 1, None)
    # SINGLE-HUE sequential ramp: shade = the within-row share, so the diagonal (0.92 to 1.00) is near
    # solid and the errors (0.02 to 0.08) are faint tints of the same hue. The diagonal is therefore
    # distinguished by value, not by hue, which prints as a clean grey ramp and is safe for red-green
    # colour-vision deficiency (the earlier green/red scheme failed both). No legend: every non-empty
    # cell prints its count and within-row percentage, so the shading only reinforces the numbers.
    HUE = (0.13, 0.31, 0.53)
    GAMMA = 0.55                    # lifts small error fractions off zero without rivalling the diagonal

    def text_color(rgb, a):
        """Pick black/white cell text by the luminance of ``rgb`` alpha-blended over white."""
        lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) * a + (1 - a)  # blended over white
        return "white" if lum < 0.5 else "#1a1a1a"

    fig, ax = plt.subplots(figsize=(COL1_IN, 3.45), layout="constrained")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(4.5, -0.5)  # keep true-species row 0 at the top
    ax.set_aspect("equal")
    for i in range(5):
        for j in range(5):
            frac = Mn[i, j]
            a = (0.06 + 0.94 * frac ** GAMMA) if M[i, j] > 0 else 0.0
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=(*HUE, a),
                                   edgecolor="#dcdcdc", linewidth=0.5))
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]}\n{100 * frac:.1f}%", ha="center", va="center",
                        fontsize=7, linespacing=1.15, color=text_color(HUE, a), fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(LAB, rotation=35, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(LAB)
    ax.set_xlabel("Predicted species", fontweight="bold")
    ax.set_ylabel("True species", fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Named for what it shows, not for an internal model revision.
    save_fig(fig, "fig_confusion_heldout", kind="vector")


def _load_daily_canonical(site, year):
    """Daily model/technician net counts for one river and year from the COMMITTED canonical table
    (``config.CANONICAL_DAILY``), the same source as Figure 4, Table 2 and the main text, so that
    Figure S2 agrees with them. Returns {species: {"model": [...], "tech": [...], "date": [...]}}."""
    d = collections.defaultdict(lambda: {"model": [], "tech": [], "date": []})
    for r in csv.DictReader(open(config.CANONICAL_DAILY)):
        if r["river"] != site or int(r["year"]) != year:
            continue
        sp = r["species"]
        d[sp]["date"].append(r["date"]); d[sp]["model"].append(float(r["model_net"])); d[sp]["tech"].append(float(r["tech_net"]))
    return d


def _pearson(a, b):
    """Pearson r between two daily series; NaN when <3 points or either series is constant."""
    a, b = np.array(a), np.array(b)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fig_daily(site_tag, site_name, title):
    """Cumulative net count through one season, model vs technician, one panel per species.

    ``site_tag`` is "<year>_<river>" (it names the output file); ``site_name`` is the river as
    written in the canonical table. Both panels of Figure S2 use the same four species in the same
    order at a fixed canvas size, with a calendar-month x-axis over a common March-December
    window, so the two rivers are directly comparable.
    """
    d = _load_daily_canonical(site_name, int(site_tag[:4]))
    all_dates = [dt for s in d for dt in d[s]["date"]]
    if not all_dates:
        raise SystemExit(f"No rows for {site_name} {site_tag[:4]} in {config.CANONICAL_DAILY}")
    yr = int(min(all_dates)[:4])
    fig, axes = plt.subplots(len(PANEL_SPECIES), 1, figsize=(COL1_IN, 6.3), squeeze=False,
                             layout="constrained")
    for ax, sp in zip(axes[:, 0], PANEL_SPECIES):
        entry = d.get(sp)
        if entry and (sum(entry["tech"]) + sum(entry["model"])) > 0:
            order = np.argsort(entry["date"])
            dt = [datetime.strptime(entry["date"][k], "%Y-%m-%d") for k in order]
            m = np.array(entry["model"])[order]; t = np.array(entry["tech"])[order]
            cm, ct = np.cumsum(m), np.cumsum(t)
            r = _pearson(entry["model"], entry["tech"])
            rtxt = "n/a" if np.isnan(r) else f"{r:.3f}"
            # grayscale-legible: technician solid dark, model dashed red
            ax.plot(dt, ct, color="#1b3a5c", lw=1.4)
            ax.plot(dt, cm, color=MODEL_C, lw=1.4, ls=(0, (5, 2)))
            ax.set_ylim(bottom=min(0.0, float(cm.min()), float(ct.min())))   # anchor the y-axis at zero
            # Season totals go in the panel title and the two series are identified once, below the
            # panels, so no legend can sit on a curve (the Ganaraska Rainbow Trout run plateaus across
            # the top of its panel from late April, under the old upper-left legend).
            ax.set_title(f"{AFS[sp]}: daily $r$ = {rtxt}\n{TECH_LABEL} {int(ct[-1]):,}  ·  "
                         f"{MODEL_LABEL} {int(cm[-1]):,}", loc="left", fontsize=7.5,
                         linespacing=1.3)
        else:
            ax.set_title(f"{AFS[sp]}: no counts", loc="left", fontsize=7.5, color="#888")
        ax.grid(alpha=0.25); ax.set_ylabel("Cumulative net-total count", fontsize=7)
        ax.set_xlim(datetime(yr, 3, 1), datetime(yr, 12, 1))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("Month", fontsize=7.5)
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=8.5)
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[Line2D([0], [0], color="#1b3a5c", lw=1.4, label=TECH_LABEL),
                        Line2D([0], [0], color=MODEL_C, lw=1.4, ls=(0, (5, 2)), label=MODEL_LABEL)],
               loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    save_fig(fig, f"fig_daily_net_{site_tag}", kind="vector")


def main():
    """Draw Figure 2 and both panels of Figure S2 into figures/."""
    set_style()   # the shared CJFAS figure style (fonts, sizes, colours); see manuscript_style.py
    fig_confusion()                                                         # Figure 2
    fig_daily("2025_credit", "Credit", "(a) Credit River, 2025")            # Figure S2 (a)
    fig_daily("2025_ganaraska", "Ganaraska", "(b) Ganaraska River, 2025")   # Figure S2 (b)


if __name__ == "__main__":
    main()

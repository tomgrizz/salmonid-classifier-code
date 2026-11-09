"""In plain terms: this script draws the paper's main results figures — the pictures that
compare the automated fish counter (a computer program that counts fish in the tunnel-camera
videos) against what human technicians counted by hand. It reads the already-finished count
tables and produces the manuscript's Figure 4 (year-by-year automated-vs-technician counts),
Figure 5 (run timing — when each species moves through), and Figure 6 (showing ~90% of the
Chinook Salmon run falls inside the new Ganaraska sanctuary's closed season).

Manuscript hero / regulation figures — the *figures* stage of the pipeline.

Pipeline position: data -> detect -> track -> eval/count -> **results** -> **figures**. This module is a
pure post-processing / plotting step: it consumes the committed canonical results tables (already produced
by the count/eval stage) and renders the publication figures. It never touches the detector, tracker, or
raw footage, so it runs in the **base environment** (matplotlib/numpy only) — NOT the DEIMv2 ``.venv_deim``
that detection/training need.

Publication-quality figures from the canonical dataset, in CJFAS/CSP production style (shared
``salmonid.figures.manuscript_style``): 2-column width, >= 7 pt embedded fonts, grayscale-legible series,
and no in-artwork title (the caption is the title). Emits a PNG preview + a vector PDF for each.

Inputs (committed, repo-local — resolved via ``salmonid.config``):
  * ``config.CANONICAL_DAILY``   — per-day model-vs-technician counts (net + activity), 2019-2025, both rivers.
  * ``config.CANONICAL_SUMMARY`` — per-(year, river, species) totals; used only to know which cells exist.

Outputs (vector PDF + PNG preview, written by ``manuscript_style.save_fig``):
  * fig_hero_performance.png / .pdf  (manuscript Fig 4) — model output (red line) vs technician
    (blue bars) annual NET counts, river x species grid, per-panel % error + inter-annual r^2.
  * fig_hero_activity.png / .pdf     — the same grid for the total ACTIVITY count.
  * fig_species_phenology.png / .pdf (manuscript Fig 5) — average run timing by species, per river.
  * fig_ganaraska_sanctuary.png / .pdf (manuscript Fig 6) — automated counter supports the new Lower
    Ganaraska Chinook sanctuary.

Manuscript claim supported: the automated counter reproduces the technician's inter-annual abundance
index (mean main-species net r^2 printed to stdout on run) and independently validates the Ganaraska
1 Sept - 14 Oct Chinook sanctuary (~90% of the run falls inside the closed window, both methods agreeing).

  python -m salmonid.figures.render_hero_regulation
"""
import csv, datetime, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from salmonid import config
from salmonid.figures.manuscript_style import (set_style, save_fig, COL2_IN, PAL, LS, MODEL_C, TECH_C,
                                               TECH_EDGE, AFS, AFS2, SPECIES_ORDER, MODEL_LABEL,
                                               TECH_LABEL, TECH_LABEL_LONG)

set_style()
OUT = config.FIGURES_DIR
DAILY = config.CANONICAL_DAILY
SUMM = config.CANONICAL_SUMMARY
ROWS = list(csv.DictReader(open(DAILY)))
SUM = {(int(r["year"]), r["river"], r["species"]): r for r in csv.DictReader(open(SUMM))}
SPS = list(SPECIES_ORDER)   # class order, matching Tables 3 and 4 and Figure 2
RIVERS = ["Ganaraska", "Credit"]
YEARS = list(range(2020, 2026))
MET = {"net": ("model_net", "tech_net", "net_r"), "activity": ("model_act", "tech_act", "act_r")}


def annual(river, species, col):
    """Sum a daily count column into a {year: total} map for one (river, species).

    Args: river/species select the daily rows; col is the CANONICAL_DAILY column to sum
    (e.g. "model_net", "tech_net", "model_act", "tech_act"). Returns a dict keyed by every
    year in YEARS (missing years total to 0)."""
    return {y: sum(int(r[col]) for r in ROWS
                   if r["river"] == river and r["species"] == species and int(r["year"]) == y) for y in YEARS}


def _pear(xs, ys):
    """Pearson correlation r between two equal-length sequences (returns NaN if n<3 or a
    series has zero variance). Squared by callers to get the abundance-index r^2."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n; my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def stat(river, sp, metric):
    """overall % error (sum/sum) and the ANNUAL abundance-index r^2 (model-vs-tech annual counts across years)
    — how well the automated line tracks the technician's year-to-year peaks and troughs."""
    mc, tc, _ = MET[metric]
    m = annual(river, sp, mc); t = annual(river, sp, tc)
    ys = [y for y in YEARS if t[y] or m[y]]
    M = sum(m[y] for y in ys); T = sum(t[y] for y in ys)
    pe = 100 * (M - T) / T if T else float("nan")
    r = _pear([m[y] for y in ys], [t[y] for y in ys])
    return pe, (r * r if not math.isnan(r) else float("nan"))


MAIN_SPS = ["Chinook", "Coho", "Rainbow Trout"]   # the auto-count-reliable species (Atlantic/Brown = review-only)


def mean_main_r2(metric):
    """Mean of the per-panel inter-annual abundance-index r^2 over the MAIN species (both rivers) — i.e. the
    average of the per-panel r^2 boxes drawn in the hero grid. Computed from the canonical data (not hardcoded)
    so the reported summary and the figure can never drift apart."""
    r2s = [stat(river, sp, metric)[1] for river in RIVERS for sp in MAIN_SPS]
    r2s = [x for x in r2s if not math.isnan(x)]
    return sum(r2s) / len(r2s) if r2s else float("nan")


def make_grid(metric, fname, ylabel):
    """River x species grid: technician annual counts as blue bars, model output as a red marked line.
    Grayscale-legible: bars are a solid mid-tone with an edge; the line carries circular markers, so the two
    series stay distinct without colour. Each panel prints its overall % error and inter-annual r^2."""
    mc, tc, _ = MET[metric]
    fig, axes = plt.subplots(2, 5, figsize=(COL2_IN, 4.1), layout="constrained")
    for i, river in enumerate(RIVERS):
        for j, sp in enumerate(SPS):
            ax = axes[i, j]
            m = annual(river, sp, mc); t = annual(river, sp, tc)
            ax.bar(YEARS, [t[y] for y in YEARS], width=0.74, color=TECH_C,
                   edgecolor=TECH_EDGE, linewidth=0.4, zorder=2)
            ax.plot(YEARS, [m[y] for y in YEARS], "-o", color=MODEL_C, lw=1.4, ms=3.2, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.5)
            pe, r2 = stat(river, sp, metric)
            # Unicode superscript, not mathtext: mathtext draws the "2" at 70% size (4.9 pt), under 7 pt.
            lbl = (f"{pe:+.0f}%\nr\u00b2 = {r2:.2f}" if not (math.isnan(pe) or math.isnan(r2)) else "")
            ax.text(0.05, 0.95, lbl, transform=ax.transAxes, va="top", ha="left", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.4, alpha=0.9))
            if i == 0:
                # full AFS common name, on two lines so it fits a five-across grid
                ax.set_title(AFS2[sp], fontweight="bold", pad=4, linespacing=1.05)
            if j == 0:
                ax.set_ylabel(f"{river}\n{ylabel}", fontweight="bold", labelpad=4, fontsize=7.5)
            ax.set_xticks(YEARS); ax.set_xticklabels([str(y)[2:] for y in YEARS])   # two-digit years; '20 crowds at 7 pt
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(labelsize=7)
            mx = max([t[y] for y in YEARS] + [m[y] for y in YEARS] + [1])
            ax.set_ylim(0, mx * 1.34); ax.margins(x=0.04); ax.set_axisbelow(True)   # headroom for the box
    handles = [Patch(facecolor=TECH_C, edgecolor=TECH_EDGE, label=TECH_LABEL_LONG),
               Line2D([0], [0], color=MODEL_C, lw=1.4, marker="o", ms=3.2, markeredgecolor="white",
                      label=MODEL_LABEL)]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save_fig(fig, Path(fname).stem, kind="vector")


def _md(d):
    """Collapse an ISO date string to a common reference year (2000) so runs from different
    years overlay on the same month-of-year axis (day-of-year alignment for the phenology/
    sanctuary plots)."""
    return datetime.date.fromisoformat(d).replace(year=2000)


def fig_sanctuary():
    """(a) cumulative Chinook run vs date, year lines on a viridis (grayscale-ordered) ramp, sanctuary window
    shaded; (b) % of run inside the window, technician (hatched blue) vs model output (solid red)."""
    fig, (axc, axs) = plt.subplots(1, 2, figsize=(COL2_IN, 3.4), layout="constrained")
    win0, win1 = datetime.date(2000, 9, 1), datetime.date(2000, 10, 14)
    yr_colors = plt.cm.viridis(np.linspace(0.04, 0.90, len(YEARS)))
    for y, c in zip(YEARS, yr_colors):
        ch = sorted([r for r in ROWS if r["river"] == "Ganaraska" and r["species"] == "Chinook"
                     and int(r["year"]) == y and _md(r["date"]) >= datetime.date(2000, 7, 1)], key=lambda r: r["date"])
        if not ch:
            continue
        dates = [_md(r["date"]) for r in ch]
        # net can be negative on downstream-dominated days; floor at 0 so the run is monotone-cumulative.
        cum = np.cumsum([max(0, int(r["model_net"])) for r in ch], dtype=float)
        axc.plot(dates, 100 * cum / (cum[-1] or 1), lw=1.4, color=c, label=str(y))
    axc.set_ylim(0, 118)
    axc.axvspan(win0, win1, color="#c65b2e", alpha=0.20, zorder=0)   # >= 10% K in grayscale
    axc.axhline(90, color="#555", ls="--", lw=0.8)
    axc.text(datetime.date(2000, 9, 22), 111, "Chinook Salmon sanctuary\n(closed 1 Sept – 14 Oct)", ha="center",
             va="top", fontsize=7.5, fontweight="bold", color="#a4441f")
    axc.xaxis.set_major_locator(mdates.MonthLocator()); axc.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axc.set_xlim(datetime.date(2000, 7, 15), datetime.date(2000, 12, 1))
    axc.set_ylabel("Cumulative % of Chinook Salmon run", fontweight="bold"); axc.set_xlabel("Date", fontweight="bold")
    axc.set_title("(a)", fontweight="bold", loc="left")   # the descriptive title lives in the caption
    axc.legend(title="Year", fontsize=7, title_fontsize=7, loc="center right", ncol=1, handlelength=1.4)
    yrs = [y for y in YEARS if (y, "Ganaraska", "Chinook") in SUM]

    def pct(col):
        """Per-year % of the Ganaraska Chinook run (column `col`, negatives clipped to 0) that
        falls inside the sanctuary window win0..win1."""
        out = []
        for y in yrs:
            ch = [r for r in ROWS if r["river"] == "Ganaraska" and r["species"] == "Chinook" and int(r["year"]) == y]
            tot = sum(max(0, int(r[col])) for r in ch) or 1
            out.append(100 * sum(max(0, int(r[col])) for r in ch if win0 <= _md(r["date"]) <= win1) / tot)
        return out
    x = np.arange(len(yrs))
    axs.bar(x - 0.2, pct("tech_net"), 0.38, color=TECH_C, edgecolor=TECH_EDGE, linewidth=0.4, hatch="////")
    axs.bar(x + 0.2, pct("model_net"), 0.38, color=MODEL_C, edgecolor="#7d241a", linewidth=0.4)
    # The "~90%" label sits in a left margin, clear of every bar (it used to overprint the 2025 bar).
    axs.axhline(90, color="#555", ls="--", lw=0.8)
    axs.text(-0.72, 91.5, "~90%", fontsize=7, ha="left")
    axs.set_xlim(-0.8, len(yrs) - 0.45)
    axs.set_xticks(x); axs.set_xticklabels(yrs); axs.set_ylim(0, 108)
    # Two lines: the one-line label ran off the top of the panel.
    axs.set_ylabel("% of Chinook Salmon run\nwithin 1 Sept – 14 Oct", fontweight="bold")
    axs.set_xlabel("Year", fontweight="bold")
    axs.set_title("(b)", fontweight="bold", loc="left")
    # Figure-level legend below both panels: the old in-axes "lower right" legend sat on the 2024-2025 bars.
    fig.legend(handles=[Patch(facecolor=TECH_C, edgecolor=TECH_EDGE, linewidth=0.4, hatch="////",
                              label=TECH_LABEL),
                        Patch(facecolor=MODEL_C, edgecolor="#7d241a", linewidth=0.4, label=MODEL_LABEL)],
               loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    save_fig(fig, "fig_ganaraska_sanctuary", kind="vector")


def fig_phenology():
    """Multi-year AVERAGE run timing by species: technician (shaded) vs model output (line), per day of
    year, normalized to each species' technician peak. Grayscale-legible: each species has a distinct dash
    pattern (LS) as well as a colourblind-safe colour. Panels (a) Ganaraska, (b) Credit."""
    species = ["Chinook", "Coho", "Rainbow Trout"]   # class order, as everywhere else
    fig, axes = plt.subplots(1, 2, figsize=(COL2_IN, 3.35), sharey=True, layout="constrained")

    def smooth(a, w=11):
        """Centred w-day moving average (boxcar) over a day-of-year abundance series, to render
        the run-timing curves as smooth phenology traces."""
        return np.convolve(a, np.ones(w) / w, mode="same")

    for k, (ax, river) in enumerate(zip(axes, RIVERS)):
        for sp in species:
            dm = np.zeros(367); dt = np.zeros(367)
            for r in ROWS:
                # 2020-2025 only, matching the caption (the canonical table also holds 2019 at the Credit).
                if r["river"] == river and r["species"] == sp and int(r["year"]) in YEARS:
                    doy = datetime.date.fromisoformat(r["date"]).timetuple().tm_yday
                    dm[doy] += max(0, int(r["model_net"])); dt[doy] += max(0, int(r["tech_net"]))
            sm, stt = smooth(dm), smooth(dt)
            pk = stt.max() or 1
            xs = [datetime.date(2000, 1, 1) + datetime.timedelta(days=i - 1) for i in range(1, 367)]
            # Technician series: a tinted fill with a thin edge in the species' dash pattern, so the
            # three overlapping autumn fills stay separable in grayscale (a bare 0.18-alpha fill did not).
            ax.fill_between(xs, (stt / pk)[1:], color=PAL[sp], alpha=0.22, lw=0, zorder=1)
            ax.plot(xs, (stt / pk)[1:], color=PAL[sp], lw=0.6, ls=LS[sp], zorder=2)
            ax.plot(xs, (sm / pk)[1:], color=PAL[sp], lw=1.6, ls=LS[sp], zorder=3, solid_capstyle="round")
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_xlim(datetime.date(2000, 3, 1), datetime.date(2000, 12, 15))
        ax.set_ylim(0, 1.4)
        ax.set_title(f"({'ab'[k]})  {river}", fontweight="bold", loc="left"); ax.set_xlabel("Month", fontweight="bold")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Relative daily net count (normalized)", fontweight="bold")
    # One figure-level legend below the panels: the in-axes species legend sat on the Credit Coho curve.
    sp_handles = [Line2D([0], [0], color=PAL[s], lw=1.8, ls=LS[s], label=AFS[s]) for s in species]
    style_handles = [Patch(facecolor="0.55", alpha=0.35, edgecolor="0.35", linewidth=0.6,
                           label=f"{TECH_LABEL} (shaded)"),
                     Line2D([0], [0], color="0.35", lw=1.8, label=f"{MODEL_LABEL} (line)")]
    fig.legend(handles=sp_handles + style_handles, loc="outside lower center", ncol=5, frameon=False,
               fontsize=7, handlelength=2.4)
    save_fig(fig, "fig_species_phenology", kind="vector")


if __name__ == "__main__":
    MR2 = mean_main_r2("net")
    print(f"mean main-species (Chinook/Coho/Rainbow, both rivers) net abundance-index r^2 = {MR2:.3f}")
    make_grid("net", "fig_hero_performance.png", "Net-total count")
    make_grid("activity", "fig_hero_activity.png", "Total activity count")
    fig_sanctuary()
    fig_phenology()

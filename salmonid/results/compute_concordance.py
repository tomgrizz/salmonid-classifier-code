"""
In plain terms: this file measures how closely the computer's fish counts agree
with the counts made by human technicians. It scores the agreement with a standard
statistic (Lin's concordance coefficient — a 0-to-1 number where 1 means the two
counts match perfectly), both day-by-day and for the whole-season totals, and it
builds the paper's agreement table (concordance_ccc.csv) and its two agreement
figures (fig_concordance and the Bland-Altman plot) that back the claim the
automated counter agrees substantially-to-almost-perfectly with the technicians.

Concordance (agreement) analysis of automated vs technician fish counts.

Pipeline stage: **results** (data -> detect -> track -> count -> *results* -> figures).
This is a leaf reporting step: it consumes the already-aggregated canonical count
tables and emits an agreement-statistic table plus two figures. It loads no model
and runs in the **base Python environment** (numpy/pandas/matplotlib only) — NOT the
DEIMv2 ``.venv_deim``, which is required only for detection/training.

What it computes
----------------
Lin's concordance correlation coefficient (CCC; Lin 1989) with a bias-correction/
accuracy factor Cb (CCC = r * Cb) and bootstrap 95% CIs, at two aggregation levels:
  - daily upstream (net) counts, pooled over corpus years, per species x river
  - annual (seasonal-total) counts, the abundance index, per species x river
plus Bland-Altman limits of agreement (Bland & Altman 1986) for daily Chinook.

Strength-of-agreement bands follow McBride (2005): <0.90 weak, 0.90-0.95
moderate, 0.95-0.99 substantial, >0.99 almost perfect.

Inputs : config.CANONICAL_DAILY, config.CANONICAL_SUMMARY (single-stage v2.6,
  2019-2025). Recovered footage-years (2019 Credit, 2020 Ganaraska) are excluded
  from agreement stats (magnitude-caveated; see manuscript caveats).
Outputs: concordance_ccc.csv (--out-dir, default config.REPO_DATA); fig_concordance.png
  + fig_bland_altman.png (--fig-dir, default = out-dir).

Manuscript claim supported: the automated counter *agrees* with the technician
counts — substantial-to-almost-perfect Lin's CCC on the annual abundance index and
daily net counts, with a small, quantified Bland-Altman bias.

Usage:
  python -m salmonid.results.compute_concordance --out-dir <dir> [--fig-dir <dir>]
Outputs: concordance_ccc.csv, fig_concordance.png, fig_bland_altman.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from salmonid import config

MAIN = ["Chinook", "Coho", "Rainbow Trout"]
# Palette matches the paper repo (make_annual_panels.py)
SPECIES_COLORS = {
    "Chinook": "#E63946", "Coho": "#E07B39", "Atlantic": "#2A9D8F",
    "Rainbow Trout": "#457B9D", "Brown Trout": "#8B5E3C",
}
RIVER_MARKER = {"Ganaraska": "o", "Credit": "^"}
# Distinct river colours (Godeaux et al. 2026 palette): Ganaraska blue, Credit amber.
# #E69F00 is the Okabe-Ito orange — a CVD- and grayscale-safe hue that stays legible
# against the blue when the figure is printed in monochrome.
RIVER_COLORS = {"Ganaraska": "#5E84B9", "Credit": "#E69F00"}
BOOT = 2000
SEED = 12345


def lin_ccc(x, y):
    """Lin's CCC, Pearson r, accuracy factor Cb, and location/scale shift."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return dict(n=n, ccc=np.nan, r=np.nan, cb=np.nan, bias=np.nan)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(), y.std()  # population SD (Lin's estimator uses 1/n)
    if sx == 0 or sy == 0:
        return dict(n=n, ccc=np.nan, r=np.nan, cb=np.nan, bias=np.nan)
    cov = ((x - mx) * (y - my)).mean()
    r = cov / (sx * sy)
    ccc = 2 * cov / (sx**2 + sy**2 + (mx - my) ** 2)
    cb = ccc / r if r != 0 else np.nan
    return dict(n=n, ccc=ccc, r=r, cb=cb, bias=my - mx)  # bias = model - tech


def boot_ci(x, y, stat="ccc", B=BOOT, seed=SEED):
    """Percentile bootstrap 95% CI for a ``lin_ccc`` field (paired resampling).

    Resamples the (x, y) pairs with replacement B times (seeded for reproducibility)
    and returns the (2.5, 97.5) percentiles of the requested ``lin_ccc`` statistic.
    Returns (nan, nan) if fewer than 3 finite pairs.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = np.array([lin_ccc(x[i], y[i])[stat]
                     for i in (rng.integers(0, n, n) for _ in range(B))])
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def mcbride(ccc):
    """Map a CCC value to its McBride (2005) strength-of-agreement band label."""
    if not np.isfinite(ccc):
        return "n/a"
    if ccc < 0.90:
        return "weak"
    if ccc < 0.95:
        return "moderate"
    if ccc < 0.99:
        return "substantial"
    return "almost perfect"


def load():
    """Load the canonical daily & summary count tables; drop recovered years.

    Merges each daily row with its summary ``recovered`` flag and keeps only
    non-recovered (year, river, species) cells in both tables, so agreement
    statistics are computed on magnitude-reliable footage-years only.

    Returns:
        (daily, summ): two DataFrames filtered to ``recovered == False``.
    Side effects: reads config.CANONICAL_DAILY and config.CANONICAL_SUMMARY.
    """
    daily = pd.read_csv(config.CANONICAL_DAILY)
    summ = pd.read_csv(config.CANONICAL_SUMMARY)
    daily = daily.merge(summ[["year", "river", "species", "recovered"]],
                        on=["year", "river", "species"], how="left")
    daily = daily[daily["recovered"] == False].copy()  # noqa: E712
    summ = summ[summ["recovered"] == False].copy()      # noqa: E712
    return daily, summ


def build_table(daily, summ):
    """Assemble the per-(species, river) CCC table at daily and annual levels.

    For each species in ``MAIN`` x each river, computes ``lin_ccc`` plus a bootstrap
    CI on the daily net counts and again on the annual (seasonal-total) counts,
    rounded to 3 dp. Returns a tidy DataFrame with one row per level x species x river.
    """
    rows = []
    for sp in MAIN:
        for rv in ["Ganaraska", "Credit"]:
            d = daily[(daily.species == sp) & (daily.river == rv)]
            res = lin_ccc(d.model_net, d.tech_net)
            lo, hi = boot_ci(d.model_net.values, d.tech_net.values)
            rows.append(dict(level="daily", species=sp, river=rv, n=res["n"],
                             pearson_r=round(res["r"], 3), ccc=round(res["ccc"], 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             cb=round(res["cb"], 3), band=mcbride(res["ccc"])))
            s = summ[(summ.species == sp) & (summ.river == rv)]
            resa = lin_ccc(s.model_net, s.tech_net)
            loa, hia = boot_ci(s.model_net.values, s.tech_net.values)
            rows.append(dict(level="annual", species=sp, river=rv, n=resa["n"],
                             pearson_r=round(resa["r"], 3), ccc=round(resa["ccc"], 3),
                             ci_lo=round(loa, 3), ci_hi=round(hia, 3),
                             cb=round(resa["cb"], 3), band=mcbride(resa["ccc"])))
    return pd.DataFrame(rows)


def make_figures(daily, fig_dir):
    """Figures styled to match Godeaux et al. 2026 (FishTracker_workflow.R):
    ggplot theme_minimal look with a black panel border, red dashed 1:1 line,
    darkgrey lm trend, and darkgreen dotted +-1.96 SD limits. The two rivers are
    drawn in distinct colours (Ganaraska blue, Credit amber)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    def theme_minimal(ax):
        ax.set_facecolor("white")
        ax.grid(True, color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():          # panel.border = black box
            sp.set_visible(True); sp.set_color("black"); sp.set_linewidth(1.0)
        ax.tick_params(length=2.5, color="0.5")

    RED, TREND, GREEN = "#E31A1C", "#A9A9A9", "#1B7837"   # 1:1, lm trend, +-1.96 SD

    # ---- Figure: per-species daily concordance (Lin plots) ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for ax, sp in zip(axes, MAIN):
        d = daily[daily.species == sp]
        hi_lim = max(d.tech_net.max(), d.model_net.max()) * 1.05
        lo_lim = min(0, d.tech_net.min(), d.model_net.min())
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color=RED, lw=1.1,
                ls="--", zorder=3, label="1:1 line")
        for rv in ["Ganaraska", "Credit"]:
            s = d[d.river == rv]
            ax.scatter(s.tech_net, s.model_net, s=15, alpha=0.7,
                       color=RIVER_COLORS[rv], edgecolor="none", label=rv, zorder=2)
        xx = d.tech_net.values.astype(float); yy = d.model_net.values.astype(float)
        b1, b0 = np.polyfit(xx, yy, 1)
        gx = np.array([lo_lim, hi_lim])
        ax.plot(gx, b0 + b1 * gx, color=TREND, lw=1.2, zorder=1, label="trend (lm)")
        ax.set_xlim(lo_lim, hi_lim); ax.set_ylim(lo_lim, hi_lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(sp, fontsize=12.5, pad=8)
        ax.set_xlabel("Technician daily net count")
        if sp == MAIN[0]:
            ax.set_ylabel("Automated daily net count")
        theme_minimal(ax)
        cg = lin_ccc(d[d.river == "Ganaraska"].model_net,
                     d[d.river == "Ganaraska"].tech_net)["ccc"]
        cc = lin_ccc(d[d.river == "Credit"].model_net,
                     d[d.river == "Credit"].tech_net)["ccc"]
        ax.text(0.05, 0.955, "Lin's CCC", transform=ax.transAxes, va="top",
                ha="left", fontsize=10, fontweight="bold")
        ax.text(0.05, 0.885, f"Ganaraska  {cg:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color=RIVER_COLORS["Ganaraska"])
        ax.text(0.05, 0.820, f"Credit  {cc:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=10, color=RIVER_COLORS["Credit"])
    axes[0].legend(frameon=True, fontsize=8.5, loc="lower right",
                   edgecolor="0.7", framealpha=0.95)
    fig.tight_layout()
    p1 = Path(fig_dir) / "fig_concordance.png"
    fig.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure: Bland-Altman, daily Chinook net (rivers in distinct colours) ----
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    d = daily[daily.species == "Chinook"]
    mean_c = (d.model_net + d.tech_net) / 2
    diff_c = d.model_net - d.tech_net
    bias, sd = diff_c.mean(), diff_c.std()
    hi, lo = bias + 1.96 * sd, bias - 1.96 * sd
    for rv in ["Ganaraska", "Credit"]:
        s = d.river == rv
        ax.scatter(mean_c[s], diff_c[s], s=15, alpha=0.7,
                   color=RIVER_COLORS[rv], edgecolor="none", label=rv)
    ax.axhline(bias, color=RED, lw=1.2, ls="--")
    ax.axhline(hi, color=GREEN, lw=0.9, ls=":")
    ax.axhline(lo, color=GREEN, lw=0.9, ls=":")
    xr = mean_c.max()
    ax.text(xr, bias, f" Mean ({bias:+.0f})", color=RED, fontsize=9,
            ha="right", va="bottom")
    ax.text(xr, hi, " +1.96 SD", color=GREEN, fontsize=9, ha="right", va="bottom")
    ax.text(xr, lo, " −1.96 SD", color=GREEN, fontsize=9, ha="right", va="top")
    ax.set_xlabel("Mean of automated & technician daily net count")
    ax.set_ylabel("Automated − technician (daily net count)")
    ax.set_title("Chinook Salmon daily net: automated vs technician", fontsize=12.5, pad=8)
    theme_minimal(ax)
    ax.legend(frameon=True, fontsize=9, loc="upper left", title="River",
              edgecolor="0.7", framealpha=0.95)
    fig.tight_layout()
    p2 = Path(fig_dir) / "fig_bland_altman.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p1, p2, dict(bias=bias, lo=lo, hi=hi, n=len(d))


def main():
    """CLI entry: load tables, write concordance_ccc.csv, render both figures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(config.REPO_DATA))
    ap.add_argument("--fig-dir", default=None)
    a = ap.parse_args()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(a.fig_dir) if a.fig_dir else out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily, summ = load()
    tbl = build_table(daily, summ)
    tbl.to_csv(out_dir / "concordance_ccc.csv", index=False)
    print(tbl.to_string(index=False))
    p1, p2, ba = make_figures(daily, fig_dir)
    print(f"\nBland-Altman (daily Chinook): bias={ba['bias']:.1f}, "
          f"95% LoA [{ba['lo']:.0f}, {ba['hi']:.0f}], n={ba['n']}")
    print("wrote:", out_dir / "concordance_ccc.csv")
    print("wrote:", p1)
    print("wrote:", p2)


if __name__ == "__main__":
    main()

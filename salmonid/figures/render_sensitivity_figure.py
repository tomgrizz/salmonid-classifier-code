"""
In plain terms: this file draws one of the paper's supplementary figures. It checks the rule the
counting software uses to decide when a fish has really moved upstream versus downstream -- a fish
must drift at least 50 pixels (dots in the video image) upward, or 300 pixels downward, before it is
tallied. Using count tables saved from earlier runs, it shows that this deployed setting reproduces
the human technicians' net fish totals more closely than any other threshold combination we tried
across 2020-2025 (the net-rule sensitivity figure, fig_net_rule_sensitivity.png).

Net-rule sensitivity figure (manuscript supplementary): shows the deployed asymmetric net rule
(up>=50 / down>=300 px) is the GLOBAL optimum for reproducing technician net counts across 2020-2025,
both rivers.

Pipeline position
    results -> figures. Terminal reporting step: it consumes the committed net-threshold sweep CSVs
    and renders a supplementary manuscript figure. It runs no model.

Inputs
    The per-(year, rule) daily net CSVs produced by the threshold sweep
    (``salmonid.count.corpus_countr_from_store --up-thresh/--down-thresh``), committed here under
    ``config.NET_RULE_SWEEP`` (data/net_rule_sweep/). Each CSV has one row per day with integer
    ``model`` and ``tech`` net-count columns.

Outputs
    - ``figures/fig_net_rule_sensitivity.png`` -- the two-panel figure.
    - ``output/tier1/net_rule_sensitivity.json`` -- the exact %-error numbers plotted. The committed
      copy is ``data/net_rule_sweep/net_rule_sensitivity.json``; run_tier1.py compares the two.

Feeds
    Manuscript Section 3.4 (the 50/300 px rule gives 3.6% mean absolute error; the alternatives
    10% to 26%). The figure itself is supporting material and is not placed in the manuscript.

Tier
    1 (runs from committed data).

Environment
    Base env (matplotlib + numpy only). Does NOT need the DEIMv2 ``.venv_deim`` -- no detector/torch here.

Manuscript claim supported
    That the deployed asymmetric net rule (up>=50 / down>=300 px) is the rule-family optimum for
    reproducing technician net counts.
      Panel A: raising the UP threshold (DOWN fixed at 300) monotonically diverges from the technician
               total -- the current up=50 sits near 0% error.
      Panel B: over the well-behaved years (2022-2025) the deployed rule has the lowest mean |net error|
               of the whole rule family (raise-up, lower-down, symmetric, no-stay).

Run
    python -m salmonid.figures.render_sensitivity_figure
"""
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salmonid import config

# Committed net-threshold sweep CSVs (data/net_rule_sweep/).
SWEEP = config.NET_RULE_SWEEP
OUT = config.FIGURES_DIR                     # figure goes to figures/ (git-ignored)
JSON_OUT = config.OUTPUT_DIR / "tier1"       # the plotted numbers, as a small JSON
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
GOOD = [2022, 2023, 2024, 2025]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150})


def overall_pcterr(fn):
    """Overall net-count %-error (model vs technician) for one sweep CSV.

    Args:
        fn: path to a ``{year}_...csv`` sweep file with integer ``model`` and ``tech`` columns.
    Returns:
        ``100 * (sum(model) - sum(tech)) / sum(tech)``, or NaN if the technician total is 0.
    """
    rows = list(csv.DictReader(open(fn, encoding="utf-8")))
    mt = sum(int(r["model"]) for r in rows); tt = sum(int(r["tech"]) for r in rows)
    return 100 * (mt - tt) / tt if tt else float("nan")


def upfile(y, t):
    """Sweep CSV for year ``y`` at UP threshold ``t`` px (DOWN threshold fixed at 300)."""
    return SWEEP / f"{y}_up{t}.csv"                                  # (up=t, down=300)


# rule family: label -> filename builder
FAMILY = [
    ("up50/dn300\n(deployed)", lambda y: SWEEP / f"{y}_up50.csv"),
    ("up50/dn150",             lambda y: SWEEP / f"{y}_u50_d150.csv"),
    ("up50/dn50\n(sym-low)",   lambda y: SWEEP / f"{y}_u50_d50.csv"),
    ("up150/dn150\n(sym)",     lambda y: SWEEP / f"{y}_u150_d150.csv"),
    ("up1/dn1\n(no-stay)",     lambda y: SWEEP / f"{y}_u1_d1.csv"),
]

UPS = [50, 150, 250, 300]
# Per-year colours. 2020/2021 are greyed (and dotted below): magnitude-caveated recovered footage,
# treated as anomalies and excluded from the Panel-B family comparison (GOOD = 2022-2025).
YCOL = {2020: "#b0b0b0", 2021: "#b0b0b0", 2022: "#1f77b4", 2023: "#2ca02c", 2024: "#ff7f0e", 2025: "#d62728"}


def main():
    """Draw the two-panel figure and write the plotted numbers (no work happens at import)."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))

    # ---- Panel A: %err vs up-threshold ----
    axA.axhspan(-15, 15, color="#4caf50", alpha=0.10, zorder=0)           # +/-15% count acceptance band
    axA.axhline(0, color="#444", lw=1, ls="--", zorder=1)
    for y in YEARS:
        vals = [overall_pcterr(upfile(y, t)) for t in UPS]
        anomaly = y in (2020, 2021)
        axA.plot(UPS, vals, marker="o", ms=5, lw=2.2 if not anomaly else 1.3,
                 ls="-" if not anomaly else ":", color=YCOL[y],
                 label=f"{y}" + (" (anomaly)" if anomaly else ""), zorder=3 if not anomaly else 2)
    axA.axvline(50, color="#333", lw=1.2, alpha=0.6)
    axA.annotate("deployed\n(up=50)", xy=(50, axA.get_ylim()[0]), xytext=(70, -58),
                 fontsize=9, color="#333")
    axA.set_xlabel("UP threshold (px)   [down fixed at 300]")
    axA.set_ylabel("net count error vs technician (%)")
    axA.set_title("A. Raising the up threshold diverges from tech")
    axA.set_xticks(UPS)
    axA.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.9)
    axA.grid(True, alpha=0.25)

    # ---- Panel B: mean|%err| over 2022-2025 for the rule family ----
    labels = [lbl for lbl, _ in FAMILY]
    means, ranges = [], []
    for _, fn in FAMILY:
        errs = [abs(overall_pcterr(fn(y))) for y in GOOD]
        means.append(np.mean(errs)); ranges.append((min(errs), max(errs)))
    colors = ["#2ca02c"] + ["#b74b4b"] * (len(FAMILY) - 1)
    x = np.arange(len(FAMILY))
    bars = axB.bar(x, means, color=colors, width=0.62, zorder=3)
    # range whiskers across the 4 years
    for i, (lo, hi) in enumerate(ranges):
        axB.plot([i, i], [lo, hi], color="#333", lw=1.2, zorder=4)
        axB.plot([i - 0.08, i + 0.08], [hi, hi], color="#333", lw=1.2, zorder=4)
    for i, m in enumerate(means):
        axB.text(i, m + 0.7, f"{m:.1f}%", ha="center", fontsize=10, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylabel("mean |net error| vs tech, 2022-2025 (%)")
    axB.set_title("B. The deployed rule is the family optimum")
    axB.set_ylim(0, max(means) * 1.25)
    axB.grid(True, axis="y", alpha=0.25)
    axB.text(0, means[0] + 2.6, "best", ha="center", fontsize=9, color="#2ca02c", fontweight="bold")

    fig.suptitle("Net-rule sensitivity: the deployed asymmetric rule (up≥50 / down≥300 px) "
                 "best reproduces technician net counts", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_net_rule_sensitivity.png"
    fig.savefig(png, bbox_inches="tight")
    print(f"wrote {png}")

    # the plotted numbers, so the figure can be checked without reading pixels
    summ = {"years": YEARS, "up_sweep_pcterr": {y: {t: overall_pcterr(upfile(y, t)) for t in UPS} for y in YEARS},
            "family_mean_abs_pcterr_2022_2025": {lbl.split(chr(10))[0]: float(np.mean([abs(overall_pcterr(fn(y))) for y in GOOD]))
                                                 for lbl, fn in FAMILY}}
    with open(JSON_OUT / "net_rule_sensitivity.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summ, fh, indent=1)
    print(f"wrote {JSON_OUT / 'net_rule_sensitivity.json'}")


if __name__ == "__main__":
    main()

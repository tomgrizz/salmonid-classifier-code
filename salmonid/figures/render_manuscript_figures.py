"""In plain terms: this file draws the finished charts that appear in the paper. It does not run
the computer-vision model itself — it just reads the already-computed results tables and turns them
into publication-ready figures: the species "confusion matrix" (a grid showing how often each
species was named correctly or mixed up, backing the paper's 0.966 accuracy headline) and the
day-by-day fish-passage charts comparing the automated counter against the human technicians
(Figure S2, including the fully held-out 2024 year the model never trained on).

Render the manuscript's held-out evaluation and daily-net figures.

Pipeline stage: **results -> figures** (the final step). This module performs no detection,
tracking, or counting itself; it reads already-computed, committed result artifacts and renders
the publication figures from them.

Environment: **base env** (CPU-only: matplotlib + numpy + pandas). It does NOT require the
DEIMv2 ``.venv_deim`` — no torch or model weights are loaded here.

Figures produced
----------------
* ``fig_confusion_v26`` — single-stage v2.6 species confusion matrix on the frozen held-out
  test set. Inputs: ``config.PREDS_V26`` (per-track predictions) + ``config.TESTSET_V6_GT``
  (per-track ground truth). Backs the manuscript's held-out species-ID claim (macro-F1 0.966).
* ``fig_daily_net_2025_{credit,ganaraska}`` — Fig S2: cumulative net-upstream passage, model vs
  technician, one panel per main species. Input: ``config.CANONICAL_DAILY`` (the committed
  BoT-SORT daily-net table, 2019-2025). Backs the daily-net counting-agreement claim (daily-r).
* ``fig_daily_net_2024_{credit,ganaraska}`` — the fully held-out 2024 year (the model never
  trained on 2024). Input: ``config.NET_2024`` (committed 2024 BoT-SORT daily net).

Non-manuscript preview figures (``fig_seasonal_timing``, ``fig_multiyear_chinook``) read external
corpus artifacts under ``config.PROJECTS_ROOT`` and are skipped cleanly when that data is absent,
so they can never block the committed manuscript figures above.

Outputs: figure files under ``config.FIGURES_DIR`` (via ``manuscript_style.save_fig`` and direct
``savefig`` for the preview panels).

Run:  python -m salmonid.figures.render_manuscript_figures
"""
import csv, json, collections
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch

from salmonid import config
from salmonid.figures.manuscript_style import set_style, save_fig, COL1_IN, MODEL_C

set_style()
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
SHORT = ["Chinook", "Coho", "Atlantic", "Rainbow", "Brown"]
# fixed panel order for every cumulative-net figure so all four come out identical in size/layout
PANEL_SPECIES = ["Chinook", "Rainbow Trout", "Coho", "Brown Trout"]
OUT = config.FIGURES_DIR; OUT.mkdir(parents=True, exist_ok=True)
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

    Diagonal (correct) cells are shaded green by recall; off-diagonal (error) cells are shaded red by
    their within-row share. Every non-empty cell prints the track count and the within-row (recall) %.

    Reads the PUBLISHED model's predictions (v2.7 / checkpoint0017), which are leak-clean and so
    cover all 366 frozen tracks rather than the 304-track remainder v2.6 had to be scored on.
    """
    gt = load_gt()
    pf = config.PUBLISHED_PREDS
    M = confusion(gt, load_preds(pf))
    row = M.sum(1, keepdims=True)
    Mn = M / np.clip(row, 1, None)
    # one green hue and one red hue; the VALUE is encoded by opacity (transparency), not a different
    # shade: higher recall -> more solid green, larger error -> more solid red (a minor error is a
    # faint red of the same colour). The count + recall printed in every cell carries the information,
    # so the matrix is fully legible in grayscale; cell text auto-contrasts to the blended cell tone.
    GREEN, RED = (0.16, 0.60, 0.34), (0.85, 0.24, 0.20)
    err_max = max([Mn[i, j] for i in range(5) for j in range(5) if i != j and M[i, j] > 0] + [1e-6])
    diag = [Mn[i, i] for i in range(5)]
    rmin, rrange = min(diag), max(max(diag) - min(diag), 1e-6)   # stretch the narrow 93-100% recall range

    def text_color(rgb, a):
        """Pick black/white cell text by the luminance of ``rgb`` alpha-blended over white."""
        lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) * a + (1 - a)  # blended over white
        return "white" if lum < 0.5 else "#1a1a1a"

    fig, ax = plt.subplots(figsize=(COL1_IN, COL1_IN * 1.12), layout="constrained")
    ax.set_xlim(-0.5, 4.5); ax.set_ylim(4.5, -0.5)  # keep true-species row 0 at the top
    ax.set_aspect("equal")
    for i in range(5):
        for j in range(5):
            frac = Mn[i, j]
            if i == j:
                a = 0.45 + 0.50 * (frac - rmin) / rrange         # more opaque green = higher recall
                rgb = GREEN
            elif M[i, j] > 0:
                a = 0.40 + 0.60 * (frac / err_max)               # more opaque red = larger error
                rgb = RED
            else:
                a, rgb = 0.0, (1, 1, 1)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=(*rgb, a),
                                   edgecolor="#dcdcdc", linewidth=0.5))
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]}\n{100 * frac:.1f}%", ha="center", va="center",
                        fontsize=7, linespacing=1.15, color=text_color(rgb, a), fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, rotation=35, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted species"); ax.set_ylabel("True species")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.legend(handles=[Patch(facecolor=(*GREEN, 0.82), label="correct (opacity ∝ recall)"),
                        Patch(facecolor=(*RED, 0.85), label="error (opacity ∝ error size)")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.1)
    # Named for what it shows, not for an internal model revision: the old "fig_confusion_v26"
    # filename leaked the model version into the manuscript's figure captions.
    save_fig(fig, "fig_confusion_heldout", kind="vector")


def _load_daily(path, site):
    """Load a per-day model/tech net CSV for one ``site`` as {species: {model, tech, date}}.

    Used by the (non-canonical) preview daily figures.
    """
    # daily CSVs carry BOTH sites (tech spans both rivers); MUST filter to this figure's site,
    # else the POOLED tech inflates the comparison (matches the gate's per-site row, not POOLED).
    d = collections.defaultdict(lambda: {"model": [], "tech": [], "date": []})
    for r in csv.DictReader(open(path)):
        if r["site"] != site:
            continue
        sp = r["species"]
        d[sp]["date"].append(r["date"]); d[sp]["model"].append(float(r["model"])); d[sp]["tech"].append(float(r["tech"]))
    return d


def _load_daily_canonical(site, year):
    """Daily model/tech net from the COMMITTED canonical dataset (``config.CANONICAL_DAILY``)
    — the same BoT-SORT source as Fig 4 and the main text, so Fig S2 agrees with them. (The old G: corpus
    file was pre-BoT-SORT and over-counted the dense Ganaraska Chinook run by ~27%, contradicting the text.)"""
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


def fig_daily(site_tag, site_name, csv_path, title, source="corpus"):
    """Cumulative net-upstream passage, model vs technician, one panel per main species.

    Every figure renders the same four species in the same order at a fixed canvas size, with a
    calendar-month x-axis over a common Mar-Dec season window, so all cumulative figures are the same
    size and directly comparable across rivers and years.
    """
    if source == "canonical":
        d = _load_daily_canonical(site_name, int(site_tag[:4]))
    elif not Path(csv_path).is_file():
        print("skip (missing)", csv_path); return
    else:
        d = _load_daily(csv_path, site_name)
    all_dates = [dt for s in d for dt in d[s]["date"]]
    if not all_dates:
        print("skip (no rows)", csv_path); return
    yr = int(min(all_dates)[:4])
    fig, axes = plt.subplots(len(PANEL_SPECIES), 1, figsize=(COL1_IN, 5.7), squeeze=False,
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
            ax.plot(dt, ct, label=f"technician ({int(ct[-1])})", color="#1b3a5c", lw=1.4)
            ax.plot(dt, cm, label=f"model ({int(cm[-1])})", color=MODEL_C, lw=1.4, ls=(0, (5, 2)))
            ax.set_title(f"{sp}: daily $r$={rtxt}", loc="left", fontsize=7.5)
            ax.legend(fontsize=6.3, loc="upper left", handlelength=1.6)
        else:
            ax.set_title(f"{sp}: no counts", loc="left", fontsize=7.5, color="#888")
        ax.grid(alpha=0.25); ax.set_ylabel("Cumulative net", fontsize=7)
        ax.set_xlim(datetime(yr, 3, 1), datetime(yr, 12, 1))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(labelsize=6.5)
    axes[-1, 0].set_xlabel("Month", fontsize=7.5)
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=8.5)
    fig.align_ylabels(axes[:, 0])
    save_fig(fig, f"fig_daily_net_{site_tag}", kind="vector")


def fig_seasonal_timing(site_name="Ganaraska",
                        csv_path=str(config.PROJECTS_ROOT / "deimv2_x_v26_corpus2025" / "gan_daily_net.csv"),
                        species=("Chinook", "Coho", "Rainbow Trout"), title=None):
    """Seasonal run timing: 7-day rolling average of daily passage, model (line) vs technician (bars)."""
    if not Path(csv_path).is_file():
        print("skip (missing)", csv_path); return
    df = pd.read_csv(csv_path)
    df = df[df["site"] == site_name].copy()
    if df.empty:
        print("skip (no site rows)", csv_path); return
    df["date"] = pd.to_datetime(df["date"])
    yr = int(df["date"].dt.year.mode()[0])
    full = pd.date_range(f"{yr}-03-01", f"{yr}-12-01", freq="D")
    sp_list = [s for s in species if s in set(df["species"])]
    colors = {"Chinook": "#c65b2e", "Coho": "#e0a13b", "Rainbow Trout": "#2f6f9f",
              "Brown Trout": "#7a5230", "Atlantic": "#3aa76d"}
    fig, axes = plt.subplots(len(sp_list), 1, figsize=(9, 2.6 * len(sp_list)), squeeze=False, sharex=True)
    for ax, sp in zip(axes[:, 0], sp_list):
        g = df[df["species"] == sp].set_index("date").reindex(full, fill_value=0)
        m = g["model"].rolling(7, center=True, min_periods=1).mean()
        t = g["tech"].rolling(7, center=True, min_periods=1).mean()
        ax.bar(full, t, width=1.0, color="#b8b8b8", label="technician (7-day avg)")
        ax.plot(full, m, color=colors.get(sp, "#c65b2e"), lw=2, label="model (7-day avg)")
        ax.set_title(sp, loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.2)
        ax.set_ylabel("daily net")
    axes[-1, 0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[-1, 0].set_xlim(full[0], full[-1]); axes[-1, 0].set_xlabel("month")
    if title is None:
        title = f"{site_name} {yr}: seasonal run timing (7-day rolling daily passage)"
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "fig_seasonal_timing.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", p)


def fig_multiyear_chinook():
    """Two-stage v6 Chinook NET record 2020-2024 (Fig 4), with the single-stage v2.6 held-out 2024
    result overlaid as a distinct marker. Two-stage values from _corpus_historical_vs_tech.py +
    _corpus_2024_vs_tech.py (extracted 2026-07-09); single-stage 2024 from corpus2024/2024_count_r_net.json.
    Ganaraska 2020 excluded (footage-coverage gap: ~1,487 clips, model activity=36)."""
    years = [2020, 2021, 2022, 2023, 2024]
    # (%err, daily-r) per river; None = excluded/absent
    gan = {2020: None, 2021: (24, 0.968), 2022: (12, 0.993), 2023: (-8, 0.969), 2024: (-7, 0.988)}
    cred = {2020: (-1, 0.992), 2021: (2, 0.966), 2022: (-8, 0.986), 2023: (-4, 0.996), 2024: (-5, 0.976)}
    # single-stage v2.6 held-out 2024 Chinook (never trained on 2024)
    ss = {r["scope"]: r for r in json.load(open(config.PROJECTS_ROOT / "deimv2_x_v26_corpus2024" / "2024_count_r_net.json"))
          if r["species"] == "Chinook"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, idx, lab in [(a1, 1, "daily-r"), (a2, 0, "seasonal net %err")]:
        for d, name, col, mk in [(gan, "Ganaraska", "#1b3a5c", "o"), (cred, "Credit", "#c65b2e", "s")]:
            xs = [y for y in years if d[y] is not None]
            ys = [d[y][idx] for y in xs]
            ax.plot(xs, ys, marker=mk, color=col, label=f"{name} (two-stage)", lw=2, ms=8)
        ax.set_xticks(years); ax.set_xlabel("year"); ax.set_title(f"Chinook net: {lab}")
        ax.grid(alpha=0.25); ax.set_xlim(2019.5, 2024.6)
    # overlay single-stage held-out 2024 points
    for name, col in [("Ganaraska", "#1b3a5c"), ("Credit", "#c65b2e")]:
        rr = ss.get(name)
        if rr:
            lbl = "single-stage 2024" if name == "Ganaraska" else None
            a1.scatter([2024], [rr["pearson_r"]], marker="*", s=240, color=col,
                       edgecolor="black", linewidth=0.8, zorder=6, label=lbl)
            a2.scatter([2024], [rr["pct_err"]], marker="*", s=240, color=col,
                       edgecolor="black", linewidth=0.8, zorder=6)
    a1.axhline(0.90, ls="--", color="grey", lw=1); a1.set_ylabel("Pearson r"); a1.set_ylim(0.8, 1.01)
    a1.text(2020, 0.905, "gate r≥0.90", fontsize=8, color="grey")
    a1.legend(fontsize=8, loc="lower left")
    a2.axhspan(-15, 15, color="green", alpha=0.08); a2.axhline(0, color="grey", lw=0.8)
    a2.set_ylim(-17, 27)
    a2.set_ylabel("% error (model minus tech)"); a2.text(2020, 16, "±15% gate", fontsize=8, color="green")
    fig.suptitle("Multi-year Chinook net validation: two-stage v6 (2020-2024) with single-stage v2.6 2024",
                 fontweight="bold")
    fig.tight_layout()
    p = OUT / "fig_multiyear_chinook_net.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    # manuscript figures first, unconditionally: Fig 2 (confusion) + Fig S2 (2025 daily net, panels a/b)
    fig_confusion()
    fig_daily("2025_credit", "Credit", "", "(a) Credit River, 2025", source="canonical")
    fig_daily("2025_ganaraska", "Ganaraska", "", "(b) Ganaraska River, 2025", source="canonical")
    # non-manuscript preview figures (results-preview doc only); skip cleanly if their data is absent so
    # they can never block the manuscript figures above.
    for fn in (fig_multiyear_chinook, fig_seasonal_timing):
        try:
            fn()
        except Exception as e:
            print(f"skip {fn.__name__}: {e}")
    # 2024 = fully HELD-OUT year (model never trained on 2024), BoT-SORT-corrected magnitude, both rivers.
    for tag, site in [("2024_credit", "Credit"), ("2024_ganaraska", "Ganaraska")]:
        try:
            fig_daily(tag, site, config.NET_2024,
                      f"{site} River, 2024 (held-out)")
        except Exception as e:
            print(f"skip {tag}: {e}")

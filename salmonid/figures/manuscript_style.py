"""In plain terms: this file is the shared "house style" that makes every figure in the
paper look consistent and meet the journal's printing rules — matching font sizes, colours
that still read correctly when printed in black-and-white, the right column widths, and
saving each chart as a print-ready file. It computes no result of its own; it takes a
finished chart handed over by another script and renders it to spec. So it produces no single
number, but every figure in the manuscript (the performance and activity panels, the species
run-timing and body-shape panels, the Ganaraska sanctuary panel, and so on) passes through it.

Shared CJFAS style helper: set_style (>=7pt, pdf.fonttype=42, 600dpi), Okabe-Ito
grayscale-safe PAL, save_fig().

This is the shared CJFAS / Canadian Science Publishing production style imported by every
manuscript figure script, so the whole set conforms to one spec instead of each script setting
its own.

Pipeline position
-----------------
Terminal ``results -> figures`` stage. It holds no analysis logic and reads no data itself: it
only configures matplotlib and writes files. The figure scripts (in this package) do the
data -> detect -> track -> eval/count -> results work, then hand their finished ``Figure`` object
to :func:`save_fig` for on-spec rendering.

Inputs / outputs
----------------
* Input: a matplotlib ``Figure`` already laid out by a caller (plus a filename stem and format).
* Output: the rendered figure written to :data:`salmonid.config.FIGURES_DIR` (the repo ``figures/``
  directory) as ``<name>.png`` (preview) plus a production ``.pdf`` (vector) or ``.tiff`` (photo).

Environment
-----------
Runs in the **base env** (matplotlib only) — no detection/training deps, so it never needs the
DEIMv2 ``.venv_deim``.

Manuscript claim
----------------
Not a number itself — it is the presentation layer for *every* manuscript figure (hero
performance/activity, species phenology/morphology, the Ganaraska sanctuary panel, the multi-year
scorecard, the net-rule sensitivity sweep, ...), guaranteeing each meets the journal's print spec.

Spec (docs/CJFAS_journal_requirements.md, docs/CSP_manuscript_requirements.txt in the paper repo):
  * width 1-column = 8.84 cm, 2-column = 18.2 cm; max height 23.7 cm
  * ONE font family, >= 7 pt at final print size, fonts embedded
  * line weights >= 0.35 pt (no hairlines)
  * grayscale-legible: separate series by marker / dash / fill pattern, not colour alone;
    solid high-contrast colours, avoid pale fills (>= ~10% K)
  * production formats EPS/TIFF/PDF; filename includes "fig"
  * a figure carries NO title — the caption is the title (kept out of the artwork)

Usage in a figure script:
    from salmonid.figures.manuscript_style import (
        set_style, save_fig, COL1_IN, COL2_IN, MAXH_IN, PAL, LS, MODEL_C, TECH_C)
    set_style()
    fig, ax = plt.subplots(figsize=(COL2_IN, 3.2), layout="constrained")
    ...
    save_fig(fig, "fig_example", kind="vector")   # -> fig_example.png (preview) + fig_example.pdf (vector)

save_fig writes at the figure's TRUE physical size (no tight-bbox rescale, so the PDF really is the
requested column width). Build figures with layout="constrained" so nothing is clipped at that size.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salmonid import config

# ---- print geometry (CJFAS) -------------------------------------------------
CM = 1.0 / 2.54
COL1_IN = 8.84 * CM     # 3.48 in  — single column
COL15_IN = 13.6 * CM    # 5.35 in  — "1.5 column" (CSP permits intermediate widths up to 2-col)
COL2_IN = 18.2 * CM     # 7.17 in  — double column
MAXH_IN = 23.7 * CM     # 9.33 in  — max height

# ---- colours ----------------------------------------------------------------
# Species palette: colourblind-safe (Okabe-Ito) AND spread across the luminance range so the
# five species stay separable in grayscale. Paired with LS (line styles) for line figures.
PAL = {
    "Chinook": "#0072B2",        # blue      (mid-dark)
    "Coho": "#E69F00",           # orange    (light)
    "Atlantic": "#009E73",       # green     (mid)
    "Rainbow Trout": "#D55E00",  # vermilion (dark)
    "Brown Trout": "#5C4033",    # brown     (darkest)
}
LS = {   # per-species dash pattern — the grayscale fallback when colour is stripped
    "Rainbow Trout": (0, ()),               # solid
    "Chinook": (0, (5, 2)),                 # dashed
    "Coho": (0, (1, 1.3)),                  # dotted
    "Atlantic": (0, (3, 1.3, 1, 1.3)),      # dash-dot
    "Brown Trout": (0, (5, 1.3, 1, 1.3, 1, 1.3)),  # dash-dot-dot
}
MK = {   # per-species marker (used where markers read better than dashes)
    "Chinook": "o", "Coho": "s", "Atlantic": "^", "Rainbow Trout": "D", "Brown Trout": "v",
}
# Class order (label ids 0-4): the one species order used by every figure and table in the paper.
SPECIES_ORDER = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]

# ---- species names and series labels -------------------------------------------
# CJFAS/AFS house style: the full American Fisheries Society common name on the artwork
# ("Chinook Salmon", not "Chinook"). Internal keys stay short (they are data values in
# canonical_daily.csv); only the displayed string changes. AFS2 is the two-line form for
# panels too narrow for the one-line name.
AFS = {
    "Chinook": "Chinook Salmon",
    "Coho": "Coho Salmon",
    "Atlantic": "Atlantic Salmon",
    "Rainbow Trout": "Rainbow Trout",
    "Brown Trout": "Brown Trout",
}
AFS2 = {k: v.replace(" ", "\n") for k, v in AFS.items()}   # "Chinook\nSalmon"
# One label pair for the two series, matching the captions ("model output" vs "technician").
MODEL_LABEL = "Model output"
TECH_LABEL = "Technician"
TECH_LABEL_LONG = "Technician (manual count)"

# Two-method comparison colours (the manuscript captions name these "red line" / "blue bars").
MODEL_C = "#C0392B"     # model output — red
TECH_C = "#6F9AC9"      # technician — steel blue (mid tone, >= 10% K, not pale)
TECH_EDGE = "#3F6390"


def set_style():
    """Apply the production rcParams. Idempotent; call once at import time in each script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 9.0,
        "axes.linewidth": 0.5,          # >= 0.35 pt
        "axes.edgecolor": "#333333",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.30,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.5,
        "hatch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "savefig.bbox": None,           # keep the exact physical size (no tight rescale)
        "pdf.fonttype": 42,             # embed TrueType subset; text stays selectable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


CSP_MIN_DPI = 600       # CSP/CJFAS: 600 dpi at print size for combination (photo + text) and line art


def save_fig(fig, name, kind="vector", outdir=None, png_dpi=CSP_MIN_DPI):
    """Write ``fig`` at its true physical size, in the repo's manuscript print format.

    Args:
        fig: the fully laid-out matplotlib ``Figure`` to render (caller owns the plotting).
        name: filename stem (no extension); the journal spec wants it to start with "fig".
        kind: "vector" -> PNG preview + PDF (line-art plots); "photo" -> PNG preview + flattened
            RGB 600 dpi TIFF (photo montages); "png" -> PNG only.
        outdir: destination directory, resolved when called; defaults to ``config.FIGURES_DIR``
            (overridable with the ``SALMONID_FIGURES_DIR`` environment variable).
        png_dpi: raster resolution at the figure's true print size (600 dpi by default).

    Side effects:
        Creates ``outdir`` if needed, writes the PNG (always) and the production PDF/TIFF, closes
        ``fig`` to free memory, and prints the path written. PDFs carry no CreationDate, so a
        rebuild from the same data is byte-reproducible.
    """
    outdir = Path(outdir) if outdir is not None else config.FIGURES_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    dpi = png_dpi
    png = outdir / f"{name}.png"
    fig.savefig(png, dpi=dpi, facecolor="white", metadata={"Software": None})
    if kind == "vector":
        fig.savefig(outdir / f"{name}.pdf", facecolor="white",
                    metadata={"CreationDate": None, "Creator": None, "Producer": None})
    elif kind == "photo":
        # CJFAS wants a flattened TIFF (no alpha channel): render to PNG in memory, drop alpha.
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        Image.open(buf).convert("RGB").save(outdir / f"{name}.tiff", dpi=(dpi, dpi),
                                            compression="tiff_deflate")
    plt.close(fig)
    print(f"wrote {png} ({dpi:.0f} dpi, {fig.get_size_inches()[0]:.2f} in wide) (+ production {kind})")

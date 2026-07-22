"""
In plain terms: this script draws the paper's Figure 6 — a side-by-side picture that lines up
one clear example of each of the five species so a reader can see how their body shapes and
depths differ. It takes real fish images that were automatically cut out from the counter-tunnel
video (background removed, leaving just the fish outline), straightens each one to horizontal,
sizes them all to the same body length, and stacks them. It only makes this comparison figure —
it does not calculate any accuracy or count number.

Figure 6 (manuscript) — five-species morphology comparison from clean SAM2 cutouts.

Pipeline position: **results -> figures** (the final, presentation end of the
data -> detect -> track -> eval/count -> results -> figures chain). This is a pure
visualisation script; it consumes no model and produces no manuscript *number* — it
renders the qualitative claim that the five reported salmonid species are visually
distinguishable by body shape / relative depth.

What it does: for each species it picks one high-quality, fully-exposed, near-lateral SAM2
cutout from the verified cutout library (``config.SAM2_CUTOUTS``), deskews it to horizontal
using its min-area-rect angle, orients it head-left, scales every fish to a common body
length, and stacks the five so their body shape / relative depth can be compared directly.
The cutouts are real fish segmented as they moved through the counter tunnel (alpha channel =
silhouette).

Inputs (Tier 2, external image data — not committed to the repo):
  * ``config.SAM2_CUTOUTS / "manifest.csv"`` — one row per cutout with quality metrics
    (border_frac, angle_deg, solidity, cc_ratio, sam_score, length_px, depth_px, ...).
  * ``config.SAM2_CUTOUTS / <png>`` — the RGBA cutout PNGs referenced by the manifest.

Outputs (two draft montages, written by ``save_fig`` into the manuscript-figures dir):
  * fig_species_morphology.png           - best clean/clear cutout per species
  * fig_species_morphology_fullbody.png  - restricted to elongated WHOLE-BODY cutouts (aspect gate),
                                           so the full fish is shown rather than a head-dominated view

Selection is visibility-aware (prefers brighter, higher-contrast silhouettes) so faint species such as
Atlantic Salmon are easy to see.

  python -m salmonid.figures.fig_species_morphology

Environment: base env (CPU-only — cv2 / numpy / pandas / matplotlib); the DEIMv2 ``.venv_deim``
is not needed. Reads the cutout PNGs + manifest from ``config.SAM2_CUTOUTS``.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from salmonid import config
from salmonid.figures.manuscript_style import set_style, save_fig, COL1_IN

set_style()
BASE = config.SAM2_CUTOUTS
OUT = Path("docs/manuscript_figures"); OUT.mkdir(parents=True, exist_ok=True)
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
# Curated cutouts for the representative figure: the original set the user approved, with the Chinook
# kept as the original (good) pick and the Atlantic swapped for a clearer, brighter individual.
CLEAN_OVERRIDE = {
    "Chinook": "ls_5d5d94cf_036015_f000024",
    "Coho": "ls_c3fdc958_002539_f000165",
    "Atlantic": "ls_4ba787a2_035605_f000234",
    "Rainbow Trout": "ls_ea0daddf_034078_f000000",
    "Brown Trout": "ls_b6fbf2e7_035934_f000252",
}
TARGET_LEN = 620          # px: every fish is scaled to this body length


def load_rgba(rel):
    """Load a cutout PNG (relative path under BASE) as an RGBA uint8 array.

    Args:
        rel: manifest ``png`` path, relative to ``config.SAM2_CUTOUTS``.
    Returns:
        HxWx4 RGBA array; a fully-opaque alpha channel is synthesised for 3-channel PNGs.
    """
    img = cv2.imread(str(BASE / rel), cv2.IMREAD_UNCHANGED)   # BGRA
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def crop_to_alpha(rgba, pad=6):
    """Tight-crop an RGBA image to its non-transparent silhouette (alpha>8), with a small pad.

    Args:
        rgba: HxWx4 RGBA array.
        pad: pixels of transparent margin kept around the silhouette bbox.
    Returns:
        The cropped RGBA view (or the input unchanged if fully transparent).
    """
    ys, xs = np.where(rgba[:, :, 3] > 8)
    if len(xs) == 0:
        return rgba
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + pad + 1)
    return rgba[y0:y1, x0:x1]


def _rot(rgba, ang):
    """Rotate an RGBA cutout by ``ang`` degrees on a transparent, diagonal-sized canvas.

    Padding to the image diagonal before rotating guarantees no corner is clipped; the result
    is re-cropped to its silhouette. Transparent borders keep the alpha silhouette intact.
    """
    h, w = rgba.shape[:2]
    diag = int(np.hypot(h, w))
    canvas = np.zeros((diag, diag, 4), np.uint8)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    canvas[oy:oy + h, ox:ox + w] = rgba
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), ang, 1.0)
    rot = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return crop_to_alpha(rot)


def deskew_horiz(rgba, angle_deg):
    """Rotate so the long body axis is horizontal (self-corrects the angle sign)."""
    best, best_ratio = rgba, 0
    for a in (-angle_deg, angle_deg):
        r = _rot(rgba, a)
        h, w = r.shape[:2]
        ratio = w / max(1, h)
        if ratio > best_ratio:
            best, best_ratio = r, ratio
    return best


def on_white(rgba):
    """Alpha-composite an RGBA cutout over a white background, returning an opaque RGB uint8 array."""
    rgb = rgba[:, :, :3].astype(float)
    a = (rgba[:, :, 3:4].astype(float)) / 255.0
    return (rgb * a + 255.0 * (1 - a)).astype(np.uint8)


def prep(row):
    """Load -> crop -> deskew-to-horizontal -> scale a manifest row's cutout to TARGET_LEN width.

    Args:
        row: a manifest row (needs ``png`` and ``angle_deg``).
    Returns:
        RGBA array whose width is exactly ``TARGET_LEN`` px (height scaled to preserve aspect),
        so every species is rendered at a common body length.
    """
    rgba = deskew_horiz(crop_to_alpha(load_rgba(row["png"])), float(row["angle_deg"]))
    h, w = rgba.shape[:2]
    return cv2.resize(rgba, (TARGET_LEN, max(1, int(round(h * TARGET_LEN / w)))), interpolation=cv2.INTER_AREA)


def visibility(rgba):
    """Clarity of the fish pixels: reward contrast/detail (std) and penalise under- or over-exposure
    (too dark, or washed-out/blown highlights). Avoids picking faint fish AND over-bright ones."""
    m = rgba[:, :, 3] > 16
    if m.sum() < 50:
        return 0.0
    px = (0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2])[m]
    mean, std = px.mean(), px.std()
    blown = (px > 245).mean()                    # fraction of clipped/blown-out pixels
    expo = max(0.0, 1.0 - abs(mean - 135.0) / 135.0)   # best near mid-grey, worse toward black/white
    return float(std * expo * (1.0 - blown))


def candidates(man, species, fullbody):
    """Return quality-filtered candidate cutouts for one species, best-first.

    Applies progressively relaxed quality tiers (centred/on-axis/solid/high-SAM) and stops at the
    first tier yielding >=3 cutouts; if none qualifies, falls back to the whole species sorted by
    SAM score. When ``fullbody`` is set, an extra length:depth aspect gate keeps only elongated
    whole-body cutouts (not head-dominated crops).

    Args:
        man: the full cutout manifest DataFrame.
        species: species name to filter on.
        fullbody: whether to additionally require the aspect (whole-body) gate.
    Returns:
        A DataFrame of candidate rows, sorted best-first.
    """
    g = man[man["species"] == species].copy()
    g = g[g["png"].apply(lambda p: (BASE / p).is_file())]
    g["aspect"] = g["length_px"] / g["depth_px"].clip(lower=1)
    tiers = [                                        # progressively relaxed if a species is sparse
        dict(border=0.0, ang=16, sol=0.92, cc=0.98, sam=0.95, length=260, aspect=2.9),
        dict(border=0.02, ang=22, sol=0.88, cc=0.96, sam=0.93, length=240, aspect=2.6),
        dict(border=0.05, ang=30, sol=0.82, cc=0.92, sam=0.90, length=200, aspect=2.2),
    ]
    for t in tiers:
        c = g[(g["border_frac"] <= t["border"]) & (g["angle_deg"].abs() <= t["ang"])
              & (g["solidity"] >= t["sol"]) & (g["cc_ratio"] >= t["cc"])
              & (g["sam_score"] >= t["sam"]) & (g["length_px"] >= t["length"])]
        if fullbody:
            c = c[c["aspect"] >= t["aspect"]]
        if len(c) >= 3:
            return c.sort_values(["sam_score", "solidity", "mask_area"], ascending=False)
    return g.sort_values("sam_score", ascending=False)


def pick(man, species, fullbody, k=16):
    """Choose the most-visible cutout among the top quality candidates."""
    c = candidates(man, species, fullbody)
    if not len(c):
        return None, None
    best, best_score, best_img = None, -1.0, None
    for _, r in c.head(k).iterrows():
        img = prep(r)
        v = visibility(img)
        if v > best_score:
            best, best_score, best_img = r, v, img
    return best, best_img


def median_ratio(manifest, species):
    """Species-level median depth:length over all reasonably clean cutouts (not one noisy individual)."""
    g = manifest[(manifest["species"] == species) & (manifest["solidity"] >= 0.7)
                 & (manifest["length_px"] >= 200) & (manifest["angle_deg"].abs() <= 30)]
    if not len(g):
        g = manifest[manifest["species"] == species]
    return float((g["depth_px"] / g["length_px"]).median()), len(g)


def build(man, fullbody, outname):
    """Assemble and save one stacked five-species morphology montage.

    For each species: use the curated ``CLEAN_OVERRIDE`` cutout (clean variant only) if present,
    else pick the most-visible quality candidate; deskew/scale it to a common body length; then
    stack all five on a white canvas with species labels and hand off to ``save_fig``.

    Args:
        man: the cutout manifest DataFrame.
        fullbody: build the whole-body variant (ignores CLEAN_OVERRIDE, applies the aspect gate).
        outname: output basename; its stem names the saved figure.
    Side effects:
        Prints per-species diagnostics (median depth:length, n, SAM score, aspect) and writes the
        figure (PNG preview + flattened TIFF) via ``save_fig``.
    """
    fish, labels = [], []
    for sp in SPECIES:
        sel = man[man["cutout_id"] == CLEAN_OVERRIDE.get(sp)] if not fullbody else man.iloc[0:0]
        if len(sel):
            row = sel.iloc[0]; img = prep(row)
        else:
            row, img = pick(man, sp, fullbody)
        if row is None:
            print("no cutout for", sp); continue
        med, n = median_ratio(man, sp)
        fish.append(img)
        labels.append(sp)          # species name only — dimensions belong in the caption, not the artwork
        # median depth:length + n are printed (not drawn) so the caption's ratios stay verifiable
        print(f"[{'fullbody' if fullbody else 'clean'}] {sp}: {row['cutout_id']}  "
              f"median depth:length={med:.2f}  n={n}  sam={row['sam_score']:.3f}  "
              f"aspect={row['length_px'] / max(1, row['depth_px']):.2f}")

    # deskewed cutouts already share a consistent facing (head to the right); no extra flip needed

    gap, top_pad = 50, 42
    heights = [f.shape[0] for f in fish]
    total_h = top_pad + sum(h + gap for h in heights)
    canvas = np.full((total_h, TARGET_LEN, 3), 255, np.uint8)
    bands, y = [], top_pad
    for f, h in zip(fish, heights):
        canvas[y:y + h, 0:TARGET_LEN] = on_white(f)
        bands.append(y)
        y += h + gap
    fig = plt.figure(figsize=(COL1_IN, COL1_IN * total_h / TARGET_LEN))
    ax = fig.add_axes([0, 0, 1, 1])           # axes fill the canvas -> output is exactly 1-column wide
    ax.imshow(canvas); ax.axis("off")
    for yb, lab in zip(bands, labels):
        ax.text(6, yb - 6, lab, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    save_fig(fig, Path(outname).stem, kind="photo")   # photo montage -> PNG preview + flattened TIFF


def main():
    """Read the cutout manifest and render both the clean and the whole-body montages."""
    man = pd.read_csv(BASE / "manifest.csv")
    build(man, fullbody=False, outname="fig_species_morphology.png")
    build(man, fullbody=True, outname="fig_species_morphology_fullbody.png")


if __name__ == "__main__":
    main()

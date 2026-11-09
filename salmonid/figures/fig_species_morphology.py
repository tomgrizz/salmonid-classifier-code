"""
Manuscript Figure 3: five-species body-shape comparison from verified SAM2 cutouts.

In plain terms: this script lines up one clear, whole-body example of each of the five species so
a reader can compare their body shapes and depths. It takes real fish segmented from the
counter-tunnel video (background removed, leaving the fish silhouette), straightens each one to
horizontal, scales every fish to the same total length, and lays the five out as a two-column
plate. It produces no manuscript number; the species median depth:length ratios it prints let the
caption's body-shape statement be checked.

Pipeline position: **results -> figures** (presentation only; no model is loaded).

Selection. ``WHOLE_BODY`` names one curated cutout per species. Each was chosen from the cutouts
whose mask clears every effective frame edge (the dark strip at x < 57 px, the 40 px header mask,
and the right and bottom edges), so the whole fish, caudal fin included, is in view. The earlier
picks ended in a straight vertical edge at the frame boundary (only the front half to two-thirds of
the fish), which inflated depth relative to length when they were scaled to a common width. Every
pick faces right, so no cutout is mirrored; each is only rotated to horizontal.

Inputs (Tier 2, external image data, not committed to the repo):
  * ``config.SAM2_CUTOUTS / "manifest.csv"``: one row per cutout with quality metrics.
  * ``config.SAM2_CUTOUTS / <png>``: the RGBA cutout PNGs referenced by the manifest.

Output: ``fig_species_morphology.png`` (preview) + ``.tiff`` (flattened, 600 dpi) via ``save_fig``.

  python -m salmonid.figures.fig_species_morphology

Environment: base env (CPU-only: cv2 / numpy / pandas / matplotlib).
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salmonid import config
from salmonid.figures.manuscript_style import set_style, save_fig, COL2_IN, MAXH_IN, AFS

set_style()
BASE = config.SAM2_CUTOUTS
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]   # class order
# One whole-body, head-right individual per species (chosen 2026-11-06 from a contact sheet of the
# cutouts that clear every frame edge, with sam_score >= 0.9 and length >= 380 px).
WHOLE_BODY = {
    "Chinook": "ls_dae863e6_036707_f000222",
    "Coho": "ls_f1917b35_073265_f000159",
    "Atlantic": "ls_c06d3b86_054857_f000141",
    "Rainbow Trout": "ls_d35eaea8_000629_f000069",
    "Brown Trout": "ls_5ed56d67_051874_f000144",
}
TARGET_LEN = 620          # px: every fish is scaled to this total length (snout to caudal fin)


def load_rgba(rel):
    """Read a cutout PNG as RGBA (adds an opaque alpha channel if the file has none)."""
    img = cv2.imread(str(BASE / rel), cv2.IMREAD_UNCHANGED)   # BGRA
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def crop_to_alpha(rgba, pad=6):
    """Crop to the silhouette's bounding box plus ``pad`` px."""
    ys, xs = np.where(rgba[:, :, 3] > 8)
    if len(xs) == 0:
        return rgba
    x0, x1 = max(0, xs.min() - pad), min(rgba.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(rgba.shape[0], ys.max() + pad + 1)
    return rgba[y0:y1, x0:x1]


def _rot(rgba, ang):
    """Rotate about the centre on a canvas large enough to hold any rotation, then re-crop."""
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
    """Composite the RGBA cutout over a white background."""
    rgb = rgba[:, :, :3].astype(float)
    a = (rgba[:, :, 3:4].astype(float)) / 255.0
    return (rgb * a + 255.0 * (1 - a)).astype(np.uint8)


def prep(row):
    """Load, deskew and scale one cutout to ``TARGET_LEN`` total length."""
    rgba = deskew_horiz(crop_to_alpha(load_rgba(row["png"])), float(row["angle_deg"]))
    h, w = rgba.shape[:2]
    return cv2.resize(rgba, (TARGET_LEN, max(1, int(round(h * TARGET_LEN / w)))), interpolation=cv2.INTER_AREA)


def median_ratio(manifest, species):
    """Species-level median depth:length over whole-body cutouts (mask clear of every frame edge)."""
    g = manifest[(manifest["species"] == species) & (manifest["solidity"] >= 0.7)
                 & (manifest["length_px"] >= 200) & (manifest["angle_deg"].abs() <= 30)
                 & (manifest["mbox_x1"] > 72) & (manifest["mbox_x2"] < 1255)
                 & (manifest["mbox_y1"] > 50) & (manifest["mbox_y2"] < 950)]
    if not len(g):
        g = manifest[manifest["species"] == species]
    return float((g["depth_px"] / g["length_px"]).median()), len(g)


def build(man, outname="fig_species_morphology"):
    """Two-column plate (18.2 cm wide): fish split between columns by stacked height, each labelled
    with its AFS common name only (dimensions belong in the caption)."""
    fish, labels = [], []
    for sp in SPECIES:
        row = man[man["cutout_id"] == WHOLE_BODY[sp]].iloc[0]
        img = prep(row)
        med, n = median_ratio(man, sp)
        fish.append(img)
        labels.append(AFS[sp])
        print(f"{sp}: {row['cutout_id']}  ({row['site']}, frame {row['frame_number']})  "
              f"median depth:length = {med:.2f}  n = {n}  sam = {row['sam_score']:.3f}")

    gap, top_pad, col_gap = 46, 40, 70
    heights = [f.shape[0] + gap for f in fish]
    split = min(range(1, len(fish)), key=lambda s: abs(sum(heights[:s]) - sum(heights[s:])))
    cols = [list(zip(fish[:split], labels[:split])), list(zip(fish[split:], labels[split:]))]
    col_w = TARGET_LEN
    canvas_w = 2 * col_w + col_gap
    col_h = [sum(f.shape[0] + gap for f, _ in c) for c in cols]
    canvas_h = top_pad + max(col_h)
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    placed = []
    for ci, c in enumerate(cols):
        x = ci * (col_w + col_gap)
        y = top_pad + (max(col_h) - col_h[ci]) // 2          # centre the shorter column
        for f, lab in c:
            h = f.shape[0]
            canvas[y:y + h, x:x + col_w] = on_white(f)
            placed.append((x, y, lab))
            y += h + gap
    fig_h = min(COL2_IN * canvas_h / canvas_w, MAXH_IN)
    fig = plt.figure(figsize=(COL2_IN, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])           # axes fill the canvas, so the output is exactly 2-column wide
    ax.imshow(canvas); ax.axis("off")
    for x, yb, lab in placed:
        ax.text(x + 6, yb - 6, lab, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    save_fig(fig, outname, kind="photo")      # photo montage -> PNG preview + flattened TIFF


def main():
    build(pd.read_csv(BASE / "manifest.csv"))


if __name__ == "__main__":
    main()

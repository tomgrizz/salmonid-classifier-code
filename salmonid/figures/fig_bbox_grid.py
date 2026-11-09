"""
In plain terms: this script builds one supporting picture for the paper (Figure S1): a grid
showing a single clear example of each of the five fish species, with a box drawn around the fish
(a "bounding box" — the outline the computer places on the fish it found) and the model's
confidence percentage. It is the paper's visual evidence that the automated system finds and
outlines fish in the right place.

Figure S1 (manuscript) — corrected-detection sample grid, one clean example per species.

Pipeline position: **results -> figures** (presentation only). This script does NOT run the
detector; it consumes detections that already exist in the SAM2 cutout manifest and renders a
publication figure. It is the source of the manuscript's "the detector localises fish correctly"
supporting claim (Figure S1) — it replaces the old figures/bbox_sample_grid.png, which was
rendered from the pre-2026-05-26 run (run_v17_ganaraska) and carries the bbox-y decode bug (boxes
drawn too high on the fish). Here we draw the boxes straight from the verified SAM2 cutout manifest,
whose detector boxes already passed the strict y-centre sanity gate (0.30-0.85), so the boxes sit
correctly. All FIVE species are shown, each labelled with the species and detector confidence.

Inputs
------
* ``config.SAM2_CUTOUTS / "manifest.csv"`` — the cutout-library manifest (one row per detection:
  box_x1..box_y2, species, det_conf, source video_id + frame_number, border/area geometry).
  External image data (Tier 2 — not committed to the repo).
* ``config.FOOTAGE_ROOT`` — the raw tunnel-camera footage the chosen source frames are read back
  from. Also external (Tier 2).

Output
------
* ``docs/manuscript_figures/bbox_sample_grid.{png,pdf/tiff}`` — manuscript Figure S1.

Environment: base Python env (cv2 + matplotlib + pandas). No detector / torch / .venv_deim needed.

  python -m salmonid.figures.fig_bbox_grid
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salmonid import config
from salmonid.figures.manuscript_style import set_style, save_fig, COL2_IN, AFS

set_style()
BASE = config.SAM2_CUTOUTS      # cutout-library root (holds manifest.csv)
FRAMES = config.FOOTAGE_ROOT    # raw footage read back for the source frames
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
COLORS = {  # per-species box colour, RGB
    "Chinook": (230, 90, 40), "Coho": (224, 161, 59), "Atlantic": (58, 167, 109),
    "Rainbow Trout": (47, 111, 159), "Brown Trout": (150, 92, 168),
}


def read_frame(video_id, frame_no):
    """Decode a single frame from a footage video and return it as RGB.

    Args:
        video_id: path (relative to ``config.FOOTAGE_ROOT``) of the source video.
        frame_no: 0-based frame index to seek to.
    Returns:
        HxWx3 uint8 RGB ndarray, or ``None`` if the file is missing or the frame can't be read.
    """
    path = FRAMES / video_id
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def annotate_and_crop(frame, r):
    """Draw the detection box, then context-crop around the fish.

    Args:
        frame: source RGB frame (HxWx3 uint8).
        r: a manifest row exposing ``box_x1..box_y2``, ``species`` and ``det_conf``.
    Returns:
        (crop, label): the RGB crop with a 0.55x-box-size context margin on each side, and the
        "<AFS name> (<detector confidence>%)" label. The label is drawn by matplotlib above the panel at >= 7 pt,
        not baked into the pixels (the baked cv2 label printed at about 2.5 pt at final size).
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = int(r.box_x1), int(r.box_y1), int(r.box_x2), int(r.box_y2)
    col = COLORS[r.species]
    bgr = tuple(int(c) for c in col[::-1])
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    th = max(3, int(round(W / 320)))          # >= 0.35 pt at final print size
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, th)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mx, my = int((x2 - x1) * 0.55), int((y2 - y1) * 0.55)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my - 12)
    cx2, cy2 = min(W, x2 + mx), min(H, y2 + my)
    return frame[cy1:cy2, cx1:cx2], f"{AFS[r.species]} ({r.det_conf * 100:.0f}%)"


def pick_panels(man):
    """Choose one clean, whole-fish example panel per species from the cutout manifest.

    Args:
        man: the manifest DataFrame (``config.SAM2_CUTOUTS / "manifest.csv"``).
    Returns:
        ``{species: (cropped RGB image, label)}`` for every species with a readable frame. Prefers big,
        well-exposed, elongated (aspect >= 2.2), high-confidence (>= 0.90), non-border detections
        so the fish is easiest to identify visually; falls back to a looser filter if none qualify.
    Side effects:
        Reads frames from footage and prints the chosen detection's stats (or a miss) per species.
    """
    man = man.copy()
    man["bw"] = man.box_x2 - man.box_x1
    man["bh"] = man.box_y2 - man.box_y1
    man["area_frac"] = (man.bw * man.bh) / (man.frame_w * man.frame_h)
    # aspect ~ full elongated fish; helps skip head-dominated / partial detections
    man["aspect"] = man.bw / man.bh.clip(lower=1)
    panels = {}
    for sp in SPECIES:
        # big, clear, well-exposed, elongated (whole fish) -> easiest to identify visually
        g = man[(man.species == sp) & (man.border_frac <= 0.03) & (man.det_conf >= 0.90)
                & (man.area_frac.between(0.05, 0.55)) & (man.aspect >= 2.2)]
        if not len(g):
            g = man[(man.species == sp) & (man.border_frac <= 0.05) & (man.det_conf >= 0.85)]
        g = g.sort_values("area_frac", ascending=False)
        for _, r in g.head(40).iterrows():
            frame = read_frame(r.video_id, r.frame_number)
            if frame is not None:
                panels[sp] = annotate_and_crop(frame, r)
                print(f"{sp}: conf={r.det_conf:.2f} site={r.site} area={r.area_frac:.2f} "
                      f"aspect={r.aspect:.1f} frame={r.frame_number}")
                break
        else:
            print("no readable frame for", sp)
    return panels


def main():
    """Build and save the 3-over-2 species grid (manuscript Figure S1).

    Reads the cutout manifest, picks one panel per species, lays them out in a balanced
    3-over-2 mosaic, and writes docs/manuscript_figures/bbox_sample_grid.* via
    :func:`salmonid.figures.manuscript_style.save_fig`.
    """
    man = pd.read_csv(BASE / "manifest.csv")
    panels = pick_panels(man)
    order = [s for s in SPECIES if s in panels]
    # balanced 3-over-2 layout: bottom row of two is centred under the top row of three
    slots = ["A", "B", "C", "D", "E"]
    layout = [["A", "A", "B", "B", "C", "C"],
              ["x", "D", "D", "E", "E", "x"]]
    # 2-column photo grid; the caption carries the title (kept out of the artwork). "compressed" layout
    # trims the whitespace between the fixed-aspect image panels.
    fig, axd = plt.subplot_mosaic(layout, figsize=(COL2_IN, COL2_IN * 0.50), empty_sentinel="x",
                                  layout="compressed")
    for k in slots:
        axd[k].axis("off")
    for k, sp in zip(slots, order):
        img, label = panels[sp]
        axd[k].imshow(img); axd[k].axis("off")
        axd[k].set_title(label, fontsize=7.5, loc="left", pad=3)
    save_fig(fig, "bbox_sample_grid", kind="photo")


if __name__ == "__main__":
    main()

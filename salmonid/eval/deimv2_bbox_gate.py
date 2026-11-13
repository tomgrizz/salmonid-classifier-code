"""In plain terms: the model draws a rectangle ("box") around each fish it finds in a video
frame. This script checks whether those boxes are in the right place. It runs the model on
150 frames from 2025 on which a person has drawn the boxes by hand, and compares the model's
boxes with the hand-drawn ones, separately for each river. It reports PASS only if, at both
rivers, the boxes overlap the hand-drawn ones closely enough and are not shifted up or down
the frame. It is a development check on box position. It says nothing about species, and it
is not a result reported in the manuscript.

Inputs
------
All are Tier-2 files that are not stored in git.

* ``--manifest``: the frozen box test set, ``bbox_test_set_v1/manifest.json`` under
  ``config.PROJECTS_ROOT``. It lists 150 frames from 2025 (112 Ganaraska River, 38 Credit
  River) with their hand-drawn boxes; 147 frames hold at least one fish, 168 boxes in all
  (126 at the Ganaraska River in 112 frames, 42 at the Credit River in 35 frames). The
  manifest marks the frames as never to be used for training.
* ``--config``: the model architecture file (default ``config.INFER_CONFIG``).
* ``--ckpt``: the trained weights (default ``config.PUBLISHED_CHECKPOINT``, the published
  model v2.7, epoch 17).
* The external DEIMv2 code and a GPU (see ``docs/DEIMV2_PIN.md``).

Outputs
-------
* Printed: one row per river with the number of hand-drawn boxes scored, the median overlap,
  the vertical-placement ratio and a verdict; then ``GATE: PASS`` or ``GATE: FAIL``.
* ``--overlay-dir`` (default ``output/deimv2_bbox_gate/``): the first ``--n-overlays`` frames
  with the hand-drawn boxes in green and the model's boxes in red, for a visual check.
* Exit code 0 for PASS, 1 for FAIL.

What it measures, and the thresholds
------------------------------------
For every hand-drawn box, the model's best-overlapping box in the same frame is found.

1. **Overlap (IoU).** IoU, "intersection over union", is the area shared by two boxes
   divided by the area covered by either: 1 means identical boxes, 0 means no overlap. The
   median IoU at each river must be at least ``--min-iou`` (default 0.80).
2. **Vertical placement (cy-ratio).** The centre height of the model's box divided by the
   centre height of the hand-drawn box. Its median at each river must be within
   ``--max-cy-dev`` (default 0.10) of 1. Image rows are numbered from the top, so a ratio
   below 1 means the model's boxes sit too high in the frame and above 1 too low. This test
   exists because an earlier pipeline once rescaled its boxes with the wrong frame size,
   which put every box at about 0.75 of its true height. A shift like that is a decoding
   error, not a loose box, and the overlap score alone does not tell the two apart.

A river fails if either test fails; the gate fails if either river fails, or if there is
nothing to score.

What it does NOT check
----------------------
* **Extra boxes.** Boxes drawn where there is no fish (false positives), or several boxes on
  one fish, are not penalised. Each hand-drawn box takes the model's best box, and one model
  box may serve several hand-drawn boxes.
* **Missed fish, beyond the median.** A missed fish scores an IoU of 0, but the test is on
  the median, so a model can miss a large minority of fish and still pass.
* **Species.** Only box position is scored.
* **Sample size.** There is no minimum number of boxes per river. The Credit River verdict
  rests on 42 boxes in 35 frames.
* **Other years.** All 150 frames are from 2025; the model is applied to 2020 to 2025.
* The vertical ratio is taken from the best-overlapping box even when that overlap is 0.

Result for the published model (v2.7, epoch 17): the gate FAILS
---------------------------------------------------------------
The published model does not pass this check. It passes at the Ganaraska River and fails
the overlap test at the Credit River:

    river        boxes   median IoU   cy-ratio   verdict
    Credit          42     0.737        1.010      FAIL (IoU below 0.80)
    Ganaraska      126     0.805        1.014      PASS
    GATE: FAIL (exit code 1)

Source: this script, run with the Run line below on 2026-11-10 against the published
weights. The two median IoU values are the ones recorded in the authors' v2.7 results note
of 2026-08 (0.737 and 0.805), which also records that all six v2.7 checkpoints that were
tried failed the overlap test at the Credit River (median IoU 0.710 to 0.737, epoch 17 the
highest) and that the vertical-placement test passed everywhere. The authors accepted the
result and proceeded.

How to read it: at the Credit River the model's boxes are looser than the hand-drawn ones
and did not meet the project's own development threshold for tightness. They are not
shifted (both cy-ratios are within 0.02 of 1). Quantities that depend on exact box edges,
for example body-size measurements (not part of the manuscript), would be affected. The
manuscript's species scores and counts use a box to say which fish is which and which way
its centre moved between the start and end of a track. This gate does not measure whether
looser boxes change those results; the operational comparison with the technician record
(manuscript Sections 3.2 to 3.6) is the evidence for the counts.

Behaviour changes relative to the original research script
-----------------------------------------------------------
The thresholds and the two measurements are unchanged.

* If no frame could be scored (the manifest lists frames whose image files are missing, for
  example because the data drive is not connected), the original printed ``GATE: PASS`` and
  exited 0. It now prints ``GATE: FAIL`` and exits 1. Frames that cannot be scored are
  counted and reported.
* A missing manifest now stops the run with one sentence naming the file.
* ``--config`` and ``--ckpt`` were required; they now default to the published model.
* Overlays are written under ``output/`` by default instead of ``figures/``; a relative
  ``--overlay-dir`` is taken relative to where the command is run.

``--mask-header-px`` keeps its original default of 0. The published model was trained with
the top 40 rows of each frame blacked out (they carry a burned-in date and river name), so
it should be checked with ``--mask-header-px 40`` as in the Run line below; a note is printed
if another value is used.

Feeds
-----
Development check; no manuscript number.

Run
---
In the DEIMv2 environment (``requirements-deim.txt``), with a GPU::

    python -m salmonid.eval.deimv2_bbox_gate --mask-header-px 40

Tier
----
2 (needs the model weights, the external DEIMv2 code, the frozen box test set and a GPU).

Status
------
Supporting check. This is the box-position check that applies to the published model.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from salmonid import config


def iou(a, b):
    """Overlap of two boxes [x1, y1, x2, y2]: shared area / area covered by either (0 to 1)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    """Score the model's boxes against the frozen box test set, print the table, and exit.

    Exits with code 0 on PASS and 1 on FAIL. Writes a few overlay pictures to ``--overlay-dir``.
    """
    ap = argparse.ArgumentParser(
        description="Box-position check for the single-stage (DEIMv2) model against the frozen "
                    "hand-boxed test frames. Development check; exit code 0 = PASS, 1 = FAIL.")
    ap.add_argument("--config", default=str(config.INFER_CONFIG),
                    help="Model architecture file (default: config.INFER_CONFIG).")
    ap.add_argument("--ckpt", default=str(config.PUBLISHED_CHECKPOINT),
                    help="Trained weights (default: the published model, config.PUBLISHED_CHECKPOINT).")
    ap.add_argument("--manifest",
                    default=str(config.PROJECTS_ROOT / "bbox_test_set_v1" / "manifest.json"),
                    help="Frozen box test set (default: <SALMONID_PROJECTS_ROOT>/bbox_test_set_v1/manifest.json).")
    ap.add_argument("--min-iou", type=float, default=0.80,
                    help="Lowest acceptable median IoU per river (default 0.80).")
    ap.add_argument("--max-cy-dev", type=float, default=0.10,
                    help="Largest acceptable distance of the median cy-ratio from 1 (default 0.10).")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Only model boxes scoring above this are considered (default 0.3).")
    ap.add_argument("--mask-header-px", type=int, default=0,
                    help="Black out the top N rows of each frame before detection. Use 40 for the "
                         "published model, which was trained with that mask (default 0).")
    ap.add_argument("--overlay-dir", default=str(config.OUTPUT_DIR / "deimv2_bbox_gate"),
                    help="Folder for the overlay pictures (default: output/deimv2_bbox_gate).")
    ap.add_argument("--n-overlays", type=int, default=8,
                    help="How many overlay pictures to write (default 8).")
    args = ap.parse_args()

    manifest = config.require(args.manifest,
                              "the frozen box test set (bbox_test_set_v1/manifest.json)",
                              "SALMONID_PROJECTS_ROOT")
    with open(manifest, encoding="utf-8") as f:
        listed = json.load(f).get("records", [])
    # Keep the frames that can be scored: the image exists and it has at least one hand-drawn
    # box. Frames without a fish are expected (nothing to locate). A missing image is not
    # expected and usually means the data drive is not connected, so it is counted and reported.
    n_missing = sum(1 for r in listed if r.get("boxes") and not Path(r["frame_path"]).is_file())
    n_nobox = sum(1 for r in listed if not r.get("boxes"))
    recs = [r for r in listed if Path(r["frame_path"]).is_file() and r.get("boxes")]
    print(f"gate samples: {len(recs)} of {len(listed)} listed frames "
          f"({n_nobox} have no fish to locate, {n_missing} image files not found)", flush=True)
    if n_missing:
        print(f"WARNING: {n_missing} listed frames are missing on disk, so the gate is running "
              f"on a subset of its test frames.", flush=True)
    if not recs:
        # Nothing to evaluate is a failure, never a pass.
        print("GATE: FAIL  (no frame could be scored: the manifest lists no frame whose image "
              "file exists)")
        raise SystemExit(1)
    if args.mask_header_px != config.HEADER_MASK_PX:
        print(f"NOTE: --mask-header-px is {args.mask_header_px}; the published model was trained "
              f"with {config.HEADER_MASK_PX}.", flush=True)

    # Resolve the overlay folder now, relative to where the command was run.
    odir = Path(args.overlay_dir).resolve()
    odir.mkdir(parents=True, exist_ok=True)

    # The detector (and with it torch and the external DEIMv2 code) is imported only here, so
    # that --help and the input checks above work in the base environment.
    from salmonid.detect import deimv2_infer as di
    det = di.DEIMv2Detector(args.config, args.ckpt, mask_header_px=args.mask_header_px)

    per = collections.defaultdict(lambda: {"iou": [], "cyr": []})
    n_saved = 0
    for r in recs:
        rgb = np.array(Image.open(r["frame_path"]).convert("RGB"))
        boxes, scores, labels = det.predict(rgb, args.thresh)
        st = r.get("site", "?")          # the stratum is the river
        pred = boxes.tolist()
        for gt in r["boxes"]:
            if not pred:
                per[st]["iou"].append(0.0); continue   # fish not detected: scored as IoU 0
            # Each hand-drawn box takes the model box that overlaps it most. Model boxes that
            # match nothing are ignored, so false positives are not counted by this gate.
            best = max(pred, key=lambda b: iou(gt, b))
            per[st]["iou"].append(iou(gt, best))
            # cy-ratio: centre height of the model's box / centre height of the hand-drawn
            # box. Rows are numbered from the top of the image, so below 1 = too high in the
            # frame, above 1 = too low. Near 1 = correct vertical placement.
            gcy, mcy = (gt[1]+gt[3])/2, (best[1]+best[3])/2
            if gcy > 0:
                per[st]["cyr"].append(mcy/gcy)
        if n_saved < args.n_overlays:
            im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
            for gt in r["boxes"]:
                dr.rectangle(list(gt), outline=(0, 255, 0), width=3)        # hand-drawn: green
            for k, b in enumerate(pred):
                dr.rectangle(list(b), outline=(255, 60, 60), width=2)        # model: red
                dr.text((b[0], max(0, b[1]-10)), f"{di.SPECIES[labels[k]]} {scores[k]:.2f}",
                        fill=(255, 255, 0))
            im.save(odir / f"gate_{n_saved:02d}_{st}.jpg", quality=85); n_saved += 1

    print(f"{'stratum':16s} {'n':>4s} {'medIoU':>7s} {'cy_ratio':>9s} {'verdict':>14s}")
    gate_pass = True
    for st in sorted(per):
        im = float(np.median(per[st]["iou"])) if per[st]["iou"] else 0.0
        # If no fish was detected at a river there is no ratio to compute; 1.0 is used so the
        # river fails on IoU (0.0) and is not also reported as shifted.
        cyr = float(np.median(per[st]["cyr"])) if per[st]["cyr"] else 1.0
        ok = im >= args.min_iou and abs(cyr - 1.0) <= args.max_cy_dev
        gate_pass &= ok
        # "FAIL iou" = boxes too loose or misplaced; "FAIL raise" = boxes shifted vertically.
        v = "PASS" if ok else ("FAIL iou" if im < args.min_iou else "FAIL raise")
        print(f"{st:16s} {len(per[st]['iou']):>4d} {im:>7.3f} {cyr:>9.3f} {v:>14s}")
    print(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}  (overlays -> {odir})")
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()

"""In plain terms: a detector draws a rectangle ("box") around each fish it finds in a video
frame. This script checks whether those boxes are in the right place. It runs a detector on
frames where a person has already drawn the boxes by hand, and compares the two sets of
boxes. It was written for the project's earlier, two-stage model (a detector followed by a
separate species classifier) and it can only load detectors of that older kind. It CANNOT
load the published model (v2.7); the published model was checked with the sister script
``salmonid.eval.deimv2_bbox_gate``, which applies the same test with the same thresholds.

Inputs
------
All are Tier-2 files that are not stored in git.

* ``--detector``: one or more folders holding a detector of the older kind (a Hugging Face
  ``AutoModelForObjectDetection`` checkpoint). None is distributed with this deposit.
* Hand-drawn boxes, from any of three sources:

  - ``--gt-testset``: the frozen box test set ``bbox_test_set_v1/manifest.json`` under
    ``config.PROJECTS_ROOT`` (150 frames from 2025, 112 Ganaraska River and 38 Credit River,
    147 of them with at least one fish, 168 boxes). Used when no source is named.
  - ``--cvat-json``: an annotation export in COCO format from the CVAT annotation tool
    (2021 to 2023 frames); up to ``--cvat-per-stratum`` frames per river and year are drawn
    with a fixed random seed, so the same frames are scored on every run.
  - ``--gt-2025``: pairs of (boxes file, frames folder) for hand-drawn 2025 frames.

Outputs
-------
Printed only; nothing is written to disk. One table per detector with a row per stratum
(river and year): number of hand-drawn boxes scored, median overlap, vertical-placement
ratio and a PASS/FAIL verdict; then a final ``GATE: PASS`` or ``GATE: FAIL`` line. The exit
code is 0 for PASS and 1 for FAIL.

What it measures, and the thresholds
------------------------------------
For every hand-drawn box, the detector's best-overlapping box in the same frame is found.

1. **Overlap (IoU).** IoU, "intersection over union", is the area shared by two boxes
   divided by the area covered by either: 1 means identical boxes, 0 means no overlap. The
   median IoU of a stratum must be at least ``--min-iou`` (default 0.80).
2. **Vertical placement (cy-ratio).** The centre height of the detector's box divided by
   the centre height of the hand-drawn box. Its median must be within ``--max-cy-dev``
   (default 0.10) of 1. This second test exists because of a specific past bug: until
   2026-05-26 the older pipeline rescaled its boxes with the wrong frame size, which placed
   every box too high in the frame (ratio about 0.75, since image rows are numbered from the
   top). A ratio below 1 means boxes sit too high; above 1, too low.

A stratum fails if either test fails. A detector fails if any stratum fails. The gate fails
if any detector fails, or if there is nothing to score (see "Behaviour changes").

What it does NOT check
----------------------
* **Extra boxes.** A detector that draws boxes where there is no fish (false positives), or
  several boxes on one fish, is not penalised. Each hand-drawn box simply takes the best
  detector box, and one detector box may serve several hand-drawn boxes.
* **Missed fish, beyond the median.** A missed fish scores an IoU of 0, but because the
  test is on the median, a detector can miss a large minority of fish and still pass.
* **Species.** Only box position is scored.
* **Sample size.** There is no minimum number of boxes per stratum; a stratum with one box
  can pass or fail on that box. Read the ``n`` column.
* **Other years.** The frozen test set holds 2025 frames only.
* The vertical ratio is taken from the best-overlapping box even when that overlap is 0.

Result for the published model
------------------------------
None from this script: it cannot load the published model. See
``salmonid.eval.deimv2_bbox_gate`` for the published model's result on the same test (it
passes at the Ganaraska River and FAILS the overlap test at the Credit River).

Behaviour changes relative to the original research script
-----------------------------------------------------------
The thresholds and the two measurements are unchanged. Three things were changed so that the
gate cannot pass by accident:

* If no frame can be scored (no source found, or every frame file missing), the original
  printed ``GATE: PASS`` and exited 0. It now prints ``GATE: FAIL`` and exits 1.
* Frames that are listed but cannot be scored (image file missing, or no box) were dropped
  silently. They are now counted and the counts are printed.
* A missing input file now stops the run with one sentence naming the file, instead of a
  traceback. If no box source is named, the frozen test set is used.

Feeds
-----
Development check; no manuscript number.

Run
---
::

    python -m salmonid.eval.bbox_regression_gate --detector <detector folder>
    python -m salmonid.eval.bbox_regression_gate --detector <detector folder> \\
        --cvat-json <annotations.json> --gt-2025 <gt_boxes_2025.json> <frames folder>

Needs ``torch`` and ``transformers`` (they are imported only when the gate actually runs, so
``--help`` works without them).

Tier
----
2 (needs a detector of the older kind and the hand-drawn box files; neither is in git).

Status
------
Legacy two-stage code kept for provenance. Not on the published path.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

from salmonid import config

# Category number -> species name in the CVAT export (numbered from 1). This gate scores box
# position only, never species, so the table is reference information and is not used below.
CAT = {1: "Chinook", 2: "Coho", 3: "Atlantic", 4: "Rainbow Trout", 5: "Brown Trout"}


def iou(a, b):
    """Overlap of two boxes: shared area divided by the area covered by either (0 to 1).

    Args:
        a, b: boxes as [x1, y1, x2, y2] in pixels (left, top, right, bottom).
    Returns:
        IoU between 0 and 1; 0.0 when the boxes do not overlap or have no area.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def site_year(vp):
    """Return the stratum label "<River> <Year>" read from a video path, or "other".

    The two rivers and each year are scored separately so that a problem at one river cannot
    be hidden by good results at the other. The pattern also accepts the folder layout
    "Ganaraska/Ganaraska 2024/...".
    """
    m = re.search(r'(Credit|Ganaraska)/(?:Ganaraska )?(\d{4})', vp or "")
    return f"{m.group(1)} {m.group(2)}" if m else "other"


def load_samples(args):
    """Collect the frames to score from every box source that was named.

    Args:
        args: parsed command-line options (``cvat_json``, ``cvat_per_stratum``, ``gt_2025``,
            ``gt_testset``).
    Returns:
        ``(samples, skipped)``. ``samples`` is a list of ``(image_path, [box, ...], stratum)``
        holding only frames whose image file exists and that carry at least one hand-drawn
        box. ``skipped`` counts, by reason, the listed frames that could not be scored.
        Frames without a fish are expected (there is nothing to locate in them); missing
        image files are not, and usually mean the data drive is not connected.
    """
    samples = []
    skipped = collections.Counter()
    if args.cvat_json:
        d = json.loads(Path(args.cvat_json).read_text(encoding="utf-8"))
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            # COCO stores a box as [left, top, width, height]; convert to [x1, y1, x2, y2].
            x, y, w, h = a["bbox"]; anns[a["image_id"]].append([x, y, x+w, y+h])
        # Draw up to --cvat-per-stratum frames for each river and year.
        bystr = collections.defaultdict(list)
        for im in d["images"]:
            bystr[site_year(im.get("video_path", ""))].append(im)
        import random; random.seed(0)  # fixed seed: the same frames are scored on every run
        for st, ims in bystr.items():
            for im in random.sample(ims, min(args.cvat_per_stratum, len(ims))):
                if Path(im["file_name"]).is_file() and im["id"] in anns:
                    samples.append((im["file_name"], anns[im["id"]], st))
                elif im["id"] in anns:
                    skipped["CVAT export: image file not found"] += 1
                else:
                    skipped["CVAT export: frame has no box"] += 1
    for pair in (args.gt_2025 or []):
        gtj, fdir = pair
        gt = json.loads(Path(gtj).read_text(encoding="utf-8"))
        man = {}
        mpath = Path(fdir).parent / "manifest.json"
        if mpath.is_file():
            man = {m["key"]: m for m in json.loads(mpath.read_text(encoding="utf-8"))}
        else:
            # Without this list the river of each frame is unknown and every frame of this
            # source is scored in one stratum called "2025 2025".
            print(f"NOTE: no manifest.json beside {fdir}; its frames are scored as one "
                  f"stratum, not per river.")
        for key, val in gt.items():
            p = Path(fdir) / key
            if not p.is_file():
                skipped["2025 hand-drawn boxes: image file not found"] += 1
                continue
            # A frame's entry is either a list of boxes or, in older files, a single box.
            if isinstance(val, list) and val and isinstance(val[0], list):
                gboxes = [[float(v) for v in b] for b in val]
            elif isinstance(val, list) and len(val) == 4 and not isinstance(val[0], list):
                gboxes = [[float(v) for v in val]]
            else:
                skipped["2025 hand-drawn boxes: frame has no box"] += 1
                continue  # a frame with no fish: nothing to locate
            site = man.get(key, {}).get("site", "")
            samples.append((str(p), gboxes, f"{site or '2025'} 2025"))
    if args.gt_testset:
        ts = json.loads(Path(args.gt_testset).read_text(encoding="utf-8"))
        for r in ts.get("records", []):
            fp = r.get("frame_path")
            boxes = r.get("boxes") or []
            if fp and Path(fp).is_file() and boxes:
                site = r.get("site", "?")
                samples.append((fp, [[float(v) for v in b] for b in boxes], f"{site} 2025"))
            elif boxes:
                skipped["frozen test set: image file not found"] += 1
            else:
                skipped["frozen test set: frame has no box"] += 1
    return samples, skipped


def main():
    """Read the options, score each detector on the hand-boxed frames, print, and exit 0 or 1."""
    ap = argparse.ArgumentParser(
        description="Box-placement check for detectors of the retired two-stage pipeline "
                    "(legacy; cannot load the published model, see deimv2_bbox_gate). "
                    "Exit code 0 = PASS, 1 = FAIL.")
    ap.add_argument("--detector", action="append", required=True,
                    help="Folder of a Hugging Face object-detection checkpoint. Repeatable.")
    ap.add_argument("--cvat-json", default=None,
                    help="CVAT annotation export in COCO format (2021-2023 frames).")
    ap.add_argument("--cvat-per-stratum", type=int, default=60,
                    help="Frames drawn per river and year from the CVAT export (default 60).")
    ap.add_argument("--gt-2025", nargs=2, action="append", metavar=("JSON", "FRAMES_DIR"), default=None,
                    help="Hand-drawn 2025 boxes file and its frames folder. Repeatable.")
    ap.add_argument("--gt-testset", default=None,
                    help="Frozen box test set manifest. Used by default when no other source is "
                         "named: <SALMONID_PROJECTS_ROOT>/bbox_test_set_v1/manifest.json.")
    ap.add_argument("--min-iou", type=float, default=0.80,
                    help="Lowest acceptable median IoU per stratum (default 0.80).")
    ap.add_argument("--max-cy-dev", type=float, default=0.10,
                    help="Largest acceptable distance of the median cy-ratio from 1 (default 0.10).")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Only detector boxes scoring above this are considered (default 0.3).")
    ap.add_argument("--max-square", action="store_true",
                    help="Diagnostic only: rescale boxes with a square target size, for a "
                         "detector whose image preparation pads frames to a square. Not the "
                         "production setting.")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: cuda if available).")
    args = ap.parse_args()

    # With no source named, use the frozen test set (the original passed with nothing to score).
    if not (args.cvat_json or args.gt_2025 or args.gt_testset):
        args.gt_testset = str(config.PROJECTS_ROOT / "bbox_test_set_v1" / "manifest.json")
    # Every external input is checked up front so a missing file gives one clear sentence.
    if args.gt_testset:
        config.require(args.gt_testset, "the frozen box test set (bbox_test_set_v1/manifest.json)",
                       "SALMONID_PROJECTS_ROOT")
    if args.cvat_json:
        config.require(args.cvat_json, "the CVAT annotation export (--cvat-json)")
    for gtj, fdir in (args.gt_2025 or []):
        config.require(gtj, "a hand-drawn 2025 boxes file (--gt-2025 JSON)")
        config.require(fdir, "the frames folder of a hand-drawn 2025 boxes file (--gt-2025 FRAMES_DIR)")
    for ckpt in args.detector:
        config.require(ckpt, "a two-stage detector checkpoint folder (--detector); none is "
                             "distributed with this deposit")

    samples, skipped = load_samples(args)
    n_boxes = collections.Counter()
    for _, gts, st in samples:
        n_boxes[st] += len(gts)
    print(f"Gate samples: {len(samples)} frames across strata {sorted(n_boxes)}")
    for st in sorted(n_boxes):
        print(f"  {st:18s} {n_boxes[st]:>4d} hand-drawn boxes")
    for reason, n in sorted(skipped.items()):
        print(f"  not scored: {n} frames ({reason})")
    if any("not found" in reason for reason in skipped):
        print("WARNING: some listed frames are missing on disk, so the gate is running on a "
              "subset of its test frames.")
    print()
    if not samples:
        # Nothing to evaluate is a failure, never a pass.
        print("GATE: FAIL  (no frame could be scored: check that the box files and the frame "
              "images they point to exist)")
        sys.exit(1)

    # The heavy libraries are imported here, not at the top, so that --help and the input
    # checks above work on a machine without them.
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    overall_pass = True
    for ckpt in args.detector:
        name = Path(ckpt).name
        proc = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(ckpt, local_files_only=True).to(device).eval()
        per = collections.defaultdict(lambda: {"iou": [], "cyr": []})
        for path, gts, st in samples:
            pil = Image.open(path).convert("RGB"); W, H = pil.width, pil.height
            inp = proc(images=pil, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            # The network works on a resized copy of the frame, so its boxes must be scaled
            # back to the original frame. Production scales by the true frame size, given as
            # [[height, width]]. --max-square scales by a square of the longer side instead (a
            # diagnostic for detectors whose image preparation pads frames to a square). Using
            # the wrong one of the two is the kind of error the cy-ratio test below detects:
            # on a 1280 x 960 frame the two differ by a factor of 0.75 in the vertical.
            ts = [[max(H, W), max(H, W)]] if args.max_square else [[H, W]]
            pp = proc.post_process_object_detection(
                out, threshold=args.thresh, target_sizes=torch.tensor(ts, device=device))[0]
            boxes = pp["boxes"].cpu().numpy().tolist()
            for gt in gts:
                if not boxes:
                    per[st]["iou"].append(0.0); continue  # fish not detected: scored as IoU 0
                # Each hand-drawn box takes the detector box that overlaps it most. Detector
                # boxes that match nothing are ignored (false positives are not counted).
                best = max(boxes, key=lambda b: iou(gt, b))
                per[st]["iou"].append(iou(gt, best))
                # cy-ratio: centre height of the detector box / centre height of the
                # hand-drawn box. Image rows are numbered from the top, so a ratio below 1
                # means the detector box sits too high and above 1 too low. This is checked
                # separately from IoU because a uniform vertical shift is a decoding bug, not
                # a sloppy box, and IoU alone does not say which one occurred.
                gcy = (gt[1]+gt[3])/2; mcy = (best[1]+best[3])/2
                if gcy > 0:
                    per[st]["cyr"].append(mcy/gcy)
        print(f"{'='*72}\nDETECTOR: {name}\n{'='*72}")
        print(f"{'stratum':18s} {'n':>4s} {'medIoU':>7s} {'cy_ratio':>9s} {'verdict':>20s}")
        det_pass = True
        for st in sorted(per):
            iou_med = float(np.median(per[st]["iou"])) if per[st]["iou"] else 0.0
            # If no fish was detected at all in a stratum there is no ratio to compute; 1.0 is
            # used so that the stratum fails on IoU (0.0) and is not also blamed for a shift.
            cyr = float(np.median(per[st]["cyr"])) if per[st]["cyr"] else 1.0
            ok_iou = iou_med >= args.min_iou
            ok_cy = abs(cyr - 1.0) <= args.max_cy_dev
            # "FAIL iou" = boxes too loose or misplaced; "FAIL raise" = boxes shifted vertically.
            v = "PASS" if (ok_iou and ok_cy) else ("FAIL iou" if not ok_iou else "") + ("" if ok_cy else " FAIL raise")
            if not (ok_iou and ok_cy): det_pass = False
            print(f"{st:18s} {len(per[st]['iou']):>4d} {iou_med:>7.3f} {cyr:>9.3f} {v:>20s}")
        verdict = "PASS" if det_pass else "FAIL"
        print(f"  -> {name}: {verdict}\n")
        overall_pass = overall_pass and det_pass
        del model
    print(f"GATE: {'PASS' if overall_pass else 'FAIL'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

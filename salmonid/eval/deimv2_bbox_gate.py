"""In plain terms: before a new fish-detector model is allowed to be used, this script checks
that it draws its boxes in the right place. (A "box" here is the rectangle the model draws around
each fish to say where it is in the video frame.) It compares the model's boxes against a fixed set
of boxes a person drew by hand, and it PASSES only if they overlap closely enough and the boxes are
not shifted too high or too low. This file does not produce a headline number or figure in the
paper itself — it is a safety check that guarantees the fish detections feeding the paper's counts
and species results are positioned correctly.

DEIMv2-specific bbox-regression gate (Ganaraska IoU / cy check) run before deploy.

Pipeline stage
--------------
``data -> detect -> track -> eval/count -> results -> figures`` — this is an **eval /
pre-deploy guard** that sits right after ``detect``. It enforces project hard-rule #2:
*every detector must PASS the bbox gate on the immutable ``bbox_test_set_v1`` before it is
used or deployed.* It replicates the verdict of ``salmonid.eval.bbox_regression_gate`` but for
the single-stage DEIMv2 model — the HuggingFace gate uses ``AutoModelForObjectDetection`` and
cannot load DEIMv2, so this wrapper drives the DEIMv2 detector instead.

Verdict
-------
PASS = for every site stratum, per-site **median IoU >= --min-iou** (default 0.80) AND the
box-centre-y ratio is within tolerance, **|cy_ratio - 1| <= --max-cy-dev** (default 0.10).
The cy_ratio guard catches the historical "y-raise" bbox decode bug: cy_ratio > 1 means the
predicted boxes sit systematically higher than ground truth. The process exit code carries the
verdict (0 = PASS, 1 = FAIL) so it can gate a deploy script.

Inputs / outputs
----------------
* INPUTS: the frozen ``bbox_test_set_v1`` manifest (frame paths + ground-truth boxes, per site),
  plus a DEIMv2 architecture config (``.yml``) and a checkpoint (``.pth``).
* OUTPUTS: a per-stratum table (n / median IoU / cy_ratio / verdict) printed to stdout, a small
  batch of GT-vs-prediction overlay JPEGs for a visual coordinate sanity check, and the PASS/FAIL
  exit code.

Environment
-----------
Runs in the **DEIMv2 ``.venv_deim``** (it imports ``salmonid.detect.deimv2_infer``, which loads
the DEIMv2 engine). ``deimv2_infer`` chdir's into the DEIMv2 repo, so the given ``--config`` /
``--ckpt`` are resolved relative to that repo, and overlay outputs are anchored to this repo's
root absolutely (see ``main``).

Example
-------
  .venv_deim/Scripts/python.exe -m salmonid.eval.deimv2_bbox_gate \\
      --config configs/deimv2/deimv2_fish_s.yml \\
      --ckpt outputs/deimv2_fish_s/best_stg1.pth --thresh 0.3
(config/ckpt paths are relative to the DEIMv2 repo; deimv2_infer chdir's there.)
"""
import argparse, json, collections
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from salmonid import config
from salmonid.detect import deimv2_infer as di


def iou(a, b):
    """Intersection-over-union of two xyxy boxes ``a``, ``b`` (returns 0.0 if disjoint)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    """Run the DEIMv2 bbox gate: score predictions against ``bbox_test_set_v1`` and exit.

    Parses CLI args, builds a ``DEIMv2Detector``, predicts boxes on every manifest frame,
    aggregates median IoU and box-centre-y ratio per site stratum, prints the verdict table,
    writes a handful of GT-vs-prediction overlay JPEGs, and raises ``SystemExit(0)`` on PASS
    or ``SystemExit(1)`` on FAIL. Side effects: creates the overlay directory and stdout output.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest",
                    default=str(config.PROJECTS_ROOT / "bbox_test_set_v1" / "manifest.json"))
    ap.add_argument("--min-iou", type=float, default=0.80)
    ap.add_argument("--max-cy-dev", type=float, default=0.10)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--mask-header-px", type=int, default=0,
                    help="Mask top N px before detection (match the model's training mask).")
    ap.add_argument("--overlay-dir", default="figures/deimv2_bbox_gate")
    ap.add_argument("--n-overlays", type=int, default=8)
    args = ap.parse_args()

    recs = json.load(open(args.manifest)).get("records", [])
    recs = [r for r in recs if Path(r["frame_path"]).is_file() and r.get("boxes")]
    print(f"gate samples: {len(recs)}", flush=True)

    det = di.DEIMv2Detector(args.config, args.ckpt, mask_header_px=args.mask_header_px)
    # deimv2_infer chdir's into the DEIMv2 repo, so a relative --overlay-dir would land there;
    # anchor it to this repo's root absolutely so overlays stay put next to the analysis code.
    odir = (Path(args.overlay_dir) if Path(args.overlay_dir).is_absolute()
            else config.REPO_ROOT / args.overlay_dir)
    odir.mkdir(parents=True, exist_ok=True)

    per = collections.defaultdict(lambda: {"iou": [], "cyr": []})
    n_saved = 0
    for r in recs:
        rgb = np.array(Image.open(r["frame_path"]).convert("RGB"))
        boxes, scores, labels = det.predict(rgb, args.thresh)
        st = r.get("site", "?")
        pred = boxes.tolist()
        for gt in r["boxes"]:
            if not pred:
                per[st]["iou"].append(0.0); continue
            # Greedy match: score each GT box against its highest-IoU prediction.
            best = max(pred, key=lambda b: iou(gt, b))
            per[st]["iou"].append(iou(gt, best))
            # cy_ratio = predicted box-centre-y / GT box-centre-y; ~1 means correct vertical
            # placement, >1 means the prediction is raised (the historical bbox-y decode bug).
            gcy, mcy = (gt[1]+gt[3])/2, (best[1]+best[3])/2
            if gcy > 0:
                per[st]["cyr"].append(mcy/gcy)
        if n_saved < args.n_overlays:
            im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
            for gt in r["boxes"]:
                dr.rectangle(list(gt), outline=(0, 255, 0), width=3)        # GT green
            for k, b in enumerate(pred):
                dr.rectangle(list(b), outline=(255, 60, 60), width=2)        # pred red
                dr.text((b[0], max(0, b[1]-10)), f"{di.SPECIES[labels[k]]} {scores[k]:.2f}",
                        fill=(255, 255, 0))
            im.save(odir / f"gate_{n_saved:02d}_{st}.jpg", quality=85); n_saved += 1

    print(f"{'stratum':16s} {'n':>4s} {'medIoU':>7s} {'cy_ratio':>9s} {'verdict':>14s}")
    gate_pass = True
    for st in sorted(per):
        im = float(np.median(per[st]["iou"])) if per[st]["iou"] else 0.0
        cyr = float(np.median(per[st]["cyr"])) if per[st]["cyr"] else 1.0
        ok = im >= args.min_iou and abs(cyr - 1.0) <= args.max_cy_dev
        gate_pass &= ok
        v = "PASS" if ok else ("FAIL iou" if im < args.min_iou else "FAIL raise")
        print(f"{st:16s} {len(per[st]['iou']):>4d} {im:>7.3f} {cyr:>9.3f} {v:>14s}")
    print(f"\nGATE: {'PASS' if gate_pass else 'FAIL'}  (overlays -> {args.overlay_dir})")
    raise SystemExit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()

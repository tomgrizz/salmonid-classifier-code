"""
In plain terms: this is an automatic safety check for the "fish finder" step -- the part of
the system that draws a box around each fish in a video frame before its species is named. It
runs a newly trained finder on frames a biologist has already hand-boxed and only lets it pass
if its boxes sit tightly on the real fish (and are not shifted up or down the frame); a finder
that fails is blocked from being used. It does not produce a headline number in the paper, but
every fish count, movement direction, and body-size measurement in the paper depends on this
check passing first.

Permanent, immutable bbox-regression GATE for the salmonid detector.

One-line role: median IoU >= 0.80 AND cy-ratio within +-0.10 vs human GT boxes; exit 0/1.

Pipeline position
-----------------
    data -> **detect [THIS = gate the detector]** -> track -> eval/count -> results -> figures

This is the go/no-go check that every newly trained *detector* must pass before it is
trusted for inference or deploy (CLAUDE.md hard rule #2 -- the one permanent bbox guard).
It does not itself emit a manuscript number; it protects the bbox-correctness assumption
that every downstream count, movement, and morphometric figure rests on. In particular it
is the standing guard against the pre-2026-05-26 y-centre "raise" bug, where boxes decoded
with the wrong target size landed ~0.75x too high on the frame.

What it checks
--------------
Runs each detector against human ground-truth boxes (CVAT 2021-23 + hand-drawn 2025, and/or
the frozen bbox_test_set_v1 manifest), using the PRODUCTION decode convention
``post_process_object_detection(target_sizes=[[H, W]])``, and PASSES/FAILS on two axes per
stratum (site x era):
  (1) localization : median IoU >= --min-iou   (default 0.80)
  (2) no y-raise   : |median(model_cy / gt_cy) - 1| <= --max-cy-dev  (default 0.10; the 0.75 raise)
Any single stratum failing either axis fails that detector; any detector failing fails the gate.

Inputs
------
* One or more detector checkpoints (Hugging Face ``AutoModelForObjectDetection`` dirs).
* Human GT boxes from up to three sources: a CVAT COCO json (``--cvat-json``), one or more
  hand-drawn 2025 (json, frames-dir) pairs (``--gt-2025``), and/or the immutable
  bbox_test_set_v1 manifest (``--gt-testset``).

Outputs
-------
A per-stratum table (n, median IoU, cy-ratio, verdict) printed to stdout, an overall
GATE PASS/FAIL line, and process exit code 0 = PASS / 1 = FAIL (usable directly as a CI gate).

Environment
-----------
Base env (transformers + torch). This gate scores the Hugging Face two-stage RT-DETR-family
detectors via ``AutoModelForObjectDetection``; it is NOT the single-stage DEIMv2 model, which
has its own decode surface (``orig_target_sizes``) and runs in the isolated ``.venv_deim``.

Usage (GT paths shown relative to config.PROJECTS_ROOT; set SALMONID_PROJECTS_ROOT to point at them):
  python -m salmonid.eval.bbox_regression_gate --detector model/<new_detector> \
      --gt-testset  <PROJECTS_ROOT>/bbox_test_set_v1/manifest.json        # the immutable anchor
  # or from explicit CVAT + hand-drawn GT sources (each --gt-2025 takes JSON then its FRAMES_DIR):
  python -m salmonid.eval.bbox_regression_gate --detector model/<new_detector> \
      --cvat-json  <PROJECTS_ROOT>/cvat_dataset/annotations.json \
      --gt-2025    <PROJECTS_ROOT>/bbox_gt_2025_v2/gt_boxes_2025.json  <PROJECTS_ROOT>/bbox_gt_2025_v2/frames \
      [--gt-2025 ...additional JSON FRAMES_DIR pairs...]
"""
from __future__ import annotations
import argparse, json, re, sys, collections
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# 1-indexed CVAT/COCO category id -> species name (this gate only scores localisation,
# not class, so CAT is reference metadata; species order mirrors salmonid.config.SPECIES).
CAT = {1: "Chinook", 2: "Coho", 3: "Atlantic", 4: "Rainbow Trout", 5: "Brown Trout"}


def iou(a, b):
    """Intersection-over-union of two axis-aligned boxes in xyxy pixel coords.

    Args:
        a, b: [x1, y1, x2, y2].
    Returns:
        float IoU in [0, 1]; 0.0 when the boxes do not overlap or the union is degenerate.
    """
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def site_year(vp):
    """Derive the "<Site> <Year>" stratum label from a video path, else "other".

    Strata keep Credit and Ganaraska separated by era so a per-site y-raise cannot be
    masked by pooling; the regex also tolerates the "Ganaraska Ganaraska/2024/..." layout.
    """
    m = re.search(r'(Credit|Ganaraska)/(?:Ganaraska )?(\d{4})', vp or "")
    return f"{m.group(1)} {m.group(2)}" if m else "other"


def load_samples(args):
    """Collect scored GT frames from every configured source.

    Args:
        args: parsed CLI namespace (uses ``cvat_json``, ``cvat_per_stratum``, ``gt_2025``,
            ``gt_testset``).
    Returns:
        list of ``(image_path, [gt_xyxy, ...], stratum)`` -- only frames that exist on disk
        and carry at least one GT box; no-fish frames are skipped (nothing to localise).
    Side effects:
        Reads the CVAT/GT json files (and their manifests) and stats candidate frame paths.
    """
    samples = []
    if args.cvat_json:
        d = json.loads(Path(args.cvat_json).read_text())
        anns = collections.defaultdict(list)
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]; anns[a["image_id"]].append([x, y, x+w, y+h])
        # sample up to --cvat-per-stratum per site-year
        bystr = collections.defaultdict(list)
        for im in d["images"]:
            bystr[site_year(im.get("video_path", ""))].append(im)
        import random; random.seed(0)  # fixed seed => the gate scores the same frames every run
        for st, ims in bystr.items():
            for im in random.sample(ims, min(args.cvat_per_stratum, len(ims))):
                if Path(im["file_name"]).is_file() and im["id"] in anns:
                    samples.append((im["file_name"], anns[im["id"]], st))
    for pair in (args.gt_2025 or []):
        gtj, fdir = pair
        gt = json.loads(Path(gtj).read_text())
        man = {}
        mpath = Path(fdir).parent / "manifest.json"
        if mpath.is_file():
            man = {m["key"]: m for m in json.loads(mpath.read_text())}
        for key, val in gt.items():
            p = Path(fdir) / key
            if not p.is_file():
                continue
            # Normalize to list-of-boxes: list-of-lists OR legacy single [x1,y1,x2,y2].
            if isinstance(val, list) and val and isinstance(val[0], list):
                gboxes = [[float(v) for v in b] for b in val]
            elif isinstance(val, list) and len(val) == 4 and not isinstance(val[0], list):
                gboxes = [[float(v) for v in val]]
            else:
                continue  # empty (no-fish frame) — no GT box to score
            site = man.get(key, {}).get("site", "")
            samples.append((str(p), gboxes, f"{site or '2025'} 2025"))
    if args.gt_testset:
        ts = json.loads(Path(args.gt_testset).read_text())
        for r in ts.get("records", []):
            fp = r.get("frame_path")
            boxes = r.get("boxes") or []
            if fp and Path(fp).is_file() and boxes:
                site = r.get("site", "?")
                samples.append((fp, [[float(v) for v in b] for b in boxes], f"{site} 2025"))
    return samples


def main():
    """Parse args, load GT frames, score each detector, print the table, and exit 0/1.

    For every ``--detector`` checkpoint: load its HF processor + model, run each GT frame,
    match every GT box to its best-IoU prediction, and aggregate median IoU + cy-ratio per
    stratum. Prints a per-detector table and a final ``GATE: PASS/FAIL``; calls ``sys.exit``
    with 0 (all detectors pass every stratum) or 1 (any failure).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", action="append", required=True)
    ap.add_argument("--cvat-json", default=None)
    ap.add_argument("--cvat-per-stratum", type=int, default=60)
    ap.add_argument("--gt-2025", nargs=2, action="append", metavar=("JSON", "FRAMES_DIR"), default=None)
    ap.add_argument("--gt-testset", default=None,
                    help="bbox_test_set_v1/manifest.json (records with frame_path+boxes+site).")
    ap.add_argument("--min-iou", type=float, default=0.80)    # localisation floor (rule #2)
    ap.add_argument("--max-cy-dev", type=float, default=0.10)  # max |cy-ratio - 1| before FAIL
    ap.add_argument("--thresh", type=float, default=0.3)       # detector score threshold to keep a box
    ap.add_argument("--max-square", action="store_true",
                    help="DIAGNOSTIC: decode with target_sizes=[[max,max]] instead of [[H,W]] "
                         "(for models trained with a square-padding processor, e.g. mar10-based).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = load_samples(args)
    print(f"Gate samples: {len(samples)} across strata "
          f"{sorted(set(s[2] for s in samples))}\n")
    overall_pass = True
    for ckpt in args.detector:
        name = Path(ckpt).name
        proc = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
        model = AutoModelForObjectDetection.from_pretrained(ckpt, local_files_only=True).to(args.device).eval()
        per = collections.defaultdict(lambda: {"iou": [], "cyr": []})
        for path, gts, st in samples:
            pil = Image.open(path).convert("RGB"); W, H = pil.width, pil.height
            inp = proc(images=pil, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = model(**inp)
            # DECODE CONVENTION: production post-process uses target_sizes=[[H, W]] (true
            # frame size). [[max, max]] (--max-square) is the DIAGNOSTIC square-pad decode; the
            # pre-2026-05-26 bug was decoding with the wrong size, which raised cy ~0.75x.
            ts = [[max(H, W), max(H, W)]] if args.max_square else [[H, W]]
            pp = proc.post_process_object_detection(
                out, threshold=args.thresh, target_sizes=torch.tensor(ts, device=args.device))[0]
            boxes = pp["boxes"].cpu().numpy().tolist()
            for gt in gts:
                if not boxes:
                    per[st]["iou"].append(0.0); continue  # no detection => worst-case IoU
                best = max(boxes, key=lambda b: iou(gt, b))  # greedy best-IoU match for this GT box
                per[st]["iou"].append(iou(gt, best))
                # cy-ratio = model box centre-y / GT centre-y; its per-stratum median deviating
                # from 1.0 is the vertical-placement (y-raise) signal, independent of IoU.
                gcy = (gt[1]+gt[3])/2; mcy = (best[1]+best[3])/2
                if gcy > 0:
                    per[st]["cyr"].append(mcy/gcy)
        print(f"{'='*72}\nDETECTOR: {name}\n{'='*72}")
        print(f"{'stratum':18s} {'n':>4s} {'medIoU':>7s} {'cy_ratio':>9s} {'verdict':>20s}")
        det_pass = True
        for st in sorted(per):
            iou_med = float(np.median(per[st]["iou"])) if per[st]["iou"] else 0.0
            cyr = float(np.median(per[st]["cyr"])) if per[st]["cyr"] else 1.0
            ok_iou = iou_med >= args.min_iou
            ok_cy = abs(cyr - 1.0) <= args.max_cy_dev
            # Verdict text names which axis failed: "FAIL iou" (localisation) and/or "FAIL raise"
            # (cy deviation). An empty cyr list defaults cyr=1.0 above, i.e. no-raise / ok_cy.
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

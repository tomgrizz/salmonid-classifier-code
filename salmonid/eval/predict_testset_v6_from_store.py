"""In plain terms: this file assigns a single species to each fish in a fixed set of test
examples. Each fish has already been followed across the video frames (a "track"), and the
detector has already saved its species guesses for every frame; this step averages those
per-frame guesses into one species per fish. Those species assignments are then graded by a
companion script to produce the paper's headline held-out accuracy for the published counter
(0.966 species agreement across the five salmon and trout species).

Per-track species predictions for the held-out test_set_v6 eval, from a DEIMv2 detect-store.

Pipeline position: **eval** stage (data -> detect -> [track] -> eval -> results). It turns an
already-computed DEIMv2 detect-store (per-frame boxes + full class-probability distributions from
the published single-stage v2.6 detector) into the per-track species predictions that
``salmonid/eval/score_testset_v6.py`` scores.

Method (no re-tracking): every frozen GT track is DEFINED by its detector boxes in the
``run_20260428`` store. For each such box we look up the SAME frame in the v2.6 detect-store,
IoU-match it (argmax IoU >= --iou-thresh) to the model's detection there, and collect that
detection's full class distribution. The matched distributions are averaged and argmax'd into one
species per track. Because the GT track identity is carried over box-for-box (never re-derived by a
tracker), predictions key cleanly to the GT ``(video_id, track_id)``.

Inputs
  --store-dir       detect-store dir, one pickle per video: {frames: {frame_no: {boxes, probs}}}.
  --gt-file         frozen test_set_v6 GT (tracks list; default config.LABEL_BATCHES/test_set_v6_gt.json).
  --detections-dir  run_20260428 per-frame parquets that supply each GT track's defining boxes.
Outputs
  --out             CSV rows video_id, track_id, pred_species (one per predicted track).
  --video-out       optional CSV rows video_id, species (the set of species predicted per video).

Environment: base Python 3.11 (numpy + pyarrow only; no torch/boxmot — the store is precomputed).

Manuscript claim: produces the predictions behind the held-out species macro-F1 (0.966) reported
for the published single-stage v2.6 counter (scored downstream by score_testset_v6.py).

Example
  python -m salmonid.eval.predict_testset_v6_from_store \
      --store-dir <detect-store dir> --out <track_preds.csv> --video-out <video_preds.csv>
  python -m salmonid.eval.score_testset_v6 --preds <track_preds.csv> --video-preds <video_preds.csv>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from salmonid import config
from salmonid.util.schema import video_id_to_parquet_stem

SP = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]   # mirrors salmonid.config.SPECIES


def store_key(site: str, rel: str) -> str:
    """Matches scripts/_deimv2_detect_store.store_key (inlined to avoid importing the DEIMv2 engine)."""
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def _iou(box, boxes):
    """IoU of one xyxy box vs an Nx4 array."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + b - inter, 1e-6, None)


def _track_boxes(video_id, track_id, det_dirs):
    """Return the run_20260428 per-frame boxes for one GT track, or None if not found.

    Reads ``<stem>.parquet`` under each dir in ``det_dirs`` (first hit wins), filtering to the given
    ``track_id`` at the Arrow layer; returns a pandas DataFrame of that track's box rows.
    """
    stem = video_id_to_parquet_stem(video_id)
    for d in det_dirs:
        pf = Path(d) / f"{stem}.parquet"
        if not pf.is_file():
            continue
        try:
            df = pq.read_table(str(pf), filters=[("track_id", "=", int(track_id))]).to_pandas()
        except Exception:
            continue
        if len(df):
            return df
    return None


def main() -> None:
    """Match every GT track to the detect-store, vote a species, and write the preds CSV(s).

    Side effects: writes ``--out`` (per-track predictions) and, if given, ``--video-out`` (per-video
    species sets); prints a coverage summary (predicted / missing-store / missing-track / no-vote).
    """
    a = _parse_args()
    gt = json.loads(Path(a.gt_file).read_text())["tracks"]
    det_dirs = [a.detections_dir]
    store_dir = Path(a.store_dir)

    # cache one store pickle per video
    store_cache: dict = {}
    def load_store(site, short):
        key = store_key(site, short)
        if key not in store_cache:
            p = store_dir / key
            store_cache[key] = pickle.load(open(p, "rb")).get("frames", {}) if p.is_file() else None
        return store_cache[key]

    rows = []                      # (video_id, track_id, pred_species)
    video_preds = defaultdict(set)
    n_ok = n_nostore = n_notrack = n_novote = 0
    by_tier_ok = defaultdict(int)
    for t in gt:
        vid, tid, tier = t["video_id"], int(t["track_id"]), t.get("tier", "?")
        site = vid.split("/", 1)[0]; short = "/".join(vid.split("/")[-2:])
        frames = load_store(site, short)
        if frames is None:
            n_nostore += 1; continue
        df = _track_boxes(vid, tid, det_dirs)
        if df is None:
            n_notrack += 1; continue
        votes = []
        for _, r in df.iterrows():
            f = int(r["frame_number"])
            fr = frames.get(f)
            if not fr or len(fr["boxes"]) == 0:
                continue
            gtb = np.array([float(r["bbox_x1"]), float(r["bbox_y1"]),
                            float(r["bbox_x2"]), float(r["bbox_y2"])])
            ious = _iou(gtb, np.asarray(fr["boxes"], dtype=np.float32))
            j = int(np.argmax(ious))               # best-overlapping model detection in this frame
            if ious[j] >= a.iou_thresh:
                # [:5] drops class 5 ("Small Salmonid") so the vote is over the 5 report species only.
                votes.append(np.asarray(fr["probs"][j], dtype=np.float32)[:5])
        if not votes:
            n_novote += 1; continue
        # Mean of the per-frame class distributions, then argmax = the track's voted species.
        sp = SP[int(np.argmax(np.mean(votes, axis=0)))]
        rows.append({"video_id": vid, "track_id": tid, "pred_species": sp})
        video_preds[vid].add(sp)
        n_ok += 1; by_tier_ok[tier] += 1

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "track_id", "pred_species"]); w.writeheader(); w.writerows(rows)
    if a.video_out:
        with open(a.video_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["video_id", "species"])
            for v, sps in video_preds.items():
                for s in sps:
                    w.writerow([v, s])
    print(f"predicted {n_ok}/{len(gt)} tracks (by tier {dict(by_tier_ok)}); "
          f"missing: store {n_nostore}, track {n_notrack}, no-vote {n_novote}")
    print(f"track preds -> {a.out}" + (f"  |  video preds -> {a.video_out}" if a.video_out else ""))


def _parse_args():
    """Parse CLI args (path defaults resolve to the original workstation locations via config)."""
    p = argparse.ArgumentParser(description="test_set_v6 per-track predictions from a DEIMv2 detect-store")
    p.add_argument("--store-dir", required=True)
    p.add_argument("--gt-file", default=str(config.LABEL_BATCHES / "test_set_v6_gt.json"))
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"))
    p.add_argument("--iou-thresh", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.add_argument("--video-out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()

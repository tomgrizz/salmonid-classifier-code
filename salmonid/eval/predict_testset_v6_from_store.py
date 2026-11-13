"""In plain terms: this script gives each of the 366 test fish one species name according to
the model, so that a second script can mark those names against the reviewers' answers. Each
test fish is a track: a sequence of boxes, one per video frame, drawn around that fish in an
earlier analysis and then frozen. For every frame of a track, the script looks at what the
published model detected in the same frame, picks the model's box that overlaps the frozen
box best, and notes the model's species scores for it. It then averages those scores over the
track's frames and names the species with the highest average. The test fish themselves and
their boxes never change; only the model's opinion of them is collected here.

Per-track species predictions for the held-out test set, from stored detections.

Pipeline stage: data -> detect -> **eval (this module)** -> ``salmonid.eval.score_testset_v6``.

Inputs : * ``--store-dir``: detections of the published model on the 364 test videos, written
           by ``salmonid.detect.detect_store`` (default ``config.EVAL_DIR/store_0017``; every
           frame examined, full precision, ``--key-mode rel``);
         * ``--gt-file``: the frozen list of 366 test tracks (``data/test_set_v6_gt.json``);
         * ``--detections-dir``: the per-frame box files of the earlier analysis run that
           defines each test track's boxes (``<projects>/model_outputs/run_20260428/detections``).
Outputs: ``--out``, a CSV with ``video_id, track_id, pred_species`` (one row per test track);
         optional ``--video-out`` (species named per video) and ``--support-out`` (how many
         frames each prediction rests on).
Feeds  : ``data/preds_v27_ckpt0017.csv``, which ``score_testset_v6`` turns into Table 1 and
         Figure 2 (macro-F1 0.961; 352 of 366 correct) and Section 3.1.
Run    : python -m salmonid.eval.predict_testset_v6_from_store
             (writes output/preds_v27_ckpt0017.csv)
         python -m salmonid.eval.score_testset_v6 --preds output/preds_v27_ckpt0017.csv
Tier   : 2 (needs the stored test-video detections and the reference box files; base Python
         with numpy and pyarrow; no GPU, about a minute).

How the published numbers were computed
---------------------------------------
* **The test tracks come from an earlier tracking run, not from the published model.** Each
  test track is the box sequence that an earlier detector and tracker produced
  (run "run_20260428") and that the reviewers labelled. The published model is asked about
  those same boxes; it does not re-detect "its own" fish or re-track anything.
* **Matching is by overlap, frame by frame.** In each frame of the track, the model's
  detection with the largest overlap with the frozen box is used, provided the overlap
  (intersection over union) is at least 0.3. Frames in which the model detected nothing, or
  nothing overlapping enough, contribute nothing.
* **The species is the highest average score.** The five species scores of the matched
  detections are averaged over the track's matched frames and the largest wins. The
  detector's sixth score ("Small Salmonid") is ignored, so every track receives one of the
  five species.
* **One matched frame is enough.** There is no minimum number of frames. ``--support-out``
  records how many frames each prediction used, so this can be reported.
* **A track with no matched frame gets no prediction** and is left out of the output file.
  The scoring script then scores fewer than 366 tracks. For the published model this did not
  happen: all 366 tracks received a prediction. This script now says so explicitly and
  returns an error code if any track is left without a prediction.
* **Video lookup uses the older "rel" naming** (river + last folder + file name), because
  the test-video store was written that way. The 364 test videos have 364 distinct names
  under this scheme.

Which research-repository version this is: the script is identical on both development
branches of the research repository. Additions for the deposit: default paths from
``salmonid.config``, a count of unreadable reference files instead of silent skipping, the
optional ``--support-out`` table, and the error code described above. The prediction logic
and the output file are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from salmonid import config
from salmonid.util.schema import video_id_to_parquet_stem

SP = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]   # same order as salmonid.config.SPECIES

_N_BAD_REFERENCE = 0   # reference box files that exist but could not be read (reported at the end)


def store_key(site: str, rel: str) -> str:
    """Store file name in "rel" naming; must match ``salmonid.detect.detect_store.store_key``.

    Repeated here so that this script does not have to import the detection code.
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def _iou(box, boxes):
    """Overlap (intersection over union) of one box with each of several boxes."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + b - inter, 1e-6, None)


def _track_boxes(video_id, track_id, det_dirs):
    """The frozen boxes of one test track (one row per frame), or None if they are not found.

    Each video has one box file in the reference run; the rows of the wanted track are
    selected from it. A file that exists but cannot be read is counted and treated as not
    found (the original script skipped it without a word).
    """
    global _N_BAD_REFERENCE
    stem = video_id_to_parquet_stem(video_id)
    for d in det_dirs:
        pf = Path(d) / f"{stem}.parquet"
        if not pf.is_file():
            continue
        try:
            df = pq.read_table(str(pf), filters=[("track_id", "=", int(track_id))]).to_pandas()
        except Exception as e:                                  # noqa: BLE001
            _N_BAD_REFERENCE += 1
            print(f"WARNING: unreadable reference box file {pf.name}: {type(e).__name__}")
            continue
        if len(df):
            return df
    return None


def main() -> None:
    """Collect the model's species scores for every test track and write the predictions."""
    a = _parse_args()
    config.require(a.gt_file, "the frozen test-track list (--gt-file)")
    config.require(a.store_dir, "the stored detections of the test videos (--store-dir), written by "
                                "salmonid.detect.detect_store with --key-mode rel", "SALMONID_PROJECTS_ROOT")
    config.require(a.detections_dir, "the reference box files that define the test tracks "
                                     "(--detections-dir, run_20260428)", "SALMONID_PROJECTS_ROOT")
    gt = json.loads(Path(a.gt_file).read_text())["tracks"]
    det_dirs = [a.detections_dir]
    store_dir = Path(a.store_dir)

    # Read each video's store file once, however many test tracks it holds.
    store_cache: dict = {}
    def load_store(site, short):
        key = store_key(site, short)
        if key not in store_cache:
            p = store_dir / key
            if p.is_file():
                with open(p, "rb") as fh:
                    store_cache[key] = pickle.load(fh).get("frames", {})
            else:
                store_cache[key] = None
        return store_cache[key]

    rows = []                      # (video_id, track_id, pred_species)
    support = []                   # per-track evidence, for --support-out
    video_preds = defaultdict(set)
    n_ok = n_nostore = n_notrack = n_novote = 0
    by_tier_ok = defaultdict(int)
    for t in gt:
        vid, tid, tier = t["video_id"], int(t["track_id"]), t.get("tier", "?")
        # "rel" name of the video: its river, and its last folder + file name.
        site = vid.split("/", 1)[0]; short = "/".join(vid.split("/")[-2:])
        frames = load_store(site, short)
        if frames is None:
            n_nostore += 1; continue          # no stored detections for this video
        df = _track_boxes(vid, tid, det_dirs)
        if df is None:
            n_notrack += 1; continue          # the track's frozen boxes were not found
        votes = []
        for _, r in df.iterrows():
            f = int(r["frame_number"])
            fr = frames.get(f)
            if not fr or len(fr["boxes"]) == 0:
                continue                      # the model detected nothing in this frame
            gtb = np.array([float(r["bbox_x1"]), float(r["bbox_y1"]),
                            float(r["bbox_x2"]), float(r["bbox_y2"])])
            ious = _iou(gtb, np.asarray(fr["boxes"], dtype=np.float32))
            j = int(np.argmax(ious))               # the model's box that overlaps the frozen box most
            if ious[j] >= a.iou_thresh:
                # [:5] keeps the five species scores and drops "Small Salmonid".
                votes.append(np.asarray(fr["probs"][j], dtype=np.float32)[:5])
        if not votes:
            n_novote += 1
            support.append({"video_id": vid, "track_id": tid, "n_reference_frames": len(df),
                            "n_matched_frames": 0, "mean_top_score": "", "pred_species": ""})
            continue
        # Average the species scores over the matched frames; the highest average is the prediction.
        mean_scores = np.mean(votes, axis=0)
        sp = SP[int(np.argmax(mean_scores))]
        rows.append({"video_id": vid, "track_id": tid, "pred_species": sp})
        support.append({"video_id": vid, "track_id": tid, "n_reference_frames": len(df),
                        "n_matched_frames": len(votes),
                        "mean_top_score": round(float(mean_scores.max()), 4), "pred_species": sp})
        video_preds[vid].add(sp)
        n_ok += 1; by_tier_ok[tier] += 1

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "track_id", "pred_species"]); w.writeheader(); w.writerows(rows)
    if a.video_out:
        Path(a.video_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.video_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["video_id", "species"])
            for v, sps in video_preds.items():
                for s in sps:
                    w.writerow([v, s])
    if a.support_out:
        Path(a.support_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.support_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "track_id", "n_reference_frames",
                                              "n_matched_frames", "mean_top_score", "pred_species"])
            w.writeheader(); w.writerows(support)
        m = sorted(s["n_matched_frames"] for s in support if s["n_matched_frames"])
        if m:
            print(f"frames behind each prediction: median {m[len(m) // 2]}, min {m[0]}, max {m[-1]} "
                  f"-> {a.support_out}")
    print(f"predicted {n_ok}/{len(gt)} tracks (by tier {dict(by_tier_ok)}); "
          f"missing: store {n_nostore}, track {n_notrack}, no-vote {n_novote}")
    print(f"track preds -> {a.out}" + (f"  |  video preds -> {a.video_out}" if a.video_out else ""))
    if _N_BAD_REFERENCE:
        print(f"WARNING: {_N_BAD_REFERENCE} reference box file(s) could not be read.")
    if n_ok != len(gt) and not a.allow_missing:
        # A track without a prediction is simply absent from the output, and the scoring
        # script would then quietly score fewer than all test tracks.
        print(f"ERROR: {len(gt) - n_ok} test track(s) received no prediction, so a score computed from "
              f"{a.out} would not cover the whole test set. Pass --allow-missing to accept this.")
        sys.exit(1)


def _parse_args():
    """Read the command-line options (defaults come from ``salmonid.config``)."""
    p = argparse.ArgumentParser(description="Species prediction for each held-out test track, from stored detections.")
    p.add_argument("--store-dir", default=str(config.EVAL_DIR / "store_0017"),
                   help="Stored detections of the test videos (default: the published model's, "
                        "config.EVAL_DIR/store_0017).")
    p.add_argument("--gt-file", default=str(config.TESTSET_V6_GT),
                   help="Frozen list of test tracks (default: data/test_set_v6_gt.json).")
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"),
                   help="Per-frame box files of the reference run that defines the test tracks.")
    p.add_argument("--iou-thresh", type=float, default=0.3,
                   help="Minimum overlap between the frozen box and the model's box (published: 0.3).")
    p.add_argument("--out", default=str(config.OUTPUT_DIR / "preds_v27_ckpt0017.csv"),
                   help="Output predictions CSV (default: output/preds_v27_ckpt0017.csv).")
    p.add_argument("--video-out", default=None, help="Optional CSV of species named per video.")
    p.add_argument("--support-out", default=None,
                   help="Optional CSV with, per track, the number of reference frames and the number of "
                        "frames that contributed to the prediction.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Do not return an error code when some test tracks receive no prediction.")
    return p.parse_args()


if __name__ == "__main__":
    main()

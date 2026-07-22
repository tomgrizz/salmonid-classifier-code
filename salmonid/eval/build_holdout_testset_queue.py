"""
In plain terms: this script picks the set of example fish that will be used to fairly grade the
model. It deliberately chooses fish the model has never seen during training, keeps only the ones
where the fish is fully in view (so a biologist can actually confirm the species), and favours
tricky look-alike cases. A person then labels these to create the fixed answer key. That answer key
is what the paper's headline held-out accuracy of 0.966 is measured against.

Build the PRISTINE held-out test-set labelling queue (test_set_v6) + the reusable BigBoxCheck
labelability gate — the eval-hygiene fix.

Pipeline position
-----------------
data -> detect -> track -> [THIS: sample a frozen held-out eval queue] -> eval/count -> results.
This is a data-preparation step that sits *before* evaluation: it selects which never-trained
tracks become the honest held-out test set, and a human (Tom) then labels them to produce the frozen
ground truth used to report the headline held-out species score.

The eval never had a real holdout: op_testset leaked into the classifier crops because prepare_crops only
excluded ls_test_set_v5. This picks NEVER-TRAINED tracks (not in labels.sqlite; optionally not in the extra
training CSVs), and — critically — only tracks that have a genuine FULL-BODY frame, so every test case is
actually LABELABLE. It then weights toward HARD chrome cases (small top-1/top-2 margin) across BOTH rivers,
and emits (a) a label_app queue for Tom to label = frozen GT, and (b) the canonical `test_set_v6_holdout.json`
(also in ls_test_set_v5 `items` shape, so it drops into any --ls-test-queue) that EVERY prep script must
exclude (prepare_dataset / prepare_crops / _build_extra_coco / _sam2_composite / _sam2_extract_cutouts).

  python -m salmonid.eval.build_holdout_testset_queue --per-species 100 --hard-frac 0.5

⚠️ The BIG-BOX gate is essential: a fish that only barely enters the frame gives a low-information,
low-margin prediction, so naive "hardest chrome margin" selection preferentially grabs UNLABELABLE partial
fish (measured: 20% of the first cut). Requiring >= --min-bigbox-frames fully-exposed frames removes them and
keeps genuinely-hard-but-labelable chrome cases (a fully-visible Coho that looks like a Rainbow). This also
cleans the Atlantic bucket: the partial "Atlantic" defaults drop out, the real fully-visible chrome
confusions (the precision test) stay.

Inputs   run_20260428 tracks.sqlite (clips already rendered) + its detections parquets (for the big-box
         check); labels.sqlite (trained-track exclusion); optional extra-train CSVs and an Atlantic
         tech-seed video list. Predicted species is used ONLY to stratify; Tom's label is the GT.
Outputs  <output-dir>/test_set_v6_queue.json (label_app queue) and
         <output-dir>/test_set_v6_holdout.json (the frozen leak-guard exclusion set).
Environment  Base Python env (sqlite3 + pyarrow only; no torch/DEIMv2) — runs on CPU.
Manuscript claim  Underpins the honest held-out species score (test_set_v6 macro-F1 0.966): this script
                  defines *which* tracks that number is computed on.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from salmonid import config
from salmonid.util.schema import video_id_to_parquet_stem

SP = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.SPECIES (index -> name)
PROB_COLS = ["mean_prob_chinook", "mean_prob_coho", "mean_prob_atlantic",
             "mean_prob_rainbow", "mean_prob_brown"]
FW, FH = 1280, 960   # all Riverwatcher frames


def _load_excludes(labels_db: str, extra_csvs) -> set:
    """Return the (video_id, track_id) keys that must NOT enter the held-out queue.

    Every human-labelled track in ``labels_db`` counts as trained, plus any (video_id, track_id)
    rows found in the optional ``extra_csvs`` training-source files. Excluding all of these keeps the
    queue pristine (never-trained). Missing extra CSVs warn and are skipped.

    Returns a set of (video_id, int(track_id)) tuples.
    """
    tracks = set()
    c = sqlite3.connect(labels_db)
    for v, t in c.execute("SELECT video_id, track_id FROM labels"):
        tracks.add((v, int(t)))
    c.close()
    for cp in extra_csvs or []:
        p = Path(cp)
        if not p.is_file():
            print(f"WARNING: --exclude-keys not found: {cp}")
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "video_id" in r and "track_id" in r:
                    try:
                        tracks.add((r["video_id"], int(r["track_id"])))
                    except ValueError:
                        pass
    return tracks


def _chrome_margin(row) -> tuple[int, float]:
    """Return (argmax class id, top1-minus-top2 probability margin) from a track's mean species probs.

    A small margin = a hard "chrome" case where two species are nearly tied — exactly the confusions
    that test the classifier — so the margin is used to stratify selection. Missing probs read as 0.
    """
    probs = [float(row[c] or 0) for c in PROB_COLS]
    order = sorted(range(5), key=lambda i: probs[i], reverse=True)  # highest prob first
    return order[0], probs[order[0]] - probs[order[1]]


class BigBoxCheck:
    """Count a track's fully-exposed (big, well-inside-frame, y-sane) frames from its parquet.

    A track qualifies only if it has >= min_frames such frames — i.e. the fish is at some point fully in
    view and thus labelable. Results are cached per (video_id, track_id)."""

    def __init__(self, det_dir, min_area_frac, edge_px, min_frames):
        """Configure the gate; no parquet is read until ok()/_count is first called for a track."""
        self.det_dir = Path(det_dir)
        self.min_area, self.edge, self.min_frames = min_area_frac, edge_px, min_frames
        self.cache: dict = {}

    def ok(self, video_id: str, track_id: int) -> bool:
        """Return True if the track has >= min_frames fully-exposed frames (result cached per track)."""
        key = (video_id, track_id)
        if key in self.cache:
            return self.cache[key]
        res = self._count(video_id, track_id) >= self.min_frames
        self.cache[key] = res
        return res

    def _count(self, video_id, track_id) -> int:
        """Count fully-exposed frames for one track by scanning its per-frame detections parquet.

        A frame qualifies when the bbox is big enough (>= min_area of the frame), sits fully inside
        the borders (>= edge px from every side) AND has a plausible vertical centre. Returns 0 when
        the parquet is missing, unreadable, or holds no rows for this track.
        """
        pf = self.det_dir / f"{video_id_to_parquet_stem(video_id)}.parquet"
        if not pf.is_file():
            return 0
        try:
            df = pq.read_table(str(pf), filters=[("track_id", "=", int(track_id))]).to_pandas()
        except Exception:
            return 0
        if df.empty:
            return 0
        x1 = df.bbox_x1.astype(float); y1 = df.bbox_y1.astype(float)
        x2 = df.bbox_x2.astype(float); y2 = df.bbox_y2.astype(float)
        area = ((x2 - x1).clip(lower=1) * (y2 - y1).clip(lower=1)) / (FW * FH)
        yc = (y1 + y2) / 2 / FH
        inside = (x1 >= self.edge) & (y1 >= self.edge) & (x2 <= FW - self.edge) & (y2 <= FH - self.edge)
        # y-centre 0.30-0.85 filters the header/date band up top and floor artefacts down low.
        ok = (area >= self.min_area) & inside & (yc >= 0.30) & (yc <= 0.85)
        return int(ok.sum())


def _select(pool, target, hard_frac, check: BigBoxCheck, rng):
    """Pick `target` LABELABLE tracks: hard_frac from the hardest chrome margins + the rest a spread,
    river-balanced. Big-box verification happens lazily (only enough parquets are read)."""
    if not pool:
        return []
    by_site = defaultdict(list)
    for it in pool:
        by_site[it["site"]].append(it)
    for s in by_site:
        by_site[s].sort(key=lambda it: it["margin"])   # hardest (lowest margin) first
    sites = [s for s in by_site if by_site[s]]
    picks, used = [], set()

    def take(iters, n):
        i, exhausted = 0, set()
        while n > 0 and len(exhausted) < len(sites):
            s = sites[i % len(sites)]; i += 1
            if s in exhausted:
                continue
            it = _next_labelable(iters[s], used, check)
            if it is None:
                exhausted.add(s); continue
            picks.append(it); used.add((it["video_id"], it["track_id"])); n -= 1

    n_hard = int(round(target * hard_frac))
    take({s: iter(by_site[s]) for s in sites}, n_hard)                       # hardest-first
    take({s: iter(rng.sample(by_site[s], len(by_site[s]))) for s in sites},  # random spread
         target - len(picks))
    return picks


def _next_labelable(seq_iter, used, check):
    """Advance seq_iter to the next not-yet-used track that passes the big-box gate; None if exhausted."""
    for it in seq_iter:
        k = (it["video_id"], it["track_id"])
        if k in used:
            continue
        if check.ok(it["video_id"], it["track_id"]):
            return it
    return None


def main() -> None:
    """Scan pristine candidate tracks, select a labelable + chrome-hard + river-balanced held-out set,
    optionally top it up with tech-confirmed Atlantic seeds, and write the label_app queue plus the
    frozen holdout-exclusion JSON. See the module docstring for the full rationale. Side effects: reads
    tracks/labels sqlite + detections parquets; creates --output-dir and writes two JSON files there."""
    a = _parse_args()
    rng = random.Random(a.seed)
    excl = _load_excludes(a.labels_db, a.exclude_keys)
    atl_vids = set()
    if a.atlantic_videos and Path(a.atlantic_videos).is_file():
        atl_vids = {l.strip() for l in open(a.atlantic_videos) if l.strip()}
        print(f"Atlantic tech seed: {len(atl_vids)} video_ids")
    check = BigBoxCheck(a.detections_dir, a.min_area_frac, a.edge_margin_px, a.min_bigbox_frames)

    t = sqlite3.connect(a.tracks_db); t.row_factory = sqlite3.Row
    by_sp: dict[int, list] = defaultdict(list)
    atl_seed: list = []
    n_scan = 0
    for r in t.execute(
        "SELECT video_id, track_id, clip_path, n_frames, mean_detection_confidence, "
        + ", ".join(PROB_COLS) + " FROM tracks WHERE clip_path IS NOT NULL "
        "AND mean_detection_confidence >= ? AND n_frames >= ?", (a.min_conf, a.min_frames)):
        vid, tid = r["video_id"], int(r["track_id"])
        if (vid, tid) in excl:
            continue
        n_scan += 1
        pred, margin = _chrome_margin(r)
        item = {"video_id": vid, "track_id": tid, "clip_path": r["clip_path"],
                "predicted": SP[pred], "confidence": round(float(r["mean_detection_confidence"]), 3),
                "margin": round(margin, 4), "site": vid.split("/", 1)[0]}
        by_sp[pred].append(item)
        if vid in atl_vids:
            atl_seed.append(item)
    t.close()
    print(f"pristine candidates (conf>={a.min_conf}, n_frames>={a.min_frames}): {n_scan} "
          f"(excluded {len(excl)} trained)")
    for cid in range(5):
        print(f"   pred {SP[cid]:<14} {len(by_sp[cid])}")
    print(f"Verifying big-box (>= {a.min_bigbox_frames} fully-exposed frames) during selection ...")

    selected: list = []
    for cid in range(5):
        picks = _select(by_sp[cid], a.per_species, a.hard_frac, check, rng)
        for it in picks:
            it["strat_species"] = SP[cid]
        selected.extend(picks)
        print(f"   {SP[cid]:<14} selected {len(picks)}/{a.per_species}")
    seen = {(it["video_id"], it["track_id"]) for it in selected}
    for it in atl_seed:
        if (it["video_id"], it["track_id"]) not in seen and check.ok(it["video_id"], it["track_id"]):
            it["strat_species"] = "Atlantic(tech-seed)"
            selected.append(it); seen.add((it["video_id"], it["track_id"]))

    rng.shuffle(selected)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_set_v6_queue.json").write_text(json.dumps(
        {"batch_id": a.batch_id,
         "purpose": "held-out test set (test_set_v6) — NEVER train on these",
         "items": selected}, indent=2))
    holdout = {"name": "test_set_v6_holdout",
               "note": "Frozen held-out test set. Exclude from ALL training: prepare_dataset, "
                       "prepare_crops, _build_extra_coco, _sam2_composite/_extract_cutouts.",
               "video_ids": sorted({it["video_id"] for it in selected}),
               "tracks": [[it["video_id"], it["track_id"]] for it in selected],
               "items": [{"video_id": it["video_id"], "track_id": it["track_id"]} for it in selected]}
    (out_dir / "test_set_v6_holdout.json").write_text(json.dumps(holdout, indent=2))

    print(f"\n=== test_set_v6 queue ===")
    print(f"  {len(selected)} tracks / {len(holdout['video_ids'])} videos (all with a full-body frame)")
    print("  by strat species: " + ", ".join(f"{k} {v}" for k, v in
          Counter(it["strat_species"] for it in selected).most_common()))
    print("  by site: " + ", ".join(f"{k} {v}" for k, v in
          Counter(it["site"] for it in selected).items()))
    print(f"  queue:   {out_dir/'test_set_v6_queue.json'}")
    print(f"  HOLDOUT: {out_dir/'test_set_v6_holdout.json'}")


def _parse_args():
    """Define and parse the CLI. Path defaults resolve via salmonid.config to the original
    workstation locations (argparse needs str, so each Path is wrapped in str())."""
    p = argparse.ArgumentParser(description="Build the pristine held-out test-set queue (test_set_v6)")
    p.add_argument("--tracks-db", default=str(config.MODEL_OUTPUTS / "run_20260428" / "tracks.sqlite"))
    p.add_argument("--labels-db", default=str(config.LABELS_DB))
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"))
    p.add_argument("--exclude-keys", nargs="*", default=None,
                   help="Extra training-source CSVs (video_id,track_id) to also exclude.")
    p.add_argument("--atlantic-videos", default=None,
                   help="File of tech-confirmed Atlantic video_ids (one/line) — seed their pristine tracks.")
    p.add_argument("--per-species", type=int, default=100)
    p.add_argument("--hard-frac", type=float, default=0.5,
                   help="Fraction from the hardest (lowest chrome-margin) LABELABLE tracks.")
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--min-frames", type=int, default=8,
                   help="Min track length — a barely-entering fish yields a very short track.")
    p.add_argument("--min-bigbox-frames", type=int, default=3,
                   help="Min fully-exposed (big, well-inside, y-sane) frames — the labelability gate.")
    p.add_argument("--min-area-frac", type=float, default=0.02,
                   help="A frame's bbox must cover >= this fraction of the frame to count as full-body.")
    p.add_argument("--edge-margin-px", type=int, default=12,
                   help="A full-body frame's bbox must be >= this many px inside every border.")
    p.add_argument("--output-dir", default=str(config.LABEL_BATCHES))
    p.add_argument("--batch-id", default="test_set_v6")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()

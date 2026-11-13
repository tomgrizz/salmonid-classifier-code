"""In plain terms: to grade a model fairly you need a set of example fish that the model was
never trained on, each with a species that a person has confirmed. This script was the first
step in making that set. It went through the fish tracks of an earlier run of the project's
previous model, kept only tracks that nobody had labelled before and in which the whole fish
is clearly visible, and wrote a list of 500 of them for a reviewer to label. The labelled
part of that list became the larger of the two parts of the manuscript's 366-fish test set.
The script is kept as the record of how those fish were chosen. An outside reader cannot
re-run it, because it reads the project's private labelling database, and re-running it
would not reproduce the same list in any case (see "Reproducibility").

How the 366-track test set was built (both steps)
-------------------------------------------------
The test set (``data/test_set_v6_gt.json``) has two parts, called tiers in the code.

1. **"pristine" tier, 215 tracks: this script, then a reviewer.** Candidate tracks came from
   one earlier run of the retired two-stage pipeline (run ``run_20260428``), which had
   already detected and tracked the fish in the archive. A track was a candidate if

   * it was not in the labelling database, i.e. no one had labelled it, so it could not
     have been used as a training example (plus any tracks listed in ``--exclude-keys``);
   * a video clip of it had been rendered, its mean detection confidence was at least 0.6
     and it had at least 8 frames;
   * it had at least 3 "full-body" frames (see ``BigBoxCheck``).

   From the candidates, 100 tracks were taken for each species *as predicted by that
   earlier model* (the prediction is used only to spread the sample across species; it is
   not the label). Within each predicted species, half were the tracks the earlier model
   was least sure about (smallest gap between its top two species scores) and half a
   random draw, alternating between the two rivers. That gave a queue of 500 tracks.
   According to the labelling database, one reviewer labelled 293 of the 500 on 2 and 3
   July 2026 in the project's labelling tool: 215 received one of the five species and
   form the tier; 78 were set aside (70 "Multiple fish", 3 "Not a fish", 5 "Small
   Salmonid"). The remaining 207 queued tracks were never labelled and are not in the
   test set. The queue file carries the earlier model's predicted species and confidence
   for every track.

   The tier is 121 Chinook Salmon, 65 Rainbow Trout, 21 Coho Salmon, 7 Atlantic Salmon and
   1 Brown Trout.

2. **"heldout" tier, 151 tracks: ``salmonid.eval.finalize_testset_v6``.** Because the first
   tier held too few of the rarer species, Coho Salmon, Atlantic Salmon and Brown Trout
   were topped up to 60 each from tracks that had ALREADY been labelled in earlier
   labelling rounds, and those tracks were then withheld from training. See that module.

In both tiers a "track" is the frozen sequence of boxes from the earlier run. The published
model is graded by asking which species it assigns to the fish inside those boxes
(``salmonid.eval.predict_testset_v6_from_store``); the tracks are not re-derived by the
published model's own tracker.

Consequences a reader should know: the test fish are fish the earlier model detected with
confidence and that are fully visible in at least three frames, so they are clearer than an
average passage; and half of the first tier was deliberately chosen to be hard for the
earlier model.

Inputs
------
All Tier 2 (not stored in git; the labelling database is private):

* ``--tracks-db``, ``--detections-dir``: the track database and the per-video detection
  tables of the earlier run (defaults under ``config.MODEL_OUTPUTS / "run_20260428"``).
* ``--labels-db``: the labelling database (``config.LABELS_DB``); every track in it is
  excluded.
* ``--exclude-keys`` (optional): further CSV files of (video_id, track_id) to exclude.
* ``--atlantic-videos`` (optional): a list of videos in which the technicians recorded an
  Atlantic Salmon; their candidate tracks are added to the queue.

Outputs
-------
Written to ``--output-dir`` (default ``output/testset_v6_rebuild/``; never ``data/``):

* ``test_set_v6_queue.json``: the list of tracks for the reviewer to label.
* ``test_set_v6_holdout.json``: the same tracks as a "do not train on these" list. The
  published list (``data/test_set_v6_holdout.json``) is the later one written by
  ``finalize_testset_v6``, which covers the final 366 tracks.

Reproducibility
---------------
The selection uses a fixed random seed (``--seed 42``), but it also depends on the state of
the labelling database on the day it was run (every track labelled since would now be
excluded) and on the order in which the database returns rows. Re-running it today gives a
different queue. The published test set is the committed file, not the output of this
script. For that reason the default output folder is ``output/`` and the script refuses to
overwrite an existing file unless ``--overwrite`` is given.

Behaviour changes relative to the original research script
-----------------------------------------------------------
Which tracks are selected is unchanged.

* A detection table that could not be read made its track silently ineligible. The track
  is still treated as ineligible, but missing and unreadable tables are now counted and
  the counts (and the first few error messages) are printed.
* Rows of an ``--exclude-keys`` file with a non-numeric track number were silently
  skipped; they are now counted and reported.
* An ``--exclude-keys`` or ``--atlantic-videos`` file that does not exist only produced a
  warning (or nothing); it now stops the run, because a missing exclusion list would let
  training tracks into the test set without notice.
* Missing databases stop the run with one sentence naming the file. Databases are opened
  read-only.
* Default output folder moved from the project data drive to ``output/``; existing files
  are not overwritten without ``--overwrite``.

Feeds
-----
Defines which tracks form the "pristine" tier of the held-out test set behind manuscript
Table 1, Figure 2 and Section 3.1 (macro-F1 0.961 on 366 tracks). It computes no reported
number itself.

Run
---
::

    python -m salmonid.eval.build_holdout_testset_queue --per-species 100 --hard-frac 0.5

Base Python with pyarrow and pandas; no GPU.

Tier
----
2, and not re-runnable outside the project (private labelling database).

Status
------
Published path (test-set construction; kept as the record of how the set was drawn).
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

SP = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # class number -> name (salmonid.config.SPECIES)
# Columns of the earlier run's track table holding each species' mean score over a track.
PROB_COLS = ["mean_prob_chinook", "mean_prob_coho", "mean_prob_atlantic",
             "mean_prob_rainbow", "mean_prob_brown"]
FW, FH = 1280, 960   # frame width and height in pixels (all footage of both rivers)


def open_readonly(db_path) -> sqlite3.Connection:
    """Open an SQLite database for reading only, so this script can never alter it."""
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def refuse_overwrite(paths, overwrite: bool) -> None:
    """Stop if an output file already exists, unless ``--overwrite`` was given.

    Re-running a test-set construction step does not reproduce the published files (see the
    module docstring), so silently replacing an existing copy would destroy the record.
    """
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Refusing to overwrite existing file(s):\n  " + "\n  ".join(existing) +
            "\nA re-run does not reproduce the published test set. Choose another "
            "--output-dir, or pass --overwrite if replacing these files is intended.")


def _load_excludes(labels_db: str, extra_csvs) -> set:
    """Return the (video_id, track_id) pairs that must NOT enter the queue.

    Every track in the labelling database is excluded: a labelled track may have been used
    as a training example. Tracks listed in the optional ``extra_csvs`` are excluded too.
    """
    tracks = set()
    c = open_readonly(labels_db)
    for v, t in c.execute("SELECT video_id, track_id FROM labels"):
        tracks.add((v, int(t)))
    c.close()
    for cp in extra_csvs or []:
        # A missing exclusion list must stop the run: carrying on would quietly allow
        # training tracks into the test set.
        p = config.require(cp, "an exclusion list given to --exclude-keys")
        n_bad = 0
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if "video_id" in r and "track_id" in r:
                    try:
                        tracks.add((r["video_id"], int(r["track_id"])))
                    except ValueError:
                        n_bad += 1       # track number is not a whole number: row cannot be used
        if n_bad:
            print(f"WARNING: {n_bad} rows of {cp} have a non-numeric track_id and were NOT "
                  f"added to the exclusion list.")
    return tracks


def _chrome_margin(row) -> tuple[int, float]:
    """Return (most likely species, gap between the top two species scores) for a track.

    A small gap means the earlier model was torn between two species. In this project those
    are typically bright silver ("chrome") fish, which several species share. The gap is
    used to steer half of the sample toward such hard cases. A missing score counts as 0.
    """
    probs = [float(row[c] or 0) for c in PROB_COLS]
    order = sorted(range(5), key=lambda i: probs[i], reverse=True)  # highest score first
    return order[0], probs[order[0]] - probs[order[1]]


class BigBoxCheck:
    """Decide whether a track shows the whole fish clearly enough for a person to label it.

    A frame counts as "full-body" when the fish's box (a) covers at least ``min_area_frac``
    of the frame, (b) stays at least ``edge_px`` pixels away from every border, so the fish
    is not cut off, and (c) has its centre between 30% and 85% of the frame height. A track
    passes when it has at least ``min_frames`` such frames. Without this check, picking the
    "hardest" tracks mostly picks fish that only poke into the picture, which nobody can
    identify. Results are remembered per track, so each detection table is read once.

    Tables that are missing or cannot be read make the track fail the check. They are
    counted (``n_missing``, ``n_unreadable``) so the caller can report them.
    """

    def __init__(self, det_dir, min_area_frac, edge_px, min_frames):
        """Store the thresholds. No file is read until a track is first checked."""
        self.det_dir = Path(det_dir)
        self.min_area, self.edge, self.min_frames = min_area_frac, edge_px, min_frames
        self.cache: dict = {}
        self.n_missing = 0            # tracks whose detection table does not exist
        self.n_unreadable = 0         # tracks whose detection table exists but could not be read
        self.errors: list[str] = []   # first few read errors, for the report

    def ok(self, video_id: str, track_id: int) -> bool:
        """True if the track has at least ``min_frames`` full-body frames."""
        key = (video_id, track_id)
        if key in self.cache:
            return self.cache[key]
        res = self._count(video_id, track_id) >= self.min_frames
        self.cache[key] = res
        return res

    def _count(self, video_id, track_id) -> int:
        """Number of full-body frames of one track, read from the video's detection table.

        Returns 0 when the table is missing, unreadable, or has no rows for this track.
        """
        pf = self.det_dir / f"{video_id_to_parquet_stem(video_id)}.parquet"
        if not pf.is_file():
            self.n_missing += 1
            return 0
        try:
            df = pq.read_table(str(pf), filters=[("track_id", "=", int(track_id))]).to_pandas()
        except Exception as e:  # noqa: BLE001  (any read failure: recorded, then reported)
            self.n_unreadable += 1
            if len(self.errors) < 5:
                self.errors.append(f"{pf.name}: {type(e).__name__}: {e}")
            return 0
        if df.empty:
            return 0
        x1 = df.bbox_x1.astype(float); y1 = df.bbox_y1.astype(float)
        x2 = df.bbox_x2.astype(float); y2 = df.bbox_y2.astype(float)
        area = ((x2 - x1).clip(lower=1) * (y2 - y1).clip(lower=1)) / (FW * FH)   # share of the frame
        yc = (y1 + y2) / 2 / FH                                                  # centre height, 0 = top
        inside = (x1 >= self.edge) & (y1 >= self.edge) & (x2 <= FW - self.edge) & (y2 <= FH - self.edge)
        # Why a centre-height test: it leaves out boxes centred in the top 30% of the frame
        # (where the burned-in date and river banner sits) and in the bottom 15% (artefacts
        # at the tunnel floor). A box there is unlikely to be a fish a reviewer can identify.
        ok = (area >= self.min_area) & inside & (yc >= 0.30) & (yc <= 0.85)
        return int(ok.sum())

    def report(self) -> None:
        """Print how many tracks failed the check only because their table could not be used."""
        if self.n_missing or self.n_unreadable:
            print(f"WARNING: full-body check: {self.n_missing} tracks had no detection table and "
                  f"{self.n_unreadable} had a table that could not be read; all were treated as "
                  f"not labelable and left out.")
            for msg in self.errors:
                print(f"    {msg}")
        else:
            print("full-body check: every detection table that was needed could be read.")


def _select(pool, target, hard_frac, check: BigBoxCheck, rng):
    """Pick ``target`` labelable tracks from ``pool``, alternating between the two rivers.

    A share ``hard_frac`` is taken from the tracks with the smallest score gap (hardest
    first); the rest is a random draw. The full-body check is applied as tracks come up, so
    only as many detection tables are read as are needed.
    """
    if not pool:
        return []
    by_site = defaultdict(list)
    for it in pool:
        by_site[it["site"]].append(it)
    for s in by_site:
        by_site[s].sort(key=lambda it: it["margin"])   # hardest (smallest gap) first
    sites = [s for s in by_site if by_site[s]]
    picks, used = [], set()

    def take(iters, n):
        # Take one track from each river in turn until n are taken or both rivers run out.
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
    take({s: iter(by_site[s]) for s in sites}, n_hard)                       # the hard half
    take({s: iter(rng.sample(by_site[s], len(by_site[s]))) for s in sites},  # the random half
         target - len(picks))
    return picks


def _next_labelable(seq_iter, used, check):
    """Next track in ``seq_iter`` that is not yet picked and passes the full-body check."""
    for it in seq_iter:
        k = (it["video_id"], it["track_id"])
        if k in used:
            continue
        if check.ok(it["video_id"], it["track_id"]):
            return it
    return None


def main() -> None:
    """Scan the candidate tracks, select the queue, and write the two JSON files."""
    a = _parse_args()
    rng = random.Random(a.seed)
    config.require(a.labels_db, "the labelling database (private; not distributed)",
                   "SALMONID_PROJECTS_ROOT")
    config.require(a.tracks_db, "the track database of the earlier run (run_20260428)",
                   "SALMONID_PROJECTS_ROOT")
    config.require(a.detections_dir, "the per-video detection tables of the earlier run "
                   "(run_20260428/detections)", "SALMONID_PROJECTS_ROOT")
    out_dir = Path(a.output_dir)
    refuse_overwrite([out_dir / "test_set_v6_queue.json", out_dir / "test_set_v6_holdout.json"],
                     a.overwrite)

    excl = _load_excludes(a.labels_db, a.exclude_keys)
    atl_vids = set()
    if a.atlantic_videos:
        p = config.require(a.atlantic_videos, "the list of technician-confirmed Atlantic Salmon "
                           "videos given to --atlantic-videos")
        with open(p, encoding="utf-8") as f:
            atl_vids = {l.strip() for l in f if l.strip()}
        print(f"Atlantic Salmon videos from the technician record: {len(atl_vids)}")
    check = BigBoxCheck(a.detections_dir, a.min_area_frac, a.edge_margin_px, a.min_bigbox_frames)

    t = open_readonly(a.tracks_db); t.row_factory = sqlite3.Row
    by_sp: dict[int, list] = defaultdict(list)
    atl_seed: list = []
    n_scan = 0
    for r in t.execute(
        "SELECT video_id, track_id, clip_path, n_frames, mean_detection_confidence, "
        + ", ".join(PROB_COLS) + " FROM tracks WHERE clip_path IS NOT NULL "
        "AND mean_detection_confidence >= ? AND n_frames >= ?", (a.min_conf, a.min_frames)):
        vid, tid = r["video_id"], int(r["track_id"])
        if (vid, tid) in excl:
            continue                              # already labelled: may have been trained on
        n_scan += 1
        pred, margin = _chrome_margin(r)
        item = {"video_id": vid, "track_id": tid, "clip_path": r["clip_path"],
                "predicted": SP[pred], "confidence": round(float(r["mean_detection_confidence"]), 3),
                "margin": round(margin, 4), "site": vid.split("/", 1)[0]}
        by_sp[pred].append(item)
        if vid in atl_vids:
            atl_seed.append(item)
    t.close()
    print(f"never-labelled candidates (conf>={a.min_conf}, n_frames>={a.min_frames}): {n_scan} "
          f"({len(excl)} labelled tracks excluded)")
    for cid in range(5):
        print(f"   predicted {SP[cid]:<14} {len(by_sp[cid])}")
    print(f"Checking for >= {a.min_bigbox_frames} full-body frames during selection ...")

    selected: list = []
    for cid in range(5):
        picks = _select(by_sp[cid], a.per_species, a.hard_frac, check, rng)
        for it in picks:
            it["strat_species"] = SP[cid]         # the species group this track was drawn for
        selected.extend(picks)
        print(f"   {SP[cid]:<14} selected {len(picks)}/{a.per_species}")
    seen = {(it["video_id"], it["track_id"]) for it in selected}
    for it in atl_seed:
        if (it["video_id"], it["track_id"]) not in seen and check.ok(it["video_id"], it["track_id"]):
            it["strat_species"] = "Atlantic(tech-seed)"
            selected.append(it); seen.add((it["video_id"], it["track_id"]))
    check.report()

    rng.shuffle(selected)      # so the reviewer does not see the tracks grouped by species
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_set_v6_queue.json").write_text(json.dumps(
        {"batch_id": a.batch_id,
         "purpose": "held-out test set (test_set_v6) — NEVER train on these",
         "items": selected}, indent=2), encoding="utf-8")
    holdout = {"name": "test_set_v6_holdout",
               "note": "Frozen held-out test set. Exclude from ALL training: prepare_dataset, "
                       "prepare_crops, _build_extra_coco, _sam2_composite/_extract_cutouts.",
               "video_ids": sorted({it["video_id"] for it in selected}),
               "tracks": [[it["video_id"], it["track_id"]] for it in selected],
               "items": [{"video_id": it["video_id"], "track_id": it["track_id"]} for it in selected]}
    (out_dir / "test_set_v6_holdout.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    print(f"\n=== test_set_v6 queue ===")
    print(f"  {len(selected)} tracks / {len(holdout['video_ids'])} videos (all with a full-body frame)")
    print("  by species group: " + ", ".join(f"{k} {v}" for k, v in
          Counter(it["strat_species"] for it in selected).most_common()))
    print("  by river: " + ", ".join(f"{k} {v}" for k, v in
          Counter(it["site"] for it in selected).items()))
    print(f"  queue:   {out_dir/'test_set_v6_queue.json'}")
    print(f"  holdout: {out_dir/'test_set_v6_holdout.json'}")
    print("  NOTE: this is a new draw. It is not the published test set (data/test_set_v6_gt.json).")


def _parse_args():
    """Define and read the command-line options. Path defaults come from ``salmonid.config``."""
    p = argparse.ArgumentParser(
        description="Draw the queue of never-labelled, fully visible tracks that a reviewer "
                    "labelled to make the 'pristine' tier of the held-out test set. Needs the "
                    "private labelling database; a re-run gives a new draw, not the published set.")
    p.add_argument("--tracks-db", default=str(config.MODEL_OUTPUTS / "run_20260428" / "tracks.sqlite"),
                   help="Track database of the earlier run.")
    p.add_argument("--labels-db", default=str(config.LABELS_DB),
                   help="Labelling database; every track in it is excluded.")
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"),
                   help="Per-video detection tables of the earlier run.")
    p.add_argument("--exclude-keys", nargs="*", default=None,
                   help="Further CSV files (video_id,track_id) of tracks to exclude.")
    p.add_argument("--atlantic-videos", default=None,
                   help="File of video_ids (one per line) with a technician-confirmed Atlantic "
                        "Salmon; their candidate tracks are added to the queue.")
    p.add_argument("--per-species", type=int, default=100,
                   help="Tracks to draw per species as predicted by the earlier model.")
    p.add_argument("--hard-frac", type=float, default=0.5,
                   help="Share drawn from the tracks with the smallest gap between the top two "
                        "species scores (the hard cases).")
    p.add_argument("--min-conf", type=float, default=0.6,
                   help="Lowest mean detection confidence of a candidate track.")
    p.add_argument("--min-frames", type=int, default=8,
                   help="Fewest frames of a candidate track (a fish that barely enters the "
                        "picture gives a very short track).")
    p.add_argument("--min-bigbox-frames", type=int, default=3,
                   help="Fewest full-body frames a track needs to count as labelable.")
    p.add_argument("--min-area-frac", type=float, default=0.02,
                   help="A full-body frame's box covers at least this share of the frame.")
    p.add_argument("--edge-margin-px", type=int, default=12,
                   help="A full-body frame's box stays at least this many pixels from every border.")
    p.add_argument("--output-dir", default=str(config.OUTPUT_DIR / "testset_v6_rebuild"),
                   help="Where to write the two JSON files (default: output/testset_v6_rebuild).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing existing output files.")
    p.add_argument("--batch-id", default="test_set_v6", help="Name of the labelling batch.")
    p.add_argument("--seed", type=int, default=42, help="Random seed of the draw.")
    return p.parse_args()


if __name__ == "__main__":
    main()

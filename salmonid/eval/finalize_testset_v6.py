"""In plain terms: this script put together the final list of 366 test fish that the
manuscript's species score is measured on. It took the fish a reviewer had just labelled
for the purpose (215 of them, almost all Chinook Salmon and Rainbow Trout), saw that the
three rarer species had too few examples, and filled each of them up to 60 with fish that
had already been labelled in earlier labelling rounds. It then wrote two files: the answer
key (which species each test fish is) and a "do not train on these" list. It does not grade
any model. It is kept as the record of how the test set was assembled; an outside reader
cannot re-run it, and re-running it would not reproduce the same list (see
"Reproducibility").

The two tiers of the test set
-----------------------------
Every entry of ``data/test_set_v6_gt.json`` carries a ``tier`` field.

    tier        Chinook   Coho   Atlantic   Rainbow   Brown   total
    pristine        121     21          7        65       1     215
    heldout           0     39         53         0      59     151
    total           121     60         60        65      60     366

* **pristine (215 tracks).** Tracks that had never been labelled before, chosen by
  ``salmonid.eval.build_holdout_testset_queue`` and then labelled by one reviewer in the
  project's labelling tool (labelling batch ``test_set_v6``). This script reads those labels
  from the labelling database and keeps the tracks labelled as one of the five species.
  Tracks the reviewer marked "Multiple fish", "Not a fish" or "Small Salmonid" are left out.
* **heldout (151 tracks).** For each of Coho Salmon, Atlantic Salmon and Brown Trout, the
  number missing to reach ``--floor`` (60) was drawn at random from tracks ALREADY in the
  labelling database with that species, i.e. from the pool of tracks labelled in earlier
  rounds, which is the pool training examples were drawn from. A track qualified if

  - one of its label rows names the species, and that row is either undisputed or was
    settled by a senior reviewer (``disagreement_resolution`` empty or ``'senior'``);
  - it is not in the older evaluation list ``ls_test_set_v5.json``;
  - a video clip of it exists, and it has at least 3 full-body frames
    (``BigBoxCheck`` with the values 0.02, 12 px, 3; see the queue module).

  These tracks were NOT looked at again when the test set was made; their existing labels
  were used as they stood. According to the labelling database, those labels were made
  between 30 April and 10 June 2026 under two reviewer accounts, and no heldout track
  carries two different species labels. Including a track here is what removes it from training: the
  "do not train" list written below is read by the training-set builders
  (``salmonid.holdout``).

Who assigned the labels, according to the code: reviewers using the project's labelling
tool, whose answers are stored in the labelling database. The code contains no step in which
a technician's field record is used as the species of a test track.

In both tiers a track is the frozen box sequence from an earlier run of the retired
two-stage pipeline (``run_20260428``); the published model is graded on the fish inside
those boxes and the tracks are not re-derived.

Inputs
------
All Tier 2 (not stored in git; the labelling database is private):

* ``--labels-db``: the labelling database (``config.LABELS_DB``).
* ``--tracks-db``, ``--detections-dir``: the track database and per-video detection tables
  of the earlier run (defaults under ``config.MODEL_OUTPUTS / "run_20260428"``).
* ``--ls-test-queue``: the older evaluation list whose tracks are kept out of the heldout
  tier (``config.LABEL_BATCHES / "ls_test_set_v5.json"``).

Outputs
-------
Written to ``--output-dir`` (default ``output/testset_v6_rebuild/``; never ``data/``):

* ``test_set_v6_gt.json``: one entry per track: video_id, track_id, species, tier, site.
* ``test_set_v6_holdout.json``: the same tracks and their videos as a "do not train" list.

The published copies are ``data/test_set_v6_gt.json`` and ``data/test_set_v6_holdout.json``.

Reproducibility
---------------
The published test set is the committed file. This script cannot regenerate it:

* it needs the private labelling database, whose contents have changed since;
* the random order of the heldout candidates is seeded with Python's ``hash()`` of the
  species name, which Python deliberately changes from one run to the next, and the
  database query has no fixed row order. The original draw therefore cannot be repeated,
  not even on the authors' machine. This is left as it was: fixing the seed now would
  suggest a reproducibility the published set does not have.

Only the final shuffle of the list uses a fixed seed (0). A re-run on the authors' machine
on 2026-11-10 returned the same 215 pristine tracks and a heldout tier of the same size and
species make-up, but only 20 of its 151 heldout tracks were the published ones. The default
output folder is ``output/`` and existing files are not overwritten unless ``--overwrite``
is given.

Behaviour changes relative to the original research script
-----------------------------------------------------------
Which tracks are selected by a given run is unchanged.

* A missing ``--ls-test-queue`` file was silently treated as "nothing to exclude"; a
  warning is now printed.
* Candidate tracks that are dropped (already used, in the older evaluation list, no clip,
  not enough full-body frames, detection table missing or unreadable) were dropped
  silently; the counts are now printed for each species.
* A top-up that cannot reach the floor because the pool is too small is now flagged.
* Missing databases stop the run with one sentence naming the file. Databases are opened
  read-only.
* Default output folder moved from the project data drive (where the frozen originals are
  kept) to ``output/``; existing files are not overwritten without ``--overwrite``.

Feeds
-----
Defines the 366-track held-out test set behind manuscript Table 1, Figure 2 and Section 3.1
(macro-F1 0.961; accuracy 97.7% on the 215 pristine and 94.0% on the 151 heldout tracks),
and the "do not train" list used when the training set was built. It computes no reported
number itself.

Run
---
::

    python -m salmonid.eval.finalize_testset_v6 --floor 60

Base Python with pyarrow and pandas; no GPU.

Tier
----
2, and not re-runnable outside the project (private labelling database).

Status
------
Published path (test-set construction; kept as the record of how the set was assembled).
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from salmonid import config
from salmonid.eval.build_holdout_testset_queue import BigBoxCheck, open_readonly, refuse_overwrite

# The five reported species, in the order of salmonid.config.SPECIES.
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]


def _load_pristine(labels_db, batch_id):
    """Return ``{(video_id, track_id): species}`` for the pristine tier.

    These are the labels the reviewer gave to the queued, never-before-labelled tracks
    (labelling batch ``batch_id``). Only labels that are one of the five species are kept;
    "Multiple fish", "Not a fish" and "Small Salmonid" are left out because such a track
    cannot be graded as one species.
    """
    conn = open_readonly(labels_db)
    out = {}
    for vid, tid, lab in conn.execute(
        "SELECT video_id, track_id, final_label FROM labels WHERE batch_id=?", (batch_id,)):
        if lab in SPECIES:
            out[(vid, int(tid))] = lab
    conn.close()
    return out


def _load_holdout_exclusions(op_path, ls_path):
    """Tracks that must not be drawn into the heldout tier: those of the older evaluation list.

    ``op_path`` (an older list kept per video, not per track) is accepted but not used: the
    heldout tier is drawn track by track, and that list has no track numbers.
    """
    excl = set()
    if ls_path and Path(ls_path).is_file():
        with open(ls_path, encoding="utf-8") as f:
            for it in json.load(f).get("items", []):
                excl.add((it["video_id"], int(it["track_id"])))
    else:
        print(f"WARNING: the older evaluation list was not found ({ls_path}); no track is "
              f"excluded from the heldout tier on its account.")
    return excl


def _clean_labelled_pool(labels_db, species, chk, already, ls_excl, tracks_db):
    """Candidate tracks of ``species`` for the heldout tier, in random order.

    Returns ``(pool, dropped)``: the usable tracks, and a count of the labelled tracks that
    were dropped, by reason.
    """
    conn = open_readonly(labels_db); conn.row_factory = sqlite3.Row
    # A track qualifies if ANY of its label rows names this species and that row is
    # undisputed or was settled by a senior reviewer. The query does not check whether
    # another row of the same track names a different species.
    rows = conn.execute(
        "SELECT DISTINCT video_id, track_id FROM labels WHERE final_label=? "
        "AND (disagreement_resolution IS NULL OR disagreement_resolution='senior')", (species,)).fetchall()
    conn.close()
    tconn = open_readonly(tracks_db)
    pool = []
    dropped = Counter()
    # NOT reproducible: hash() of a text value changes from one Python run to the next, and
    # the query above has no ORDER BY. See "Reproducibility" in the module docstring.
    order = list(rows); random.Random(hash(species) & 0xffff).shuffle(order)
    miss0, bad0 = chk.n_missing, chk.n_unreadable
    for r in order:
        key = (r["video_id"], int(r["track_id"]))
        if key in already:
            dropped["already in the test set"] += 1
            continue
        if key in ls_excl:
            dropped["in the older evaluation list"] += 1
            continue
        cp = tconn.execute("SELECT clip_path FROM tracks WHERE video_id=? AND track_id=?",
                           (key[0], key[1])).fetchone()
        if not (cp and cp[0]):
            # Without a rendered clip the track could not be looked at again later.
            dropped["no video clip in the earlier run"] += 1
            continue
        if chk.ok(key[0], key[1]):
            pool.append(key)
        else:
            dropped["fewer than 3 full-body frames"] += 1
    tconn.close()
    # How many of the "fewer than 3 full-body frames" were in fact tables that could not be used.
    if chk.n_missing - miss0:
        dropped["  of which: detection table missing"] = chk.n_missing - miss0
    if chk.n_unreadable - bad0:
        dropped["  of which: detection table unreadable"] = chk.n_unreadable - bad0
    return pool, dropped


def main() -> None:
    """Assemble the two tiers and write the answer key and the "do not train" list."""
    a = _parse_args()
    config.require(a.labels_db, "the labelling database (private; not distributed)",
                   "SALMONID_PROJECTS_ROOT")
    config.require(a.tracks_db, "the track database of the earlier run (run_20260428)",
                   "SALMONID_PROJECTS_ROOT")
    config.require(a.detections_dir, "the per-video detection tables of the earlier run "
                   "(run_20260428/detections)", "SALMONID_PROJECTS_ROOT")
    out_dir = Path(a.output_dir)
    refuse_overwrite([out_dir / "test_set_v6_gt.json", out_dir / "test_set_v6_holdout.json"],
                     a.overwrite)

    pristine = _load_pristine(a.labels_db, a.batch_id)
    ls_excl = _load_holdout_exclusions(a.op_testset, a.ls_test_queue)
    # Full-body check: box covers >= 2% of the frame, stays >= 12 px from every border,
    # in at least 3 frames (the same values as the queue step).
    chk = BigBoxCheck(a.detections_dir, 0.02, 12, 3)

    gt = []           # one entry per test track: video_id, track_id, species, tier, site
    used = set()
    for (vid, tid), sp in pristine.items():
        gt.append({"video_id": vid, "track_id": tid, "species": sp,
                   "tier": "pristine", "site": vid.split("/", 1)[0]})
        used.add((vid, tid))
    pri_counts = Counter(g["species"] for g in gt)
    print("pristine tier: " + ", ".join(f"{s} {pri_counts[s]}" for s in SPECIES))
    if not gt:
        print(f"WARNING: the labelling database holds no species label for batch '{a.batch_id}'; "
              f"the pristine tier is empty.")

    # Top up each rare species to the floor from the previously labelled pool.
    for sp in a.topup_species:
        need = a.floor - pri_counts.get(sp, 0)
        if need <= 0:
            print(f"  {sp}: pristine {pri_counts.get(sp,0)} >= floor {a.floor}, no top-up")
            continue
        pool, dropped = _clean_labelled_pool(a.labels_db, sp, chk, used, ls_excl, a.tracks_db)
        take = pool[:need]
        for (vid, tid) in take:
            gt.append({"video_id": vid, "track_id": tid, "species": sp,
                       "tier": "heldout", "site": vid.split("/", 1)[0]})
            used.add((vid, tid))
        print(f"  {sp}: +{len(take)} held-out (needed {need}, pool {len(pool)}) -> {pri_counts.get(sp,0)+len(take)}")
        for reason, n in dropped.items():
            print(f"      labelled {sp} tracks not usable: {n} ({reason})")
        if len(take) < need:
            print(f"  WARNING: {sp} stays below the floor of {a.floor}: the pool is too small.")

    random.Random(0).shuffle(gt)   # fixed seed: the order of the list is the same on every run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_set_v6_gt.json").write_text(json.dumps(
        {"name": "test_set_v6", "note": "FROZEN held-out test set. GT species per track; two tiers.",
         "tracks": gt}, indent=2), encoding="utf-8")
    # The same tracks in three shapes, so that every training-set builder can read the list:
    # by video, by (video, track) pair, and as a list of items.
    holdout = {"name": "test_set_v6_holdout",
               "note": "Exclude from ALL training (prepare_dataset/prepare_crops/_build_extra_coco/"
                       "_merge_synth_into_coco/_sam2_composite).",
               "video_ids": sorted({g["video_id"] for g in gt}),
               "tracks": [[g["video_id"], g["track_id"]] for g in gt],
               "items": [{"video_id": g["video_id"], "track_id": g["track_id"]} for g in gt]}
    (out_dir / "test_set_v6_holdout.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    print(f"\n=== test_set_v6 (new draw) ===  {len(gt)} tracks / {len(holdout['video_ids'])} videos")
    for tier in ("pristine", "heldout"):
        c = Counter(g["species"] for g in gt if g["tier"] == tier)
        print(f"  {tier:8s}: " + ", ".join(f"{s} {c[s]}" for s in SPECIES) + f"  (total {sum(c.values())})")
    tot = Counter(g["species"] for g in gt)
    print("  TOTAL   : " + ", ".join(f"{s} {tot[s]}" for s in SPECIES))
    print(f"  gt:      {out_dir/'test_set_v6_gt.json'}")
    print(f"  holdout: {out_dir/'test_set_v6_holdout.json'}")
    print("  NOTE: this is a new draw. It is not the published test set (data/test_set_v6_gt.json).")


def _parse_args():
    """Define and read the command-line options. Path defaults come from ``salmonid.config``."""
    p = argparse.ArgumentParser(
        description="Assemble the two-tier held-out test set (reviewer-labelled new tracks plus a "
                    "rare-species top-up from previously labelled tracks). Needs the private "
                    "labelling database; a re-run gives a new draw, not the published set.")
    p.add_argument("--labels-db", default=str(config.LABELS_DB), help="Labelling database.")
    p.add_argument("--tracks-db", default=str(config.MODEL_OUTPUTS / "run_20260428" / "tracks.sqlite"),
                   help="Track database of the earlier run.")
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"),
                   help="Per-video detection tables of the earlier run.")
    p.add_argument("--batch-id", default="test_set_v6",
                   help="Labelling batch that holds the pristine-tier labels.")
    p.add_argument("--floor", type=int, default=60,
                   help="Fewest tracks per topped-up species (pristine + heldout).")
    p.add_argument("--topup-species", nargs="+", default=["Coho", "Atlantic", "Brown Trout"],
                   help="Species topped up to the floor from the previously labelled pool.")
    p.add_argument("--op-testset", default=str(config.LABEL_BATCHES / "op_testset_v1.json"),
                   help="Older per-video evaluation list (accepted for compatibility; not used).")
    p.add_argument("--ls-test-queue", default=str(config.LABEL_BATCHES / "ls_test_set_v5.json"),
                   help="Older evaluation list whose tracks are kept out of the heldout tier.")
    p.add_argument("--output-dir", default=str(config.OUTPUT_DIR / "testset_v6_rebuild"),
                   help="Where to write the two JSON files (default: output/testset_v6_rebuild).")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing existing output files.")
    return p.parse_args()


if __name__ == "__main__":
    main()

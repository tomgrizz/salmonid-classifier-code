"""
In plain terms: this file assembles the fixed "exam paper" the model is graded on — a locked-in
set of 366 example fish, each with a species that a biologist has confirmed. It doesn't grade the
model itself; it just gathers those examples (balancing the rarer species so every kind is well
represented) and freezes them so the accuracy test is fair and repeatable. This frozen set is the
held-out evaluation set behind the paper's headline 0.966 species-accuracy figure.

Finalize the held-out test set (test_set_v6) into a FROZEN, publishable artifact.

Pipeline position: **eval / evaluation-set construction** (data -> detect -> track ->
*eval*/count -> results -> figures). This is a one-shot build step that turns human review
labels into the immutable ground-truth artifact the scorer (``salmonid.eval.score_testset_v6``)
consumes and the leak guard (``salmonid.holdout``) reads. Runs in the **base environment**
(sqlite/json only — no torch/DEIMv2), so it does not need the ``.venv_deim`` detector venv.

Two tiers, both excluded from v2.5+ training (via the holdout key):
  * PRISTINE  — the never-trained tracks Tom labelled in label_app (batch_id=test_set_v6). Honest on ANY
                model (incl. current deploy), but the pristine pool is Chinook/Rainbow-heavy so rare classes
                are thin (Atlantic ~7, Brown ~1).
  * HELDOUT   — a top-up of the rare classes (Coho/Atlantic/Brown) to a floor, drawn from ALREADY-VERIFIED
                labelled tracks (big-box clean, no review needed) and EXCLUDED from v2.5 so its numbers are
                honest for v2.5+. Standard rare-class train/test split — the "creative Atlantic" answer.

Inputs (all env-overridable via ``salmonid.config``):
  * ``config.LABELS_DB``    — human review labels (pristine tracks + verified rare-class pool).
  * a tracks/detections run under ``config.MODEL_OUTPUTS`` — supplies clip paths + big-box QC.
  * ``config.LABEL_BATCHES`` — the older eval-set manifests kept disjoint from this test set.

Outputs (written under ``--output-dir``, default ``config.LABEL_BATCHES``):
  test_set_v6_gt.json       — frozen GT: [{video_id, track_id, species, tier, site}] (the scorer reads THIS).
  test_set_v6_holdout.json  — exclusion key (video_ids + tracks + ls-test-style items) for EVERY prep path.

Manuscript claim supported: this frozen 366-track, class-balanced held-out set is the honest
eval behind the headline single-stage macro-F1 of **0.966** (v2.6, ``test_set_v6``).

  python -m salmonid.eval.finalize_testset_v6 --floor 60
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

from salmonid import config
from salmonid.eval.build_holdout_testset_queue import BigBoxCheck

# mirrors salmonid.config.SPECIES (the 5 report species; kept inline for faithfulness)
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]


def _load_pristine(labels_db, batch_id):
    """Return {(video_id, track_id): species} for the never-trained pristine tier.

    Args:
        labels_db: path to the human-review labels sqlite (``config.LABELS_DB``).
        batch_id: label_app batch tag for the pristine tracks (default "test_set_v6").
    Returns:
        dict keyed by (video_id, int track_id) -> final_label, restricted to the 5 report species.
    """
    conn = sqlite3.connect(labels_db)
    out = {}
    for vid, tid, lab in conn.execute(
        "SELECT video_id, track_id, final_label FROM labels WHERE batch_id=?", (batch_id,)):
        if lab in SPECIES:
            out[(vid, int(tid))] = lab
    conn.close()
    return out


def _load_holdout_exclusions(op_path, ls_path):
    """Keys we should NOT pull into the held-out tier (keep test sets disjoint from the old eval sets)."""
    excl = set()
    if ls_path and Path(ls_path).is_file():
        for it in json.loads(Path(ls_path).read_text()).get("items", []):
            excl.add((it["video_id"], int(it["track_id"])))
    # op_testset is video-level; skip (labelled-track keys are what matters for the labelled top-up)
    return excl


def _clean_labelled_pool(labels_db, species, chk, already, ls_excl, tracks_db):
    """Random-ordered clean (big-box) labelled tracks of `species` not already used, with a valid clip."""
    conn = sqlite3.connect(labels_db); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT video_id, track_id FROM labels WHERE final_label=? "
        "AND (disagreement_resolution IS NULL OR disagreement_resolution='senior')", (species,)).fetchall()
    conn.close()
    tconn = sqlite3.connect(tracks_db)
    pool = []
    order = list(rows); random.Random(hash(species) & 0xffff).shuffle(order)
    for r in order:
        key = (r["video_id"], int(r["track_id"]))
        if key in already or key in ls_excl:
            continue
        cp = tconn.execute("SELECT clip_path FROM tracks WHERE video_id=? AND track_id=?",
                           (key[0], key[1])).fetchone()
        if not (cp and cp[0]):          # need a clip for the frozen test set to be reviewable later
            continue
        if chk.ok(key[0], key[1]):
            pool.append(key)
    tconn.close()
    return pool


def main() -> None:
    """Build and write the two frozen artifacts (GT + holdout key).

    Loads the pristine tier, tops up each rare class from the verified big-box pool up to the
    per-class floor, shuffles with a fixed seed for a stable order, and writes both JSON files.
    Side effects: prints per-tier class tallies and writes test_set_v6_gt.json /
    test_set_v6_holdout.json under ``--output-dir``.
    """
    a = _parse_args()
    out_dir = Path(a.output_dir)
    pristine = _load_pristine(a.labels_db, a.batch_id)
    ls_excl = _load_holdout_exclusions(a.op_testset, a.ls_test_queue)
    chk = BigBoxCheck(a.detections_dir, 0.02, 12, 3)

    gt = []           # [{video_id, track_id, species, tier, site}]
    used = set()
    for (vid, tid), sp in pristine.items():
        gt.append({"video_id": vid, "track_id": tid, "species": sp,
                   "tier": "pristine", "site": vid.split("/", 1)[0]})
        used.add((vid, tid))
    pri_counts = Counter(g["species"] for g in gt)
    print("pristine tier: " + ", ".join(f"{s} {pri_counts[s]}" for s in SPECIES))

    # Held-out top-up for the rare classes below the floor.
    for sp in a.topup_species:
        need = a.floor - pri_counts.get(sp, 0)
        if need <= 0:
            print(f"  {sp}: pristine {pri_counts.get(sp,0)} >= floor {a.floor}, no top-up")
            continue
        pool = _clean_labelled_pool(a.labels_db, sp, chk, used, ls_excl, a.tracks_db)
        take = pool[:need]
        for (vid, tid) in take:
            gt.append({"video_id": vid, "track_id": tid, "species": sp,
                       "tier": "heldout", "site": vid.split("/", 1)[0]})
            used.add((vid, tid))
        print(f"  {sp}: +{len(take)} held-out (needed {need}, pool {len(pool)}) -> {pri_counts.get(sp,0)+len(take)}")

    random.Random(0).shuffle(gt)   # fixed seed -> deterministic track order in the frozen artifact
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_set_v6_gt.json").write_text(json.dumps(
        {"name": "test_set_v6", "note": "FROZEN held-out test set. GT species per track; two tiers.",
         "tracks": gt}, indent=2))
    holdout = {"name": "test_set_v6_holdout",
               "note": "Exclude from ALL training (prepare_dataset/prepare_crops/_build_extra_coco/"
                       "_merge_synth_into_coco/_sam2_composite).",
               "video_ids": sorted({g["video_id"] for g in gt}),
               "tracks": [[g["video_id"], g["track_id"]] for g in gt],
               "items": [{"video_id": g["video_id"], "track_id": g["track_id"]} for g in gt]}
    (out_dir / "test_set_v6_holdout.json").write_text(json.dumps(holdout, indent=2))

    print(f"\n=== FROZEN test_set_v6 ===  {len(gt)} tracks / {len(holdout['video_ids'])} videos")
    for tier in ("pristine", "heldout"):
        c = Counter(g["species"] for g in gt if g["tier"] == tier)
        print(f"  {tier:8s}: " + ", ".join(f"{s} {c[s]}" for s in SPECIES) + f"  (total {sum(c.values())})")
    tot = Counter(g["species"] for g in gt)
    print("  TOTAL   : " + ", ".join(f"{s} {tot[s]}" for s in SPECIES))
    print(f"  gt:      {out_dir/'test_set_v6_gt.json'}")
    print(f"  holdout: {out_dir/'test_set_v6_holdout.json'}")


def _parse_args():
    """Parse CLI args. Path defaults resolve via ``salmonid.config`` to the original
    workstation locations (argparse wants strings, so each is ``str(config.X)``)."""
    p = argparse.ArgumentParser(description="Finalize the frozen held-out test_set_v6")
    p.add_argument("--labels-db", default=str(config.LABELS_DB))
    p.add_argument("--tracks-db", default=str(config.MODEL_OUTPUTS / "run_20260428" / "tracks.sqlite"))
    p.add_argument("--detections-dir", default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"))
    p.add_argument("--batch-id", default="test_set_v6")
    p.add_argument("--floor", type=int, default=60, help="Minimum tracks per class (pristine + held-out).")
    p.add_argument("--topup-species", nargs="+", default=["Coho", "Atlantic", "Brown Trout"])
    p.add_argument("--op-testset", default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    p.add_argument("--ls-test-queue", default=str(config.LABEL_BATCHES / "ls_test_set_v5.json"))
    p.add_argument("--output-dir", default=str(config.LABEL_BATCHES))
    return p.parse_args()


if __name__ == "__main__":
    main()

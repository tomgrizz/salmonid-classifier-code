"""In plain terms: this script measures how often the model names a fish's species correctly.
It compares the model's guesses against the species a biologist assigned to each fish in a fixed,
never-trained-on set of test examples, and reports the accuracy. This is where the paper's headline
0.966 species score comes from, along with the table showing which species get mistaken for which
(the confusion matrix).

Held-out evaluation — THE headline metric of the manuscript.

Pipeline stage: results / eval  (data -> detect -> track -> **eval/count** -> results -> figures).
Environment: base Python (pure stdlib + salmonid.config). It consumes ALREADY-COMPUTED predictions,
so it never loads torch / DEIMv2 and does NOT need the .venv_deim detection env.

Inputs : the frozen ground truth (test_set_v6_gt.json / labels.sqlite) + one of the prediction
         sources below (a per-track CSV, a per-video CSV, or a run's tracks.sqlite).
Outputs: printed TRACK species F1 + confusion matrix + VIDEO presence F1 (FULL and PRISTINE-only).
Manuscript claim: the published single-stage DEIMv2-X model (v2.6) scores held-out **macro
         species-F1 0.966** on this test set — this script is what computes that number.

Score a model on the held-out test set (test_set_v6). Defines the GT + two metrics precisely so the number
is honest and reproducible:

  TRACK species F1 (primary)  — for each labelled test track, does the model name the BOXED fish correctly?
                                Uses Tom's per-track label as GT -> a real confusion matrix (Coho->Chinook,
                                Atlantic<->Rainbow, ...), per-species P/R/F1, macro, accuracy.
  VIDEO presence F1 (op-style) — per video, does the model's species SET match GT presence? GT presence =
                                (boxed species of the video's test tracks) UNION (Tom's background flags),
                                so multi-species/overlap clips are scored honestly. Same TP/FP/FN-per-species
                                formulation as scripts/_score_tracked.py (comparable to the 0.894 op number),
                                but with HUMAN GT instead of tech.

GT (from labels.sqlite, batch_id=test_set_v6, written by label_app during review):
  * final_label in the 5 species          -> track species GT (the boxed fish).
  * final_label 'Multiple fish'/'Not a fish' -> track EXCLUDED from species F1 (can't grade one species).
  * 'Unsure' is never written, Skip -> skipped_tasks -> both absent -> excluded.
  * background_observations(species)       -> other species present -> folded into video presence GT.

Predictions are model-agnostic:
  --preds        CSV video_id,track_id,pred_species   (per GT track; two-stage = classify the track's frames
                 e.g. via eval_clip_avg; single-stage = IoU-match detections to the GT track's boxes, vote)
  --video-preds  CSV video_id,species                 (the model's full-video species set, for presence F1)
  --from-tracks-db DB   convenience: read predicted_class per track from a run's tracks.sqlite (species F1
                 needs matching track_ids; presence unions predicted species per video — track_ids don't matter)

  python scripts/_score_testset_v6.py --preds v6_track_preds.csv --video-preds v6_video_species.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
CLASS_TO_SP = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_gt_file(gt_path: str, labels_db: str | None = None, batch_id: str = "test_set_v6"):
    """Preferred: read the FROZEN test set (test_set_v6_gt.json). Returns (track_gt, tiers, video_presence).

    Presence GT = each video's track species ∪ any background flags recorded during pristine review.
    """
    data = json.loads(Path(gt_path).read_text())
    track_gt, tiers, video_presence = {}, {}, defaultdict(set)
    for t in data["tracks"]:
        key = (t["video_id"], int(t["track_id"]))
        if t["species"] in SPECIES:
            track_gt[key] = t["species"]
            tiers[key] = t.get("tier", "pristine")
            video_presence[t["video_id"]].add(t["species"])
    if labels_db and Path(labels_db).is_file():
        conn = sqlite3.connect(labels_db)
        if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='background_observations'"
                        ).fetchone():
            gt_videos = {t["video_id"] for t in data["tracks"]}
            for vid, sp in conn.execute(
                "SELECT video_id, species FROM background_observations WHERE batch_id=?", (batch_id,)):
                if sp in SPECIES and vid in gt_videos:
                    video_presence[vid].add(sp)
        conn.close()
    return track_gt, tiers, dict(video_presence)


def load_gt(labels_db: str, batch_id: str):
    """Return (track_gt {(vid,tid):sp}, excluded {(vid,tid):reason}, video_presence {vid:set(sp)})."""
    conn = sqlite3.connect(labels_db)
    track_gt, excluded = {}, {}
    for vid, tid, lab, act in conn.execute(
        "SELECT video_id, track_id, final_label, label_action FROM labels WHERE batch_id=?", (batch_id,)):
        key = (vid, int(tid))
        if lab in SPECIES:
            track_gt[key] = lab
        elif lab in ("Multiple fish", "Not a fish"):
            excluded[key] = lab
    video_presence = defaultdict(set)
    for (vid, _), sp in track_gt.items():
        video_presence[vid].add(sp)
    # background flags -> presence (table may not exist if nothing was flagged)
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='background_observations'"
                    ).fetchone():
        for vid, sp in conn.execute(
            "SELECT video_id, species FROM background_observations WHERE batch_id=?", (batch_id,)):
            if sp in SPECIES:
                video_presence[vid].add(sp)
    conn.close()
    return track_gt, excluded, dict(video_presence)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def load_track_preds(path: str) -> dict:
    """Read a per-track prediction CSV -> {(video_id, track_id): species}.

    Accepts either a ``pred_species`` or ``species`` column; rows whose value is not one of the
    five graded species are dropped. Used for the TRACK species-F1 metric.
    """
    preds = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r.get("pred_species") or r.get("species")
            if sp in SPECIES:
                preds[(r["video_id"], int(r["track_id"]))] = sp
    return preds


def load_video_preds(path: str) -> dict:
    """Read a per-video prediction CSV -> {video_id: set(species)} for the presence-F1 metric.

    Accepts either a ``species`` or ``pred_species`` column; one row per (video, species) predicted
    present. Non-graded species are dropped.
    """
    vp = defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp = r.get("species") or r.get("pred_species")
            if sp in SPECIES:
                vp[r["video_id"]].add(sp)
    return dict(vp)


def preds_from_tracks_db(db: str, gt_videos: set):
    """Per-track predicted species (for species F1, by matching track_id) + per-video species set (presence)."""
    conn = sqlite3.connect(db)
    track_preds, video_preds = {}, defaultdict(set)
    for vid, tid, pc in conn.execute("SELECT video_id, track_id, predicted_class FROM tracks"):
        if vid not in gt_videos:
            continue
        sp = CLASS_TO_SP.get(pc)
        if sp:
            track_preds[(vid, int(tid))] = sp
            video_preds[vid].add(sp)
    conn.close()
    return track_preds, dict(video_preds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def species_f1(track_gt: dict, track_preds: dict):
    """Per-track species classification score over tracks present in BOTH gt and preds.

    Args:
        track_gt:    {(video_id, track_id): true_species}.
        track_preds: {(video_id, track_id): predicted_species}.
    Returns:
        (macro_f1, accuracy, per-species rows [(sp, tp, fp, fn, prec, rec, f1)], confusion matrix
        cm[true][pred], n_scored). Only keys shared by both dicts are graded.
    """
    common = [k for k in track_gt if k in track_preds]
    cm = {g: Counter() for g in SPECIES}          # cm[true][pred]
    for k in common:
        cm[track_gt[k]][track_preds[k]] += 1
    tp = Counter(); fp = Counter(); fn = Counter()
    for g in SPECIES:
        for p, n in cm[g].items():
            if p == g:
                tp[g] += n
            else:
                fn[g] += n; fp[p] += n
    rows, f1s = [], []
    for sp in SPECIES:
        prec = tp[sp] / max(tp[sp] + fp[sp], 1)
        rec = tp[sp] / max(tp[sp] + fn[sp], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1); rows.append((sp, tp[sp], fp[sp], fn[sp], prec, rec, f1))
    acc = sum(tp.values()) / max(len(common), 1)
    return (sum(f1s) / len(f1s)), acc, rows, cm, len(common)


def presence_f1(video_presence_gt: dict, video_preds: dict):
    """Op-style per-video species-presence score (same TP/FP/FN formulation as _score_tracked.py).

    For each video, compare the predicted species SET against the GT presence set (boxed species
    UNION background flags). Returns (macro_f1, per-species rows). Comparable to the 0.894 tracked-op
    number but graded against HUMAN GT rather than technician labels.
    """
    tp = Counter(); fp = Counter(); fn = Counter()
    for vid, gt_set in video_presence_gt.items():
        pred_set = video_preds.get(vid, set())
        for sp in SPECIES:
            if sp in pred_set and sp in gt_set:
                tp[sp] += 1
            elif sp in pred_set:
                fp[sp] += 1
            elif sp in gt_set:
                fn[sp] += 1
    rows, f1s = [], []
    for sp in SPECIES:
        prec = tp[sp] / max(tp[sp] + fp[sp], 1)
        rec = tp[sp] / max(tp[sp] + fn[sp], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1); rows.append((sp, tp[sp], fp[sp], fn[sp], prec, rec, f1))
    return (sum(f1s) / len(f1s)), rows


def _print_prf(title, rows, macro, extra=""):
    """Pretty-print a per-species precision/recall/F1 table with the macro-F1 footer (side effect: stdout)."""
    print(f"\n== {title} ==  {extra}")
    print(f"{'species':14s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for sp, t, f, n, p, r, f1 in rows:
        print(f"{sp:14s} {t:>4d} {f:>4d} {n:>4d} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    print(f"{'MACRO F1':14s} {macro:>46.3f}")


def _print_cm(cm):
    """Pretty-print the species confusion matrix (rows=true, cols=pred; side effect: stdout)."""
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(f"{'':14s}" + "".join(f"{s[:5]:>7s}" for s in SPECIES))
    for g in SPECIES:
        print(f"{g:14s}" + "".join(f"{cm[g][p]:>7d}" for p in SPECIES))


def main() -> None:
    """CLI entry point: load GT + predictions, then print the species-F1, confusion matrix and
    presence-F1 report. GT comes from the frozen JSON (``--gt-file``) when present, else a
    labels.sqlite batch scan; predictions come from ``--from-tracks-db`` / ``--preds`` /
    ``--video-preds`` (later flags override earlier). Prints to stdout; returns nothing."""
    a = _parse_args()
    tiers = {}
    if a.gt_file:
        track_gt, tiers, video_presence = load_gt_file(a.gt_file, a.labels_db, a.batch_id)
        print(f"GT (frozen {Path(a.gt_file).name}): {len(track_gt)} tracks across {len(video_presence)} videos")
        print("  by tier: " + ", ".join(f"{t} {c}" for t, c in Counter(tiers.values()).most_common()))
    else:
        track_gt, excluded, video_presence = load_gt(a.labels_db, a.batch_id)
        print(f"GT: {len(track_gt)} labelled test tracks across {len(video_presence)} videos "
              f"({len(excluded)} excluded: {dict(Counter(excluded.values()))})")
    print("  GT species: " + ", ".join(f"{s} {c}" for s, c in Counter(track_gt.values()).most_common()))
    if not track_gt:
        print("No test_set_v6 GT — finalize/label first."); return

    gt_videos = set(video_presence)
    track_preds, video_preds = {}, {}
    if a.from_tracks_db:
        track_preds, video_preds = preds_from_tracks_db(a.from_tracks_db, gt_videos)
    if a.preds:
        track_preds = load_track_preds(a.preds)
    if a.video_preds:
        video_preds = load_video_preds(a.video_preds)

    if track_preds:
        macro, acc, rows, cm, n = species_f1(track_gt, track_preds)
        _print_prf("TRACK species F1 (boxed fish, FULL set)", rows, macro, f"n={n} scored, acc={acc:.3f}")
        _print_cm(cm)
        # Pristine-only subset — honest on ANY model (incl. baselines that trained on the held-out tier).
        if tiers:
            pri_gt = {k: v for k, v in track_gt.items() if tiers.get(k) == "pristine"}
            pmacro, pacc, prows, _, pn = species_f1(pri_gt, track_preds)
            _print_prf("TRACK species F1 (PRISTINE tier only — cross-model honest)", prows, pmacro,
                       f"n={pn} scored, acc={pacc:.3f}")
    else:
        print("\n(no per-track preds given -> skipping species F1)")
    if video_preds:
        pmacro, prows = presence_f1(video_presence, video_preds)
        _print_prf("VIDEO presence F1 (species set per video, GT = boxed + flags)", prows, pmacro)
    else:
        print("(no per-video preds given -> skipping presence F1)")


def _parse_args():
    """Define and parse the command-line arguments (GT source + prediction sources)."""
    p = argparse.ArgumentParser(description="Score a model on test_set_v6 (species F1 + presence F1)")
    p.add_argument("--gt-file", default=str(config.TESTSET_V6_GT),
                   help="Frozen test GT (test_set_v6_gt.json). If absent, falls back to labels.sqlite batch scan.")
    p.add_argument("--labels-db", default=str(config.LABELS_DB))
    p.add_argument("--batch-id", default="test_set_v6")
    p.add_argument("--preds", default=None, help="CSV video_id,track_id,pred_species (track species F1).")
    p.add_argument("--video-preds", default=None, help="CSV video_id,species (presence F1).")
    p.add_argument("--from-tracks-db", default=None,
                   help="Score a run's tracks.sqlite predicted_class directly (species F1 needs matching "
                        "track_ids; presence unions per video).")
    return p.parse_args()


if __name__ == "__main__":
    main()

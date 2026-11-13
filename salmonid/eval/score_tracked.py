"""In plain terms: this is a development tool for trying out settings. Once the model has
been run on a set of videos and its fish have been followed from frame to frame, there are
still choices to make about how to turn each followed fish (a "track") into one species
call: how many frames a track needs, whether to ignore frames where the fish touches the
edge of the picture, and whether to trust the average of all frames or the single clearest
one. This script re-scores the same stored tracks under different choices, in seconds and
without a GPU. For each video it asks a simple question: do the species the model reports
match the species the technician recorded for that video? It was used while earlier models
were being developed. It was not run for the published model (v2.7), and none of its numbers
appear in the manuscript.

Inputs
------
All are Tier-2 files that are not stored in git.

* ``--agg``: a track file written by ``salmonid.detect.track_from_store``. It holds, per
  video, the frame size, the river, the set of species the technician recorded
  (``tech_set``) and, for each track, its boxes and its per-frame species scores.
  ``tech_set`` is filled only when the video list given to the tracking step carried the
  technicians' species per video. The season-long count files were built from lists
  without them, so their ``tech_set`` is empty and they cannot be scored here.
* ``--keep-keys`` / ``--drop-keys`` (optional): text files with one video key per line, to
  score only some videos or to leave some out. During development these separated a tuning
  half from a confirmation half of the videos, and removed videos that had contributed
  training images. The key files are not part of this deposit.

Outputs
-------
Printed only; nothing is written to disk. For each setting: per species the number of
videos scored as hit, false alarm and miss, with precision, recall and F1; the macro-F1; and
a rough upstream-minus-downstream tally per species.

What "presence F1" means
------------------------
Each video is scored once per species. If the model reports a species in the video and the
technician recorded it, that is a hit (TP). Model yes, technician no: a false alarm (FP).
Technician yes, model no: a miss (FN). Precision is hits / (hits + false alarms); recall is
hits / (hits + misses); F1 combines the two (their harmonic mean); macro-F1 is the plain
average of the five species' F1 values. The number of fish does not enter: a video with one
Chinook Salmon and a video with ten both count as "Chinook Salmon present". The score is
therefore blind to under- or over-counting, which is why the manuscript validates counts
separately against the technicians' daily totals.

What it does NOT tell you
-------------------------
* Nothing about counts or direction (see above). The "net upstream" line it prints uses one
  distance threshold for both directions (``--min-disp-px``, default 300 px). The published
  counting rule is different (50 px upstream, 300 px downstream; ``docs/COUNTING_METHOD.md``),
  so that line is a rough diagnostic and is not a published count.
* A species whose only tracks have all-zero scores is not reported for that video (such
  tracks are skipped; they are not assigned to the first species in the list).
* ``--sweep`` picks the best of six settings on the videos it is given. The best value is
  optimistic for those same videos; it should be confirmed on other videos.

Result for the published model
------------------------------
None. The published checkpoint (v2.7, epoch 17) was chosen on the held-out species score
(``salmonid.eval.score_testset_v6``), and no track file with technician species was built
for it.

Behaviour changes relative to the original research script
-----------------------------------------------------------
The scoring is unchanged.

* A ``--keep-keys`` or ``--drop-keys`` file that does not exist was silently ignored, so a
  run meant to exclude videos would score all of them without saying so. It now stops with
  a message.
* If no video is left to score, or no video carries technician species, the script stops
  with exit code 1 instead of printing a table that looks like a result.
* The ``--sweep`` summary no longer prints three fixed "baseline" numbers that belonged to
  other, earlier models.

Feeds
-----
Development check; no manuscript number.

Run
---
::

    python -m salmonid.eval.score_tracked --agg <track file.pkl> \\
        --min-frames 5 --edge-margin-px 8 --vote-mode best-frame
    python -m salmonid.eval.score_tracked --agg <track file.pkl> --sweep

Base Python with numpy; no GPU.

Tier
----
2 (needs a track file with technician species; none is stored in git).

Status
------
Supporting check (development tool; not run for the published model).
"""
import argparse, pickle
from pathlib import Path
import numpy as np

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # same order as salmonid.config.SPECIES
# Which way is upstream on screen at each river: fish swim left-to-right ("Right") to go
# upstream at the Ganaraska River and right-to-left at the Credit River. Same convention as
# salmonid.config.UP_SIGN.
UP_DIR = {"Ganaraska": "Right", "Credit": "Left"}


def score(agg, min_frames, edge_px, vote_mode, min_disp):
    """Score all videos of a track file under one group of settings.

    Args:
        agg: ``{video_key: video}`` as loaded from the ``--agg`` file. Each video has ``fw``,
            ``fh`` (frame width and height), ``site``, ``tech_set`` (species the technician
            recorded) and ``tracks`` = ``{track_id: {"boxes": N x 4, "pvs": N x 5}}`` where
            ``pvs`` holds the five species scores of each frame.
        min_frames: tracks with fewer detections than this are ignored (very short tracks
            are often debris or a fish seen in passing).
        edge_px: frames in which the box comes within this many pixels of any border do not
            vote on the species, because a fish that is entering or leaving is only partly
            visible. 0 switches this off.
        vote_mode: ``"best-frame"`` uses the scores of the frame with the largest box (the
            fish at its closest and most complete); ``"mean"`` adds the scores of all voting
            frames.
        min_disp: a track counts as moving up or down only if its centre moved at least this
            many pixels horizontally; otherwise it counts in neither direction.

    Returns:
        ``(macro_f1, rows, net)``: ``rows`` is a list of (species, TP, FP, FN, precision,
        recall, F1) and ``net`` maps species to upstream minus downstream tracks.
    """
    import collections
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    m_up = collections.Counter(); m_dn = collections.Counter()
    for vk, vd in agg.items():
        fw, fh, site = vd["fw"], vd["fh"], vd["site"]
        tech_set = set(vd["tech_set"]); model_set = set()
        for tid, t in vd["tracks"].items():
            boxes, pvs = t["boxes"], t["pvs"]
            n = len(boxes)
            if n < min_frames:
                continue
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0     # horizontal centre of each box
            # Frames near the border keep their place in the track but do not vote.
            if edge_px > 0:
                keep = ~((boxes[:, 0] <= edge_px) | (boxes[:, 1] <= edge_px)
                         | (boxes[:, 2] >= fw - edge_px) | (boxes[:, 3] >= fh - edge_px))
            else:
                keep = np.ones(n, bool)
            vpvs = pvs[keep]
            if vote_mode == "best-frame":
                vb = boxes[keep]
                if len(vb):
                    # Largest box = fish closest to the camera and most fully in view.
                    areas = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
                    vote = vpvs[int(np.argmax(areas))]
                else:
                    vote = np.zeros(5, np.float32)
            else:
                vote = vpvs.sum(0) if len(vpvs) else np.zeros(5, np.float32)
            if vote.sum() == 0:                      # every frame was near the border:
                vote = pvs.sum(0)                    # fall back to all frames
            if vote.sum() == 0:
                # No species score at all. Skip the track; without this line the "largest
                # score" below would be a tie that silently resolves to Chinook Salmon.
                continue
            sp = int(np.argmax(vote)); model_set.add(SPECIES[sp])
            w = min(3, n)
            # Direction: mean centre of the last three boxes minus that of the first three.
            dx = float(np.mean(cx[-w:]) - np.mean(cx[:w]))
            if abs(dx) >= min_disp:                   # moved far enough to count as a passage
                d = "Right" if dx > 0 else "Left"
                (m_up if d == UP_DIR.get(site) else m_dn)[sp] += 1
        # One comparison per video and species: is the species present on both sides?
        for sp in SPECIES:
            if sp in model_set and sp in tech_set: tp[sp] += 1      # hit
            elif sp in model_set: fp[sp] += 1                       # false alarm
            elif sp in tech_set: fn[sp] += 1                        # miss
    f1s = []
    rows = []
    for sp in SPECIES:
        # max(..., 1) avoids dividing by zero for a species that never occurs; its
        # precision, recall and F1 are then 0 and still enter the macro average.
        p = tp[sp] / max(tp[sp] + fp[sp], 1); r = tp[sp] / max(tp[sp] + fn[sp], 1)
        f1 = 2 * p * r / max(p + r, 1e-9); f1s.append(f1)
        rows.append((sp, tp[sp], fp[sp], fn[sp], p, r, f1))
    return float(np.mean(f1s)), rows, {SPECIES[s]: m_up[s] - m_dn[s] for s in range(5)}


def report(agg, min_frames, edge_px, vote_mode, min_disp):
    """Score one group of settings, print its table, and return the macro-F1."""
    macro, rows, net = score(agg, min_frames, edge_px, vote_mode, min_disp)
    print(f"\n== min_frames={min_frames} edge={edge_px} vote={vote_mode} min_disp={min_disp} ==")
    print(f"{'species':14s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for sp, t, f, n, p, r, f1 in rows:
        print(f"{sp:14s} {t:>4d} {f:>4d} {n:>4d} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    print(f"{'MACRO F1':14s} {macro:>46.3f}")
    print("net upstream (one symmetric threshold; not the published counting rule):", net)
    return macro


def _read_keys(path, option):
    """Read a one-key-per-line file; stop with a clear message if it does not exist."""
    p = config.require(path, f"the video key list given to {option}")
    with open(p, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def main():
    """Load the track file, apply the optional key lists, then score one setting or sweep six."""
    ap = argparse.ArgumentParser(
        description="Re-score stored tracks against the technicians' per-video species under "
                    "different settings (development tool; no manuscript number).")
    ap.add_argument("--agg", required=True,
                    help="track file from salmonid.detect.track_from_store (must carry tech_set)")
    ap.add_argument("--min-frames", type=int, default=5, help="ignore tracks shorter than this")
    ap.add_argument("--edge-margin-px", type=int, default=0,
                    help="frames with a box within this many px of a border do not vote (0 = off)")
    ap.add_argument("--vote-mode", choices=["mean", "best-frame"], default="mean",
                    help="species from all voting frames (mean) or from the largest box (best-frame)")
    ap.add_argument("--min-disp-px", type=float, default=300.0,
                    help="horizontal movement needed for the rough up/down tally (default 300)")
    ap.add_argument("--sweep", action="store_true",
                    help="try edge margins 0, 8, 16 px with both vote modes and name the best")
    ap.add_argument("--drop-keys", default=None,
                    help="file of video keys to LEAVE OUT (one per line), e.g. videos that "
                         "contributed training images")
    ap.add_argument("--keep-keys", default=None,
                    help="file of video keys to score ONLY (one per line), e.g. the tuning half "
                         "or the confirmation half of the videos")
    args = ap.parse_args()
    agg_path = config.require(args.agg, "the track file to score (--agg)", "SALMONID_PROJECTS_ROOT")
    with open(agg_path, "rb") as f:
        agg = pickle.load(f)
    print(f"loaded {len(agg)} videos from {Path(args.agg).name}")
    if args.keep_keys:
        keep = _read_keys(args.keep_keys, "--keep-keys")
        before = len(agg)
        agg = {k: v for k, v in agg.items() if k in keep}
        print(f"restricted to {len(agg)} videos in {Path(args.keep_keys).name} (from {before})")
    if args.drop_keys:
        drop = _read_keys(args.drop_keys, "--drop-keys")
        before = len(agg)
        agg = {k: v for k, v in agg.items() if k not in drop}
        print(f"dropped {before - len(agg)} videos listed in {Path(args.drop_keys).name} "
              f"-> {len(agg)} remain")
    if not agg:
        raise SystemExit("Nothing to score: no video is left after applying the key lists "
                         "(or the track file is empty).")
    # Without technician species there is nothing to compare with: every species the model
    # reports would be scored as a false alarm and the table would look like a real result.
    if not any(v.get("tech_set") for v in agg.values()):
        raise SystemExit("Nothing to score against: no video in this track file carries the "
                         "technicians' species (tech_set is empty). Use a track file built from "
                         "a video list that includes them.")
    if args.sweep:
        best = (None, -1)
        for edge in (0, 8, 16):
            for vm in ("mean", "best-frame"):
                m = report(agg, args.min_frames, edge, vm, args.min_disp_px)
                if m > best[1]:
                    best = ((edge, vm), m)
        print(f"\nBEST: edge={best[0][0]} vote={best[0][1]} -> macro {best[1]:.3f}  "
              f"(best of six settings on these videos; confirm on videos not used to choose it)")
    else:
        report(agg, args.min_frames, args.edge_margin_px, args.vote_mode, args.min_disp_px)


if __name__ == "__main__":
    main()

"""In plain terms: the manuscript checks the model against a person who re-watched a sample
of 2025 video clips without seeing the model's answers (the "blind review"). This script
chose which clips that person was given. It did not pick clips at random from the whole
season, because nearly all clips are single Chinook Salmon or Rainbow Trout on quiet days.
Instead it sorted the clips into groups by river, by time of season and by whether the clip
was crowded, drew clips from every group, and then added extra clips containing the rarest
species. It was run on the tracks of the earlier model (v2.6) and produced a list of 206
clips, of which the reviewer completed 155. The draw uses a fixed random seed; whether a
re-run returns the same 206 clips depends on having the same files (see "Reproducibility").

What it does, step by step
--------------------------
1. **The pool.** Every 2025 clip in which the model has at least one countable track (a
   track with at least 3 detections). Clips in which the model found no fish are not in the
   pool, so the blind review cannot reveal fish in clips the model missed entirely.
2. **Groups (strata).** Each clip gets three tags: the river; the run phase (early, peak or
   late, obtained by cutting that river's list of dates with countable clips into thirds);
   and crowding ("multi" if the model's track diagnostics show two or more fish in view at
   once in that clip, otherwise "single"). That gives 2 x 3 x 2 = 12 groups.
3. **Draw.** From each group, a number of clips proportional to the group's share of the
   pool, aiming at ``--target`` (180) clips in total, but never fewer than 6 per group.
4. **Rare-species top-up.** For Coho Salmon, Atlantic Salmon and Brown Trout, if fewer than
   ``--rare-min`` (10) drawn clips contain the species according to the model, more clips
   containing it are drawn from the whole pool until there are 10 or none are left.
5. **Output.** One entry per drawn clip with the model's tracks (species, direction,
   confidence), in the layout read by the review tool. The review tool is not part of this
   deposit; according to the manuscript the reviewer was not shown the model's species.

The list that was actually reviewed has 206 clips: 194 from step 3 and 12 from step 4 (10
for Atlantic Salmon, 2 for Brown Trout).

Limitations a reader should know
--------------------------------
* **The ``weight`` field is not a valid sampling weight.** It is written as "clips in the
  group divided by clips drawn from the group". That would scale a group's sample up to
  the season if every clip in a group had the same chance of being drawn. The top-up in
  step 4 breaks this: a clip containing a rare species had a higher chance than its
  neighbours. The scorer does not use the field, and the manuscript's blind-review figures
  are plain (unweighted) percentages over the reviewed fish.
* **The sample is tilted toward rare species on purpose**, so its overall agreement rate is
  not an estimate of the season-wide rate.
* **Top-up clips are chosen by the model's own call.** A clip is added for Atlantic Salmon
  because the model reported an Atlantic Salmon in it. In the reviewed sample all six fish
  the reviewer identified as Atlantic Salmon are in top-up clips, so the agreement on them
  measures how often a reported Atlantic Salmon was confirmed, not how many Atlantic Salmon
  the model finds.
* **Not every drawn clip was reviewed.** 155 of the 206 were. Of the 51 that were not, 48
  are late-phase Ganaraska River clips (62 drawn there, 14 reviewed).
* **Tags come from the model.** Species content and crowding are the earlier model's, not a
  person's; the phase comes from the dates of the video files.
* **Clip dates** are the files' last-modified dates. The footage must be mounted to read
  them, and a clip whose file is not found is left out of the pool (now counted and
  reported).

Reproducibility
---------------
The draw is seeded (``--seed``, default 20260713). On the authors' machine a re-run with
the default settings on 2026-11-10 returned exactly the 206 clips of the reviewed sample.
Two things outside the seed also decide the result, so the same is not guaranteed elsewhere:

* the order in which candidate clips enter the draw follows the order in which the
  operating system lists the track files (the list is not sorted by the script; Windows
  lists them alphabetically);
* a clip is in the pool only if its video file is found, and its phase depends on that
  file's last-modified date, which copying footage to another disk can change.

The script was left as it was run. The clips that were reviewed are listed in
``data/paper/blind_v27_clips.csv``. The default output folder is ``output/`` and an
existing file is not overwritten unless ``--overwrite`` is given, so a re-run cannot
replace the reviewed list.

Inputs
------
All Tier 2 (not stored in git):

* ``--manifest``: the 2025 video list for both rivers
  (``config.LABEL_BATCHES / "corpus_2025_both.json"``).
* ``--corpus-dir`` (default ``config.BLIND_REVIEW_DIR``, the earlier model's 2025 run):
  the track files ``agg_*shard*.pkl`` (tracks from the BoT-SORT tracker, i.e. the tracks
  the counts are made from) and the per-track diagnostics ``credit_track_diag.csv`` and
  ``gan_track_diag.csv`` (used only for the crowding tag).
* The 2025 footage under ``config.FOOTAGE_ROOT`` (for the clip dates).

Outputs
-------
``--out`` (default ``output/blind_review/reference_sample_2025.json``): one entry per drawn
clip: video, river, date, group, ``weight`` (see the limitation above), which rare species
it was added for, the model's tracks and their up/down/stay tally.

Behaviour changes relative to the original research script
-----------------------------------------------------------
The sampling itself (pool, groups, numbers drawn, random draws) is unchanged.

* A missing diagnostics file was skipped silently, which would have tagged every clip of
  that river "single". Missing inputs now stop the run with one sentence naming the file.
* Clips left out because their video file was not found, and diagnostics rows with an
  unreadable fish count, were dropped or defaulted silently; both are now counted and
  printed. An empty pool stops the run.
* The input folder is ``config.BLIND_REVIEW_DIR`` instead of a hard-coded folder name, and
  the output goes to ``output/`` instead of into the run folder (where the reviewed list
  is kept); existing output is not overwritten without ``--overwrite``.

Feeds
-----
The clip sample behind the blind review in manuscript Sections 2.4, 3.2 and 4.4. The
published figures (196 of 208 paired fish, 94.2%; kappa 0.90; 150 clips with at least one
paired fish) were computed for the published model (v2.7) from the reviewer's record by the
blind-review recomputation in ``salmonid/paper/``, not by this script.

Run
---
::

    python -m salmonid.results.build_reference_sample --target 180 --seed 20260713

Base Python with numpy; no GPU.

Tier
----
2 (needs the earlier run's track files, the 2025 video list and the 2025 footage).

Status
------
Supporting (the record of how the blind-review clips were drawn, on the earlier model's
tracks).
"""
import argparse, csv, glob, hashlib, pickle, collections, json, random
from pathlib import Path
import numpy as np

from salmonid import config
from salmonid.count.corpus_countr_from_store import UP_SIGN, FRAME_W, SPECIES  # noqa

# Folder of the run the sample is drawn from. The published sample was drawn from the earlier
# (v2.6) 2025 run, which is what config.BLIND_REVIEW_DIR points at by default.
CORP = config.BLIND_REVIEW_DIR
MANIFEST = config.LABEL_BATCHES / "corpus_2025_both.json"
# The counting rule (docs/COUNTING_METHOD.md): a track counts as an UPSTREAM passage if its
# centre moved at least 50 px in the upstream direction, and as a DOWNSTREAM passage only if
# it moved at least 300 px the other way (the stricter downstream limit keeps fish that mill
# about in front of the camera from being counted). A track needs at least 3 detections.
# Same values as config.NET_UP_THRESH / NET_DOWN_THRESH / MIN_FRAMES.
UP_THR, DOWN_THR, MIN_FRAMES = 50.0, 300.0, 3
DIRMAP = {1: "up", -1: "down", 0: "stay"}


def store_key(site, rel):
    """Return the name under which a clip is filed in the track files.

    ``<river>__<first 12 characters of the MD5 fingerprint of the clip's path>.pkl``, the
    same rule the detection step used, so a clip in the video list can be matched to its
    tracks.
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def track_dir(boxes, site):
    """Classify a track as ``"up"``, ``"down"`` or ``"stay"`` from where its boxes moved.

    The horizontal movement is the mean box centre of the last three frames minus that of
    the first three. Upstream is left-to-right at the Ganaraska River and right-to-left at
    the Credit River (``UP_SIGN``). The 50 px / 300 px limits are explained above.

    Args:
        boxes: the track's boxes, one ``[x1, y1, x2, y2]`` per frame.
        site:  ``"Credit"`` or ``"Ganaraska"``.
    """
    b = np.asarray(boxes, float); cx = (b[:, 0] + b[:, 2]) / 2.0
    w = min(3, len(cx)); dx = float(cx[-w:].mean() - cx[:w].mean())
    sgn = dx * UP_SIGN[site]
    nc = 1 if (sgn > 0 and abs(dx) >= UP_THR) else (-1 if (sgn < 0 and abs(dx) >= DOWN_THR) else 0)
    return DIRMAP[nc]


def load_crowd(corp):
    """Return ``{video_id: largest number of fish in view at once}`` from the track diagnostics.

    The diagnostics have one row per track with a ``max_cooccur`` column; the largest value
    over a clip's tracks is the clip's crowding. Clips that do not appear count as 1.
    """
    crowd = {}
    n_bad = 0
    for name in ("credit_track_diag.csv", "gan_track_diag.csv"):
        fn = config.require(Path(corp) / name, f"the per-track diagnostics of the sampled run ({name})",
                            "SALMONID_BLIND_REVIEW_DIR")
        with open(fn, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    mc = int(float(r.get("max_cooccur") or 1))
                except ValueError:
                    mc = 1                # unreadable value: treated as a single fish
                    n_bad += 1
                vid = r["video_id"]
                crowd[vid] = max(crowd.get(vid, 1), mc)
    if n_bad:
        print(f"NOTE: {n_bad} diagnostics rows had an unreadable max_cooccur and were taken as 1.")
    return crowd


def main():
    """Build the pool, tag and draw the clips, and write the sample file."""
    ap = argparse.ArgumentParser(
        description="Draw the stratified clip sample for the blind review from the earlier "
                    "model's 2025 tracks. Output goes to output/, never over the reviewed sample.")
    ap.add_argument("--target", type=int, default=180, help="clips aimed at before the rare-species top-up")
    ap.add_argument("--rare-min", type=int, default=10, help="fewest drawn clips CONTAINING each rare species")
    ap.add_argument("--seed", type=int, default=20260713,
                    help="random seed of the draw (the result also depends on the input files; "
                         "see the docstring)")
    ap.add_argument("--corpus-dir", default=str(CORP),
                    help="run folder with agg_*shard*.pkl and the track diagnostics "
                         "(default: config.BLIND_REVIEW_DIR)")
    ap.add_argument("--manifest", default=str(MANIFEST), help="the 2025 video list for both rivers")
    ap.add_argument("--agg-glob", default=None,
                    help="track files to read (default: <corpus-dir>/agg_*shard*.pkl)")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "blind_review" / "reference_sample_2025.json"),
                    help="output file (default: output/blind_review/reference_sample_2025.json)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    corp = Path(a.corpus_dir)
    agg_glob = a.agg_glob or f"{corp}/agg_*shard*.pkl"

    config.require(a.manifest, "the 2025 video list (corpus_2025_both.json)", "SALMONID_PROJECTS_ROOT")
    config.require(corp, "the run folder the sample is drawn from", "SALMONID_BLIND_REVIEW_DIR")
    if Path(a.out).exists() and not a.overwrite:
        raise SystemExit(f"Refusing to overwrite the existing file {a.out}. Choose another "
                         f"--out, or pass --overwrite.")

    with open(a.manifest, encoding="utf-8") as f:
        vids = json.load(f)["videos"]
    vid2rel = {v.get("video_id"): (v["site"], v["video_rel"].replace("\\", "/")) for v in vids if v.get("video_id")}
    rel2vid = {store_key(s, rel): vid for vid, (s, rel) in vid2rel.items()}
    crowd = load_crowd(corp)

    # ---- the pool: every clip with at least one countable track ----
    clips = []
    import os
    from salmonid.count.corpus_countr_from_store import _resolve
    import datetime
    agg_files = glob.glob(agg_glob)       # in the order the operating system lists them (not sorted)
    if not agg_files:
        raise SystemExit(f"Nothing to sample from: no track file matches {agg_glob}")
    n_not_in_manifest = n_no_video = 0
    for gf in agg_files:
        with open(gf, "rb") as fh:
            agg = pickle.load(fh)
        for k, rec in agg.items():
            vid = rel2vid.get(k)
            if vid is None:
                n_not_in_manifest += 1
                continue
            site = vid.split("/", 1)[0]
            vp = _resolve(vid)
            if not vp:
                n_no_video += 1           # video file not found: no date, so it cannot be placed in a phase
                continue
            # The clip's date is the file's last-modified date.
            date = datetime.date.fromtimestamp(os.path.getmtime(vp)).isoformat()
            tracks = []
            for t in rec.get("tracks", {}).values():
                if len(t["boxes"]) < MIN_FRAMES:
                    continue
                # Species of a track: average its per-frame scores and take the highest of the
                # five reported species ([:5] ignores a sixth "Small Salmonid" column if present).
                pvs = np.asarray(t["pvs"], float); sp = int(np.argmax(pvs.mean(0)[:5]))
                tracks.append({"sp": SPECIES[sp], "dir": track_dir(t["boxes"], site),
                               "conf": round(float(pvs.mean(0)[sp]), 2)})
            if not tracks:                          # the pool holds only clips with a countable fish
                continue
            clips.append({"video_id": vid, "site": site, "date": date, "tracks": tracks,
                          "n_fish": len(tracks), "sp_set": sorted({t["sp"] for t in tracks}),
                          "crowd": "multi" if crowd.get(vid, 1) >= 2 else "single"})
    print(f"counted clips available: {len(clips)}")
    if n_not_in_manifest:
        print(f"  left out: {n_not_in_manifest} clips in the track files that are not in the video list")
    if n_no_video:
        print(f"  WARNING: {n_no_video} clips left out because their video file was not found under "
              f"the footage folder (SALMONID_FOOTAGE_ROOT); the pool is incomplete.")
    if not clips:
        raise SystemExit("Nothing to sample from: no clip with a countable track could be dated. "
                         "Check that the 2025 footage is available (SALMONID_FOOTAGE_ROOT).")

    # ---- run phase: cut each river's list of dates into thirds ----
    for site in ("Credit", "Ganaraska"):
        ds = sorted({c["date"] for c in clips if c["site"] == site})
        if not ds:
            continue
        t1, t2 = ds[len(ds)//3], ds[2*len(ds)//3]
        for c in clips:
            if c["site"] == site:
                c["phase"] = "early" if c["date"] < t1 else ("peak" if c["date"] < t2 else "late")

    # ---- groups (strata) = river x phase x crowding ----
    for c in clips:
        c["stratum"] = f"{c['site']}|{c['phase']}|{c['crowd']}"
    strata = collections.defaultdict(list)
    for c in clips:
        strata[c["stratum"]].append(c)

    # ---- draw: in proportion to group size, at least 6 per group ----
    total = len(clips)
    chosen, chosen_ids = [], set()
    for st, pool in strata.items():
        n = min(len(pool), max(6, round(a.target * len(pool) / total)))
        for c in rng.sample(pool, n):
            chosen.append(c); chosen_ids.add(c["video_id"])

    # ---- rare-species top-up: at least rare_min drawn clips must CONTAIN each rare species ----
    # The extra clips come from anywhere in the pool, so clips with a rare species end up with
    # a higher chance of selection than other clips of the same group.
    for rare in ("Coho", "Atlantic", "Brown Trout"):
        have = sum(1 for c in chosen if rare in c["sp_set"])
        if have >= a.rare_min:
            continue
        extra_pool = [c for c in clips if rare in c["sp_set"] and c["video_id"] not in chosen_ids]
        for c in rng.sample(extra_pool, min(a.rare_min - have, len(extra_pool))):
            c["rare_topup"] = rare
            chosen.append(c); chosen_ids.add(c["video_id"])

    # ---- "weight" = clips in the group / clips drawn from the group (after the top-up) ----
    # Within a group the weights add up to the group's size. Because of the top-up above the
    # clips of a group did not all have the same chance of being drawn, so this is NOT a valid
    # design weight; it is kept because the reviewed sample file contains it. Nothing uses it.
    n_sampled = collections.Counter(c["stratum"] for c in chosen)
    for c in chosen:
        c["weight"] = round(len(strata[c["stratum"]]) / n_sampled[c["stratum"]], 3)

    # ---- write the file for the review tool ----
    out = []
    for c in sorted(chosen, key=lambda x: (x["site"], x["date"])):
        vh = hashlib.md5(c["video_id"].encode()).hexdigest()[:8]
        clip = f"ref_{c['site']}_{c['date']}_{vh}.mp4"
        tally = collections.Counter(t["dir"] for t in c["tracks"])
        out.append({"clip": clip, "video_id": c["video_id"], "site": c["site"], "date": c["date"],
                    "stratum": c["stratum"], "phase": c["phase"], "crowd": c["crowd"],
                    "weight": c["weight"], "rare_topup": c.get("rare_topup", ""),
                    "n_fish": c["n_fish"], "tracks": c["tracks"],
                    "model_tally": {"up": tally["up"], "down": tally["down"], "stay": tally["stay"]},
                    "day_divergence": f"{c['site']} {c['date']} — reference sample ({c['stratum']})"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"\nsampled {len(out)} clips -> {a.out}")
    print("per group (river|phase|crowding):")
    bc = collections.Counter(c["stratum"] for c in out)
    for st in sorted(bc):
        print(f"   {st:34s} {bc[st]:3d}")
    print("clips containing each rare species:", {r: sum(1 for c in out if r in {t['sp'] for t in c['tracks']})
                                                   for r in ("Coho", "Atlantic", "Brown Trout")})
    print(f"model tracks in the sample (fish for the reviewer to check) ~ {sum(c['n_fish'] for c in out)}")
    print("NOTE: the sample the blind review was done on is the file kept with the review "
          "records; compare before treating this draw as the same.")


if __name__ == "__main__":
    main()

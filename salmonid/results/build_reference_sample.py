"""In plain terms: this file chooses a fair, representative set of video clips for a human expert to
double-check by hand. It spreads the picks evenly across river, time of season, and how crowded each
scene is (and deliberately adds extra clips of the rarer species) so the check-set is not lopsided. A
technician later reviews these clips without seeing the model's answers, and comparing their verdicts
to the model's is what produces the paper's 94.7% blind-human accuracy figure.

Draw the stratified REFERENCE-STANDARD accuracy sample (design:
docs/stratified_sampling_design_2026-07-13.md).

Pipeline stage: RESULTS (data -> detect -> track -> eval/count -> **results** -> figures). Runs in the
base Python environment -- it only reads pickles / CSVs / JSON and uses numpy; it does NOT load torch or
the DEIMv2 stack, so the .venv_deim detection env is not needed here.

What it does: draws a probability sample of *counted* clips, stratified on the covariates that drive error
(river x run-phase x crowding), allocated proportionally within strata (floor of 6/stratum) plus a
rare-species top-up. Draws are taken from the BoT-SORT aggregate tracks -- the REPORTED counts
(~1.02 tracks/fish) -- NOT the IoU store.

Inputs:
  * config.LABEL_BATCHES / "corpus_2025_both.json"  -- corpus video manifest (site + rel path per clip).
  * <corpus>/agg_*shard*.pkl                         -- BoT-SORT aggregate tracks (boxes + per-frame probs).
  * <corpus>/{credit,gan}_track_diag.csv            -- per-track diagnostics (max co-present fish -> crowd tag).
Output:
  * <corpus>/reference_sample_2025.json  -- one entry per sampled clip: the design weight (for reweighting the
    sample back up to the corpus), the app-ready model tracks (species / direction / confidence), stratum tags.

Downstream / which claim: the review app serves this JSON in --blind mode and a technician adjudicates every
fish independently (adding any the model missed). Scoring the adjudicated sample against the model is what
produces the manuscript's blind-human reference-standard accuracy figure (94.7%).

  python -m salmonid.results.build_reference_sample --target 180 --seed 20260713
"""
import argparse, csv, glob, hashlib, pickle, collections, json, random
from pathlib import Path
import numpy as np

from salmonid import config
from salmonid.count.corpus_countr_from_store import UP_SIGN, FRAME_W, SPECIES  # noqa

# Corpus detect/track store for the 2025 both-rivers run (holds the BoT-SORT aggs + track-diag CSVs).
CORP = config.PROJECTS_ROOT / "deimv2_x_v26_corpus2025"
MANIFEST = config.LABEL_BATCHES / "corpus_2025_both.json"
# Asymmetric net rule (docs/counting_and_movement.md): count a track UPSTREAM at |dx|>=50px but DOWNSTREAM
# only at |dx|>=300px (strict downstream suppresses milling); MIN_FRAMES=3 detections for a track to count.
UP_THR, DOWN_THR, MIN_FRAMES = 50.0, 300.0, 3   # mirror config.NET_UP_THRESH / NET_DOWN_THRESH / MIN_FRAMES
DIRMAP = {1: "up", -1: "down", 0: "stay"}


def store_key(site, rel):
    """Return the detect-store pickle key for a clip: ``<site>__<md5(rel)[:12]>.pkl``.

    Mirrors the naming used when the aggregate store was written, so a manifest (site, rel) pair can be
    matched back to its pickled tracks.
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def track_dir(boxes, site):
    """Classify a track's net direction as ``'up'`` / ``'down'`` / ``'stay'`` from its box centroids.

    ``dx`` = mean of the last-3 box x-centroids minus the mean of the first-3, oriented to upstream by
    ``UP_SIGN[site]`` (Ganaraska upstream = +x, Credit = -x). The asymmetric thresholds (up >= 50px,
    down >= 300px) match the technicians' convention: liberal upstream, strict downstream to reject
    fish milling in front of the tunnel.

    Args:
        boxes: iterable of ``[x1, y1, x2, y2]`` boxes across the track's frames.
        site:  ``"Credit"`` or ``"Ganaraska"`` (selects the upstream sign).
    Returns:
        One of ``"up"``, ``"down"``, ``"stay"``.
    """
    b = np.asarray(boxes, float); cx = (b[:, 0] + b[:, 2]) / 2.0
    w = min(3, len(cx)); dx = float(cx[-w:].mean() - cx[:w].mean())
    sgn = dx * UP_SIGN[site]
    nc = 1 if (sgn > 0 and abs(dx) >= UP_THR) else (-1 if (sgn < 0 and abs(dx) >= DOWN_THR) else 0)
    return DIRMAP[nc]


def load_crowd():
    """video_id -> max co-present fish (from the per-track diag CSVs; a per-video property)."""
    crowd = {}
    for fn in (f"{CORP}/credit_track_diag.csv", f"{CORP}/gan_track_diag.csv"):
        if not Path(fn).is_file():
            continue
        for r in csv.DictReader(open(fn, encoding="utf-8")):
            try:
                mc = int(float(r.get("max_cooccur") or 1))
            except ValueError:
                mc = 1
            vid = r["video_id"]
            crowd[vid] = max(crowd.get(vid, 1), mc)
    return crowd


def main():
    """Parse args, build the stratified sample, and write ``reference_sample_2025.json``.

    Gathers every *counted* clip from the BoT-SORT aggs (each track needs >= MIN_FRAMES boxes), tags each
    clip with ``site x run-phase(date tercile) x crowding``, allocates draws proportionally within strata
    (floor of 6) with a rare-species top-up, computes uniform Horvitz-Thompson design weights per stratum,
    and emits the app-ready JSON. Side effects: reads the manifest / agg pickles / diag CSVs, writes the
    ``--out`` JSON, and prints a per-stratum + rare-coverage summary.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=180, help="core sample size (clips) before rare top-up")
    ap.add_argument("--rare-min", type=int, default=10, help="min sampled clips CONTAINING each rare species")
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--agg-glob", default=f"{CORP}/agg_*shard*.pkl")
    ap.add_argument("--out", default=f"{CORP}/reference_sample_2025.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    vids = json.load(open(MANIFEST))["videos"]
    vid2rel = {v.get("video_id"): (v["site"], v["video_rel"].replace("\\", "/")) for v in vids if v.get("video_id")}
    rel2vid = {store_key(s, rel): vid for vid, (s, rel) in vid2rel.items()}
    crowd = load_crowd()

    # gather counted clips from the BoT-SORT aggs
    clips = []                         # per clip: dict
    import os
    from salmonid.count.corpus_countr_from_store import _resolve
    import datetime
    for gf in glob.glob(a.agg_glob):
        for k, rec in pickle.load(open(gf, "rb")).items():
            vid = rel2vid.get(k)
            if vid is None:
                continue
            site = vid.split("/", 1)[0]
            vp = _resolve(vid)
            if not vp:
                continue
            date = datetime.date.fromtimestamp(os.path.getmtime(vp)).isoformat()
            tracks = []
            for t in rec.get("tracks", {}).values():
                if len(t["boxes"]) < MIN_FRAMES:
                    continue
                # Mean per-frame probability vote over the track; [:5] keeps only the 5 reported species
                # (drops class 5 = Small Salmonid / background), matching the corpus count convention.
                pvs = np.asarray(t["pvs"], float); sp = int(np.argmax(pvs.mean(0)[:5]))
                tracks.append({"sp": SPECIES[sp], "dir": track_dir(t["boxes"], site),
                               "conf": round(float(pvs.mean(0)[sp]), 2)})
            if not tracks:                          # only clips with >=1 counted fish
                continue
            clips.append({"video_id": vid, "site": site, "date": date, "tracks": tracks,
                          "n_fish": len(tracks), "sp_set": sorted({t["sp"] for t in tracks}),
                          "crowd": "multi" if crowd.get(vid, 1) >= 2 else "single"})
    print(f"counted clips available: {len(clips)}")

    # run-phase = per-river date tercile
    for site in ("Credit", "Ganaraska"):
        ds = sorted({c["date"] for c in clips if c["site"] == site})
        if not ds:
            continue
        t1, t2 = ds[len(ds)//3], ds[2*len(ds)//3]
        for c in clips:
            if c["site"] == site:
                c["phase"] = "early" if c["date"] < t1 else ("peak" if c["date"] < t2 else "late")

    # strata = site x phase x crowd
    for c in clips:
        c["stratum"] = f"{c['site']}|{c['phase']}|{c['crowd']}"
    strata = collections.defaultdict(list)
    for c in clips:
        strata[c["stratum"]].append(c)

    # proportional allocation with a floor of 6/stratum
    total = len(clips)
    chosen, chosen_ids = [], set()
    for st, pool in strata.items():
        n = min(len(pool), max(6, round(a.target * len(pool) / total)))
        for c in rng.sample(pool, n):
            chosen.append(c); chosen_ids.add(c["video_id"])

    # rare-species top-up: ensure >= rare_min sampled clips CONTAIN each rare species (adds MORE samples from
    # the rare species' strata — the design weight below accounts for the extra draws)
    for rare in ("Coho", "Atlantic", "Brown Trout"):
        have = sum(1 for c in chosen if rare in c["sp_set"])
        if have >= a.rare_min:
            continue
        extra_pool = [c for c in clips if rare in c["sp_set"] and c["video_id"] not in chosen_ids]
        for c in rng.sample(extra_pool, min(a.rare_min - have, len(extra_pool))):
            c["rare_topup"] = rare
            chosen.append(c); chosen_ids.add(c["video_id"])

    # Horvitz-Thompson design weight: UNIFORM per stratum = population / (total sampled in that stratum),
    # computed AFTER all sampling (incl. top-ups) so each stratum's weights sum to its population exactly.
    n_sampled = collections.Counter(c["stratum"] for c in chosen)
    for c in chosen:
        c["weight"] = round(len(strata[c["stratum"]]) / n_sampled[c["stratum"]], 3)

    # emit
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
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\nsampled {len(out)} clips -> {a.out}")
    print("per-stratum:")
    bc = collections.Counter(c["stratum"] for c in out)
    for st in sorted(bc):
        print(f"   {st:34s} {bc[st]:3d}")
    print("rare-species clip coverage:", {r: sum(1 for c in out if r in {t['sp'] for t in c['tracks']})
                                          for r in ("Coho", "Atlantic", "Brown Trout")})
    print(f"total fish to adjudicate ~ {sum(c['n_fish'] for c in out)}")


if __name__ == "__main__":
    main()

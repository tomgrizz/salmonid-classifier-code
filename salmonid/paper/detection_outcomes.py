"""In plain terms: this script asks what the model did with every 2025 video in which a
technician logged a fish. Each video falls into one of three outcomes: the model counted a
fish, the model saw something but too briefly to count it, or the model saw nothing at all.
It also checks the reverse case, videos the technician marked as containing no fish (false
triggers), and counts how often the model nevertheless counted one. Together these show
whether the model's counts are short because fish are missed, and whether empty videos add
fish that are not there.

Inputs
------
Tier 1 (default), committed in ``data/paper/``:

* ``stepB_credit.csv``, ``stepB_ganaraska.csv``: one row per 2025 video present in the
  technician's video-linked record, with the technician's fish / no-fish flag and the
  model's outcome.

Tier 2 (``--from-external``), not stored in git:

* the 2025 technician review exports (``config.TECH_2025_DIR``; third-party OMNR records);
* the list of 2025 videos given to the model
  (``<SALMONID_PROJECTS_ROOT>/label_batches/corpus_2025_<river>_full.json``);
* the counting pass's per-video tracks for 2025
  (``<SALMONID_PROJECTS_ROOT>/deimv2_x_v27_corpus2025/agg_<river>_shard*.pkl``, written by
  ``salmonid.detect.track_from_store``).

Outputs
-------
* ``detection_outcomes_2025.csv`` in ``--out-dir`` (default ``output/paper/``): counts and
  percentages by river, technician flag and model outcome.
* with ``--from-external --write-tables``: the per-video ``stepB_<river>.csv`` tables.

Feeds
-----
Section 3.2, "Few technician-logged fish went uncounted": a counted track in 96.7% of 9,042
Credit River videos and 96.6% of 16,910 Ganaraska River videos; nothing detected in 2.4% and
1.8%; detections short of the three-frame minimum in 0.9% and 1.7%; a track counted in 6 of
195 and 22 of 250 false-trigger videos. Section 4.4 discusses the same two ways a fish can go
uncounted.

Run
---
::

    python -m salmonid.paper.detection_outcomes                    # Tier 1
    python -m salmonid.paper.detection_outcomes --from-external    # Tier 2

Tier
----
1 for the percentages. Building the per-video table is Tier 2 (``build_table``).

Provenance
----------
The summary is the "Step B" block of ``recompute_v27.py`` (manuscript repository,
2026-11-06). ``build_table`` is the per-video classification of
``scripts/_credit_stepB_no_track.py`` in the research repository, unchanged; see the note in
``build_table`` on how the committed tables encode the technician flag.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from salmonid import config
from salmonid.paper._common import TECHMAP, keyof, add_out_dir, out_dir

RIVERS = ("credit", "ganaraska")
TABLES = {r: config.PAPER_DATA / f"stepB_{r}.csv" for r in RIVERS}

# ---------------------------------------------------------------- Tier-2 inputs
LABELS = config.LABEL_BATCHES
O = config.corpus_dir(2025)
# A track must hold at least this many processed frames to be counted (Methods 2.3). A fish
# glimpsed in one or two processed frames is "detected" but not "counted".
MIN_FRAMES = config.MIN_FRAMES
CANON = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
TECH_GLOB = {
    "Credit": str(config.TECH_2025_DIR / "Credit_Counter_Analysis_2025" / "data" / "raw_data"
                  / "fish_counts_export_*.csv"),
    "Ganaraska": str(config.TECH_2025_DIR / "Ganaraska_Counter_Analysis_2025" / "data" / "raw"
                     / "fish_counts_export_*.csv"),
}


def store_key_vid(v):
    """File name under which the counting pass stores one video's tracks."""
    return f"{hashlib.md5(v.encode()).hexdigest()[:16]}.pkl"


def build_table(site):
    """Tier 2. Classify every 2025 video of one river by technician flag and model outcome.

    The loop is the one in the research repository's Step B script. A video is "technician:
    fish" if ANY of its records names one of the five species, so a video holding a fish and a
    piece of debris still counts as a fish video. The model outcome uses the counting pass's
    tracks: "counted track" needs one track of at least MIN_FRAMES frames; "detections, no
    track" means something was detected but no track reached that length; "no detection"
    means the detector returned nothing above its threshold.
    """
    man = config.require(LABELS / f"corpus_2025_{site.lower()}_full.json",
                         "list of 2025 videos given to the model", "SALMONID_PROJECTS_ROOT")
    vids = json.load(open(man, encoding="utf-8"))["videos"]

    # technician: which videos carry a real fish
    if not glob.glob(TECH_GLOB[site]):
        config.require(Path(TECH_GLOB[site]).parent, "2025 technician review exports "
                       "(third-party OMNR records)", "SALMONID_TECH_2025_DIR")
    tech = collections.defaultdict(list)
    for f in glob.glob(TECH_GLOB[site]):
        for r in csv.DictReader(open(f, encoding="latin-1")):
            tech[keyof(r["video_rel"])].append(TECHMAP.get(r["species"].strip().lower(),
                                                           r["species"].strip()))

    agg = {}
    shards = sorted(glob.glob(str(O / f"agg_{site.lower()}_shard*.pkl")))
    if not shards:
        config.require(O / f"agg_{site.lower()}_shard0.pkl", "2025 counting-pass tracks",
                       "SALMONID_PROJECTS_ROOT")
    for fn in shards:
        with open(fn, "rb") as fh:
            agg.update(pickle.load(fh))

    rows = []
    for v in vids:
        k = store_key_vid(v["video_id"])
        tk = keyof(v["video_rel"])
        entries = tech.get(tk)
        if entries is None:
            has_fish = "unlogged"          # absent from the technician export entirely
        else:
            has_fish = any(e in CANON for e in entries)
        rec = agg.get(k)
        trk = rec["tracks"] if rec else {}
        n_track = sum(1 for t in trk.values() if len(t["boxes"]) >= MIN_FRAMES)
        n_any = len(trk)
        if n_track:
            state = "counted track"
        elif n_any:
            state = "detections, no track"
        else:
            state = "no detection"
        rows.append(dict(river=site, key=tk, tech_fish=has_fish, state=state,
                         n_tracks_any=n_any, n_tracks_counted=n_track))
    d = pd.DataFrame(rows)
    # The committed tables hold only videos that appear in the technician export, with the
    # flag written as 1 (fish) / 0 (no fish). Videos absent from the export ("unlogged") have
    # no technician outcome to compare against and are dropped here in the same way.
    n_unlogged = int((d.tech_fish == "unlogged").sum())
    d = d[d.tech_fish != "unlogged"].copy()
    d["tech_fish"] = d.tech_fish.astype(bool).astype(int)
    return d.reset_index(drop=True), n_unlogged


def summarize(tables):
    """The Step B block of the reconciliation script, with the same printout, plus one tidy
    table of the same numbers. ``tables`` maps river name -> per-video DataFrame."""
    rows = []
    print("\n## Step B (v2.7): technician-fish videos by model outcome")
    for river in RIVERS:
        s = tables[river]
        tf = s[s.tech_fish == 1].state.value_counts()
        nf = s[s.tech_fish == 0].state.value_counts()
        n = tf.sum()
        print(f"  {river}: tech-fish n={n:,}: " + ", ".join(f"{k} {v:,} ({100*v/n:.1f}%)" for k, v in tf.items()))
        m = nf.sum()
        print(f"  {river}: tech non-fish n={m}: counted {nf.get('counted track', 0)} ({100*nf.get('counted track', 0)/m:.1f}%)")
        for flag, vc, tot in (("fish", tf, n), ("no fish (false trigger)", nf, m)):
            for state in ("counted track", "detections, no track", "no detection"):
                k = int(vc.get(state, 0))
                rows.append(dict(river=river.capitalize(), technician=flag, model_outcome=state,
                                 videos=k, of_videos=int(tot), percent=round(100 * k / tot, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    ap.add_argument("--from-external", action="store_true",
                    help="Tier 2: rebuild the per-video tables from the technician exports, the 2025 "
                         "video list and the counting-pass tracks.")
    ap.add_argument("--write-tables", action="store_true",
                    help="With --from-external: also write stepB_<river>.csv to --out-dir.")
    args = ap.parse_args()
    out = out_dir(args)

    tables = {}
    for river in RIVERS:
        if args.from_external:
            tables[river], n_unlogged = build_table(river.capitalize())
            print(f"{river}: {len(tables[river]):,} videos matched to the technician export "
                  f"({n_unlogged:,} videos given to the model are absent from the export)")
            if args.write_tables:
                tables[river].to_csv(out / TABLES[river].name, index=False)
        else:
            tables[river] = pd.read_csv(config.require(TABLES[river], "committed per-video outcome table"))
    summary = summarize(tables)
    summary.to_csv(out / "detection_outcomes_2025.csv", index=False)
    print(f"\nwrote: {out / 'detection_outcomes_2025.csv'}")


if __name__ == "__main__":
    main()

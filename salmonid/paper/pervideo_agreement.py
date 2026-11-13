"""In plain terms: for every 2025 video in which a technician logged one fish, this script
asks whether the model named the same species. It reports the percentage of videos on which
they agree at each river, the same figure corrected for luck (Cohen's kappa, because most
videos are Chinook Salmon and two observers who both say "Chinook" will often agree by
chance), and how well each species is found (recall) and how trustworthy each species call
is (precision). It also shows where the disagreements sit: which species pair, which month,
and whether the model had split the video into several fish.

Inputs
------
Tier 1 (default), committed in ``data/paper/``:

* ``pervideo_2025_credit.csv``, ``pervideo_2025_ganaraska.csv``: one row per video with the
  technician's species and count and the model's dominant track (columns are described in
  ``data/paper/README.md``).

Tier 2 (``--from-external``), not stored in git:

* the 2025 technician review exports (``config.TECH_2025_DIR``; third-party OMNR records);
* the per-track diagnostics of the published model for 2025
  (``<SALMONID_PROJECTS_ROOT>/deimv2_x_v27_corpus2025/{credit,ganaraska}_track_diag_v27.csv``,
  written by ``salmonid.count.corpus_countr_from_store``).

Outputs
-------
Written to ``--out-dir`` (default ``output/paper/``):

* ``pervideo_summary_v27.csv``: one row per river and denominator (n, agreement, kappa and
  its interval, chance agreement, one-track and multi-track subsets, disagreement make-up).
* ``pervideo_per_species_v27.csv``: per-species videos, recall and precision.
* with ``--from-external --write-tables``: the two ``pervideo_2025_*.csv`` tables themselves.

The published copies of the first two files are committed in ``data/paper/`` for comparison.

Feeds
-----
* Section 3.2, first two paragraphs: 98.6% (5,131 of 5,203) and 90.6% (7,396 of 8,160);
  Chinook Salmon share 88% and 81%; chance agreement 79% and 68%; kappa 0.93 (0.84 to 0.97)
  and 0.71 (0.54 to 0.86); 612 of 764 disagreements; 86% in September; one-track subset
  99.1% and 94.7% (kappa 0.96 and 0.82); multi-track videos about a fifth, 96.7% and 75.2%.
* Table 2, columns Videos, Recall and Precision (rows named ``*_fish`` in the outputs).
* Section 3.2, Atlantic Salmon and Coho Salmon paragraphs (84% of 19 videos; recall 54%,
  precision 72%; 439 missed, 414 called Chinook Salmon, 409 in September, 269 split;
  Credit River 16 videos, 11 found, four over-calls).
* Methods 2.3, "No per-species confidence floor is applied": the floor sweep printed here is
  the evidence that a floor trades recall for precision.
* Sections 4.4 and 5 repeat 90.6%, 75.2% and 98.6%.

Run
---
::

    python -m salmonid.paper.pervideo_agreement                    # Tier 1
    python -m salmonid.paper.pervideo_agreement --from-external    # Tier 2, rebuilds from the exports

Tier
----
1 for every statistic. The step that builds the two per-video tables from the technician
exports and the model's tracks is Tier 2 (``build_table``).

Provenance
----------
Ported from ``recompute_v27.py`` (manuscript repository, 2026-11-06 reconciliation). The
functions ``load_tech``, ``load_diag``, ``dominant``, ``pervideo`` and ``summarize`` are
copied unchanged. Two things were added for the deposit and are marked "ADDED" below: the
per-video table that lets the statistics run without the technician exports, and a printout
of the Coho Salmon misses on the single-fish denominator (the original printed them on the
older one-record denominator only).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from salmonid import config
from salmonid.paper._common import (SP, TECHMAP, keyof, cm_from, kappa, block_boot_kappa,
                                    add_out_dir, out_dir)

RIVERS = ("Credit", "Ganaraska")

# ---------------------------------------------------------------- Tier-1 inputs
TABLES = {r: config.PAPER_DATA / f"pervideo_2025_{r.lower()}.csv" for r in RIVERS}
TABLE_COLUMNS = ["river", "key", "date", "tech_species", "tech_count", "n_tracks",
                 "dom_species", "dom_conf_top", "dom_conf2_species"]

# ---------------------------------------------------------------- Tier-2 inputs
# 2025 technician review exports, one row per identification, linked to a video.
GLOBS = {
    "Credit": str(config.TECH_2025_DIR / "Credit_Counter_Analysis_2025" / "data" / "raw_data"
                  / "fish_counts_export_*.csv"),
    "Ganaraska": str(config.TECH_2025_DIR / "Ganaraska_Counter_Analysis_2025" / "data" / "raw"
                     / "fish_counts_export_*.csv"),
}
# Per-track output of the published model for the 2025 season.
V27 = config.corpus_dir(2025)


# ---------------------------------------------------------------- loaders (Tier 2, unchanged)
# load_tech: every technician record, grouped by video. A record with a blank count is one fish.
def load_tech(river):
    tech = collections.defaultdict(list)
    for f in sorted(glob.glob(GLOBS[river])):
        for r in csv.DictReader(open(f, encoding="latin-1")):
            try:
                cnt = float(r.get("count") or 1)
            except ValueError:
                cnt = float("nan")
            tech[keyof(r["video_rel"])].append({"species": TECHMAP.get(r["species"].strip().lower(), r["species"].strip()),
                                                "count": cnt, "movement": r.get("movement", "")})
    return tech


def load_diag(river):
    d = pd.read_csv(V27 / f"{river.lower()}_track_diag_v27.csv")
    d["key"] = d.video_id.map(keyof)
    return d


# The model's call for a video is its DOMINANT track: the track seen in the most frames.
# A video can hold several tracks (a second fish, or one fish split in two), and the longest
# one is the model's best-supported identification. The optional ``floor`` is only used by the
# sensitivity sweep at the end: a Coho Salmon call below that confidence is replaced by the
# model's second choice.
def dominant(tracks: pd.DataFrame, floor=None):
    t = tracks.loc[tracks.n_frames.idxmax()]   # first max, as in the original script
    sp = t.pred_species
    if floor is not None and sp == "Coho" and t.conf_top < floor:
        sp = t.conf2_species
    return sp


# ---------------------------------------------------------------- per-video (Tier 2, unchanged)
# Which videos enter the comparison:
#   * the technician logged exactly ONE record for the video, and it names one of the five
#     species (videos with two or more records hold several fish, and matching each
#     technician fish to a model track would be a second, separate problem);
#   * with single_fish_only, that record's count is exactly one. A record can carry a count
#     of two or more (several fish entered on one line), so "one record" alone is not "one
#     fish". The manuscript's primary denominator is one record WITH a count of one;
#   * the model produced at least one counted track (videos with none are reported
#     separately by salmonid.paper.detection_outcomes).
def pervideo(river, tech, diag, single_fish_only, floor=None):
    by = {k: g for k, g in diag.groupby("key")}
    rows = []
    for k, v in tech.items():
        if len(v) != 1 or v[0]["species"] not in SP:
            continue
        if single_fish_only and v[0]["count"] != 1:
            continue
        g = by.get(k)
        if g is None or len(g) == 0:
            continue
        dom = dominant(g, floor)
        if dom not in SP:
            continue
        rows.append(dict(key=k, tech=v[0]["species"], dom=dom, n_tracks=len(g), date=g.date.iloc[0],
                         tech_count=v[0]["count"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ADDED: the per-video table
def build_table(river):
    """Tier 2. One row per comparable video, so the statistics can be recomputed without
    the technician exports. Applies exactly the selection in ``pervideo`` (one technician
    record naming a species; at least one model track; a species call), keeps the count so
    either denominator can be formed, and keeps the dominant track's confidence and second
    choice so the confidence-floor sweep can be repeated."""
    if not glob.glob(GLOBS[river]):
        folder = Path(GLOBS[river]).parent
        config.require(folder, "2025 technician review exports (third-party OMNR records)",
                       "SALMONID_TECH_2025_DIR")
        raise SystemExit(f"No fish_counts_export_*.csv files in {folder}")
    config.require(V27 / f"{river.lower()}_track_diag_v27.csv",
                   "per-track diagnostics of the published model, 2025", "SALMONID_PROJECTS_ROOT")
    tech = load_tech(river)
    diag = load_diag(river)
    by = {k: g for k, g in diag.groupby("key")}
    rows = []
    for k, v in tech.items():
        if len(v) != 1 or v[0]["species"] not in SP:
            continue
        g = by.get(k)
        if g is None or len(g) == 0:
            continue
        t = g.loc[g.n_frames.idxmax()]          # the dominant track, as in dominant()
        if t.pred_species not in SP:
            continue
        rows.append(dict(river=river, key=k, date=g.date.iloc[0], tech_species=v[0]["species"],
                         tech_count=v[0]["count"], n_tracks=len(g), dom_species=t.pred_species,
                         dom_conf_top=t.conf_top, dom_conf2_species=t.conf2_species))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS), tech, diag


def pervideo_from_table(table, single_fish_only, floor=None):
    """Tier 1. The same rows ``pervideo`` returns, taken from the committed table."""
    d = table
    if single_fish_only:
        d = d[d.tech_count == 1]
    dom = d.dom_species.copy()
    if floor is not None:
        low = (dom == "Coho") & (d.dom_conf_top < floor)
        dom = dom.where(~low, d.dom_conf2_species)
    out = pd.DataFrame(dict(key=d.key, tech=d.tech_species, dom=dom, n_tracks=d.n_tracks,
                            date=d.date, tech_count=d.tech_count))
    return out[out.dom.isin(SP)].reset_index(drop=True)


# ---------------------------------------------------------------- summary (unchanged)
# For one river and one denominator: agreement, kappa with its 10-day moving-block interval,
# the subset the model resolved into exactly one track (identification without tracking
# error) against the subset it split into two or more, and the make-up of the disagreements.
# Per species: recall = of the technician's videos of that species, the share the model also
# called that species; precision = of the videos the model called that species, the share
# the technician agreed with.
def summarize(name, d, report, with_ci=True):
    n = len(d)
    ok = int((d.tech == d.dom).sum())
    cm = cm_from(d.tech, d.dom, SP)
    k, po, pe = kappa(cm)
    lo = hi = float("nan")
    if with_ci:
        lo, hi = block_boot_kappa(d.tech, d.dom, d.date, SP)
    one = d[d.n_tracks == 1]
    multi = d[d.n_tracks > 1]
    k1 = kappa(cm_from(one.tech, one.dom, SP))[0] if len(one) else float("nan")
    res = dict(name=name, n=n, correct=ok, acc=ok / n, kappa=k, k_lo=lo, k_hi=hi, pe=pe,
               n_1to1=len(one), acc_1to1=(one.tech == one.dom).mean(), kappa_1to1=k1,
               n_multi=len(multi), share_multi=len(multi) / n, acc_multi=(multi.tech == multi.dom).mean())
    dis = d[d.tech != d.dom]
    pair = dis[dis.apply(lambda r: {r.tech, r.dom} == {"Chinook", "Coho"}, axis=1)]
    res["n_disagree"] = len(dis)
    res["n_chin_coho"] = len(pair)
    res["share_dis_sept"] = (pd.to_datetime(dis.date).dt.month == 9).mean() if len(dis) else float("nan")
    res["share_ref_chinook"] = (d.tech == "Chinook").mean()
    per = []
    for s in SP:
        tp = int(((d.tech == s) & (d.dom == s)).sum())
        nt = int((d.tech == s).sum())
        npred = int((d.dom == s).sum())
        per.append(dict(name=name, species=s, n_tech=nt, n_called=npred, tp=tp,
                        recall=tp / nt if nt else float("nan"), precision=tp / npred if npred else float("nan")))
    report.append(res)
    return res, pd.DataFrame(per)


def coho_misses(d):
    """ADDED. The Section 3.2 breakdown of technician Coho Salmon videos the model did not
    call Coho Salmon, on whichever denominator ``d`` holds."""
    miss = d[(d.tech == "Coho") & (d.dom != "Coho")]
    return dict(n=len(miss), called=miss.dom.value_counts().to_dict(),
                september=int((pd.to_datetime(miss.date).dt.month == 9).sum()),
                multi_track=int((miss.n_tracks > 1).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    ap.add_argument("--from-external", action="store_true",
                    help="Tier 2: rebuild the per-video tables from the technician exports and the "
                         "model's 2025 tracks instead of reading data/paper/.")
    ap.add_argument("--write-tables", action="store_true",
                    help="With --from-external: also write pervideo_2025_<river>.csv to --out-dir.")
    args = ap.parse_args()
    out = out_dir(args)

    report, per_all = [], []
    pv = {}
    for river in RIVERS:
        if args.from_external:
            table, tech, diag = build_table(river)
            # The original script's opening line: how many one-record videos carry a count other
            # than one. These are the videos the single-fish denominator leaves out.
            nrows = sum(1 for v in tech.values() if len(v) == 1 and v[0]["species"] in SP)
            nmulti_cnt = sum(1 for v in tech.values() if len(v) == 1 and v[0]["species"] in SP and v[0]["count"] != 1)
            print(f"\n## {river}: single-row species videos {nrows:,}; of which count != 1: {nmulti_cnt:,}")
            # Check that the table route gives exactly what the original route gives.
            for sfo in (False, True):
                a, b = pervideo(river, tech, diag, sfo), pervideo_from_table(table, sfo)
                assert len(a) == len(b) and (a.tech.values == b.tech.values).all() \
                    and (a.dom.values == b.dom.values).all(), "table route differs from original route"
            if args.write_tables:
                table.to_csv(out / TABLES[river].name, index=False)
                print(f"   wrote {out / TABLES[river].name}")
        else:
            table = pd.read_csv(config.require(TABLES[river], "committed per-video table"))
            print(f"\n## {river}: {len(table):,} comparable videos in {TABLES[river].name} "
                  f"(one technician record naming a species, model returned a counted track)")
        # "row"  = one technician record of any count (the older denominator, kept for comparison)
        # "fish" = one technician record with a count of one (the manuscript's primary denominator)
        for tag, sfo in (("row", False), ("fish", True)):
            d = pervideo_from_table(table, sfo)
            pv[(river, tag)] = d
            res, per = summarize(f"{river}_{tag}", d, report)
            per_all.append(per)
            print(f"  [{tag}] n={res['n']:,} correct={res['correct']:,} acc={res['acc']:.4f} kappa={res['kappa']:.3f} "
                  f"[{res['k_lo']:.2f},{res['k_hi']:.2f}] pe={res['pe']:.3f} | 1to1 n={res['n_1to1']:,} acc={res['acc_1to1']:.4f} "
                  f"k={res['kappa_1to1']:.3f} | multi n={res['n_multi']:,} ({res['share_multi']:.3f}) acc={res['acc_multi']:.4f} "
                  f"| disagree {res['n_disagree']} chin-coho {res['n_chin_coho']} sept {res['share_dis_sept']:.2f} "
                  f"| ref Chinook share {res['share_ref_chinook']:.3f}")
            print(per.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        # Coho floor sweep on the single-fish denominator. A "floor" would refuse a Coho Salmon
        # call below a set confidence. The sweep shows what that costs: at the Credit River
        # precision rises but recall falls, and at the Ganaraska River no floor helps. This is
        # why the manuscript applies no floor (Methods 2.3).
        print(f"  Coho floor sweep ({river}, single-fish denominator):")
        for fl in (None, 0.50, 0.55, 0.60, 0.65, 0.70):
            d = pervideo_from_table(table, True, floor=fl)
            tp = int(((d.tech == "Coho") & (d.dom == "Coho")).sum())
            nt = int((d.tech == "Coho").sum()); npred = int((d.dom == "Coho").sum())
            fpc = d[(d.dom == "Coho") & (d.tech != "Coho")].tech.value_counts().to_dict()
            print(f"    floor={fl}: Coho tech {nt}, called {npred}, tp {tp}, recall {tp/nt:.3f}, precision {tp/npred:.3f}, "
                  f"overall acc {(d.tech == d.dom).mean():.4f}; over-calls by tech species {fpc}")
        # Coho misses: what are they? (original printout, one-record denominator)
        d = pv[(river, "row")]
        miss = d[(d.tech == "Coho") & (d.dom != "Coho")]
        print(f"  Coho apparent misses (row denominator): {len(miss)}; called as {miss.dom.value_counts().to_dict()}; "
              f"multi-track {int((miss.n_tracks > 1).sum())}; tech count>1 {int((miss.tech_count != 1).sum())}")
        # ADDED: the same on the single-fish denominator, which is what Section 3.2 quotes.
        cm_ = coho_misses(pv[(river, "fish")])
        print(f"  Coho misses (single-fish denominator, as in Section 3.2): {cm_['n']}; called as {cm_['called']}; "
              f"in September {cm_['september']}; multi-track {cm_['multi_track']}")

    pd.DataFrame(report).to_csv(out / "pervideo_summary_v27.csv", index=False)
    pd.concat(per_all).to_csv(out / "pervideo_per_species_v27.csv", index=False)
    print(f"\nwrote: {out / 'pervideo_summary_v27.csv'}\nwrote: {out / 'pervideo_per_species_v27.csv'}")


if __name__ == "__main__":
    main()

"""STATUS: development record. NOT used for any number in the manuscript.
This script belongs to the earlier two-stage pipeline (a separate detector and species
classifier, with per-frame output files that are not part of this deposit). It is kept
because it documents how the counting rules were arrived at; see
docs/history/counting_and_movement_development_notes.md. The published counts come from
salmonid/count/corpus_countr_from_store.py, and the method of record is
docs/COUNTING_METHOD.md. Statements below about "the paper" describe an earlier draft.
Tier: 2 (and its inputs are not deposited).

In plain terms: this script works out WHY the computer's tally of fish moving
upstream doesn't perfectly match the human technicians' tally. For each species and river it
separates every fish's movement into "went upstream" versus "went downstream" and compares those
two totals against the technicians'. It shows the mismatch comes mostly from the computer
over-counting fish that drift back downstream (fish milling about or re-crossing in front of the
camera) rather than from missing fish heading upstream — which is the reason the paper uses a
lopsided counting rule: count an upstream crossing readily, but only count a downstream crossing
when the fish travels a long way (the 50/300-pixel "asymmetric net rule"). See
``docs/history/counting_and_movement_development_notes.md``.

Up-vs-down decomposition of the counter's net error (the downstream over-count insight).

Pipeline stage: **eval / count** (data -> detect -> track -> **eval/count** -> results -> figures).
This diagnostic answers *why* the model's net (upstream) count differs from the technicians':
is the gap from MISSING UPSTREAM passages or from EXCESS DOWNSTREAM ones? It splits both the
model and the technician tallies into separate UP and DOWN totals, per species and river, swept
across a range of ``|delta_x|`` (net-displacement) thresholds. The finding it surfaces — that the
net error is dominated by the model over-counting DOWNSTREAM (milling / re-crossing) fish rather
than missing upstream ones — is the empirical justification for the **asymmetric net rule**
(count a track upstream at a liberal threshold, downstream only at a strict one). See
``docs/history/counting_and_movement_development_notes.md``.

Inputs
------
* ``--shard-dirs``: glob(s) of DEIMv2 detect-store shard directories, each holding a
  ``tracks.sqlite`` with one row per BoT-SORT track (``predicted_class``, ``delta_x`` = mean of
  last-3 minus first-3 box-centroid x, and per-species ``mean_prob_*``). These live under
  ``config.MODEL_OUTPUTS`` on the original workstation.
* ``--tech-dir``: the technician ground-truth CSV tree (Ganaraska ``all_counts_gan.csv`` +
  Credit raw-data exports), defaulting to the committed ``docs/Riverwatcher_2025``.

Output
------
Plain-text tables printed to stdout: (1) per-species model-vs-tech UP / DOWN / NET across the
``--thresholds`` sweep, and (2) an ASYMMETRIC block scoring several (up_thr / down_thr) combos —
the sweep that motivates the deployed 50 / 300 px rule.

Environment: base Python (stdlib only — sqlite3 / csv / glob; no torch/boxmot). It reads the
detect-store SQLite that the detection stage (``.venv_deim``) produced.

Example (module invocation; shard dirs live under ``config.MODEL_OUTPUTS``)::

  python -m salmonid.count.net_direction_decomp \\
      --shard-dirs "<MODEL_OUTPUTS>/run_corpus_v6_2025_shard_*"
"""
from __future__ import annotations
import argparse, csv, glob, sqlite3, os, collections, datetime
from pathlib import Path

from salmonid import config

# Domain constants are kept INLINE (verbatim from the research script) for faithfulness; the
# mirror comments point at the single authority in salmonid.config.
SPECIES={0:"Chinook",1:"Coho",2:"Atlantic",3:"Rainbow Trout",4:"Brown Trout"}  # mirrors config.CLASS_TO_SP
AUTO_FLOOR={0:0.50,1:0.60,2:1.01,3:0.60,4:0.60}  # mirrors config.AUTO_FLOOR (Atlantic 1.01 => never auto-counted)
_PROB=["mean_prob_chinook","mean_prob_coho","mean_prob_atlantic","mean_prob_rainbow","mean_prob_brown"]
UP_SIGN={"Ganaraska":1.0,"Credit":-1.0}  # mirrors config.UP_SIGN (sign of delta_x that means UPSTREAM)

def _vd(d):
    """True if ``d`` is a valid ``YYYY-MM-DD`` date prefix (used to drop malformed CSV rows)."""
    return bool(d) and len(d)==10 and d[4]=="-" and d[7]=="-" and d[:4].isdigit()
def _sp(n):
    """Map a free-text species name to our class index 0-4, or None if unrecognised."""
    n=(n or "").strip().lower()
    if "chinook" in n: return 0
    if "coho" in n: return 1
    if "atlantic" in n: return 2
    if "rainbow" in n or "steelhead" in n: return 3
    if "brown" in n: return 4
    return None
def _dir(s):
    """Normalise a technician direction cell to ``"up"`` / ``"down"`` / None."""
    s=(s or "").strip().strip('"').lower(); return "up" if s=="up" else ("down" if s=="down" else None)

def load_tech(rw):
    """Load technician UP / DOWN passage totals per (river, species) from the GT CSV tree.

    Args:
        rw: root Path of the technician CSV tree (the ``--tech-dir``). Expected layout is the
            Ganaraska ``all_counts_gan.csv`` (``;``-delimited) plus the Credit raw-data exports.

    Returns:
        ``(up, dn)`` — two nested dicts ``{river: {class_idx: count}}`` of upstream and
        downstream passage counts. Rows flagged as false triggers, undated, or of an
        unrecognised species are skipped; the ``count`` field (fish per event) is summed where
        present, else 1 per row (Credit ``Credit_river_Counts_*`` exports are 1-per-row).
    """
    up=collections.defaultdict(lambda: collections.defaultdict(int))
    dn=collections.defaultdict(lambda: collections.defaultdict(int))
    g=rw/"Ganaraska_Counter_Analysis_2025"/"all_counts_gan.csv"
    for r in csv.DictReader(open(g,encoding="utf-8",errors="replace"),delimiter=";"):
        if (r.get("false_trigger") or "").strip().strip('"')=="1": continue
        sp=_sp(r.get("species_main") or "");
        if sp is None or not _vd((r.get("ts") or "").strip().strip('"')[:10]): continue
        try: n=int(float(r.get("count") or 1))
        except ValueError: n=1
        d=_dir(r.get("movement") or "")
        if d=="up": up["Ganaraska"][sp]+=n
        elif d=="down": dn["Ganaraska"][sp]+=n
    cdir=rw/"Credit_Counter_Analysis_2025"/"data"/"raw_data"
    for cf in glob.glob(str(cdir/"Credit_river_Counts_*.csv")):
        for r in csv.DictReader(open(cf,encoding="utf-8",errors="replace")):
            sp=_sp(r.get("sortname") or "")
            if sp is None or not _vd((r.get("DateTime") or "")[:10]): continue
            d=_dir(r.get("Direction") or "")
            if d=="up": up["Credit"][sp]+=1
            elif d=="down": dn["Credit"][sp]+=1
    for cf in glob.glob(str(cdir/"fish_counts_export_*.csv")):
        if "vaki_matched" in cf: continue
        for r in csv.DictReader(open(cf,encoding="utf-8",errors="replace")):
            if (r.get("false_trigger") or "0").strip()=="1": continue
            sp=_sp(r.get("species") or "")
            if sp is None or not _vd((r.get("ts") or "")[:10]): continue
            try: n=int(float(r.get("count") or 1))
            except ValueError: n=1
            d=_dir(r.get("movement") or "")
            if d=="up": up["Credit"][sp]+=n
            elif d=="down": dn["Credit"][sp]+=n
    return up,dn

def main():
    """CLI entry: load model tracks + technician GT and print the up/down/net decomposition tables.

    Reads each shard's ``tracks.sqlite``, keeps only tracks whose species passes its ``AUTO_FLOOR``
    triage, then prints (1) the per-threshold model-vs-tech UP/DOWN/NET sweep and (2) the
    asymmetric (up_thr / down_thr) combo table. Output goes to stdout only; nothing is written.
    """
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-dirs",nargs="+",required=True)
    ap.add_argument("--footage-root",default=str(config.FOOTAGE_ROOT_FAST))
    ap.add_argument("--tech-dir",default="docs/Riverwatcher_2025")
    ap.add_argument("--thresholds",default="0,125,300")
    args=ap.parse_args()
    Ts=[float(x) for x in args.thresholds.split(",")]
    froot=Path(args.footage_root)
    tup,tdn=load_tech(Path(args.tech_dir))

    rows=[]; cache={}
    cols="video_id, predicted_class, delta_x, "+", ".join(_PROB)
    for d in sorted({x for pat in args.shard_dirs for x in glob.glob(pat)}):
        db=Path(d)/"tracks.sqlite"
        if not db.exists(): continue
        c=sqlite3.connect(str(db))
        try:
            for r in c.execute(f"SELECT {cols} FROM tracks"):
                vid,sp,dx=r[0],r[1],r[2]
                if sp is None or not (0<=sp<5) or dx is None: continue
                # Species-specific triage: use the predicted class's own mean_prob (column 3+sp)
                # and drop the track if it is below that class's AUTO_FLOOR.
                conf=r[3+sp]
                if conf is None or conf<AUTO_FLOOR[sp]: continue
                site=vid.split("/")[0]
                if site in UP_SIGN: rows.append((site,sp,float(dx)))
        finally: c.close()

    for site in ["Ganaraska","Credit"]:
        print(f"\n############### {site} ###############")
        for sp in [0,1,3,4]:
            tu,td=tup[site][sp],tdn[site][sp]
            print(f"\n{SPECIES[sp]}   tech: up={tu:,} down={td:,} net={tu-td:,}")
            print(f"  {'thresh':>6}{'mUP':>9}{'mUP/tUP':>9}{'mDOWN':>8}{'mDN/tDN':>9}{'mNET':>9}{'net%err':>9}")
            for T in Ts:
                # dx*UP_SIGN[site] > 0 => upstream (Ganaraska +x, Credit -x); same threshold T
                # applied symmetrically to both directions in this diagnostic sweep.
                mu=sum(1 for (s,ss,dx) in rows if s==site and ss==sp and abs(dx)>=T and dx*UP_SIGN[site]>0)
                md=sum(1 for (s,ss,dx) in rows if s==site and ss==sp and abs(dx)>=T and dx*UP_SIGN[site]<0)
                ur=f"{100*mu/tu:.0f}%" if tu else "-"
                dr=f"{100*md/td:.0f}%" if td else "-"
                ne=mu-md; pe=f"{100*(ne-(tu-td))/(tu-td):.0f}%" if (tu-td) else "-"
                print(f"  {int(T):>6}{mu:>9,}{ur:>9}{md:>8,}{dr:>9}{ne:>9,}{pe:>9}")

    # ---- ASYMMETRIC up/down thresholds: liberal up, strict down ----
    # Separate thresholds for the two directions: a low up-threshold keeps genuine upstream
    # passages while a high down-threshold kills milling/re-crossing over-counts. The deployed
    # rule is 50/300 (see NET_UP_THRESH/NET_DOWN_THRESH in salmonid.config).
    print("\n\n################ ASYMMETRIC (up_thr / down_thr) net %err ################")
    combos=[("sym125",125,125),("50/200",50,200),("50/300",50,300),("0/200",0,200),("0/300",0,300)]
    for site in ["Ganaraska","Credit"]:
        print(f"\n{site}:")
        print(f"  {'species':14}" + "".join(f"{c[0]:>10}" for c in combos))
        for sp in [0,1,3,4]:
            tn=tup[site][sp]-tdn[site][sp]
            cells=[]
            for (_,uT,dT) in combos:
                # up counted at |dx|>=uT, down counted at the (stricter) |dx|>=dT
                mu=sum(1 for (s,ss,dx) in rows if s==site and ss==sp and dx*UP_SIGN[site]>0 and abs(dx)>=uT)
                md=sum(1 for (s,ss,dx) in rows if s==site and ss==sp and dx*UP_SIGN[site]<0 and abs(dx)>=dT)
                ne=mu-md; pe=f"{100*(ne-tn)/tn:+.0f}%" if tn else "-"
                cells.append(pe)
            print(f"  {SPECIES[sp]:14}" + "".join(f"{c:>10}" for c in cells))

if __name__=="__main__":
    main()

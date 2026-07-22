"""
In plain terms: this file works out how far a fish has to travel across the video (measured in
pixels of sideways movement) before we count it as heading upstream or downstream, and it tallies
the resulting net-upstream count for each species and river. It tests a range of distance cut-offs
and checks which one best matches the technicians' full-year hand counts. The finding -- that no
single cut-off fits both rivers at once -- is what justifies the paper's "asymmetric" counting rule
(count a fish moving upstream after 50 pixels, but downstream only after 300 pixels).

Net-upstream min_displacement ("up vector pixel") SWEEP vs full-year tech, 2025, PER RIVER.

Re-derives model net at each |delta_x| threshold from the saved corpus tracks (no re-inference):
upstream = Ganaraska Right(dx>0) / Credit Left(dx<0); |dx| < T -> Unknown (excluded, like tech "Stay").
Compares per-river net totals (%err) + daily net-r to full-year tech, across thresholds, so we can
pick the threshold that best matches BOTH rivers. Also prints a per-species/river calibration factor
(tech/model) at --apply-thresh. Tech: Ganaraska all_counts_gan (movement Up-Down); Credit raw counter
(Direction) + reviewed fall exports (movement). Triage floor applied (--raw to disable).

Pipeline stage
--------------
eval/count. Consumes the already-produced BoT-SORT corpus tracks (one ``tracks.sqlite`` per detect
shard, written by the detect->track stage) plus the 2025 technician CSVs, and produces a printed
table. Runs in the BASE Python env: it only reads sqlite + CSV, and never loads torch / DEIMv2.

Inputs
------
* ``--shard-dirs`` glob(s) of corpus run dirs, each holding ``tracks.sqlite`` with columns
  ``video_id, predicted_class, delta_x, mean_prob_*``.
* ``--footage-root`` footage tree (default ``config.FOOTAGE_ROOT_FAST``); read only to date each
  video via its file mtime -- no frames are decoded.
* ``--tech-dir`` the 2025 technician counter analyses (Ganaraska ``all_counts_gan.csv`` +
  Credit raw counter / reviewed fall exports).

Output
------
A per-species, per-river table of model/tech net totals, %err and daily-net Pearson-r at each
threshold, plus a calibration-factor (tech/model) table at ``--apply-thresh``. Printed to stdout
only -- no files are written.

Manuscript claim
----------------
Justifies the deployed ASYMMETRIC net rule (count a track UPSTREAM at |dx|>=50px, DOWNSTREAM only
at |dx|>=300px). This sweep of a single SYMMETRIC |dx| threshold shows no one value matches both
rivers' net simultaneously, which is the empirical motivation for the asymmetric family optimum
(docs/counting_and_movement.md).

  # --shard-dirs is a glob under config.MODEL_OUTPUTS; --footage-root defaults to
  # config.FOOTAGE_ROOT_FAST (set the SALMONID_* env vars to point at your data).
  python -m salmonid.count.net_threshold_sweep \
    --shard-dirs "<MODEL_OUTPUTS>/run_corpus_v6_2025_shard_*"
"""
from __future__ import annotations
import argparse, csv, glob, sqlite3, os, collections, datetime
from pathlib import Path
import numpy as np

from salmonid import config  # centralised paths (F:/ footage default); logic unchanged

try: from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    def pearsonr(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float)
        if len(a)<2 or a.std()==0 or b.std()==0: return (float("nan"),1.0)
        return (float(np.corrcoef(a,b)[0,1]),0.0)

SPECIES={0:"Chinook",1:"Coho",2:"Atlantic",3:"Rainbow Trout",4:"Brown Trout"}  # mirrors salmonid.config.SPECIES (as index->name)
AUTO_FLOOR={0:0.50,1:0.60,2:1.01,3:0.60,4:0.60}  # mirrors salmonid.config.AUTO_FLOOR (Atlantic 1.01 => never auto-counted)
_PROB=["mean_prob_chinook","mean_prob_coho","mean_prob_atlantic","mean_prob_rainbow","mean_prob_brown"]  # per-class conf columns, class-index order
UP_SIGN={"Ganaraska":1.0,"Credit":-1.0}  # mirrors salmonid.config.UP_SIGN (sign of dx that means UPSTREAM)


def _vd(d):
    """True if ``d`` is a well-formed ISO date string (``YYYY-MM-DD``)."""
    return bool(d) and len(d)==10 and d[4]=="-" and d[7]=="-" and d[:4].isdigit()
def _sp(n):
    """Map a free-text technician species name to its class index (0-4), else None.

    Substring match on lower-cased text; 'steelhead' is treated as Rainbow (3).
    """
    n=(n or "").strip().lower()
    if "chinook" in n: return 0
    if "coho" in n: return 1
    if "atlantic" in n: return 2
    if "rainbow" in n or "steelhead" in n: return 3
    if "brown" in n: return 4
    return None
def _mv(s):
    """Map a movement string to a signed unit: +1 'up', -1 'down', 0 stay/blank/unknown."""
    s=(s or "").strip().strip('"').lower(); return 1 if s=="up" else (-1 if s=="down" else 0)


def load_tech_net(rw: Path):
    """Load 2025 technician NET counts per (river, species) from the counter analyses.

    Args:
        rw: the ``--tech-dir`` root containing ``Ganaraska_Counter_Analysis_2025/`` and
            ``Credit_Counter_Analysis_2025/``.

    Returns:
        (net_d, net_t) where ``net_d[(site, date)][sp]`` is the count-weighted daily net
        (Up minus Down) and ``net_t[site][sp]`` is the season total. Each event contributes
        ``count * (+1 up / -1 down / 0 stay)``; false-triggers and 'Stay' rows drop out.

    Ganaraska uses the complete ';'-delimited event list (all_counts_gan.csv). Credit is spliced
    from the raw Vaki counter (Apr..Oct14, ``sortname``/``Direction``) plus the human-reviewed fall
    exports (Oct14..Nov, ``species``/``movement``); ``vaki_matched`` files are skipped to avoid
    double-counting the reviewed period.
    """
    net_d=collections.defaultdict(lambda: collections.defaultdict(int))
    net_t=collections.defaultdict(lambda: collections.defaultdict(int))
    # Ganaraska: complete event list
    g=rw/"Ganaraska_Counter_Analysis_2025"/"all_counts_gan.csv"
    for r in csv.DictReader(open(g,encoding="utf-8",errors="replace"),delimiter=";"):
        if (r.get("false_trigger") or "").strip().strip('"')=="1": continue
        sp=_sp(r.get("species_main") or ""); date=(r.get("ts") or "").strip().strip('"')[:10]
        if sp is None or not _vd(date): continue
        try: n=int(float(r.get("count") or 1))
        except ValueError: n=1
        v=n*_mv(r.get("movement") or "")
        net_d[("Ganaraska",date)][sp]+=v; net_t["Ganaraska"][sp]+=v
    # Credit: raw counter (Apr..Oct14, Vaki sortname/Direction)
    cdir=rw/"Credit_Counter_Analysis_2025"/"data"/"raw_data"
    for cf in glob.glob(str(cdir/"Credit_river_Counts_*.csv")):
        for r in csv.DictReader(open(cf,encoding="utf-8",errors="replace")):
            sp=_sp(r.get("sortname") or ""); date=(r.get("DateTime") or "")[:10]
            if sp is None or not _vd(date): continue
            v=_mv(r.get("Direction") or "")
            net_d[("Credit",date)][sp]+=v; net_t["Credit"][sp]+=v
    # Credit: reviewed fall exports (Oct14..Nov, human species + movement)
    for cf in glob.glob(str(cdir/"fish_counts_export_*.csv")):
        if "vaki_matched" in cf: continue
        for r in csv.DictReader(open(cf,encoding="utf-8",errors="replace")):
            if (r.get("false_trigger") or "0").strip()=="1": continue
            sp=_sp(r.get("species") or ""); date=(r.get("ts") or "")[:10]
            if sp is None or not _vd(date): continue
            try: n=int(float(r.get("count") or 1))
            except ValueError: n=1
            v=n*_mv(r.get("movement") or "")
            net_d[("Credit",date)][sp]+=v; net_t["Credit"][sp]+=v
    return net_d,net_t


def main():
    """CLI entry: sweep the |delta_x| net threshold per river and print the comparison tables."""
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-dirs",nargs="+",required=True)
    ap.add_argument("--footage-root",default=str(config.FOOTAGE_ROOT_FAST))  # F:/RiverWatcher NVMe mirror
    ap.add_argument("--tech-dir",default="docs/Riverwatcher_2025")
    ap.add_argument("--thresholds",default="0,50,100,125,150,200,300")
    ap.add_argument("--apply-thresh",type=float,default=125.0)
    ap.add_argument("--raw",action="store_true")
    args=ap.parse_args()
    Ts=[float(x) for x in args.thresholds.split(",")]
    froot=Path(args.footage_root)
    tnet_d,tnet_t=load_tech_net(Path(args.tech_dir))

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
                if not args.raw:
                    conf=r[3+sp]  # r[3:] are the mean_prob_* cols in class-index order
                    if conf is None or conf<AUTO_FLOOR[sp]: continue  # per-species deploy triage floor
                site=vid.split("/")[0]
                if site not in UP_SIGN: continue
                date=cache.get(vid)
                if date is None:
                    # date the video from its file mtime (footage has no reliable embedded timestamp)
                    try: date=datetime.date.fromtimestamp(os.path.getmtime(froot/vid)).isoformat()
                    except OSError: date=""
                    cache[vid]=date
                if date: rows.append((site,date,sp,float(dx)))
        finally: c.close()
    print(f"loaded {len(rows):,} triaged tracks\n")

    def model_net(sp, site, T):
        """Re-derive the model net (total, per-day) for one species/river at threshold ``T``.

        A track counts +1 if it moved >=T px in the site's upstream direction, -1 if downstream,
        and is skipped when |dx| < T (mirrors the technician 'Stay' exclusion). This is the
        symmetric threshold whose per-river inadequacy motivates the asymmetric deploy rule.
        """
        tot=0; daily=collections.defaultdict(int)
        for (s,date,ss,dx) in rows:
            if ss!=sp or s!=site or abs(dx)<T: continue
            up=1 if dx*UP_SIGN[site]>0 else -1  # UP_SIGN flips Credit so +up is always upstream
            tot+=up; daily[date]+=up
        return tot, daily

    for sp in [0,1,3,4]:
        print(f"=== {SPECIES[sp]} ===")
        print(f"  {'thresh':>6} | {'GANARASKA model/tech':>22} {'%err':>6} {'r':>6} | {'CREDIT model/tech':>20} {'%err':>6} {'r':>6}")
        for T in Ts:
            line=f"  {int(T):>6} |"
            for site in ["Ganaraska","Credit"]:
                tt=tnet_t[site][sp]
                td={dt:tnet_d[(site,dt)][sp] for (s,dt) in tnet_d if s==site}
                mt,md=model_net(sp,site,T)
                pct=100*(mt-tt)/tt if tt else float("nan")
                dts=sorted(set(md)|set(td))
                r=pearsonr([md.get(x,0) for x in dts],[td.get(x,0) for x in dts])[0] if (mt and sum(td.values())) else float("nan")
                if site=="Ganaraska": line+=f" {mt:>9,}/{tt:<9,}{pct:>6.0f}%{r:>6.2f} |"
                else: line+=f" {mt:>8,}/{tt:<8,}{pct:>6.0f}%{r:>6.2f}"
            print(line)
        print()

    # calibration at apply-thresh
    print(f"=== CALIBRATION FACTORS at min_displacement={int(args.apply_thresh)}px (tech_net / model_net) ===")
    print(f"  {'species':14}{'Ganaraska (model->tech)':>30}{'Credit (model->tech)':>28}")
    for sp in [0,1,3,4]:
        cells=[]
        for site in ["Ganaraska","Credit"]:
            mt,_=model_net(sp,site,args.apply_thresh); tt=tnet_t[site][sp]
            f=tt/mt if mt else float("nan")
            cells.append(f"{mt:,}->{tt:,}  x{f:.2f}")
        print(f"  {SPECIES[sp]:14}{cells[0]:>30}{cells[1]:>28}")


if __name__=="__main__":
    main()

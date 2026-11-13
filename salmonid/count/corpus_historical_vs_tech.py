"""STATUS: only ``load_tech()`` in this file is on the published path.
``load_tech()`` reads the technician count records for the season(s) named below and is
imported by salmonid/count/corpus_countr_from_store.py, which produced the published counts.
Everything else in this file (the command-line entry point, the per-frame file reading, the
species-confidence floors) is the counting engine of the earlier two-stage pipeline: it is
kept for provenance, its inputs are not deposited, and none of its numbers are in the
manuscript. Statements below about "the paper" describe an earlier draft. The method of
record is docs/COUNTING_METHOD.md.
Tier: 2 (technician records are external; see docs/DATA_AVAILABILITY.md).

In plain terms: this script checks whether the automated fish counter's totals agree with the counts
that fisheries technicians recorded by hand in the earlier years (2018-2023), for both rivers. For each
species it works out the net number of fish that moved upstream (fish going up minus fish going down),
compares that to the official technician total, and prints a validation table plus a running-total chart
for each river. These results support the paper's claim that the automated counter reproduces the
historical run record — both the size of each run and its day-to-day timing — all the way back to 2018.

HISTORICAL year (2018-2023) model (v6 full run) vs technician counts — BOTH rivers, NET upstream.

Pipeline position
-----------------
This is an **eval/count** stage tool (data -> detect -> track -> *eval/count* -> results -> figures).
It consumes the already-computed per-frame detection+track parquets from the multi-year corpus run and
reduces them to per-(site, species) daily/season NET-upstream counts, then compares those to the
official OMNRF technician NET totals for the pre-2024 years. It produces the printed 2018-2023 NET
validation table and per-river cumulative-net figures that support the manuscript claim that the
automated counter reproduces the historical run record (magnitude + daily correlation) back to 2018.

For the pre-2024 years both rivers report only the official **daily NET** count (no activity / up+down
split), from two OMNRF sources:
  * GANARASKA = `lomu_interview/data/Ganny_Total.csv`  (SPC code, signed daily CNT, 2017-2024)
  * CREDIT    = `model_pro/tech_data/data/tech_reviewed/2023-2018/<year>/FINAL_CREDIT_<year>.csv`
                (SPC code, signed daily CNT, DATE = MM/DD/YYYY)
Both net totals reconcile with the published `ganaraska_annual_counts.csv` (Ganaraska) / OMNRF Credit
annual counts. So this is a NET comparison (the metric OMNRF reports); model ACTIVITY is printed for
reference only (no tech activity exists for these years).

Same validated method as the 2024/2025 tools: counts RE-DERIVED from per-frame PARQUETS at --min-frames
(default 3), deploy species_conf triage, file-mtime dates, ASYMMETRIC net (up>=--up-thresh,
down>=--down-thresh).

Inputs
------
* Detect-store parquets (per-frame boxes + per-class probs) under ``config.MODEL_OUTPUTS`` — external.
* Technician truth: ``config.LOMU_INTERVIEW_DIR/data/Ganny_Total.csv`` (Ganaraska) and
  ``config.MODEL_PRO_DIR/tech_data/data/tech_reviewed/2023-2018/`` (Credit) — external sibling repos.
* Raw footage roots (``config.FOOTAGE_ROOT_FAST`` / ``config.FOOTAGE_ROOT``) — used only for file-mtime
  dates, never re-read for pixels here.

Outputs
-------
* A printed NET-upstream table (model / tech / %err) + model activity + per-species daily-r, and
  two per-river cumulative-net PNGs written under ``--out-dir`` (default ``figures/corpus_v6_<year>``).

Environment
-----------
Base env (pyarrow / numpy / pandas / optional scipy + matplotlib). No detector/DEIMv2 GPU code runs
here, so the ``.venv_deim`` environment is NOT required for this step.

  python -m salmonid.count.corpus_historical_vs_tech --year 2023
"""
from __future__ import annotations
import argparse, glob, os, collections, datetime
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from salmonid import config
try:
    from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    # Faithful fallback so daily-r still works without scipy: a guarded Pearson r that
    # returns NaN for degenerate (constant / <2-point) series instead of raising.
    def pearsonr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0: return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}  # mirrors salmonid.config.AUTO_FLOOR (Atlantic 1.01 = never auto-counted)
_PROB = ["prob_chinook", "prob_coho", "prob_atlantic", "prob_rainbow", "prob_brown"]
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}       # mirrors salmonid.config.UP_SIGN (per-site sign of +delta_x = upstream)
SPC = {73: 1, 75: 0, 76: 3, 77: 2, 78: 4}          # OMNRF SPC -> class idx (70 = Salmonid, excluded); mirrors salmonid.config.SPC_TO_CLASS
# Technician NET truth for the pre-2024 years (external sibling OMNRF repos; see config).
GANA_TECH = config.LOMU_INTERVIEW_DIR / "data" / "Ganny_Total.csv"
CREDIT_HIST = config.MODEL_PRO_DIR / "tech_data" / "data" / "tech_reviewed" / "2023-2018"


def _vd(d):
    """True iff ``d`` is a well-formed ISO date string 'YYYY-MM-DD' (cheap validity gate)."""
    return bool(d) and len(d) == 10 and d[4] == "-" and d[7] == "-" and d[:4].isdigit()


def load_tech(year: int, with_activity: bool = False):
    """t_net_d[('site',date)][sp], t_net_t[site][sp]; both rivers NET. If with_activity, also returns
    t_act_d, t_act_t = daily/season ACTIVITY (total fish regardless of travel) = Σ|CNT| per (date,species).
    NOTE the archived Vaki aggregate is upstream-dominant (down <1% for the salmonids), so tech activity ≈ net."""
    import pandas as pd
    t_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    uncl = collections.defaultdict(int)

    def add(site, date, code, cnt):
        if code == 70:
            uncl[site] += cnt; return
        if code not in SPC:
            return
        sp = SPC[code]
        t_net_d[(site, date)][sp] += cnt; t_net_t[site][sp] += cnt
        t_act_d[(site, date)][sp] += abs(cnt); t_act_t[site][sp] += abs(cnt)

    g = pd.read_csv(GANA_TECH); g = g[g.YEAR == year]
    for _, r in g.iterrows():
        date = str(r["DATE"])[:10]
        if _vd(date): add("Ganaraska", date, int(r["SPC"]), int(r["CNT"]))
    cf = CREDIT_HIST / str(year) / f"FINAL_CREDIT_{year}.csv"
    c = pd.read_csv(cf); c["d"] = pd.to_datetime(c["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    for _, r in c.iterrows():
        date = r["d"]
        if _vd(date): add("Credit", date, int(r["SPC"]), int(r["CNT"]))
    if with_activity:
        return t_net_d, t_net_t, uncl, t_act_d, t_act_t
    return t_net_d, t_net_t, uncl


def _r(md, td):
    """Pearson daily-r between two {date: count} maps over their union of dates.

    Missing dates on either side are treated as 0. Returns NaN when one series is all-zero
    (no signal to correlate). ``md`` = model daily counts, ``td`` = tech daily counts.
    """
    dates = sorted(set(md) | set(td))
    m = [md.get(x, 0) for x in dates]; t = [td.get(x, 0) for x in dates]
    return pearsonr(m, t)[0] if (any(m) and any(t)) else float("nan")


def main() -> None:
    """Reduce the corpus parquets for one year to NET-upstream counts and score vs tech truth.

    Reads CLI args (``--year`` required), loads the technician NET truth via ``load_tech``, walks the
    detect-store parquet shards, re-derives per-track species (mean-prob vote + deploy floor) and net
    direction (asymmetric delta_x rule), then prints the model/tech NET table, model activity, and
    per-river daily-r, and writes two per-river cumulative-net PNGs. Side effects: creates ``--out-dir``
    and writes figures there; prints to stdout. Returns None.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--shard-dirs", nargs="+", default=None)
    # Footage roots are consulted only for file-mtime dates: NVMe mirror first, HDD archive second.
    ap.add_argument("--footage-roots", nargs="+",
                    default=[str(config.FOOTAGE_ROOT_FAST), str(config.FOOTAGE_ROOT)])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--up-thresh", type=float, default=50.0)
    ap.add_argument("--down-thresh", type=float, default=300.0)
    args = ap.parse_args()
    yr = args.year
    shard_dirs = args.shard_dirs or [str(config.MODEL_OUTPUTS / f"run_corpus_v6_{yr}_shard_*")]
    out = Path(args.out_dir or f"figures/corpus_v6_{yr}"); out.mkdir(parents=True, exist_ok=True)
    froots = [Path(p) for p in args.footage_roots]
    MF, uT, dT = args.min_frames, args.up_thresh, args.down_thresh

    t_net_d, t_net_t, t_uncl = load_tech(yr)

    m_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    cache: dict[str, str] = {}
    cols = ["video_id", "track_id", "frame_number", "bbox_x1", "bbox_x2", *_PROB]
    n_tr = nfiles = 0
    for d in sorted({x for pat in shard_dirs for x in glob.glob(pat)}):
        for pf in glob.glob(os.path.join(d, "detections", "*.parquet")):
            t = pq.read_table(pf, columns=cols)
            if t.num_rows == 0: continue
            vid = t.column("video_id")[0].as_py(); site = vid.split("/")[0]
            if site not in UP_SIGN: continue
            date = cache.get(vid)
            if date is None:
                date = ""
                for fr in froots:
                    try: date = datetime.date.fromtimestamp(os.path.getmtime(fr / vid)).isoformat(); break
                    except OSError: continue
                cache[vid] = date
            if not _vd(date): continue
            nfiles += 1
            tid = t.column("track_id").to_numpy()
            frame = t.column("frame_number").to_numpy()
            cx = (t.column("bbox_x1").to_numpy().astype(np.float64) + t.column("bbox_x2").to_numpy().astype(np.float64)) / 2.0
            probs = np.column_stack([t.column(c).to_numpy() for c in _PROB])
            # Sort rows by (track_id, frame_number) so each track is a contiguous run.
            order = np.lexsort((frame, tid))
            tid_s, cx_s, probs_s = tid[order], cx[order], probs[order]
            uniq, idx, counts = np.unique(tid_s, return_index=True, return_counts=True)
            # Per-track species vote = argmax of the track's MEAN per-class probability (not a
            # per-frame majority): reduceat sums each track's rows, divided by its frame count.
            means = np.add.reduceat(probs_s, idx, axis=0) / counts[:, None]
            spk = means.argmax(axis=1); confk = means[np.arange(len(uniq)), spk]
            for k in range(len(uniq)):
                s = int(spk[k])
                # Deploy triage: drop tracks below the species conf floor (Atlantic floor 1.01 => never
                # auto-counted) or shorter than min_frames (min_frames=3 re-derived from the parquets).
                if confk[k] < AUTO_FLOOR[s] or counts[k] < MF: continue
                # Direction = mean-of-first-3 vs mean-of-last-3 box-centre x (robust to single-frame jitter).
                seg = cx_s[idx[k]: idx[k] + counts[k]]; m3 = min(3, len(seg))
                delta_x = seg[-m3:].mean() - seg[:m3].mean()
                n_tr += 1
                m_act_t[site][s] += 1
                sgn = delta_x * UP_SIGN[site]  # +ve => moving UPSTREAM for this site's orientation
                # ASYMMETRIC net rule: count +1 upstream only if travel >= up-thresh (50px, liberal),
                # -1 downstream only if travel >= down-thresh (300px, strict — suppresses milling); else 0.
                nc = 1 if (sgn > 0 and abs(delta_x) >= uT) else (-1 if (sgn < 0 and abs(delta_x) >= dT) else 0)
                m_net_d[(site, date)][s] += nc; m_net_t[site][s] += nc

    print(f"\n=== {yr} model v6 vs tech (HELD-OUT)  [min_frames={MF}, net up>={int(uT)}/down>={int(dT)}px] ===")
    print(f"re-derived tracks={n_tr:,} from {nfiles:,} videos | tech NET only (both rivers). "
          f"unclassified Salmonid net: Gan {t_uncl['Ganaraska']:,}, Cred {t_uncl['Credit']:,}")

    print("\nNET UPSTREAM (model / tech / %err):")
    for sp in range(5):
        cells = []
        for site in ["Ganaraska", "Credit"]:
            m, t = m_net_t[site][sp], t_net_t[site][sp]
            pct = f"{100*(m-t)/t:+.0f}%" if t else "n/a"
            cells.append(f"{m:>7,} / {t:>7,} ({pct})")
        print(f"  {SPECIES[sp]:13} Ganaraska {cells[0]:>24}   Credit {cells[1]:>24}")
    print("\nmodel ACTIVITY totals (no tech activity for these years):")
    for sp in range(5):
        print(f"  {SPECIES[sp]:13} Ganaraska {m_act_t['Ganaraska'][sp]:>8,}   Credit {m_act_t['Credit'][sp]:>8,}")

    for site in ["Ganaraska", "Credit"]:
        print(f"\n{site.upper()} {yr} net daily-r:")
        for sp in range(5):
            rn = _r({dt: m_net_d[(s, dt)][sp] for (s, dt) in m_net_d if s == site},
                    {dt: t_net_d[(s, dt)][sp] for (s, dt) in t_net_d if s == site})
            print(f"  {SPECIES[sp]:13} net r={rn:6.3f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for site in ["Ganaraska", "Credit"]:
            fig, axes = plt.subplots(5, 1, figsize=(12, 13))
            for sp in range(5):
                ax = axes[sp]
                md = {dt: m_net_d[(s, dt)][sp] for (s, dt) in m_net_d if s == site}
                td = {dt: t_net_d[(s, dt)][sp] for (s, dt) in t_net_d if s == site}
                dates = sorted(d for d in (set(md) | set(td)) if _vd(d))
                x = [np.datetime64(z) for z in dates]
                ym = np.cumsum([md.get(z, 0) for z in dates]); yt = np.cumsum([td.get(z, 0) for z in dates])
                ax.plot(x, yt, "-", lw=1.4, label="tech", color="#333")
                ax.plot(x, ym, "-", lw=1.4, label="model v6", color="#1f77b4")
                ax.set_title(f"{site} {SPECIES[sp]} cumulative net  r={_r(md, td):.3f}  (model {sum(md.values()):,} / tech {sum(td.values()):,})", fontsize=10)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.suptitle(f"{yr} {site} NET upstream (min_frames={MF}, asym up{int(uT)}/down{int(dT)}): model v6 vs tech [HELD-OUT]", y=1.0)
            fig.tight_layout()
            fp = out / f"{yr}_{site.lower()}_net_mf{MF}.png"
            fig.savefig(fp, dpi=140, bbox_inches="tight"); print(f"wrote {fp}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

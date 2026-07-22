"""
In plain terms: this file is a fair test of the automated fish counter for the year 2024, a year the
model was never trained on. It re-counts the fish the model found in the 2024 Credit and Ganaraska
footage and compares those counts — both the total number of each species seen and the net upstream
movement — against the counts human technicians made by hand that same year. Its numbers and day-by-day
comparison plots support the paper's multi-year validation of the counter (the abundance-index and
daily-correlation results and the reproduced run-timing).

Held-out 2024 both-river validation: model (v6 full run) vs technician counts — model never saw 2024.

Pipeline position
-----------------
This is an **eval/count** stage (data -> detect -> track -> **eval/count** -> results -> figures). It sits
downstream of the corpus detect+track run: the per-frame parquet detect-stores already exist; this module
re-derives per-track species and movement from them and scores the result against the 2024 technician truth.
It shares its counting method with `salmonid.count.corpus_fullyear_vs_tech`.

Method (identical to the validated fullyear path)
-------------------------------------------------
Counts are RE-DERIVED from the per-frame PARQUETS at --min-frames (default 3), using the deploy species_conf
triage floors, file-mtime video dates, and the ASYMMETRIC net rule (up>=--up-thresh / down>=--down-thresh).
Per-track delta_x is re-derived from bbox centroids (mean of last 3 frames minus mean of first 3).

Inputs
------
* Corpus detect-stores: ``config.MODEL_OUTPUTS / run_corpus_v6_2024_shard_*/detections/*.parquet`` (the
  parquet columns read are video_id, track_id, frame_number, bbox_x1/x2, and the five prob_* species scores).
* Raw footage roots (``config.FOOTAGE_ROOT_FAST`` NVMe mirror, then ``config.FOOTAGE_ROOT`` HDD) — used only
  for the video file mtime that dates each clip.
* Two 2024 technician sources (all that exists for 2024):
    - CREDIT  = Vaki "AbsDb" reviewed export (``sortname``=human species, ``Direction``), two season xlsx under
                ``config.MODEL_PRO_DIR / tech_data/data/tech_reviewed/2024``. One reviewed row = one fish ->
                gives ACTIVITY *and* net (raw-AbsDb net over-logs downstream, so net is a loose secondary).
    - GANARASKA = ``config.LOMU_INTERVIEW_DIR / data/Ganny_Total.csv`` — official OMNRF daily counts (SPC code,
                signed CNT = daily NET). Reported NET only (no separate up+down), so Ganaraska gets a NET
                comparison and model activity is printed for reference. NET total matches the published
                ganaraska_annual_counts.csv (2024 Chinook 19,514, Rainbow 9,032, Coho 3,201).

Outputs
-------
* Console: activity + net totals (model/tech/%err) per species per river, and per-river daily-r (Pearson).
* Figures: ``<out-dir>/2024_<site>_<metric>_mf<MF>.png`` for each (site, metric) that has tech (Credit
  activity, Credit net, Ganaraska net).

Manuscript claim
----------------
This is the held-out 2024 leg of the multi-year automated-counter validation (species never trained on 2024):
it supports the abundance-index / daily-correlation results and the reproduced run-timing phenology.

Environment
-----------
Base env (numpy / pyarrow / pandas / matplotlib). No detector or CUDA is loaded — this reads the already-written
parquet detect-store, so it does NOT need the DEIMv2 ``.venv_deim`` used for detection/training.

  python -m salmonid.count.corpus_2024_vs_tech --out-dir figures/corpus_v6_2024
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
    # Fallback Pearson r if SciPy is unavailable: returns (r, p) like scipy.stats.pearsonr, but
    # degenerate inputs (n<2 or a constant series, which have undefined correlation) yield NaN.
    def pearsonr(a, b):
        """Minimal (r, p) Pearson correlation via numpy; NaN r when correlation is undefined."""
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0: return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP
# Deploy species-confidence triage floors; Atlantic=1.01 => never auto-counted (review-only).
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}  # mirrors salmonid.config.AUTO_FLOOR
_PROB = ["prob_chinook", "prob_coho", "prob_atlantic", "prob_rainbow", "prob_brown"]
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}            # Gana up=Right(+dx), Credit up=Left(-dx); mirrors salmonid.config.UP_SIGN
SPC = {73: 1, 75: 0, 76: 3, 77: 2, 78: 4}               # OMNRF SPC code -> our class idx; mirrors salmonid.config.SPC_TO_CLASS
# Credit AbsDb reviewed exports (two seasons) under the sibling model_pro tech-data repo.
CREDIT_TECH = config.MODEL_PRO_DIR / "tech_data" / "data" / "tech_reviewed" / "2024"
CREDIT_FILES = [("Credit_2024_05_16_to_2024_09_03.xlsx", None, "2024-09-03"),
                ("Credit_2024_09_03_to_2024_11_27.xlsx", "2024-09-03", None)]
# Ganaraska official OMNRF daily NET counts (signed CNT per SPC code) in the sibling lomu_interview repo.
GANA_TECH = config.LOMU_INTERVIEW_DIR / "data" / "Ganny_Total.csv"


def _vd(d):
    """True iff ``d`` is a valid ``YYYY-MM-DD`` date string (cheap format check, not a calendar check)."""
    return bool(d) and len(d) == 10 and d[4] == "-" and d[7] == "-" and d[:4].isdigit()
def _sp(n):
    """Map a technician species name (``sortname``) to our class index 0-4, or None if unrecognised."""
    n = (n or "").strip().lower()
    if "chinook" in n: return 0
    if "coho" in n: return 1
    if "atlantic" in n: return 2
    if "rainbow" in n or "steelhead" in n: return 3
    if "brown" in n: return 4
    return None


def load_tech():
    """t_act_d/t_net_d keyed ('site',date)->sp; t_act_t/t_net_t keyed site->sp. Gana has net only."""
    import pandas as pd
    t_act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    uncl = collections.defaultdict(int)
    # --- Credit: AbsDb reviewed rows (activity + net) ---
    for fname, dmin, dmax in CREDIT_FILES:
        df = pd.read_excel(CREDIT_TECH / fname, sheet_name="AbsDb")
        df["d"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.strftime("%Y-%m-%d")
        for _, r in df.iterrows():
            date = r["d"]
            if not _vd(date): continue
            if dmin is not None and date < dmin: continue
            if dmax is not None and date >= dmax: continue
            sp = _sp(str(r.get("sortname") or ""))
            if sp is None:
                if "salmonid" in str(r.get("sortname") or "").lower(): uncl["Credit"] += 1
                continue
            # Signed movement per reviewed passage: +1 upstream, -1 downstream, 0 if direction unknown.
            mv = 1 if str(r.get("Direction") or "").strip().lower() == "up" else (
                -1 if str(r.get("Direction") or "").strip().lower() == "down" else 0)
            # Each reviewed row is one fish -> +1 activity always; net accumulates the signed movement.
            t_act_d[("Credit", date)][sp] += 1; t_act_t["Credit"][sp] += 1
            t_net_d[("Credit", date)][sp] += mv; t_net_t["Credit"][sp] += mv
    # --- Ganaraska: Ganny_Total daily signed NET ---
    g = pd.read_csv(GANA_TECH)
    g = g[g.YEAR == 2024]
    for _, r in g.iterrows():
        code = int(r["SPC"]); date = str(r["DATE"])[:10]
        if not _vd(date): continue
        if code not in SPC:
            uncl["Ganaraska"] += int(r["CNT"]) if code == 70 else 0
            continue
        sp = SPC[code]; cnt = int(r["CNT"])
        t_net_d[("Ganaraska", date)][sp] += cnt; t_net_t["Ganaraska"][sp] += cnt
    return t_act_d, t_net_d, t_act_t, t_net_t, uncl


def _r(md, td):
    """Pearson r between model and tech daily series (date->count dicts), aligned on the union of dates.

    Missing dates on either side count as 0. Returns NaN if either series is all-zero (no signal to correlate).
    """
    dates = sorted(set(md) | set(td))
    m = [md.get(x, 0) for x in dates]; t = [td.get(x, 0) for x in dates]
    return pearsonr(m, t)[0] if (any(m) and any(t)) else float("nan")


def main() -> None:
    """Re-derive 2024 counts from the parquet detect-store, score vs tech, and write per-panel figures.

    Reads the CLI args (shard globs, footage roots for dating, min-frames, net thresholds), loads the tech
    truth via ``load_tech``, aggregates per-track species/movement from every parquet, prints the
    activity/net totals and daily-r table, and saves one PNG per (site, metric) that has tech. No return value
    (side effects: stdout + files under --out-dir).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+",
                    default=[str(config.MODEL_OUTPUTS / "run_corpus_v6_2024_shard_*")])
    ap.add_argument("--footage-roots", nargs="+",
                    default=[str(config.FOOTAGE_ROOT_FAST), str(config.FOOTAGE_ROOT)])
    ap.add_argument("--out-dir", default="figures/corpus_v6_2024")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--up-thresh", type=float, default=50.0)
    ap.add_argument("--down-thresh", type=float, default=300.0)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    froots = [Path(p) for p in args.footage_roots]
    MF, uT, dT = args.min_frames, args.up_thresh, args.down_thresh

    t_act_d, t_net_d, t_act_t, t_net_t, t_uncl = load_tech()

    m_act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    cache: dict[str, str] = {}
    cols = ["video_id", "track_id", "frame_number", "bbox_x1", "bbox_x2", *_PROB]
    n_tr = nfiles = 0
    for d in sorted({x for pat in args.shard_dirs for x in glob.glob(pat)}):
        for pf in glob.glob(os.path.join(d, "detections", "*.parquet")):
            t = pq.read_table(pf, columns=cols)
            if t.num_rows == 0: continue
            vid = t.column("video_id")[0].as_py()
            site = vid.split("/")[0]
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
            # Track x-centroid per frame = midpoint of the bbox (used for the up/downstream delta_x).
            cx = (t.column("bbox_x1").to_numpy().astype(np.float64) + t.column("bbox_x2").to_numpy().astype(np.float64)) / 2.0
            probs = np.column_stack([t.column(c).to_numpy() for c in _PROB])
            # Sort rows by track then by frame so each track's frames are contiguous and time-ordered.
            order = np.lexsort((frame, tid))
            tid_s, cx_s, probs_s = tid[order], cx[order], probs[order]
            uniq, idx, counts = np.unique(tid_s, return_index=True, return_counts=True)
            # Per-track species vote = argmax of the frame-averaged prob vector (reduceat sums each
            # contiguous track block at its start offset; /counts turns the block sum into a mean).
            means = np.add.reduceat(probs_s, idx, axis=0) / counts[:, None]
            spk = means.argmax(axis=1)
            confk = means[np.arange(len(uniq)), spk]
            for k in range(len(uniq)):
                s = int(spk[k])
                # Deploy triage: drop below the species AUTO_FLOOR (Atlantic 1.01 => always dropped) and
                # any track seen in fewer than MIN_FRAMES frames (re-derived min_frames=3 stability gate).
                if confk[k] < AUTO_FLOOR[s] or counts[k] < MF: continue
                # delta_x = mean of last-3 centroids minus mean of first-3 (net travel across the tunnel).
                seg = cx_s[idx[k]: idx[k] + counts[k]]; m3 = min(3, len(seg))
                delta_x = seg[-m3:].mean() - seg[:m3].mean()
                n_tr += 1
                m_act_d[(site, date)][s] += 1; m_act_t[site][s] += 1
                # Asymmetric net rule: sign of delta_x*UP_SIGN gives direction; a track counts UPSTREAM only
                # if it moved >= up-thresh (50px, liberal) and DOWNSTREAM only if >= down-thresh (300px,
                # strict) — the wider down bar suppresses milling fish being logged as downstream passages.
                sgn = delta_x * UP_SIGN[site]
                nc = 1 if (sgn > 0 and abs(delta_x) >= uT) else (-1 if (sgn < 0 and abs(delta_x) >= dT) else 0)
                m_net_d[(site, date)][s] += nc; m_net_t[site][s] += nc

    print(f"\n=== 2024 model v6 vs tech (HELD-OUT)  [min_frames={MF}, net up>={int(uT)}/down>={int(dT)}px] ===")
    print(f"re-derived tracks={n_tr:,} from {nfiles:,} videos | tech unclassified: Cred 'Salmonid' {t_uncl['Credit']:,}, Gana SPC70 {t_uncl['Ganaraska']:,}")
    print("  (Credit tech = AbsDb reviewed rows: activity+net.  Ganaraska tech = Ganny_Total daily NET only.)")

    def line(site, sp, mt, tt, has):
        """Format one 'model / tech (%err)' cell; 'n/a' when this site has no tech for the metric (``has``)."""
        m, t = mt[site][sp], tt[site][sp]
        if not has: return f"{m:>8,} / {'n/a':>8}"
        pct = f"{100*(m-t)/t:+.0f}%" if t else "n/a"
        return f"{m:>8,} / {t:>8,} ({pct})"

    print("\nACTIVITY (model / tech / %err):   [Ganaraska tech activity not in historical file]")
    for sp in range(5):
        print(f"  {SPECIES[sp]:13} Credit {line('Credit',sp,m_act_t,t_act_t,True):>22}   Ganaraska {line('Ganaraska',sp,m_act_t,t_act_t,False):>14}")
    print("\nNET UPSTREAM (model / tech / %err):")
    for sp in range(5):
        print(f"  {SPECIES[sp]:13} Credit {line('Credit',sp,m_net_t,t_net_t,True):>22}   Ganaraska {line('Ganaraska',sp,m_net_t,t_net_t,True):>22}")

    for site in ["Ganaraska", "Credit"]:
        print(f"\n{site.upper()} 2024 daily-r:")
        for sp in range(5):
            ra = _r({dt: m_act_d[(s, dt)][sp] for (s, dt) in m_act_d if s == site},
                    {dt: t_act_d[(s, dt)][sp] for (s, dt) in t_act_d if s == site})
            rn = _r({dt: m_net_d[(s, dt)][sp] for (s, dt) in m_net_d if s == site},
                    {dt: t_net_d[(s, dt)][sp] for (s, dt) in t_net_d if s == site})
            act_str = f"activity r={ra:6.3f}" if site == "Credit" else "activity r=  (n/a)"
            print(f"  {SPECIES[sp]:13} {act_str}   net r={rn:6.3f}")

    # figures: one per (site, metric) that has tech
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # (site, metric, model-dict, tech-dict, cumulative?) — net panels plot the running cumulative
        # net (the seasonal run curve); the activity panel plots raw daily counts.
        panels = [("Credit", "activity", m_act_d, t_act_d, False),
                  ("Credit", "net", m_net_d, t_net_d, True),
                  ("Ganaraska", "net", m_net_d, t_net_d, True)]
        for site, kind, md_all, td_all, cum in panels:
            fig, axes = plt.subplots(5, 1, figsize=(12, 13))
            for sp in range(5):
                ax = axes[sp]
                md = {dt: md_all[(s, dt)][sp] for (s, dt) in md_all if s == site}
                td = {dt: td_all[(s, dt)][sp] for (s, dt) in td_all if s == site}
                dates = sorted(d for d in (set(md) | set(td)) if _vd(d))
                x = [np.datetime64(z) for z in dates]
                yt = [td.get(z, 0) for z in dates]; ym = [md.get(z, 0) for z in dates]
                if cum: yt = np.cumsum(yt); ym = np.cumsum(ym)
                ax.plot(x, yt, "-", lw=1.4, label="tech", color="#333")
                ax.plot(x, ym, "-", lw=1.4, label="model v6", color=("#1f77b4" if cum else "#d62728"))
                ax.set_title(f"{site} {SPECIES[sp]} {kind}  r={_r(md, td):.3f}  (model {sum(md.values()):,} / tech {sum(td.values()):,})", fontsize=10)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.suptitle(f"2024 {site} {kind} (min_frames={MF}{', asym up'+str(int(uT))+'/down'+str(int(dT)) if kind=='net' else ''}): model v6 vs tech [HELD-OUT]", y=1.0)
            fig.tight_layout()
            fp = out / f"2024_{site.lower()}_{kind}_mf{MF}.png"
            fig.savefig(fp, dpi=140, bbox_inches="tight"); print(f"wrote {fp}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

"""STATUS: only ``load_tech()`` in this file is on the published path.
``load_tech()`` reads the technician count records for the season(s) named below and is
imported by salmonid/count/corpus_countr_from_store.py, which produced the published counts.
Everything else in this file (the command-line entry point, the per-frame file reading, the
species-confidence floors) is the counting engine of the earlier two-stage pipeline: it is
kept for provenance, its inputs are not deposited, and none of its numbers are in the
manuscript. Statements below about "the paper" describe an earlier draft. The method of
record is docs/COUNTING_METHOD.md.
Tier: 2 (technician records are external; see docs/DATA_AVAILABILITY.md).

In plain terms: this file tallies how many fish of each species the automated system counted
each day and checks those daily tallies against the official counts made by the field
technicians. For every species it reports two numbers — total fish seen (activity) and net
upstream movement (fish moving up minus fish moving down) — and measures how closely the
automated tallies track the technicians' day-by-day counts. This is where the paper's
abundance-index agreement comes from (the Ganaraska daily correlation of about 0.98-0.99 for
activity, and the up-vs-down net rule that matches how the technicians count).

Reference counting engine — re-derive per-species counts from the per-frame detection
parquets and compare them, by day, to the deduplicated technician ground truth.

Pipeline position
-----------------
    data -> detect -> track -> [THIS: eval/count] -> results -> figures

This is the counting stage of the two-stage (RT-DETR detector + DINOv3 v6 crop classifier)
full-corpus run. It does **not** run any model: it reads the per-frame PARQUETS a
detection/tracking run already wrote and re-derives counts on CPU using footage-drive file
metadata only (no GPU, no torch). Runs in the base Python env, NOT the DEIMv2 .venv_deim.

Videos are dated by file mtime (validated 100% == the technician date).

What it computes
----------------
Counts are RE-DERIVED from the per-frame PARQUETS at --min-frames (default 3), which recovers
the short tracks that the min_frames=5 tracks.sqlite filter dropped (Rainbow especially). For
every (river, species) it reports two metrics:
  * ACTIVITY (total fish) — every qualifying track counts once; and
  * NET UPSTREAM (up - down) — under an ASYMMETRIC direction rule (liberal up, strict down):
    a track counts UPSTREAM if |delta_x| >= --up-thresh (50 px) and its motion sign matches the
    site's upstream convention, DOWNSTREAM if |delta_x| >= --down-thresh (300 px), else
    Stay/Unknown. delta_x is re-derived per track from the bbox centroids (mean of last 3 -
    mean of first 3, by frame order), matching salmonid/util/schema.py. The deploy species_conf
    triage floors (AUTO_FLOOR; Atlantic = 1.01 => never auto-counted, review-only) are applied.

This module also defines the canonical ``load_tech()`` reused across the count package: the
deduplicated official technician counts are the ground truth (raw per-event Vaki rows count
passages / re-triggers and are NOT used as a fish count).

Inputs
------
  * --shard-dirs   glob(s) of detection-store shard dirs (each has detections/*.parquet),
                   e.g. under config.MODEL_OUTPUTS.
  * --footage-root footage tree, used only to date each video by file mtime; defaults to
                   config.FOOTAGE_ROOT_FAST.
  * --tech-dir     technician CSV directory (Ganaraska_/Credit_Counter_Analysis_2025).

Outputs
-------
  * Printed ACTIVITY / NET totals tables (model / tech / %err) and per-species Ganaraska daily
    Pearson-r (activity | net).
  * Per-species Ganaraska activity + cumulative-net PNG figures in --out-dir.
  * Optional daily CSVs (--dump-daily-prefix) feeding salmonid.eval.count_r_gate.

Manuscript claim
----------------
Supports the abundance-index result: Ganaraska daily-r 0.98-0.99 (activity) and the asymmetric
up>=50/down>=300 px net rule as the family optimum matching the technicians' convention.

Example
-------
    python -m salmonid.count.corpus_fullyear_vs_tech \\
        --shard-dirs "$SALMONID_PROJECTS_ROOT/model_outputs/run_corpus_v6_2025_shard_*" \\
        --footage-root "$SALMONID_FOOTAGE_ROOT_FAST" --year 2025 \\
        --out-dir figures/corpus_v6_2025_fullyear
"""
from __future__ import annotations
import argparse, csv, glob, os, collections, datetime
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from salmonid import config
try:
    from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    def pearsonr(a, b):
        """Minimal Pearson-r fallback when scipy is absent: returns (r, p) with p unused here.

        Mirrors scipy's return shape; yields nan for degenerate input (n<2 or a constant
        series) so the caller's ``any(m) and any(t)`` guard still behaves identically.
        """
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0: return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}  # mirrors salmonid.config.AUTO_FLOOR (Atlantic 1.01 => review-only)
_PROB = ["prob_chinook", "prob_coho", "prob_atlantic", "prob_rainbow", "prob_brown"]
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}  # mirrors salmonid.config.UP_SIGN (upstream sign of delta_x per site)


def _vd(d):
    """True iff ``d`` is a valid ISO date string ``YYYY-MM-DD`` (10 chars, dashes, numeric year)."""
    return bool(d) and len(d) == 10 and d[4] == "-" and d[7] == "-" and d[:4].isdigit()
def _sp(n):
    """Map a free-text species name to our class index (0-4), or None if unrecognised.

    "steelhead" is folded into Rainbow (3); unmatched names return None so the caller can
    tally them as "unclassified" rather than mis-assigning a species.
    """
    n = (n or "").strip().lower()
    if "chinook" in n: return 0
    if "coho" in n: return 1
    if "atlantic" in n: return 2
    if "rainbow" in n or "steelhead" in n: return 3
    if "brown" in n: return 4
    return None
def _mv(s):
    """Map a technician movement string to +1 (up) / -1 (down) / 0 (stay/unknown)."""
    s = (s or "").strip().strip('"').lower(); return 1 if s == "up" else (-1 if s == "down" else 0)


def load_tech(rw: Path):
    """Load the canonical deduplicated technician counts from the tech CSV directory ``rw``.

    This is the count package's single authoritative reader of the technician ground truth.
    It parses each river's official counter export, drops false triggers and rows with an
    invalid date, maps species/movement via ``_sp``/``_mv``, and honours the per-row ``count``
    field (a technician row can represent >1 fish, so counts are summed, never row-counted).

    Sources:
      * Ganaraska ``all_counts_gan.csv`` (';'-delimited): count-weighted, movement-aware.
      * Credit ``Credit_river_Counts_*.csv``: one fish per row (Direction, sortname).
      * Credit ``fish_counts_export_*.csv`` (skipping ``vaki_matched``): count-weighted.

    Args:
        rw: technician CSV directory (e.g. ``docs/Riverwatcher_2025``).

    Returns:
        (act_d, net_d, act_t, net_t, uncl):
          * act_d / net_d — {(site, date): {species: activity | net}} daily dicts.
          * act_t / net_t — {site: {species: total}} season totals.
          * uncl          — {site: count} of rows whose species could not be classified.
    """
    act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    uncl = collections.defaultdict(int)
    g = rw / "Ganaraska_Counter_Analysis_2025" / "all_counts_gan.csv"
    if g.exists():
        for r in csv.DictReader(open(g, encoding="utf-8", errors="replace"), delimiter=";"):
            if (r.get("false_trigger") or "").strip().strip('"') == "1": continue
            sp = _sp(r.get("species_main") or ""); date = (r.get("ts") or "").strip().strip('"')[:10]
            try: n = int(float(r.get("count") or 1))
            except ValueError: n = 1
            if not _vd(date): continue
            if sp is None: uncl["Ganaraska"] += n; continue
            mv = _mv(r.get("movement") or "")
            act_d[("Ganaraska", date)][sp] += n; act_t["Ganaraska"][sp] += n
            net_d[("Ganaraska", date)][sp] += n * mv; net_t["Ganaraska"][sp] += n * mv
    cdir = rw / "Credit_Counter_Analysis_2025" / "data" / "raw_data"
    for cf in glob.glob(str(cdir / "Credit_river_Counts_*.csv")):
        for r in csv.DictReader(open(cf, encoding="utf-8", errors="replace")):
            date = (r.get("DateTime") or "")[:10]; sp = _sp(r.get("sortname") or "")
            if not _vd(date): continue
            if sp is None: uncl["Credit"] += 1; continue
            mv = _mv(r.get("Direction") or "")
            act_d[("Credit", date)][sp] += 1; act_t["Credit"][sp] += 1
            net_d[("Credit", date)][sp] += mv; net_t["Credit"][sp] += mv
    for cf in glob.glob(str(cdir / "fish_counts_export_*.csv")):
        if "vaki_matched" in cf: continue
        for r in csv.DictReader(open(cf, encoding="utf-8", errors="replace")):
            if (r.get("false_trigger") or "0").strip() == "1": continue
            date = (r.get("ts") or "")[:10]; sp = _sp(r.get("species") or "")
            if sp is None or not _vd(date): continue
            try: n = int(float(r.get("count") or 1))
            except ValueError: n = 1
            mv = _mv(r.get("movement") or "")
            act_d[("Credit", date)][sp] += n; act_t["Credit"][sp] += n
            net_d[("Credit", date)][sp] += n * mv; net_t["Credit"][sp] += n * mv
    return act_d, net_d, act_t, net_t, uncl


def _r(md, td):
    """Pearson-r between two {date: value} series over their union of dates (missing => 0).

    Returns nan unless both aligned series contain a non-zero value (empty/all-zero series
    have no meaningful correlation).
    """
    dates = sorted(set(md) | set(td))
    m = [md.get(x, 0) for x in dates]; t = [td.get(x, 0) for x in dates]
    return pearsonr(m, t)[0] if (any(m) and any(t)) else float("nan")


def main() -> None:
    """Parse args, re-derive model counts from the parquets, and emit tables + figures.

    Side effects: prints the ACTIVITY / NET totals tables and Ganaraska daily-r to stdout;
    writes per-species Ganaraska PNG figures to --out-dir; optionally dumps daily CSVs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True)
    ap.add_argument("--footage-root", default=str(config.FOOTAGE_ROOT_FAST))
    ap.add_argument("--tech-dir", default="docs/Riverwatcher_2025")
    ap.add_argument("--year", default="2025")
    ap.add_argument("--out-dir", default="figures/corpus_fullyear")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--up-thresh", type=float, default=50.0)
    ap.add_argument("--down-thresh", type=float, default=300.0)
    ap.add_argument("--dump-daily-prefix", default="",
                    help="Write <prefix>_activity.csv and <prefix>_net.csv (date,site,species,model,tech) so the "
                         "two-stage counts flow through the same salmonid.eval.count_r_gate as the single-stage adapter.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    froot = Path(args.footage_root)
    MF, uT, dT = args.min_frames, args.up_thresh, args.down_thresh

    t_act_d, t_net_d, t_act_t, t_net_t, t_uncl = load_tech(Path(args.tech_dir))

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
                # Date each video by file mtime (validated 100% == the technician date); cache per vid.
                try: date = datetime.date.fromtimestamp(os.path.getmtime(froot / vid)).isoformat()
                except OSError: date = ""
                cache[vid] = date
            if not _vd(date): continue
            nfiles += 1
            tid = t.column("track_id").to_numpy()
            frame = t.column("frame_number").to_numpy()
            cx = (t.column("bbox_x1").to_numpy().astype(np.float64) + t.column("bbox_x2").to_numpy().astype(np.float64)) / 2.0
            probs = np.column_stack([t.column(c).to_numpy() for c in _PROB])
            order = np.lexsort((frame, tid))            # sort by track, then frame
            tid_s, cx_s, probs_s = tid[order], cx[order], probs[order]
            # One species vote per track: mean of the per-frame probabilities, then argmax.
            uniq, idx, counts = np.unique(tid_s, return_index=True, return_counts=True)
            means = np.add.reduceat(probs_s, idx, axis=0) / counts[:, None]
            spk = means.argmax(axis=1)
            confk = means[np.arange(len(uniq)), spk]
            for k in range(len(uniq)):
                s = int(spk[k])
                # Deploy species_conf triage floor AND min_frames re-derivation (default 3):
                # drops low-confidence / too-short tracks (recovering the min_frames=5 dropouts).
                if confk[k] < AUTO_FLOOR[s] or counts[k] < MF:
                    continue
                seg = cx_s[idx[k]: idx[k] + counts[k]]   # centroids in frame order
                # delta_x = mean of last-3 centroids - mean of first-3 (matches schema.py).
                m3 = min(3, len(seg))
                delta_x = seg[-m3:].mean() - seg[:m3].mean()
                n_tr += 1
                m_act_d[(site, date)][s] += 1; m_act_t[site][s] += 1
                sgn = delta_x * UP_SIGN[site]
                # Asymmetric net rule: UP if moved upstream & |delta_x|>=50px (liberal), DOWN if
                # moved downstream & |delta_x|>=300px (strict; kills milling), else Stay (0).
                nc = 1 if (sgn > 0 and abs(delta_x) >= uT) else (-1 if (sgn < 0 and abs(delta_x) >= dT) else 0)
                m_net_d[(site, date)][s] += nc; m_net_t[site][s] += nc

    if args.dump_daily_prefix:
        for metric, md in [("activity", m_act_d), ("net", m_net_d)]:
            td = t_act_d if metric == "activity" else t_net_d
            rows = []
            for (site, date) in sorted(set(md) | set(td)):
                for si in range(5):
                    mv = md.get((site, date), {}).get(si, 0)
                    tv = td.get((site, date), {}).get(si, 0)
                    if mv == 0 and tv == 0:
                        continue
                    rows.append({"date": date, "site": site, "species": SPECIES[si], "model": mv, "tech": tv})
            outp = f"{args.dump_daily_prefix}_{metric}.csv"
            with open(outp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["date", "site", "species", "model", "tech"])
                w.writeheader(); w.writerows(rows)
            print(f"dumped {len(rows)} daily rows ({metric}) -> {outp}")

    sites = ["Ganaraska", "Credit"]
    print(f"\n=== {args.year} FULL-YEAR  model v6 vs tech  [min_frames={MF}, net up>={int(uT)}/down>={int(dT)}px] ===")
    print(f"re-derived tracks={n_tr:,} from {nfiles:,} videos | tech unclassified: Gan {t_uncl['Ganaraska']:,}, Cred {t_uncl['Credit']:,}")

    def tbl(title, mt, tt):
        """Print a per-species model/tech totals table (Ganaraska | Credit) with signed %err."""
        print(f"\n{title} totals (model / tech / %err):")
        print(f"  {'species':13}{'Ganaraska':>24}{'Credit':>24}")
        for sp in range(5):
            cells = []
            for site in sites:
                m, t = mt[site][sp], tt[site][sp]
                pct = f"{100*(m-t)/t:+.0f}%" if t else "n/a"
                cells.append(f"{m:,}/{t:,} ({pct})")
            print(f"  {SPECIES[sp]:13}{cells[0]:>24}{cells[1]:>24}")

    tbl("ACTIVITY", m_act_t, t_act_t)
    tbl("NET UPSTREAM (up-down)", m_net_t, t_net_t)
    print("\nGANARASKA full-year daily-r  (activity | net):")
    for sp in range(5):
        ra = _r({dt: m_act_d[("Ganaraska", dt)][sp] for (s, dt) in m_act_d if s == "Ganaraska"},
                {dt: t_act_d[("Ganaraska", dt)][sp] for (s, dt) in t_act_d if s == "Ganaraska"})
        rn = _r({dt: m_net_d[("Ganaraska", dt)][sp] for (s, dt) in m_net_d if s == "Ganaraska"},
                {dt: t_net_d[("Ganaraska", dt)][sp] for (s, dt) in t_net_d if s == "Ganaraska"})
        print(f"  {SPECIES[sp]:13} activity r={ra:6.3f}   net r={rn:6.3f}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for kind, md_all, td_all, cum in [("activity", m_act_d, t_act_d, False), ("net", m_net_d, t_net_d, True)]:
            fig, axes = plt.subplots(5, 1, figsize=(12, 13))
            for sp in range(5):
                ax = axes[sp]
                md = collections.defaultdict(int); td = collections.defaultdict(int)
                for (s, dt) in md_all:
                    if s == "Ganaraska": md[dt] += md_all[(s, dt)][sp]
                for (s, dt) in td_all:
                    if s == "Ganaraska": td[dt] += td_all[(s, dt)][sp]
                dates = sorted(d for d in (set(md) | set(td)) if _vd(d))
                x = [np.datetime64(z) for z in dates]
                yt = [td[z] for z in dates]; ym = [md[z] for z in dates]
                if cum: yt = np.cumsum(yt); ym = np.cumsum(ym)
                ax.plot(x, yt, "-", lw=1.4, label="tech", color="#333")
                ax.plot(x, ym, "-", lw=1.4, label="model v6", color=("#1f77b4" if cum else "#d62728"))
                rr = _r(md, td)
                ax.set_title(f"Ganaraska {SPECIES[sp]} {kind}  r={rr:.3f}  (model {sum(md.values()):,} / tech {sum(td.values()):,})", fontsize=10)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            fig.suptitle(f"{args.year} {kind} (min_frames={MF}{', asym up'+str(int(uT))+'/down'+str(int(dT)) if kind=='net' else ''}): model v6 vs tech (Ganaraska)", y=1.0)
            fig.tight_layout()
            fp = out / f"{args.year}_fullyear_{kind}_mf{MF}_ganaraska.png"
            fig.savefig(fp, dpi=140, bbox_inches="tight"); print(f"wrote {fp}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

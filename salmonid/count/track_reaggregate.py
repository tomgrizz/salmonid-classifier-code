"""In plain terms: this file tests how many extra fish we can recover by counting
shorter camera sightings. When the counter first ran it ignored any fish it saw for
fewer than 5 video frames; here we re-count the already-saved sightings at looser
cut-offs (2, 3, or 5 frames) and check how close each choice comes to the technicians'
full-season counts. It doesn't make a figure of its own, but it is the check that led
the paper to keep fish seen in at least 3 frames — the counting setting used throughout
the counts-and-movement results.

min_frames sweep quantifying short-track recovery.

Post-processing re-aggregation (**NO re-inference**): re-derive per-species *activity* counts
directly from the saved per-frame detection PARQUETS at a tunable ``min_frames`` threshold,
recovering the short tracks that the deploy-time ``min_frames=5`` filter dropped when it wrote
``tracks.sqlite``. For each candidate threshold it applies the deploy species-confidence triage
(``AUTO_FLOOR``), dates each clip by its footage-file mtime, and reports activity %err plus the
Ganaraska daily correlation-r against the FULL-YEAR technician truth — quantifying how much of the
Coho / Rainbow under-count is recoverable from the already-saved data alone. This motivates the
manuscript's ``min_frames=3`` re-derivation choice (see docs/counting_and_movement.md).

Pipeline position: post-track / eval-count. It consumes the detect -> track outputs (the per-frame
parquets) and produces a diagnostic count sweep; it feeds no figure directly but underpins the
min_frames count knob used throughout the count/movement results.

Inputs   : per-frame detection parquets under a corpus run's ``detections/`` dir (one per video);
           raw footage (for file mtimes -> clip dates); the full-year technician CSVs (--tech-dir).
Outputs  : a stdout table of {min_frames x site x species} model totals, %err and Ganaraska
           daily-r. No files are written.
Environment: base Python env (CPU only, plus G:/F: filesystem metadata — no GPU / DEIMv2 needed).

  python scripts/_track_reaggregate.py --year 2025 --min-frames 2,3,5
"""
from __future__ import annotations
import argparse, glob, os, collections, datetime
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from salmonid import config
from salmonid.count.corpus_fullyear_vs_tech import load_tech, SPECIES, AUTO_FLOOR, _vd
try: from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    def pearsonr(a, b):
        """Fallback Pearson r if SciPy is absent: returns (r, p) but only r is consumed here."""
        a=np.asarray(a,float); b=np.asarray(b,float)
        if len(a)<2 or a.std()==0 or b.std()==0: return (float("nan"),1.0)
        return (float(np.corrcoef(a,b)[0,1]),0.0)

_PROB = ["prob_chinook","prob_coho","prob_atlantic","prob_rainbow","prob_brown"]


def main():
    """Sweep min_frames over the saved parquets and print the count-recovery table.

    Reads each corpus shard's per-frame detection parquets, re-aggregates them into per-track
    species votes at every requested ``min_frames``, applies the deploy AUTO_FLOOR triage, and
    prints per-species Ganaraska/Credit model totals, %err vs full-year tech, and (Ganaraska)
    the daily correlation-r. Side effects: reads the filesystem (parquets + footage mtimes) and
    writes a summary table to stdout; no files are produced.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2025")
    # {year} is left as a literal placeholder here and filled by .format(year=...) below; the
    # dir resolves to the same G:/Projects/model_outputs corpus-run location as before.
    ap.add_argument("--shard-glob", default=str(config.MODEL_OUTPUTS / "run_corpus_v6_{year}_shard_*"))
    ap.add_argument("--footage-root", default=str(config.FOOTAGE_ROOT_FAST))
    ap.add_argument("--tech-dir", default="docs/Riverwatcher_2025")
    ap.add_argument("--min-frames", default="2,3,5")
    args = ap.parse_args()
    mfs = sorted({int(x) for x in args.min_frames.split(",")})
    froot = Path(args.footage_root)

    t_act_d, _, t_act_t, _, _ = load_tech(Path(args.tech_dir))

    # model: per min_frames -> totals[site][sp] and daily[(site,date)][sp]
    tot = {mf: collections.defaultdict(lambda: collections.defaultdict(int)) for mf in mfs}
    day = {mf: collections.defaultdict(lambda: collections.defaultdict(int)) for mf in mfs}
    mtime_cache: dict[str, str] = {}
    nfiles = 0
    for d in sorted(glob.glob(args.shard_glob.format(year=args.year))):
        for pf in glob.glob(os.path.join(d, "detections", "*.parquet")):
            t = pq.read_table(pf, columns=["video_id", "track_id", *_PROB])
            if t.num_rows == 0:
                continue
            nfiles += 1
            vid = t.column("video_id")[0].as_py()
            site = vid.split("/")[0]
            if site not in ("Ganaraska", "Credit"):
                continue
            date = mtime_cache.get(vid)
            if date is None:
                try: date = datetime.date.fromtimestamp(os.path.getmtime(froot / vid)).isoformat()
                except OSError: date = ""
                mtime_cache[vid] = date
            if not _vd(date):
                continue
            tid = t.column("track_id").to_numpy()
            probs = np.column_stack([t.column(c).to_numpy() for c in _PROB])  # (N,5)
            # Per-track species vote = argmax of the MEAN per-frame probability over the track's
            # frames. Sort by track_id so reduceat can sum contiguous per-track blocks in one pass.
            order = np.argsort(tid, kind="stable")
            tid_s, probs_s = tid[order], probs[order]
            uniq, idx, counts = np.unique(tid_s, return_index=True, return_counts=True)
            means = np.add.reduceat(probs_s, idx, axis=0) / counts[:, None]
            sp = means.argmax(axis=1)
            conf = means[np.arange(len(uniq)), sp]
            for k in range(len(uniq)):
                s = int(sp[k])
                if conf[k] < AUTO_FLOOR[s]:        # deploy triage
                    continue
                # counts[k] = frames in this track; this is the re-aggregation lever — a track
                # counts once for every candidate min_frames it survives (recovering short tracks).
                for mf in mfs:
                    if counts[k] >= mf:
                        tot[mf][site][s] += 1
                        day[mf][(site, date)][s] += 1
    print(f"\n=== {args.year} post-reaggregation from parquets ({nfiles:,} non-empty videos) ===")
    sites = ["Ganaraska", "Credit"]
    # Class 2 (Atlantic) is omitted: AUTO_FLOOR[2]=1.01 means it is never auto-counted (review-only),
    # so its recovery sweep is degenerate. Report only the four auto-counted species.
    for sp in (0, 1, 3, 4):
        print(f"\n{SPECIES[sp]}  (tech: Gana {t_act_t['Ganaraska'][sp]:,} / Credit {t_act_t['Credit'][sp]:,}):")
        print(f"  {'min_frames':>10}{'Gana model':>12}{'Gana %err':>11}{'Gana day-r':>12}{'Credit model':>14}{'Credit %err':>12}")
        for mf in mfs:
            row = f"  {mf:>10}"
            for site in sites:
                m = tot[mf][site][sp]; tt = t_act_t[site][sp]
                pct = 100*(m-tt)/tt if tt else float("nan")
                if site == "Ganaraska":
                    md = {dt: day[mf][("Ganaraska", dt)][sp] for (s, dt) in day[mf] if s == "Ganaraska"}
                    td = {dt: t_act_d[("Ganaraska", dt)][sp] for (s, dt) in t_act_d if s == "Ganaraska"}
                    dates = sorted(set(md) | set(td))
                    r = pearsonr([md.get(x,0) for x in dates], [td.get(x,0) for x in dates])[0] if (m and sum(td.values())) else float("nan")
                    row += f"{m:>12,}{pct:>10.0f}%{r:>12.3f}"
                else:
                    row += f"{m:>14,}{pct:>11.0f}%"
            print(row)


if __name__ == "__main__":
    main()

"""In plain terms: this script is an automatic safety check that runs while the computer counts
fish across many years of footage. After the software has counted a year, it compares its daily
fish tallies against the counts the human technicians recorded for the same days and gives a simple
GO / NO-GO signal, so a year where the automated counter has drifted off track gets caught and
paused before it spoils the results. It does not produce a number or figure in the paper itself, but
it protects the day-to-day agreement between automated and technician counts (the "activity daily-r",
around 0.95-0.998) that the paper's abundance-index results rest on.

GO/NO-GO activity daily-r gate from a corpus tracks.sqlite vs reviewed technician counts.

Pipeline position
-----------------
data -> detect -> track -> **eval/count (this module)** -> results -> figures. A cheap, CPU-only
*checkpoint* that runs between shards/years of the multi-year corpus count: once a year's
detect+track pass has written its ``tracks.sqlite``, this gate decides whether the automated
counter is still tracking the technicians well enough to keep going, or whether the run should be
paused and investigated. It is an operational safety gate, not a reported manuscript figure — but
it guards the very ACTIVITY daily-r that underpins the manuscript's abundance-index / daily-count
claims (2025 baseline activity r: Chinook .998 / Rainbow .994 / Brown .973 / Coho .945).

What it does
------------
Reads an ALREADY-COMPUTED corpus ``tracks.sqlite`` (no GPU, no re-inference) and reuses the eval's
exact tech-matching ((site, timestamp_folder, mp4) -> tech event date) so the daily-r is directly
comparable to the v2 baseline. Applies the deploy review-triage species_conf floors (``AUTO_FLOOR``,
mirroring inference/config auto_count_*), restricts model counts to the tech-referenced videos (the
same set the technicians reviewed), correlates per-species daily counts, and prints a verdict.

GATE = pass if every gated auto-species (Chinook/Coho/Rainbow/Brown) with enough tech data has
Pearson daily-r >= --r-min AND model/tech count ratio within [1/--count-tol, --count-tol]. ATLANTIC
is review-only (reported, never gated). NET is intentionally NOT gated (Ganaraska-Rainbow net is a
known milling limitation, not a model failure).

Inputs
------
* ``--shard-dirs``: one or more corpus run dirs (globs ok), each holding a ``tracks.sqlite``. On the
  original workstation these live under ``config.MODEL_OUTPUTS`` (e.g. ``run_corpus_v6_2025_shard_*``).
* ``--tech-dir``: a tree of technician ``fish_counts_export_*.csv`` exports (default
  ``docs/Riverwatcher_2025``).

Outputs
-------
Prints a per-species table (daily-r, model/tech totals, %err, active-day count, verdict) and a
GO / NO-GO line, then exits 0 on GO and 2 on NO-GO — so a shell loop over years stops on the first
failure.

Environment: base env (numpy + optional scipy). No torch/boxmot/DEIMv2 — this reads SQLite only, so
it does NOT need the ``.venv_deim`` detection/training environment.

Usage (``--shard-dirs`` points at the corpus detect-store runs under ``config.MODEL_OUTPUTS``):
  python -m salmonid.eval.corpus_year_gate \
    --shard-dirs "$SALMONID_PROJECTS_ROOT/model_outputs/run_corpus_v6_2025_shard_*" \
    --tech-dir docs/Riverwatcher_2025 --year 2025
"""
from __future__ import annotations
import argparse, csv, glob, sqlite3, collections
from pathlib import Path
import numpy as np

try:
    from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    def pearsonr(a, b):
        """scipy-free Pearson r fallback.

        Returns a ``(r, p)`` 2-tuple matching scipy's signature; only ``r`` is consumed here, so
        the p-value is a placeholder (1.0 for a degenerate/constant series, 0.0 otherwise). A
        constant or <2-sample series has undefined correlation -> ``nan``.
        """
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP
# free-text technician-CSV species strings -> our 0..4 class index
TECH_MAP = {"chinook": 0, "coho": 1, "atlantic": 2, "atlantic salmon": 2,
            "rainbow": 3, "rainbow trout": 3, "brown": 4, "brown trout": 4}
# deploy review-triage floors; mirrors salmonid.config.AUTO_FLOOR / inference/config.py::
# auto_count_species_conf. Atlantic 1.01 => never clears the floor => never auto-counted.
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}
GATE_SPECIES = (0, 1, 3, 4)  # Atlantic excluded (review-only)
# per-class name of the tracks.sqlite mean-probability column (mean prob over the track's frames)
_PROB_COL = {0: "mean_prob_chinook", 1: "mean_prob_coho", 2: "mean_prob_atlantic",
             3: "mean_prob_rainbow", 4: "mean_prob_brown"}


def _infer_site(s: str):
    """Infer the river ('Credit' / 'Ganaraska') from a path or string; None if neither matches."""
    s = s.lower()
    return "Credit" if "credit" in s else ("Ganaraska" if "ganaraska" in s else None)


def _key(path_like: str):
    """Return the ``(timestamp_folder, mp4)`` tail (last two components) of a path.

    Together with the site, this pair is the join key between a technician CSV row's ``video_rel``
    and a model track's ``video_id`` — both reference the same tunnel-camera clip. Returns
    ``(None, None)`` when the path has fewer than two components.
    """
    p = path_like.replace("\\", "/").rstrip("/").split("/")
    return (p[-2], p[-1]) if len(p) >= 2 else (None, None)


def main() -> None:
    """Run the gate end to end.

    Loads technician daily activity counts + the (site, ts_folder, mp4) -> date lookup from the
    ``--tech-dir`` CSV exports, tallies the triage-filtered model tracks on the tech-referenced
    videos, prints a per-species daily-r table plus a GO/NO-GO verdict, and exits 0 (GO) / 2 (NO-GO).
    Side effects: reads SQLite + CSV from disk, writes to stdout, and raises ``SystemExit``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dirs", nargs="+", required=True,
                    help="corpus run dirs (globs ok), each containing tracks.sqlite")
    ap.add_argument("--tech-dir", default="docs/Riverwatcher_2025")
    ap.add_argument("--year", default="?")
    ap.add_argument("--r-min", type=float, default=0.85)
    ap.add_argument("--count-tol", type=float, default=2.0)
    ap.add_argument("--min-events", type=int, default=20)
    ap.add_argument("--min-days", type=int, default=5)
    ap.add_argument("--raw", action="store_true",
                    help="count ALL tracks (skip the deploy species_conf triage floor) — "
                         "comparable to the v2 baseline's raw daily-r.")
    args = ap.parse_args()

    # --- tech: daily activity counts + (site, ts_folder, mp4) -> date ---
    tech_daily = collections.defaultdict(lambda: collections.defaultdict(int))
    vid_date: dict[tuple, str] = {}
    n_ev = 0
    for csvp in sorted(Path(args.tech_dir).rglob("fish_counts_export_*.csv")):
        if "vaki_matched" in csvp.name:
            continue
        site = _infer_site(str(csvp))
        if not site:
            continue
        with open(csvp, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if (row.get("false_trigger", "0") or "0").strip() == "1":
                    continue
                rel = (row.get("video_rel", "") or "").strip()
                if not rel:
                    continue
                sp = TECH_MAP.get((row.get("species", "") or "").strip().lower())
                if sp is None:
                    continue
                date = (row.get("ts", "") or "")[:10]
                if not date:
                    continue
                ts_folder, mp4 = _key(rel)
                if ts_folder is None:
                    continue
                tech_daily[date][sp] += 1
                vid_date[(site, ts_folder, mp4)] = date  # join key: restricts model to tech-reviewed clips
                n_ev += 1

    # --- model: triage-filtered tracks restricted to tech-referenced videos ---
    model_daily = collections.defaultdict(lambda: collections.defaultdict(int))
    dirs = sorted({d for pat in args.shard_dirs for d in glob.glob(pat)})
    cols = "video_id, predicted_class, " + ", ".join(_PROB_COL[i] for i in range(5))
    n_tracks = n_match = 0
    for d in dirs:
        db = Path(d) / "tracks.sqlite"
        if not db.exists():
            continue
        c = sqlite3.connect(str(db))
        try:
            for r in c.execute(f"SELECT {cols} FROM tracks"):
                vid, sp = r[0], r[1]
                if sp is None or not (0 <= sp < 5):
                    continue
                n_tracks += 1
                parts = vid.replace("\\", "/").split("/")
                if len(parts) < 2:
                    continue
                date = vid_date.get((parts[0], parts[-2], parts[-1]))
                if date is None:
                    continue                      # not a tech-reviewed video
                if not args.raw:
                    conf = r[2 + sp]               # mean_prob for predicted class
                    if conf is None or conf < AUTO_FLOOR[sp]:
                        continue                   # deploy review-triage floor
                model_daily[date][sp] += 1
                n_match += 1
        finally:
            c.close()

    # --- per-species daily-r + verdict ---
    dates = sorted(set(tech_daily) | set(model_daily))
    print(f"\n=== {args.year} ACTIVITY daily-r GATE ===")
    print(f"tech events={n_ev:,} | model tracks read={n_tracks:,} | "
          f"auto-counted in tech-videos={n_match:,} | days={len(dates)} | shard dirs={len(dirs)}\n")
    print(f"{'species':15}{'r':>8}{'model':>8}{'tech':>8}{'%err':>8}{'days':>6}   verdict")
    overall_go = True
    fails = []
    for sp in range(5):
        mt = [model_daily[d].get(sp, 0) for d in dates]
        tt = [tech_daily[d].get(sp, 0) for d in dates]
        msum, tsum = sum(mt), sum(tt)
        ndays = sum(1 for x in tt if x > 0)
        r = pearsonr(mt, tt)[0] if (msum > 0 and tsum > 0) else float("nan")
        pct = 100 * (msum - tsum) / tsum if tsum else float("nan")
        ratio = msum / tsum if tsum else float("inf")
        if sp in GATE_SPECIES and tsum >= args.min_events and ndays >= args.min_days:
            ok = (r >= args.r_min) and ((1 / args.count_tol) <= ratio <= args.count_tol)
            verdict = "GO" if ok else "** NO-GO **"
            if not ok:
                overall_go = False
                fails.append(f"{SPECIES[sp]} (r={r:.3f}, ratio={ratio:.2f})")
        elif sp == 2:
            verdict = "(review-only)"
        else:
            verdict = "(low-n: informational)"
        print(f"{SPECIES[sp]:15}{r:8.3f}{msum:8,}{tsum:8,}{pct:8.1f}{ndays:6}   {verdict}")

    print("\n" + "=" * 60)
    if overall_go:
        print(f"VERDICT: GO  -- v6 tracks the tech counts (all gated species r>={args.r_min}, "
              f"counts within {args.count_tol}x). Safe to continue the remaining years.")
    else:
        print(f"VERDICT: NO-GO  -- {'; '.join(fails)}. Model may be getting lost; pause + investigate.")
    print("=" * 60)
    raise SystemExit(0 if overall_go else 2)


if __name__ == "__main__":
    main()

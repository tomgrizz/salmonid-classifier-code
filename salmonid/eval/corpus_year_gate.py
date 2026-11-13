"""In plain terms: counting fish in a whole season of video takes days of computer time, and
the project's earlier (two-stage) model was run one season after another. This script was the
checkpoint between seasons. After a season had been processed, it compared the model's daily
number of fish passages with the technicians' daily number for the same videos and printed
GO or NO-GO, so that a model that had gone wrong would be noticed before more seasons were
processed. It reads the track database that the earlier model wrote. The published model
(v2.7) does not write that kind of database, so this script was never run on it and cannot
be; it is kept as a record of how the earlier runs were supervised.

Inputs
------
All are Tier-2 files that are not stored in git.

* ``--shard-dirs``: one or more run folders of the two-stage pipeline (wildcards allowed),
  each holding a ``tracks.sqlite`` database with one row per tracked fish. On the authors'
  machine these are under ``config.MODEL_OUTPUTS`` (for example ``run_corpus_v6_2025_shard_*``;
  "v6" there is the version of the earlier species classifier, not of the published model).
* ``--tech-dir``: the folder of technician review exports (``fish_counts_export_*.csv``),
  searched recursively. Default ``config.TECH_2025_DIR``.

Outputs
-------
Printed only. A table with one row per species (daily correlation r, model and technician
totals, percentage difference, number of days with technician fish, verdict), then
``VERDICT: GO`` or ``VERDICT: NO-GO``. Exit code 0 for GO and 2 for NO-GO, so a loop over
seasons stops at the first failure. A missing input folder stops the run with exit code 1.

What it measures, and the thresholds
------------------------------------
"Activity" means every fish passage counted once, whatever its direction. For each species
the model's passages per day are compared with the technicians' passages per day.

* Only videos in which the technician recorded at least one fish are used, and a model
  track is matched to the technician's date for that video (river, recording folder, file
  name).
* Unless ``--raw`` is given, a model track is counted only if its mean species score
  reaches the review floor for its species (``AUTO_FLOOR``: 0.50 Chinook Salmon, 0.60 Coho
  Salmon, Rainbow Trout and Brown Trout).
* A species is judged only if the technicians recorded at least ``--min-events`` (20) fish
  on at least ``--min-days`` (5) days. It passes if the daily Pearson correlation r is at
  least ``--r-min`` (0.85) AND the model total is within a factor ``--count-tol`` (2.0) of
  the technician total.
* The verdict is GO only if every judged species passes and at least one species was judged.

What it does NOT check
----------------------
* **False detections on empty videos.** Videos in which the technician recorded no fish
  (including videos marked as false triggers) are left out, so fish the model "finds" in
  them are invisible here.
* **The size of the count, except coarsely.** A factor of 2.0 lets a model that counts 1.9
  times too many fish pass.
* **Net counts.** Upstream-minus-downstream counts, the quantity the manuscript reports, are
  not tested, only total passages.
* **Atlantic Salmon.** It is reported but never judged (its review floor is set above 1 so
  that no Atlantic Salmon is counted without a person's review).
* **Rare species.** A species below the ``--min-events`` or ``--min-days`` floor is shown
  as "low-n: informational" and does not affect the verdict.

Result for the published model
------------------------------
None. The published single-stage model stores its tracks as per-season aggregate files, not
as ``tracks.sqlite``, so this gate has no input for it. The corresponding check on the
published model's daily counts is ``salmonid.eval.count_r_gate``; the published comparison
with the technician record is ``salmonid.results.build_canonical_results`` and the
statistics built on it.

Behaviour changes relative to the original research script
-----------------------------------------------------------
The thresholds, the matching rule and the correlation are unchanged.

* With no technician files, no run folders or no tracks, the original judged no species
  and printed ``VERDICT: GO`` with exit code 0. Each of these cases is now NO-GO (exit 2),
  as is the case where data exist but no species has enough technician fish to be judged.
* A technician folder that does not exist stops the run with one sentence naming it.
* The technician folder default is ``config.TECH_2025_DIR`` instead of a path relative to
  the folder the command is run from.
* Technician rows and run folders that are left out are counted, and the counts printed.

Feeds
-----
Development check; no manuscript number.

Run
---
::

    python -m salmonid.eval.corpus_year_gate --year 2025 \\
        --shard-dirs "<SALMONID_PROJECTS_ROOT>/model_outputs/run_corpus_v6_2025_shard_*"

Base Python (numpy; scipy if installed). No GPU.

Tier
----
2 (needs the two-stage run folders and the technician review exports).

Status
------
Legacy two-stage code kept for provenance. Not on the published path.
"""
from __future__ import annotations
import argparse, csv, glob, sqlite3, collections
from pathlib import Path
import numpy as np

from salmonid import config

try:
    from scipy.stats import pearsonr
except Exception:  # noqa: BLE001  (scipy is optional here; the fallback below gives the same r)
    def pearsonr(a, b):
        """Pearson correlation without scipy.

        Returns ``(r, p)`` like scipy's function; only ``r`` is used in this script, so the
        second value is a placeholder. r is NaN for a constant series or fewer than 2 points.
        """
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # same as salmonid.config.CLASS_TO_SP
# Species names as typed in the technician exports -> class number 0..4.
TECH_MAP = {"chinook": 0, "coho": 1, "atlantic": 2, "atlantic salmon": 2,
            "rainbow": 3, "rainbow trout": 3, "brown": 4, "brown trout": 4}
# Review floors (same values as salmonid.config.AUTO_FLOOR): a track is counted without human
# review only if its mean score for its species reaches the floor. Atlantic Salmon is set
# above 1, which no score can reach, so every Atlantic Salmon goes to a person.
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}
GATE_SPECIES = (0, 1, 3, 4)  # the species that are judged; Atlantic Salmon (2) is not
# Column of tracks.sqlite holding each species' mean score over the frames of a track.
_PROB_COL = {0: "mean_prob_chinook", 1: "mean_prob_coho", 2: "mean_prob_atlantic",
             3: "mean_prob_rainbow", 4: "mean_prob_brown"}


def _infer_site(s: str):
    """Return the river ('Credit' or 'Ganaraska') named in a path, or None."""
    s = s.lower()
    return "Credit" if "credit" in s else ("Ganaraska" if "ganaraska" in s else None)


def _key(path_like: str):
    """Return the last two parts of a video path: (recording folder, file name).

    Together with the river these identify one clip. They are what links a row of the
    technician export to the model's tracks for the same video. Returns ``(None, None)`` if
    the path has fewer than two parts.
    """
    p = path_like.replace("\\", "/").rstrip("/").split("/")
    return (p[-2], p[-1]) if len(p) >= 2 else (None, None)


def main() -> None:
    """Run the check and exit 0 (GO) or 2 (NO-GO)."""
    ap = argparse.ArgumentParser(
        description="GO / NO-GO check of a two-stage season run against the technician record "
                    "(legacy; the published model has no input for it). Exit 0 = GO, 2 = NO-GO.")
    ap.add_argument("--shard-dirs", nargs="+", required=True,
                    help="two-stage run folders (wildcards allowed), each holding tracks.sqlite")
    ap.add_argument("--tech-dir", default=str(config.TECH_2025_DIR),
                    help="folder of technician review exports (default: config.TECH_2025_DIR)")
    ap.add_argument("--year", default="?", help="label printed in the heading")
    ap.add_argument("--r-min", type=float, default=0.85, help="lowest acceptable daily r")
    ap.add_argument("--count-tol", type=float, default=2.0,
                    help="model total must be within this factor of the technician total")
    ap.add_argument("--min-events", type=int, default=20,
                    help="a species is judged only with at least this many technician fish")
    ap.add_argument("--min-days", type=int, default=5,
                    help="a species is judged only with technician fish on at least this many days")
    ap.add_argument("--raw", action="store_true",
                    help="count every track, without the species-score review floors")
    args = ap.parse_args()

    tech_dir = config.require(args.tech_dir, "the folder of technician review exports "
                              "(fish_counts_export_*.csv)", "SALMONID_TECH_2025_DIR")

    # --- technicians: fish per day and species, and the date of every video with a fish ---
    tech_daily = collections.defaultdict(lambda: collections.defaultdict(int))
    vid_date: dict[tuple, str] = {}
    n_ev = 0
    n_files = 0
    left_out = collections.Counter()      # reason -> technician rows not used
    for csvp in sorted(Path(tech_dir).rglob("fish_counts_export_*.csv")):
        if "vaki_matched" in csvp.name:
            continue
        site = _infer_site(str(csvp))
        if not site:
            left_out["file whose path names neither river (whole file)"] += 1
            continue
        n_files += 1
        with open(csvp, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if (row.get("false_trigger", "0") or "0").strip() == "1":
                    left_out["marked as a false trigger"] += 1
                    continue
                rel = (row.get("video_rel", "") or "").strip()
                if not rel:
                    left_out["no video linked to the row"] += 1
                    continue
                sp = TECH_MAP.get((row.get("species", "") or "").strip().lower())
                if sp is None:
                    left_out["species is not one of the five"] += 1
                    continue
                date = (row.get("ts", "") or "")[:10]
                if not date:
                    left_out["no date"] += 1
                    continue
                ts_folder, mp4 = _key(rel)
                if ts_folder is None:
                    left_out["video path too short to identify a clip"] += 1
                    continue
                tech_daily[date][sp] += 1
                # Remember the date of this video. The model's tracks are counted only for
                # videos that appear here, i.e. videos in which the technician recorded a fish.
                vid_date[(site, ts_folder, mp4)] = date
                n_ev += 1

    # --- model: tracks in those same videos, above the review floor ---
    model_daily = collections.defaultdict(lambda: collections.defaultdict(int))
    dirs = sorted({d for pat in args.shard_dirs for d in glob.glob(pat)})
    cols = "video_id, predicted_class, " + ", ".join(_PROB_COL[i] for i in range(5))
    n_tracks = n_match = n_db = 0
    no_db = []
    for d in dirs:
        db = Path(d) / "tracks.sqlite"
        if not db.exists():
            no_db.append(d)
            continue
        n_db += 1
        # Opened read-only: this script must never alter a run's track database.
        c = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
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
                    continue                      # the technician recorded no fish in this video
                if not args.raw:
                    conf = r[2 + sp]               # mean score for the track's own species
                    if conf is None or conf < AUTO_FLOOR[sp]:
                        continue                   # below the review floor: not counted
                model_daily[date][sp] += 1
                n_match += 1
        finally:
            c.close()

    # --- daily correlation per species, and the verdict ---
    dates = sorted(set(tech_daily) | set(model_daily))
    print(f"\n=== {args.year} ACTIVITY daily-r GATE ===")
    print(f"tech events={n_ev:,} (from {n_files} export files) | model tracks read={n_tracks:,} "
          f"(from {n_db} track databases) | counted in tech-videos={n_match:,} | "
          f"days={len(dates)} | shard dirs={len(dirs)}")
    for reason, n in sorted(left_out.items()):
        print(f"  technician rows not used: {n:,} ({reason})")
    for d in no_db:
        print(f"  WARNING: no tracks.sqlite in {d} (folder ignored)")
    print()
    print(f"{'species':15}{'r':>8}{'model':>8}{'tech':>8}{'%err':>8}{'days':>6}   verdict")
    overall_go = True
    fails = []
    n_judged = 0
    for sp in range(5):
        mt = [model_daily[d].get(sp, 0) for d in dates]
        tt = [tech_daily[d].get(sp, 0) for d in dates]
        msum, tsum = sum(mt), sum(tt)
        ndays = sum(1 for x in tt if x > 0)
        r = pearsonr(mt, tt)[0] if (msum > 0 and tsum > 0) else float("nan")
        pct = 100 * (msum - tsum) / tsum if tsum else float("nan")
        ratio = msum / tsum if tsum else float("inf")
        if sp in GATE_SPECIES and tsum >= args.min_events and ndays >= args.min_days:
            n_judged += 1
            # A NaN r (model counted nothing) compares as False here, so it is a NO-GO.
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

    # Nothing to evaluate is a NO-GO, never a GO. In the original, each of these cases left
    # every species "informational" and the verdict defaulted to GO.
    nothing = []
    if n_ev == 0:
        nothing.append(f"no usable technician rows were found under {tech_dir}")
    if not dirs:
        nothing.append("--shard-dirs matched no folder")
    elif n_db == 0:
        nothing.append("none of the run folders holds a tracks.sqlite")
    elif n_tracks == 0:
        nothing.append("the track databases hold no tracks of the five species")
    if not nothing and n_judged == 0:
        nothing.append(f"no species has at least {args.min_events} technician fish on at least "
                       f"{args.min_days} days, so no species could be judged")

    print("\n" + "=" * 60)
    if nothing:
        print(f"VERDICT: NO-GO  -- nothing to evaluate: {'; '.join(nothing)}.")
        overall_go = False
    elif overall_go:
        print(f"VERDICT: GO  -- the model follows the technician counts ({n_judged} judged species, "
              f"all with r>={args.r_min} and counts within {args.count_tol}x).")
    else:
        print(f"VERDICT: NO-GO  -- {'; '.join(fails)}. Pause and investigate before processing "
              f"more seasons.")
    print("=" * 60)
    raise SystemExit(0 if overall_go else 2)


if __name__ == "__main__":
    main()

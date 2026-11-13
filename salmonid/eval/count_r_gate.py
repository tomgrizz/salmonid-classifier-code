"""In plain terms: this script compares the model's daily fish counts with the technicians'
daily counts, river by river and species by species. For each pair it reports how closely
the two series rise and fall together (a correlation, where 1.0 means they move in step)
and how far apart the season totals are (a percentage). It then marks every river-species
pair that falls short of two fixed development thresholds, and fails if any pair is marked.
It was written as a quick pass/fail check during model development. The agreement
statistics printed in the manuscript are computed elsewhere, with more careful methods (see
"Feeds").

Inputs
------
``--daily-csv``: one or more daily count tables with one row per date, river and species and
the columns ``date, site, species, model, tech``. The per-season files written by
``salmonid.count.corpus_countr_from_store`` have this layout. The published model's daily
counts for each season are committed in ``data/``, so the gate runs on them without any
download: NET counts (upstream minus downstream) as ``data/net_rule_sweep/<year>_up50.csv``
and ACTIVITY counts (every passage counted once) as ``data/activity_sweep/<year>_activity.csv``.
Rows that share a date, river and species are added together, so several files can be given.

Outputs
-------
* Printed: a table with one row per river and species, plus a POOLED row per species (both
  rivers added together by date): number of active days, model and technician totals,
  Pearson correlation r with a 95% interval, percentage error of the total with a 95%
  interval, and flags; then ``GATE: PASS`` or ``GATE: FAIL``.
* ``--out`` (default ``output/count_r_gate.json``): the same rows as a JSON list.
* Exit code 0 for PASS, 1 for FAIL.

What it measures, and the thresholds
------------------------------------
* **r** is the Pearson correlation between the model's and the technicians' daily counts
  over all dates in the table for that river and species. Flag ``r_below_gate`` if r is
  below ``--r-gate`` (default 0.90).
* **%err** is 100 x (model total - technician total) / technician total. Flag
  ``pct_below_gate`` if its absolute value is ABOVE ``--pct-gate`` (default 15). The flag
  name is historical; it means "fails the percentage test".
* ``low_n`` marks a row with fewer than ``--min-days`` (default 8) active days, an active
  day being one on which either side counted at least one fish. It is a caution, not a
  failure: with so few days r says little.
* ``r_undefined`` marks a row that has enough active days but whose r cannot be computed
  because one of the two series never changes (for example a model that counts zero every
  day).

The gate FAILS if any row carries ``r_below_gate``, ``pct_below_gate`` or ``r_undefined``,
or if the table holds no rows for the five species.

What it does NOT check
----------------------
* **Agreement in the strict sense.** A correlation of 1.0 is reached by a model that
  counts exactly three times too many fish every day; only the percentage test catches
  that. Neither test looks at individual days.
* **Realistic uncertainty.** The 95% intervals resample single days as if each day were
  independent of the one before. Daily fish passage is not: runs last several days. These
  intervals are therefore too narrow. The manuscript's intervals use a moving-block
  resample that keeps neighbouring days together (``salmonid.results.agreement_stats``).
* **Days missing from the table.** Only the dates present in the input are compared.
* **Rare species sensibly.** The thresholds were set with the abundant species in mind.
  A species with a few dozen fish in a season can fail the percentage test on a
  difference of five or six fish.

Result for the published model (v2.7)
-------------------------------------
On the committed 2025 daily NET counts (``data/net_rule_sweep/2025_up50.csv``), run on
2026-11-10 with the default thresholds, the gate FAILS: 9 of the 15 rows are flagged.

    passes at both rivers and pooled:  Chinook Salmon (r 0.941 Credit, 0.978 Ganaraska;
                                       -12.6% and +3.6%), Rainbow Trout (r 0.947, 0.990;
                                       -5.2% and +3.3%)
    flagged:  Coho Salmon      Credit r 0.850 and -28.1%; Ganaraska -20.0% (r 0.981)
              Atlantic Salmon  Credit r 0.824 and -18.9%; Ganaraska +100% on 6 against 3
                               fish (also low_n); pooled r 0.848
              Brown Trout      Credit r 0.703; Ganaraska +41.1% (r 0.957)

The other seasons' net counts also FAIL (2020: 14 of 15 rows flagged; 2021: 12; 2022: 7;
2023: 9; 2024: 9), as do the 2025 activity counts (8 of 15). Two things to know when
reading those: the 2020 file holds the regular 2020 Ganaraska River footage only, without
the recovered footage that the master tables blend in, so its Ganaraska River rows compare
a handful of model fish with the technicians' full season; and in 2021 abundant species
are flagged as well (model totals 30% to 60% above the technicians' for Ganaraska River
Chinook Salmon and Rainbow Trout and Credit River Rainbow Trout, with daily r of 0.96 to
0.99). The gate's verdict was not used to accept or reject the published model. For the
agreement actually reported, with intervals, see the modules listed under "Feeds".

Behaviour changes relative to the original research script
-----------------------------------------------------------
The two thresholds and the definitions of r, %err and the existing flags are unchanged.

* The original only printed flags and always exited 0, so it could not stop an automated
  run. It now prints ``GATE: PASS`` or ``GATE: FAIL`` and exits 1 on FAIL.
* An input with no usable rows now FAILS instead of printing an empty table.
* New flag ``r_undefined`` (see above). In the original, a correlation that could not be
  computed raised no flag.
* A missing input file stops the run with one sentence naming it.
* The JSON summary is written under ``output/`` by default instead of ``figures/``.

Feeds
-----
Development check; no manuscript number. The manuscript's daily and seasonal agreement
figures come from ``salmonid.results.build_canonical_results`` (season totals, percentage
errors and daily r), ``salmonid.results.compute_concordance`` (Lin's concordance) and
``salmonid.results.agreement_stats``.

Run
---
::

    python -m salmonid.eval.count_r_gate --daily-csv data/net_rule_sweep/2025_up50.csv
    python -m salmonid.eval.count_r_gate --daily-csv <a.csv> <b.csv> --pooled-only

Tier
----
1 for the committed daily count files (base Python with numpy; no model, no GPU). 2 for
any other daily table.

Status
------
Supporting check.
"""
from __future__ import annotations
import argparse, csv, json, collections
from pathlib import Path
import numpy as np

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # same order as salmonid.config.SPECIES
# Flags that make the gate fail. "low_n" is deliberately not among them: it is a caution.
FAIL_FLAGS = ("r_below_gate", "pct_below_gate", "r_undefined")


def pearson(a, b):
    """Pearson correlation r between two equally long series.

    Returns NaN (not a number) when there are fewer than 3 points or when either series is
    constant, because r is undefined in both cases.
    """
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(m, t, n_boot, seed, stat):
    """95% interval of ``stat(model, tech)``, by redrawing days with replacement.

    Each of the ``n_boot`` rounds draws as many days as there are in the table, at random
    and with repeats, and recomputes the statistic; the interval runs from the 2.5th to the
    97.5th percentile of those values. Days are drawn independently of one another, which
    ignores the day-to-day persistence of a fish run, so the interval is narrower than it
    should be (see the module docstring).
    """
    rng = np.random.default_rng(seed)
    n = len(m)
    if n < 3:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat(m[idx], t[idx])
        if not np.isnan(v):
            vals.append(v)
    # If more than half of the rounds gave no value (for example all drawn days had zero
    # technician fish), the interval would rest on an unrepresentative remainder: report none.
    if len(vals) < n_boot // 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pct_err(m, t):
    """Signed percentage error of the total: 100 x (model - technician) / technician.

    Positive means the model counted more than the technicians. NaN when the technician
    total is 0.
    """
    st = t.sum()
    return float((m.sum() - st) / st * 100.0) if st > 0 else float("nan")


def load(csv_paths):
    """Read the daily tables into ``{(river, species): {date: [model, technician]}}``.

    Rows with the same river, species and date are added together, so the files of several
    seasons (or several parts of one run) can be passed at once. Counts written as "12.0"
    are accepted.
    """
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for cp in csv_paths:
        with open(cp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["site"], r["species"])
                cell = d[key][r["date"]]
                cell[0] += int(float(r["model"])); cell[1] += int(float(r["tech"]))
    return d


def series(daily_by_group, groups):
    """Add the given (river, species) groups together by date; return model and technician arrays."""
    merged = collections.defaultdict(lambda: [0, 0])
    for g in groups:
        for date, (mv, tv) in daily_by_group[g].items():
            merged[date][0] += mv; merged[date][1] += tv
    dates = sorted(merged)
    m = np.array([merged[dt][0] for dt in dates], float)
    t = np.array([merged[dt][1] for dt in dates], float)
    return m, t


def main():
    """Load the daily table(s), score every river-species pair, print, write JSON, and exit."""
    ap = argparse.ArgumentParser(
        description="Daily model-vs-technician count check: correlation and percentage error "
                    "per river and species. Development check; exit code 0 = PASS, 1 = FAIL.")
    ap.add_argument("--daily-csv", nargs="+", required=True,
                    help="daily count table(s) with columns date,site,species,model,tech "
                         "(e.g. data/net_rule_sweep/2025_up50.csv)")
    ap.add_argument("--bootstrap", type=int, default=2000, help="resampling rounds for the intervals")
    ap.add_argument("--seed", type=int, default=0, help="random seed of the resampling")
    ap.add_argument("--min-days", type=int, default=8, help="rows with fewer active days get the low_n caution")
    ap.add_argument("--r-gate", type=float, default=0.90, help="a row fails if r is below this")
    ap.add_argument("--pct-gate", type=float, default=15.0, help="a row fails if |%%err| is above this")
    ap.add_argument("--pooled-only", action="store_true", help="skip the per-river rows")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "count_r_gate.json"),
                    help="where to write the JSON summary (default: output/count_r_gate.json)")
    a = ap.parse_args()

    for cp in a.daily_csv:
        config.require(cp, "a daily model-vs-technician count table (--daily-csv)")
    d = load(a.daily_csv)
    sites = sorted({s for (s, _) in d})
    rows, out = [], []

    def evaluate(label, site_key, sp):
        """Score one river-species pair (or the POOLED pair) and store its row.

        Does nothing when the table has no rows for this pair.
        """
        groups = [(site_key, sp)] if site_key != "POOLED" else [(s, sp) for s in sites]
        groups = [g for g in groups if g in d]
        if not groups:
            return
        m, t = series(d, groups)
        active = int(((m > 0) | (t > 0)).sum())   # active day: either side counted a fish
        r = pearson(m, t)
        r_ci = bootstrap_ci(m, t, a.bootstrap, a.seed, pearson)
        pe = pct_err(m, t)
        # seed + 1 gives the percentage-error interval its own random draws, so it is not
        # computed from the very same resampled days as the interval for r.
        pe_ci = bootstrap_ci(m, t, a.bootstrap, a.seed + 1, pct_err)
        flags = []
        if active < a.min_days: flags.append("low_n")
        if not np.isnan(r) and r < a.r_gate: flags.append("r_below_gate")
        if not np.isnan(pe) and abs(pe) > a.pct_gate: flags.append("pct_below_gate")
        # r could not be computed although there are enough days: one series is constant.
        if np.isnan(r) and active >= a.min_days: flags.append("r_undefined")
        rows.append((label, sp, active, int(m.sum()), int(t.sum()), r, r_ci, pe, pe_ci, flags))
        out.append({"scope": label, "species": sp, "active_days": active,
                    "model_total": int(m.sum()), "tech_total": int(t.sum()),
                    "pearson_r": r, "r_ci95": r_ci, "pct_err": pe, "pct_err_ci95": pe_ci,
                    "flags": flags})

    scopes = ([] if a.pooled_only else sites) + ["POOLED"]
    for site_key in scopes:
        label = site_key if site_key != "POOLED" else "POOLED"
        for sp in SPECIES:
            evaluate(label, site_key, sp)

    print(f"count-r gate  (r>={a.r_gate}, |%err|<={a.pct_gate}, min_days={a.min_days}, "
          f"bootstrap={a.bootstrap})\n")
    print(f"{'scope':9s} {'species':13s} {'days':>4s} {'model':>7s} {'tech':>7s} "
          f"{'r':>6s} {'r 95% CI':>16s} {'%err':>7s} {'%err 95% CI':>18s}  flags")
    for lab, sp, days, mt, tt, r, rci, pe, peci, fl in rows:
        rs = f"{r:6.3f}" if not np.isnan(r) else "   nan"
        rcis = f"[{rci[0]:5.2f},{rci[1]:5.2f}]" if not np.isnan(rci[0]) else "        --      "
        pes = f"{pe:6.1f}" if not np.isnan(pe) else "   nan"
        pecis = f"[{peci[0]:6.1f},{peci[1]:6.1f}]" if not np.isnan(peci[0]) else "        --       "
        print(f"{lab:9s} {sp:13s} {days:>4d} {mt:>7d} {tt:>7d} {rs} {rcis:>16s} "
              f"{pes} {pecis:>18s}  {','.join(fl)}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")

    # ---- verdict -------------------------------------------------------------------------
    if not rows:
        # Nothing to evaluate is a failure, never a pass.
        print("\nGATE: FAIL  (the input holds no rows for any of the five species: check the "
              "file and its column names date,site,species,model,tech)")
        raise SystemExit(1)
    failed = [(lab, sp, fl) for lab, sp, *_, fl in rows if any(f in FAIL_FLAGS for f in fl)]
    if failed:
        print(f"\nGATE: FAIL  ({len(failed)} of {len(rows)} rows flagged)")
        for lab, sp, fl in failed:
            print(f"  {lab} {sp}: {', '.join(f for f in fl)}")
        raise SystemExit(1)
    print(f"\nGATE: PASS  (none of the {len(rows)} rows is flagged; low_n rows, if any, are cautions)")


if __name__ == "__main__":
    main()

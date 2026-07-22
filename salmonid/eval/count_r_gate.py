"""In plain terms: this file checks whether the computer's daily fish counts agree with the
counts the human technicians recorded, comparing them day by day for each river and species. It
reports a correlation number (how closely the two sets of counts rise and fall together, where
1.0 is perfect agreement), the total percent count error, and it flags any species where the
agreement is too weak to trust. These are the numbers behind the paper's daily-count claims --
per-species daily correlation of 0.94-0.99, and Chinook counts within 5-10% of the technicians'.

Deploy count gate: tech-count Pearson r with bootstrap CIs on a daily model-vs-tech CSV.

Pipeline position: the eval/count stage (data -> detect -> track -> **eval/count** -> results ->
figures). This is the go/no-go gate the project's hard rule demands -- "operational accuracy =
tech-count Pearson r, not clip-avg presence-F1" -- computed with BOOTSTRAP confidence intervals,
so a candidate is judged on daily count AGREEMENT rather than presence-F1 (which is blind to the
multi-fish under-counting: Coho -27%, Rainbow -12..-29%).

INPUT: one or more daily-count CSVs already produced by the corpus-vs-tech reducers
(salmonid.count.corpus_fullyear_vs_tech and the sibling corpus_*_vs_tech modules). Long format,
one row per (date, site, species):
    date,site,species,model,tech
Model-agnostic -- it gates WHICHEVER candidate produced that daily CSV. (A stratified per-clip
op_testset CANNOT yield a meaningful count-r; this needs a day-complete corpus run.)

OUTPUT: a printed per-(site | POOLED, species) table -- active days, Pearson r + 95% bootstrap CI
(resampling days with replacement), model/tech totals, %-error + its 95% CI, and deploy flags --
plus a JSON summary written to --out.

This is the honest count-agreement complement to the held-out species-F1: it backs the
manuscript's daily-count claims (per-species daily Pearson r 0.94-0.99; Chinook |count-err|
5-10%). Runs in the base Python environment -- numpy only, no torch / DEIMv2 / boxmot.

  python -m salmonid.eval.count_r_gate --daily-csv path/to/daily_model_vs_tech.csv
  python -m salmonid.eval.count_r_gate --daily-csv shardA.csv shardB.csv --pooled-only
"""
from __future__ import annotations
import argparse, csv, json, collections
from pathlib import Path
import numpy as np

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES


def pearson(a, b):
    """Pearson r between arrays ``a`` and ``b``.

    Returns NaN when there are fewer than 3 points or when either array has zero variance
    (r is undefined there); the callers treat NaN as "not scored" rather than a failure.
    """
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(m, t, n_boot, seed, stat):
    """Percentile 95% CI of `stat(m,t)` by resampling day-indices with replacement."""
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
    if len(vals) < n_boot // 2:          # too many degenerate resamples -> untrustworthy
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pct_err(m, t):
    """Signed percent count error ``100*(sum(model) - sum(tech)) / sum(tech)``.

    Positive = model over-counts, negative = under-counts. NaN when tech sums to 0.
    """
    st = t.sum()
    return float((m.sum() - st) / st * 100.0) if st > 0 else float("nan")


def load(csv_paths):
    """Read the long-format daily CSV(s) into ``(site, species) -> {date: [model_sum, tech_sum]}``.

    Args:
        csv_paths: iterable of CSV paths, each with columns date,site,species,model,tech.
    Returns:
        Nested defaultdict keyed by (site, species) then date, holding [model, tech] running sums.
    Rows sharing a (site, species, date) key are ACCUMULATED, so several shard CSVs from one run
    may be passed together; counts go through ``int(float(...))`` to tolerate float-formatted ints.
    """
    # (site, species) -> {date: [model, tech]}
    d = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for cp in csv_paths:
        for r in csv.DictReader(open(cp)):
            key = (r["site"], r["species"])
            cell = d[key][r["date"]]
            cell[0] += int(float(r["model"])); cell[1] += int(float(r["tech"]))
    return d


def series(daily_by_group, groups):
    """Merge the given (site,species) groups into aligned per-date model/tech arrays."""
    merged = collections.defaultdict(lambda: [0, 0])
    for g in groups:
        for date, (mv, tv) in daily_by_group[g].items():
            merged[date][0] += mv; merged[date][1] += tv
    dates = sorted(merged)
    m = np.array([merged[dt][0] for dt in dates], float)
    t = np.array([merged[dt][1] for dt in dates], float)
    return m, t


def main():
    """CLI entry point: load the daily CSV(s), score every (scope, species), print + write JSON.

    Side effects: prints the gate table to stdout and writes the per-group JSON summary to ``--out``
    (creating parent dirs). Returns None.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-csv", nargs="+", required=True,
                    help="long-format daily count CSV(s): date,site,species,model,tech")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-days", type=int, default=8, help="flag groups with fewer active days")
    ap.add_argument("--r-gate", type=float, default=0.90, help="deploy r threshold for the flag")
    ap.add_argument("--pct-gate", type=float, default=15.0, help="deploy abs %-err threshold for the flag")
    ap.add_argument("--pooled-only", action="store_true", help="skip per-site rows")
    ap.add_argument("--out", default="figures/count_r_gate.json")
    a = ap.parse_args()

    d = load(a.daily_csv)
    sites = sorted({s for (s, _) in d})
    rows, out = [], []

    def evaluate(label, site_key, sp):
        """Score one (scope, species) group; append its row to ``rows`` and dict to ``out``.

        ``site_key`` is a single site name, or "POOLED" to merge every site for this species.
        Sets deploy flags: low_n (fewer than --min-days active days), r_below_gate (r < --r-gate),
        pct_below_gate (|%err| > --pct-gate). No-op when the group has no rows in the data.
        """
        groups = [(site_key, sp)] if site_key != "POOLED" else [(s, sp) for s in sites]
        groups = [g for g in groups if g in d]
        if not groups:
            return
        m, t = series(d, groups)
        active = int(((m > 0) | (t > 0)).sum())   # active day = model OR tech saw the species
        r = pearson(m, t)
        r_ci = bootstrap_ci(m, t, a.bootstrap, a.seed, pearson)
        pe = pct_err(m, t)
        # seed+1: an independent day-resample stream so the %-err CI is not coupled to the r CI
        pe_ci = bootstrap_ci(m, t, a.bootstrap, a.seed + 1, pct_err)
        flags = []
        if active < a.min_days: flags.append("low_n")
        if not np.isnan(r) and r < a.r_gate: flags.append("r_below_gate")
        if not np.isnan(pe) and abs(pe) > a.pct_gate: flags.append("pct_below_gate")
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
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

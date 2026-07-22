"""
In plain terms: this script builds the master results table the paper's numbers rest on. For every
year (2019-2025), river, species, and day, it lines the automated counter's fish tally up next to
the human technicians' tally and works out how well they agree (season totals, percent error, and a
day-by-day correlation). The table it writes is the source for the paper's multi-year abundance-index
result (annual r-squared of 0.96 and 0.98, Chinook counts within 5-10%) and it feeds the run-timing
and Ganaraska-sanctuary figures.

Assemble the single canonical daily results dataset for the manuscript: model (single-stage v2.6 BoT-SORT)
vs technician, NET + ACTIVITY, per (year, river, species, date), 2019-2025 both rivers — using the RECOVERED
counts for 2019 Credit and 2020 Ganaraska. Tech = per-event net (validated to match the deduplicated official
counts) + activity. Emits canonical_daily.csv + a per-cell seasonal summary (totals, %err, daily-r).

Pipeline position — RESULTS assembly
------------------------------------
This is the join/aggregation step that sits *after* the count stage (data -> detect -> track ->
count -> **results** -> figures). It does not touch a model or footage; it reads the already-computed
per-year model-vs-tech daily count CSVs and folds them into one tidy long table plus one wide seasonal
summary that every downstream manuscript figure and statistic reads from.

Inputs
------
* Per-year corpus sweeps under ``config.PROJECTS_ROOT / "deimv2_x_v26_corpus2025"``:
  ``net_rule_sweep/{year}_up50.csv`` (NET, asymmetric up>=50/down>=300 rule) and
  ``activity_sweep/{year}_activity.csv`` (ACTIVITY). Each row = (site, date, species, model, tech).
* Recovered-footage overrides for the two cells whose 2025-corpus sweeps are absent because the
  footage was recovered separately: 2019 Credit (``deimv2_x_v26_corpus2019``) and 2020 Ganaraska
  (``deimv2_x_v26_corpus2020``). These *replace* the affected (site,date,species) cells.

Outputs
-------
* ``config.CANONICAL_DAILY``   — long per-(year,river,species,date) model/tech NET+ACTIVITY table.
* ``config.CANONICAL_SUMMARY`` — one row per (year,river,species): season totals, %err, daily-r.
Both land in ``config.REPO_DATA`` (the committed Tier-1 ``data/`` dir), so the figures and the
concordance / abundance-index statistics reproduce on a fresh clone.

Manuscript claim
----------------
This is the canonical source for the multi-year abundance-index result (annual r² 0.96/0.98,
Chinook count |err| 5-10%, daily-r 0.94-0.99) and it feeds the phenology / sanctuary figures.

Environment
-----------
Base Python env only — pure standard library, no torch/boxmot. Run it after the per-year count
sweeps exist.

  python -m salmonid.results.build_canonical_results
"""
import csv, json, glob, math, collections
from pathlib import Path

from salmonid import config

# Corpus sweep root for the main 2019-2025 run (per-year net + activity CSVs live under here).
SW = config.PROJECTS_ROOT / "deimv2_x_v26_corpus2025"
SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
OUT_DAILY = config.CANONICAL_DAILY
OUT_SUMMARY = config.CANONICAL_SUMMARY


def countr_daily(year, metric):
    """Load one year's daily model-vs-tech counts for one metric.

    Args:
        year: calendar year of the sweep (2019-2025).
        metric: ``"net"`` (asymmetric up>=50/down>=300 net-upstream rule) or ``"activity"``.

    Returns:
        dict mapping ``(site, date, species) -> (model_count, tech_count)``.

    Tech = the DEDUPLICATED official count (Ganny_Total / FINAL_CREDIT / 2025 daily) that the count-r
    used -- the fish-truth the model matches, NOT the re-trigger-inflated per-event log. The recovered
    CSVs (2019 Credit, 2020 Ganaraska) override the affected site because those cells are absent from
    the main 2025-corpus sweep (their footage was recovered and counted separately).
    """
    tag = "net" if metric == "net" else "activity"
    out = {}
    base = f"{SW}/net_rule_sweep/{year}_up50.csv" if metric == "net" else f"{SW}/activity_sweep/{year}_activity.csv"
    if Path(base).is_file():
        for r in csv.DictReader(open(base)):
            out[(r["site"], r["date"], r["species"])] = (int(r["model"]), int(r["tech"]))
    rec = []
    if year == 2019:
        rec = [(config.PROJECTS_ROOT / "deimv2_x_v26_corpus2019" / f"2019credit_{tag}_botsort.csv", "Credit")]
    if year == 2020:
        rec = [(config.PROJECTS_ROOT / "deimv2_x_v26_corpus2020" / f"2020gan_recovered_{tag}_botsort.csv", "Ganaraska")]
    for fn, site in rec:
        if Path(fn).is_file():
            for r in csv.DictReader(open(fn)):
                if r["site"] == site:
                    out[(r["site"], r["date"], r["species"])] = (int(r["model"]), int(r["tech"]))
    return out


def pearson(xs, ys):
    """Pearson correlation r of two equal-length sequences.

    Returns NaN when n < 3 or either series has zero variance (the daily-r is reported only where
    it is meaningful; callers substitute an empty cell for NaN).
    """
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def main():
    """Build and write the canonical daily + summary CSVs, then print a Chinook-NET headline table.

    Side effects: creates ``config.REPO_DATA`` if needed and writes ``config.CANONICAL_DAILY`` and
    ``config.CANONICAL_SUMMARY``; prints row/cell counts and the per-(year,river) Chinook net table.
    """
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)  # ensure config.REPO_DATA (the committed data/ dir) exists
    rows = []
    summ = []
    RECOVERED = {(2019, "Credit"), (2020, "Ganaraska")}
    SKIP = {(2019, "Ganaraska")}                     # tech-only: no footage -> no model
    for year in range(2019, 2026):
        cnet = countr_daily(year, "net"); cact = countr_daily(year, "activity")
        for site in ["Ganaraska", "Credit"]:
            if (year, site) in SKIP:
                continue
            for sp in SPECIES:
                dates = sorted({d for (s, d, x) in cnet if s == site and x == sp})
                if not dates:
                    continue
                mn = [cnet.get((site, d, sp), (0, 0))[0] for d in dates]
                tn = [cnet.get((site, d, sp), (0, 0))[1] for d in dates]
                ma = [cact.get((site, d, sp), (0, 0))[0] for d in dates]
                ta = [cact.get((site, d, sp), (0, 0))[1] for d in dates]
                for i, d in enumerate(dates):
                    rows.append({"year": year, "river": site, "species": sp, "date": d,
                                 "model_net": mn[i], "tech_net": tn[i], "model_act": ma[i], "tech_act": ta[i]})
                Mn, Tn, Ma, Ta = sum(mn), sum(tn), sum(ma), sum(ta)
                if Mn == 0 and Tn == 0 and Ma == 0 and Ta == 0:
                    continue
                summ.append({"year": year, "river": site, "species": sp,
                             "model_net": Mn, "tech_net": Tn,
                             "net_pct_err": round(100 * (Mn - Tn) / Tn, 1) if Tn else "",
                             "net_r": round(pearson(mn, tn), 3) if not math.isnan(pearson(mn, tn)) else "",
                             "model_act": Ma, "tech_act": Ta,
                             "act_pct_err": round(100 * (Ma - Ta) / Ta, 1) if Ta else "",
                             "act_r": round(pearson(ma, ta), 3) if not math.isnan(pearson(ma, ta)) else "",
                             "active_days": sum(1 for x in tn if x != 0) or sum(1 for x in mn if x != 0),
                             "recovered": (year, site) in RECOVERED})
    with open(OUT_DAILY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "river", "species", "date", "model_net", "tech_net", "model_act", "tech_act"])
        w.writeheader(); w.writerows(rows)
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        cols = ["year", "river", "species", "model_net", "tech_net", "net_pct_err", "net_r",
                "model_act", "tech_act", "act_pct_err", "act_r", "active_days", "recovered"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(summ)
    print(f"canonical_daily: {len(rows)} rows -> {OUT_DAILY}")
    print(f"canonical_summary: {len(summ)} cells -> {OUT_SUMMARY}")
    # headline: Chinook net across years
    print("\nChinook NET model vs tech (both rivers):")
    for s in summ:
        if s["species"] == "Chinook":
            rec = " [recovered]" if s["recovered"] else ""
            print(f"  {s['year']} {s['river']:10s} model {s['model_net']:6d} tech {s['tech_net']:6d} "
                  f"({s['net_pct_err']}%, r={s['net_r']}){rec}")


if __name__ == "__main__":
    main()

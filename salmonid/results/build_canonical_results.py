"""In plain terms: this script builds the two master tables that the paper's count results
are read from. The count step leaves one small file per season; this script stacks them into
one long table with a row for every river, species and day (model count beside technician
count), and a short table with one row per river, species and season giving the season
totals, the percentage difference, and how closely the daily counts track each other. It does
no counting of its own and never touches video or the model: it only gathers and summarises.

Assemble ``canonical_daily.csv`` and ``canonical_summary.csv`` from the per-season count files.

Pipeline stage: data -> detect -> track -> count -> **results (this module)** -> figures.

Inputs : per-season files written by ``salmonid.count.corpus_countr_from_store``
         (run for all seasons by ``salmonid.count.run_corpus``):
           <sweep-root>/net_rule_sweep/<year>_up50.csv      net counts, 50/300 px rule
           <sweep-root>/activity_sweep/<year>_activity.csv  activity counts
           <recovered-dir>/2020gan_recovered_{net,activity}_botsort.csv
                                                            the recovered 2020 Ganaraska footage
Outputs: ``canonical_daily.csv`` and ``canonical_summary.csv`` in ``--out-dir`` (default
         ``output/``). The published copies are ``data/canonical_daily.csv`` and
         ``data/canonical_summary.csv``; this script never writes into ``data/``.
Feeds  : Table 2 (count columns), Table 3, Table 4, Table 5 and Figures 4-7, plus every
         statistic computed from them (``salmonid.results.agreement_stats``,
         ``compute_concordance``); manuscript Sections 3.2, 3.3, 3.5 and 3.6.
Run    : python -m salmonid.results.build_canonical_results
             (reads output/corpus_v27/, as written by salmonid.count.run_corpus)
         python -m salmonid.results.build_canonical_results --source projects
             (reads the authors' original per-season files on the project data drive)
Tier   : 2. The activity files and the recovered-2020 files are not stored in git, so the
         tables cannot be rebuilt from the committed ``data/`` folder alone. Base Python,
         standard library only, runs in under a second.

Columns of canonical_daily.csv
  year, river, species, date; model_net, tech_net (upstream minus downstream);
  model_act, tech_act (every passage counted once, whatever its direction).
Columns of canonical_summary.csv
  model_net, tech_net, model_act, tech_act   season totals (sums of the daily rows)
  net_pct_err, act_pct_err   100 x (model - technician) / technician, one decimal; blank if
                             the technician total is 0
  net_r, act_r               Pearson correlation between model and technician daily counts
  active_days                number of days with a non-zero technician net count (if there
                             are none: days with a non-zero model net count)
  recovered                  True for the season counted from recovered footage (2020 Ganaraska)

How the published numbers were computed
---------------------------------------
* **Which days are in the table.** The rows for a river and species are the days present in
  that season's NET file, that is, days on which the model's net count or the technician's
  net count is not zero. A day that appears only in the activity file (fish passed, but the
  net count was zero on both sides) is not in the table.
* **The daily correlation uses all of those rows.** ``net_r`` and ``act_r`` are computed over
  every row for that river, species and season. Days on which only the model counted fish
  are included; so are days on which the technician counted fish and the model has no video
  at all, with the model at 0. Such days pull the correlation and the model total down; they
  are not treated as missing data. ``active_days`` is a different, smaller number (see
  above) and is not the number of rows behind ``net_r``.
* **2020 Ganaraska is a blend of two footage sets.** Most of that season's footage was
  recovered from an archive drive and processed separately. Its counts replace the regular
  2020 Ganaraska counts cell by cell (river, date, species). A cell that exists only in the
  regular file is kept as it is. The ``recovered`` flag marks the whole river-season, and the
  paper sets it aside from its agreement statistics.
* **2019 is not included.** The published (v2.7) model was not run on 2019. The code path
  for the 2019 Credit River footage is kept only for the superseded model (``--tag v26
  --years 2019 ...``).
* **Percentage error uses season sums over all rows**, including days with fish on one side
  only.

Which research-repository version this is: the version with the model switch (v2.6 / v2.7),
from the lineage that produced the v2.7 counts. The switch is ``--tag`` here; the arithmetic
is unchanged.
"""
import argparse, csv, math
from pathlib import Path

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # same order as salmonid.config.SPECIES

# Where the per-season count files are read from. Set in main() from the command line.
SW = None             # folder holding net_rule_sweep/ and activity_sweep/
REC_2019_DIR = None   # folder with 2019credit_{net,activity}_botsort.csv (superseded model only)
REC_2020_DIR = None   # folder with 2020gan_recovered_{net,activity}_botsort.csv
OUT_DAILY = None
OUT_SUMMARY = None


def countr_daily(year, metric):
    """Read one season's daily counts for one metric ("net" or "activity").

    Returns ``{(river, date, species): (model_count, technician_count)}``. For 2020 the
    recovered-footage file then overwrites the Ganaraska cells it contains (see the module
    notes); cells it does not contain are left as read from the regular file.
    """
    tag = "net" if metric == "net" else "activity"
    out = {}
    # The published net count is the "up50" file: upstream >= 50 px, downstream >= 300 px.
    base = f"{SW}/net_rule_sweep/{year}_up50.csv" if metric == "net" else f"{SW}/activity_sweep/{year}_activity.csv"
    if Path(base).is_file():
        for r in csv.DictReader(open(base)):
            out[(r["site"], r["date"], r["species"])] = (int(r["model"]), int(r["tech"]))
    rec = []
    if year == 2019 and REC_2019_DIR:
        rec = [(f"{REC_2019_DIR}/2019credit_{tag}_botsort.csv", "Credit")]
    if year == 2020:
        rec = [(f"{REC_2020_DIR}/2020gan_recovered_{tag}_botsort.csv", "Ganaraska")]
    for fn, site in rec:
        if Path(fn).is_file():
            for r in csv.DictReader(open(fn)):
                if r["site"] == site:                 # only that river's rows are taken from the recovered file
                    out[(r["site"], r["date"], r["species"])] = (int(r["model"]), int(r["tech"]))
    return out


def pearson(xs, ys):
    """Pearson correlation of two equally long lists.

    Returns "not a number" for fewer than 3 points or when either list never varies; the
    summary table then shows a blank.
    """
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def main():
    """Read the per-season files, write the two tables, and print the Chinook net summary."""
    global SW, REC_2019_DIR, REC_2020_DIR, OUT_DAILY, OUT_SUMMARY
    ap = argparse.ArgumentParser(description="Assemble canonical_daily.csv and canonical_summary.csv.")
    ap.add_argument("--tag", default=config.CORPUS_TAG,
                    help="Model whose counts are assembled (default: the published model, v27). "
                         "v26 is the superseded model.")
    ap.add_argument("--source", choices=["output", "projects"], default="output",
                    help="output (default): read output/corpus_<tag>/, as written by "
                         "salmonid.count.run_corpus. projects: read the original per-season files on the "
                         "project data drive (config.corpus_dir(2025) and config.corpus_dir(2020)).")
    ap.add_argument("--sweep-root", default=None,
                    help="Folder holding net_rule_sweep/ and activity_sweep/ (overrides --source).")
    ap.add_argument("--recovered-dir", default=None,
                    help="Folder holding 2020gan_recovered_{net,activity}_botsort.csv (overrides --source).")
    ap.add_argument("--years", nargs="+", type=int, default=list(config.CORPUS_YEARS),
                    help="Seasons to include (default: 2020-2025).")
    ap.add_argument("--out-dir", default=str(config.OUTPUT_DIR),
                    help="Where to write the two tables (default: output/). Never data/.")
    a = ap.parse_args()

    if a.source == "projects":
        # The original run kept all seasons' count files in the newest season's folder.
        SW = config.corpus_dir(2025, a.tag)
        REC_2020_DIR = config.corpus_dir(2020, a.tag)
    else:
        SW = config.OUTPUT_DIR / f"corpus_{a.tag}"
        REC_2020_DIR = SW
    if a.sweep_root:
        SW = Path(a.sweep_root)
        if not a.recovered_dir and a.source != "projects":
            REC_2020_DIR = SW
    if a.recovered_dir:
        REC_2020_DIR = Path(a.recovered_dir)
    # The 2019 Credit River footage was counted only with the superseded model. Under the
    # published model that season is left out, not borrowed from the older model's files.
    REC_2019_DIR = config.corpus_dir(2019, "v26") if a.tag == "v26" else None

    config.require(Path(SW) / "net_rule_sweep",
                   "the per-season count files (folder 'net_rule_sweep'); create them with "
                   "'python -m salmonid.count.run_corpus', or pass --source projects",
                   "SALMONID_PROJECTS_ROOT" if a.source == "projects" else None)
    out_dir = Path(a.out_dir)
    if out_dir.resolve() == config.REPO_DATA.resolve():
        raise SystemExit("Refusing to write into data/: that folder holds the published copies. "
                         "Choose another --out-dir (default: output/).")
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT_DAILY = out_dir / "canonical_daily.csv"
    OUT_SUMMARY = out_dir / "canonical_summary.csv"

    rows = []
    summ = []
    RECOVERED = {(2019, "Credit"), (2020, "Ganaraska")}
    SKIP = {(2019, "Ganaraska")}                     # technician record only; no footage, so no model count
    if REC_2019_DIR is None:
        SKIP.add((2019, "Credit"))
    for year in a.years:
        cnet = countr_daily(year, "net"); cact = countr_daily(year, "activity")
        if not cnet:
            print(f"WARNING: no net count file found for {year} under {SW}; that season is left out")
        for site in ["Ganaraska", "Credit"]:
            if (year, site) in SKIP:
                continue
            for sp in SPECIES:
                # The days listed for this river and species are those in the NET file.
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
                             # Correlation over ALL listed days for this river, species and season.
                             "net_r": round(pearson(mn, tn), 3) if not math.isnan(pearson(mn, tn)) else "",
                             "model_act": Ma, "tech_act": Ta,
                             "act_pct_err": round(100 * (Ma - Ta) / Ta, 1) if Ta else "",
                             "act_r": round(pearson(ma, ta), 3) if not math.isnan(pearson(ma, ta)) else "",
                             # Days with a non-zero technician net count (else: non-zero model net count).
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
    # Headline: Chinook net counts by season and river (manuscript Table 3).
    print("\nChinook NET model vs tech (both rivers):")
    for s in summ:
        if s["species"] == "Chinook":
            rec = " [recovered]" if s["recovered"] else ""
            print(f"  {s['year']} {s['river']:10s} model {s['model_net']:6d} tech {s['tech_net']:6d} "
                  f"({s['net_pct_err']}%, r={s['net_r']}){rec}")


if __name__ == "__main__":
    main()

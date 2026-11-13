"""In plain terms: this script answers two questions about the daily counts. First, when all
five species are added together, by how many fish does the model's count for a day typically
differ from the technician's? Second, the run is concentrated on a few very busy days, so
how does the model do on exactly those days: the set of busiest days that together carry two
thirds of all the Chinook Salmon the technicians counted?

Inputs
------
``data/canonical_daily.csv`` (committed, Tier 1): one row per year, river, species and day
with the model's and the technician's net count. The 2020 Ganaraska River season is set
aside, as everywhere in the agreement statistics (Methods 2.6).

Outputs
-------
Printed, and ``daily_totals_and_peak_days.csv`` in ``--out-dir`` (default ``output/paper/``).

Feeds
-----
* Abstract and Section 3.5: "daily totals across species differed by a median of three fish"
  (1,702 days at the two rivers).
* Section 3.5 and Conclusions: "The 127 days (15%) that carried two thirds of the
  technician-counted Chinook Salmon run had a model total 7% below the technician's. Ranked
  within each river, the shortfall was 7% at the Ganaraska River and 11% at the Credit River."

Run
---
::

    python -m salmonid.paper.daily_totals_and_peak_days

Tier
----
1.

Provenance
----------
Ported from ``conclusions_check.py`` (manuscript repository, 2026-11-09 review fixes). The
calculations and the printout are unchanged; the input path, this docstring, the comments
and the CSV output were added.
"""
from __future__ import annotations

import argparse

import pandas as pd

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir


def top(x, name, rows):
    """The busiest days of ``x`` that together carry two thirds of the technician count.

    Days are ranked from the largest technician count down and taken until their running
    share of the technician total first reaches two thirds. On those days the model's total
    is divided by the technician's: 0.93 means the model counted 7% fewer fish than the
    technician on the days that matter most for the size of the run.
    """
    x = x.sort_values("tech_net", ascending=False)
    cum = x.tech_net.cumsum() / x.tech_net.sum()
    k = int((cum < 2/3).sum()) + 1
    t = x.iloc[:k]
    print(f"{name}: days {len(x)}, top days {k} ({k/len(x):.1%}) carry {t.tech_net.sum()/x.tech_net.sum():.1%}; "
          f"model/tech on them {t.model_net.sum()/t.tech_net.sum():.3f}; min tech count on those days {t.tech_net.min()}")
    rows.append(dict(statistic="busiest days carrying two thirds of the technician Chinook count",
                     scope=name, days=len(x), top_days=k, share_of_days=round(k / len(x), 4),
                     share_of_run=round(t.tech_net.sum() / x.tech_net.sum(), 4),
                     model_over_tech=round(t.model_net.sum() / t.tech_net.sum(), 4),
                     min_tech_count=int(t.tech_net.min())))
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    args = ap.parse_args()
    out = out_dir(args)
    rows = []

    d = pd.read_csv(config.CANONICAL_DAILY)
    print(d.groupby(["river", "year"]).size().unstack(0))
    print(d.species.unique())
    # Year-inclusion rule (Methods 2.6): 2020 at the Ganaraska River is set aside.
    keep = d[~((d.river == "Ganaraska") & (d.year == 2020))]
    for label, x in [("all species rows", keep), ("3 abundant", keep[keep.species.isin(["Chinook", "Coho", "Rainbow Trout", "Rainbow"])])]:
        # One total per river and day: the species are added together, so a Coho Salmon that
        # the model called a Chinook Salmon is not a difference here. The median of the
        # absolute difference is quoted because a few very busy days would dominate a mean.
        tot = x.groupby(["river", "date"])[["model_net", "tech_net"]].sum()
        ad = (tot.model_net - tot.tech_net).abs()
        print(f"\n{label}: river-days {len(tot)}; median |daily total diff| {ad.median()}, mean {ad.mean():.1f}; by river:",
              ad.groupby(level=0).median().to_dict())
        sd = (x.model_net - x.tech_net).abs()
        print("   per species-day median", sd.median())
        rows.append(dict(statistic="median absolute difference in the daily total", scope=label,
                         days=len(tot), median_abs_diff=float(ad.median()),
                         mean_abs_diff=round(float(ad.mean()), 1)))
    c = keep[keep.species == "Chinook"]
    print()
    allt = top(c, "pooled (days ranked across both rivers)", rows)
    for r, g in allt.groupby("river"):
        print(f"   of pooled top days, {r}: n {len(g)}, model/tech {g.model_net.sum()/g.tech_net.sum():.3f}")
    for r, g in c.groupby("river"):
        top(g, f"{r} ranked within river", rows)
    for yrs, nm in [((2022, 2025), "2022-2025")]:
        cc = c[(c.year >= yrs[0]) & (c.year <= yrs[1])]
        t = top(cc, f"pooled {nm}", rows)
        for r, g in t.groupby("river"):
            print(f"   {r}: n {len(g)}, model/tech {g.model_net.sum()/g.tech_net.sum():.3f}")

    pd.DataFrame(rows).to_csv(out / "daily_totals_and_peak_days.csv", index=False)
    print(f"\nwrote: {out / 'daily_totals_and_peak_days.csv'}")


if __name__ == "__main__":
    main()

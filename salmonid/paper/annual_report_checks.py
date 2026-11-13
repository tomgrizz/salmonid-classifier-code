"""In plain terms: the Methods quote ranges and shares taken from the technicians' published
annual totals (Tables S1 and S2), for example "5,000 to 28,000 Chinook Salmon a year" and
"unidentified salmonids were equivalent to 27% to 36% of the Chinook Salmon count in 2020 to
2022". This script recomputes those figures from the two tables, so a reader can see which
years each range covers and how each share was formed.

Inputs
------
Committed in ``data/paper/`` (Tier 1): ``table_s1_ganaraska_annual_counts.csv`` and
``table_s2_credit_annual_counts.csv``, the machine-readable form of Tables S1 and S2 (annual
upstream totals published by the Lake Ontario Management Unit).

Outputs
-------
Printed, and ``annual_report_checks.csv`` in ``--out-dir`` (default ``output/paper/``).

Feeds
-----
* Section 2.1: the per-species annual ranges at each river.
* Section 2.2: unidentified salmonids as 27% to 36% of the Ganaraska River Chinook Salmon
  count in 2020 to 2022 and 3% or less from 2023.
* Table 4 caption: Credit River Coho Salmon, 31 to 65 fish a year since 2023 against 431.
* Table S1 and S2 column totals.

Run
---
::

    python -m salmonid.paper.annual_report_checks

Tier
----
1.

Provenance
----------
WRITTEN FOR THE DEPOSIT. These figures were read off the tables by hand; no earlier script
existed. Nothing is estimated: each value is a minimum, maximum, sum or ratio of table cells.
"""
from __future__ import annotations

import argparse

import pandas as pd

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir

FILES = {"Ganaraska": "table_s1_ganaraska_annual_counts.csv", "Credit": "table_s2_credit_annual_counts.csv"}
FIVE = ["Rainbow Trout", "Chinook Salmon", "Coho Salmon", "Brown Trout", "Atlantic Salmon"]


def load(river):
    """Species-by-year counts as numbers (rows = species, columns = years)."""
    t = pd.read_csv(config.require(config.PAPER_DATA / FILES[river], "committed annual-count table"),
                    dtype=str).set_index("Species")
    years = [c for c in t.columns if c.isdigit()]
    counts = t.loc[["All Salmon & Trout"] + FIVE, years].apply(pd.to_numeric, errors="coerce")
    return counts, t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    args = ap.parse_args()
    out = out_dir(args)
    rows = []
    for river in FILES:
        counts, raw = load(river)
        counts = counts.dropna(axis=1, how="all")
        print(f"\n== {river} River, {counts.columns[0]} to {counts.columns[-1]} ==")
        for sp in FIVE:
            s = counts.loc[sp].dropna()
            total_ok = int(s.sum()) == int(pd.to_numeric(raw.loc[sp, "Total"]))
            print(f"  {sp:16s} {int(s.min()):>6,} ({s.idxmin()}) to {int(s.max()):>6,} ({s.idxmax()}); "
                  f"sum {int(s.sum()):,} {'= printed total' if total_ok else '!= printed total'}")
            rows.append(dict(river=river, quantity=f"{sp} annual range", low=int(s.min()), low_year=s.idxmin(),
                             high=int(s.max()), high_year=s.idxmax()))
        # Unidentified salmonids are not a column of the tables. They are what is left of the
        # "All Salmon & Trout" total after the five species columns are subtracted. Expressed
        # against the Chinook Salmon count because the reference used to validate the model
        # leaves these fish out while the model assigns every counted fish to a species.
        unid = counts.loc["All Salmon & Trout"] - counts.loc[FIVE].sum()
        share = 100 * unid / counts.loc["Chinook Salmon"]
        print("  unidentified salmonids (All S & T minus the five species), and as % of the Chinook Salmon count:")
        for y in counts.columns:
            print(f"    {y}: {int(unid[y]):>6,}  {share[y]:5.1f}%")
            rows.append(dict(river=river, quantity="unidentified salmonids", low_year=y, low=int(unid[y]),
                             pct_of_chinook=round(float(share[y]), 1)))
    pd.DataFrame(rows).to_csv(out / "annual_report_checks.csv", index=False)
    print(f"\nwrote: {out / 'annual_report_checks.csv'}")


if __name__ == "__main__":
    main()

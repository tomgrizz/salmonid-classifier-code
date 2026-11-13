"""In plain terms: the manuscript's count tables compare how many fish the model counted with
how many the technicians counted, by species, river and year. This script rebuilds those
tables from the two committed count files, so every cell can be traced to a row of data. It
prints Table 2 (count columns), Table 3 and Table 4, and the handful of count statistics that
are quoted in the Results text but are not in any table: the per-year size of the daily
difference, what happens on quiet days, and the share of the Chinook Salmon run that falls
inside the Ganaraska River sanctuary window.

Inputs
------
Committed (Tier 1):

* ``data/canonical_daily.csv``: one row per year, river, species and day on which either
  the model or the technician recorded a passage; net counts for both.
* ``data/canonical_summary.csv``: one row per year, river and species; annual net totals,
  the daily correlation, and the flag for the set-aside season (2020, Ganaraska River).
* ``data/paper/table_s2_credit_annual_counts.csv``: the published annual totals (Table S2),
  used only for the 2025 Credit River comparison against the published figure.

Outputs
-------
Printed, and written to ``--out-dir`` (default ``output/paper/``):
``table2_count_columns.csv``, ``table3_chinook_multiyear.csv``, ``table4_three_species.csv``,
``fig4_panel_annotations.csv``, ``chinook_daily_difference_by_year.csv``,
``quiet_day_sums.csv``, ``sanctuary_window_share.csv``.

Feeds
-----
* Table 2, columns Daily r, Days, 95% CI and % error (and the caption's -6% against the
  published Credit River total; Ganaraska River Atlantic Salmon 6 against 3).
* Table 3 (Chinook Salmon, % error and daily r by river and year) and its note.
* Table 4 (mean absolute net error and median daily r, 2022 to 2025).
* Abstract, Sections 3, 3.2, 3.3, 4, 4.2 and 5: the 6% and 10% Chinook Salmon errors; daily r
  0.94 to 0.99; within about 4%, 3%, 13%, 6%, 14%, 11%, 41%.
* Figure 4 caption: the Ganaraska River Chinook Salmon r-squared of 0.96 over six years.
* Section 3.5: Ganaraska River difference of +22.8 fish per day in 2021 and -1.1 over 2022
  to 2025; the quiet-day sums (27% above at the Ganaraska River, equal at the Credit River).
* Section 3.6: 85% to 98% of the run inside the sanctuary window; within 4 percentage points.
* Section 2.2: 10,533 against 9,748 (8% more).

Run
---
::

    python -m salmonid.paper.count_tables

Tier
----
1.

Provenance
----------
WRITTEN FOR THE DEPOSIT. No single earlier script produced these tables: the cells were
read from ``canonical_summary.csv`` and the text statistics were computed in scratch
sessions that were not kept. Each block below implements the definition given in the
manuscript (the section is cited beside it) and nothing else. Where the manuscript's wording
admits two readings, both are printed and the difference is stated, so nothing here was
chosen to match a printed number. The sanctuary block repeats the formula inside
``salmonid.figures.render_hero_regulation`` (which draws the values but does not print
them).
"""
from __future__ import annotations

import argparse
import datetime
import math

import numpy as np
import pandas as pd
from scipy import stats

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
ABUNDANT = ["Chinook", "Coho", "Rainbow Trout"]
RIVERS = ["Ganaraska", "Credit"]


def pct(model, tech):
    """Percentage error of a model total against the technician total (model minus technician)."""
    return 100.0 * (model - tech) / tech if tech else float("nan")


def fisher_ci(r, n):
    """95% confidence interval for a Pearson r on n days (Fisher z-transformation), and the
    two-sided p-value. r is stretched to z = atanh(r), whose error is near-normal with
    SE = 1 / sqrt(n - 3); the interval is taken there and mapped back. Needs n >= 4."""
    if n < 4 or not np.isfinite(r):
        return float("nan"), float("nan"), float("nan")
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r * r)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se), 2 * stats.t.sf(abs(t), n - 2)


def daily_r(g):
    """Pearson correlation of the model's and the technician's daily net counts."""
    return float(np.corrcoef(g.model_net, g.tech_net)[0, 1]) if len(g) > 2 else float("nan")


# ------------------------------------------------------------------------------- Table 2
def table2(daily):
    """Count columns of Table 2 (2025). Methods 2.6: the correlation is "computed over all
    days on which either series recorded a passage", and "its degrees of freedom, two-sided
    p-value, and 95% confidence interval use the same days". ``canonical_daily.csv`` holds
    exactly those days (a day on which neither recorded a fish of that species has no row),
    so Days is the number of rows."""
    rows = []
    for sp in SPECIES:
        for rv in RIVERS:
            g = daily[(daily.year == 2025) & (daily.river == rv) & (daily.species == sp)]
            n, r = len(g), daily_r(g)
            lo, hi, p = fisher_ci(r, n)
            rows.append(dict(species=sp, river=rv, daily_r=round(r, 3), days=n, df=n - 2,
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3), p=p,
                             model_net=int(g.model_net.sum()), tech_net=int(g.tech_net.sum()),
                             pct_error=round(pct(g.model_net.sum(), g.tech_net.sum()), 1)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- Table 3
def table3(daily):
    """Chinook Salmon by river and year: annual net percentage error and daily net r."""
    rows = []
    for y in range(2020, 2026):
        row = dict(year=y)
        for rv in RIVERS:
            g = daily[(daily.year == y) & (daily.river == rv) & (daily.species == "Chinook")]
            row[f"{rv}_pct_error"] = round(pct(g.model_net.sum(), g.tech_net.sum()), 1)
            row[f"{rv}_daily_r"] = round(daily_r(g), 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- Table 4
def table4(daily):
    """Three abundant species over 2022 to 2025: the mean of the four years' ABSOLUTE annual
    percentage errors (so a +10% year and a -10% year average to 10%, not 0%), and the median
    of the four years' daily correlations. With four values the median is the mean of the
    middle two."""
    rows = []
    for sp in ABUNDANT:
        for rv in RIVERS:
            errs, rs = [], []
            for y in range(2022, 2026):
                g = daily[(daily.year == y) & (daily.river == rv) & (daily.species == sp)]
                errs.append(pct(g.model_net.sum(), g.tech_net.sum()))
                rs.append(daily_r(g))
            rows.append(dict(species=sp, river=rv,
                             mean_abs_net_error_pct=round(float(np.mean(np.abs(errs))), 2),
                             median_daily_r=round(float(np.median(rs)), 4),
                             yearly_pct_error="; ".join(f"{e:+.1f}" for e in errs),
                             yearly_daily_r="; ".join(f"{r:.3f}" for r in rs)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- Figure 4 labels
def fig4_annotations(daily):
    """What each Figure 4 panel prints: the percentage error of the count pooled over all six
    plotted years, and the r-squared between the two series of six annual totals. These
    include the set-aside 2020 Ganaraska River season, which is why the Chinook Salmon
    r-squared there (0.96) is below the 0.998 of Section 3.3 (five years)."""
    rows = []
    for rv in RIVERS:
        for sp in SPECIES:
            a = (daily[(daily.river == rv) & (daily.species == sp) & daily.year.between(2020, 2025)]
                 .groupby("year")[["model_net", "tech_net"]].sum())
            r = stats.pearsonr(a.model_net, a.tech_net)[0] if len(a) >= 3 else float("nan")
            rows.append(dict(river=rv, species=sp, years=len(a),
                             pooled_pct_error=round(pct(a.model_net.sum(), a.tech_net.sum()), 1),
                             interannual_r2=round(r * r, 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Section 3.5, by year
def chinook_difference_by_year(daily, keep):
    """Mean daily difference (model minus technician, fish per day) for Chinook Salmon by
    river and year, under the year-inclusion rule. The Ganaraska River mean of +2.9 is
    almost entirely the 2021 season."""
    c = daily[(daily.species == "Chinook") & keep].copy()
    c["diff"] = c.model_net - c.tech_net
    t = c.groupby(["river", "year"])["diff"].agg(days="size", mean_diff="mean").reset_index()
    t["mean_diff"] = t.mean_diff.round(2)
    extra = []
    for rv in RIVERS:
        g = c[c.river == rv]
        extra.append(dict(river=rv, year="all included years", days=len(g), mean_diff=round(g["diff"].mean(), 2)))
    g = c[(c.river == "Ganaraska") & (c.year >= 2022)]
    extra.append(dict(river="Ganaraska", year="2022-2025", days=len(g), mean_diff=round(g["diff"].mean(), 2)))
    extra.append(dict(river="both", year="all included years", days=len(c), mean_diff=round(c["diff"].mean(), 2)))
    return pd.concat([t, pd.DataFrame(extra)], ignore_index=True)


# ---------------------------------------------------------------- Section 3.5, quiet days
def quiet_day_sums(daily, keep):
    """Section 3.5: on quiet days the MEDIAN ratio of model to technician is 1.00, but that
    is partly an artefact of small whole numbers (2 against 2, 3 against 3). Summing the
    counts over the same days shows the level: "27% above the technician's at the Ganaraska
    River, a difference of one to two fish per day, and equal to it at the Credit River".

    "Quiet" is up to about 20 fish per day at the Ganaraska River and 50 at the Credit
    River. The manuscript does not say whether a day's size is the technician's count or
    the mean of the two counts (the count bands in ``agreement_ba_by_count_band.csv`` use
    the mean), so both are given. Days with a count of zero on the chosen measure are left
    out, as they are from the median ratio."""
    c = daily[(daily.species == "Chinook") & keep]
    rows = []
    for rv, top in (("Ganaraska", 20), ("Credit", 50)):
        g = c[c.river == rv]
        for basis, size in (("mean of the two counts", (g.model_net + g.tech_net) / 2.0),
                            ("technician count", g.tech_net)):
            q = g[(size >= 1) & (size < top)]
            rows.append(dict(river=rv, day_size=f"1 to {top - 1} fish", size_measured_by=basis,
                             days=len(q), model_sum=int(q.model_net.sum()), tech_sum=int(q.tech_net.sum()),
                             model_over_tech=round(q.model_net.sum() / q.tech_net.sum(), 3),
                             mean_diff_per_day=round(float((q.model_net - q.tech_net).mean()), 2)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Section 3.6, sanctuary
def sanctuary_share(daily):
    """Share of each year's Ganaraska River Chinook Salmon upstream run that falls inside
    the 1 September to 14 October sanctuary window, from the technician's and from the
    model's daily counts. Days with a negative net count (more fish down than up) are set to
    zero first, so the run is a running total that never decreases; this is the formula in
    ``salmonid.figures.render_hero_regulation.fig_sanctuary`` (Figure 7b)."""
    rows = []
    g = daily[(daily.river == "Ganaraska") & (daily.species == "Chinook")].copy()
    d = pd.to_datetime(g.date)
    md = pd.Series([datetime.date(2000, m, dd) for m, dd in zip(d.dt.month, d.dt.day)], index=g.index)
    inside = (md >= datetime.date(2000, 9, 1)) & (md <= datetime.date(2000, 10, 14))
    for y in range(2020, 2026):
        row = dict(year=y)
        for col, name in (("tech_net", "technician_pct"), ("model_net", "model_pct")):
            v = g.loc[g.year == y, col].clip(lower=0)
            row[name] = round(100.0 * v[inside[g.year == y]].sum() / (v.sum() or 1), 1)
        row["model_minus_technician_points"] = round(row["model_pct"] - row["technician_pct"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    args = ap.parse_args()
    out = out_dir(args)
    pd.set_option("display.width", 220)

    daily = pd.read_csv(config.CANONICAL_DAILY)
    summ = pd.read_csv(config.CANONICAL_SUMMARY)
    # Year-inclusion rule (Methods 2.6): a season whose archived video was saved as shorter
    # files is set aside from the agreement statistics. The summary file flags it
    # (recovered = True); only 2020 at the Ganaraska River carries the flag.
    flagged = {(int(r.year), r.river) for r in summ[summ.recovered.astype(str) == "True"].itertuples()}
    keep = pd.Series([(y, rv) not in flagged for y, rv in zip(daily.year, daily.river)], index=daily.index)
    print(f"set-aside seasons (year, river): {sorted(flagged)}")

    # Cross-check: recomputing from the daily rows must give the committed summary cells.
    chk = (daily.groupby(["year", "river", "species"])[["model_net", "tech_net"]].sum().reset_index()
           .merge(summ, on=["year", "river", "species"], suffixes=("_d", "")))
    assert (chk.model_net_d == chk.model_net).all() and (chk.tech_net_d == chk.tech_net).all(), \
        "canonical_daily.csv and canonical_summary.csv disagree"

    t2 = table2(daily)
    print("\n== Table 2, count columns (2025) ==")
    print(t2.drop(columns="p").to_string(index=False))
    print(f"   largest p-value among rows with a CI: {t2.p.max():.2g}  (caption: all p < 0.001)")
    t2.to_csv(out / "table2_count_columns.csv", index=False)

    t3 = table3(daily)
    print("\n== Table 3, Chinook Salmon by year (% error, daily r) ==")
    print(t3.to_string(index=False))
    r21 = pd.concat([t3[t3.year >= 2021][f"{rv}_daily_r"] for rv in RIVERS])
    print(f"   daily r, 2021 to 2025, both rivers: {r21.min():.3f} to {r21.max():.3f}  (Section 3.3: 0.94 to 0.99)")
    t3.to_csv(out / "table3_chinook_multiyear.csv", index=False)

    t4 = table4(daily)
    print("\n== Table 4, three abundant species, 2022 to 2025 ==")
    print(t4.to_string(index=False))
    t4.to_csv(out / "table4_three_species.csv", index=False)

    f4 = fig4_annotations(daily)
    print("\n== Figure 4 panel annotations (all six plotted years) ==")
    print(f4.to_string(index=False))
    f4.to_csv(out / "fig4_panel_annotations.csv", index=False)

    by = chinook_difference_by_year(daily, keep)
    print("\n== Section 3.5: Chinook Salmon mean daily difference, model minus technician (fish per day) ==")
    print(by.to_string(index=False))
    by.to_csv(out / "chinook_daily_difference_by_year.csv", index=False)

    q = quiet_day_sums(daily, keep)
    print("\n== Section 3.5: quiet days, counts summed (Chinook Salmon) ==")
    print(q.to_string(index=False))
    q.to_csv(out / "quiet_day_sums.csv", index=False)

    s = sanctuary_share(daily)
    print("\n== Section 3.6: % of the Ganaraska River Chinook Salmon run inside 1 Sept to 14 Oct ==")
    print(s.to_string(index=False))
    print(f"   model: {s.model_pct.min():.0f}% to {s.model_pct.max():.0f}%, median {s.model_pct.median():.0f}%; "
          f"largest gap to the technician {s.model_minus_technician_points.abs().max():.1f} points")
    s.to_csv(out / "sanctuary_window_share.csv", index=False)

    # Section 2.2 / Table 2 caption: the 2025 Credit River Chinook Salmon reference summed from
    # the technicians' event record against the published annual total of Table S2.
    row = summ[(summ.year == 2025) & (summ.river == "Credit") & (summ.species == "Chinook")].iloc[0]
    s2 = pd.read_csv(config.PAPER_DATA / "table_s2_credit_annual_counts.csv").set_index("Species")
    published = int(s2.loc["Chinook Salmon", "2025"])
    print("\n== Section 2.2: 2025 Credit River Chinook Salmon, two references ==")
    print(f"   event-based reference {int(row.tech_net):,}; published annual total {published:,}; "
          f"event-based is {pct(row.tech_net, published):+.1f}% against published")
    print(f"   model {int(row.model_net):,}: {pct(row.model_net, row.tech_net):+.1f}% against the event-based "
          f"reference, {pct(row.model_net, published):+.1f}% against the published total")

    # Sections 2.1 and 3.2: how busy the busiest days are (all species together).
    tot = daily.groupby(["river", "date"])[["model_net", "tech_net"]].sum()
    print("\n== Busiest single day, all species (net fish) ==")
    print(tot.groupby(level=0).max().to_string())
    print(f"\nwrote 7 tables to {out}")


if __name__ == "__main__":
    main()

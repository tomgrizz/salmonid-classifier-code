"""In plain terms: a correlation coefficient says how closely the model's daily counts rise
and fall with the technician's, but a single number hides how sure we can be of it. This
script attaches the uncertainty: for each 2025 daily correlation it gives the number of days
it rests on, a significance test and a 95% confidence interval. It also checks whether the
model reproduces the year-to-year ups and downs of the run, which is the property a
monitoring index needs.

Inputs
------
``data/canonical_summary.csv`` (committed, Tier 1): one row per river, species and season
with the daily net-count correlation (``net_r``), the number of days with a technician count
(``active_days``) and the two annual net totals.

Outputs
-------
Printed only.

Feeds
-----
* Section 3.3: the Chinook Salmon year ranking (Spearman's rho = 1.00 at both rivers).
* The history of Table 2: see "Which day count" below.

Run
---
::

    python -m salmonid.paper.correlation_stats

Tier
----
1.

Which day count (read before quoting)
-------------------------------------
This is the script as it stood when the manuscript's correlation intervals were first
computed, copied unchanged. It takes the number of days from ``active_days``, the days on
which the TECHNICIAN recorded a passage. The manuscript now counts days on which EITHER
series recorded a passage (Methods 2.6), so the Days column and the intervals of Table 2 as
printed come from ``salmonid.paper.count_tables``, not from here. The two differ by a few
days per row and by at most 0.03 in an interval end. Likewise the inter-annual r-squared
and the pooled rank correlation printed here use all six seasons, including the set-aside
2020 Ganaraska River season; the values in Section 3.3 follow the year-inclusion rule and
come from ``salmonid.results.agreement_stats``.

Provenance
----------
Ported from ``scripts/compute_correlation_stats.py`` (manuscript repository). ``daily_stats``,
``fmt_p`` and ``main`` are unchanged; only the input path was changed.
"""
import csv
import math

from scipy import stats
import numpy as np

from salmonid import config

CSV = config.CANONICAL_SUMMARY


# daily_stats: the test and interval for a Pearson correlation r on n days.
#   * df = n - 2 and the t statistic give the two-sided p-value for "no linear relation".
#   * The interval uses Fisher's z-transformation. r cannot exceed 1, so its sampling
#     error is lopsided near 1; z = atanh(r) stretches it onto an unbounded scale where the
#     error is close to normal with SE = 1 / sqrt(n - 3). The interval is taken there and its
#     ends are mapped back with tanh.
#   * Days within a season are autocorrelated, so the true uncertainty is somewhat wider
#     than this; the manuscript says the p-values are indicative.
def daily_stats(r, n):
    """Return (df, p, ci_low, ci_high) for a Pearson r on n paired observations."""
    df = n - 2
    if n < 4:  # too few for an interval
        return df, None, None, None
    t = r * math.sqrt(df) / math.sqrt(1 - r * r)
    p = 2 * stats.t.sf(abs(t), df)
    z = math.atanh(r)
    se = 1 / math.sqrt(n - 3)
    return df, p, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def fmt_p(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

    print("2025 daily net-upstream correlations (Section 3.3, Table 4)")
    print(f"{'river':10s}{'species':16s}{'r':>7s}{'n':>5s}{'df':>5s}{'p':>9s}  95% CI")
    for row in rows:
        if row["year"] != "2025":
            continue
        r, n = float(row["net_r"]), int(row["active_days"])
        df, p, lo, hi = daily_stats(r, n)
        ci = "n/a" if lo is None else f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{row['river']:10s}{row['species']:16s}{r:+7.3f}{n:5d}{df:5d}{fmt_p(p):>9s}  {ci}")

    print("\nInter-annual Chinook agreement, 2020-2025 (Section 3.5)")
    years = {"2020", "2021", "2022", "2023", "2024", "2025"}
    for river in ["Ganaraska", "Credit"]:
        pts = [(float(x["model_net"]), float(x["tech_net"])) for x in rows
               if x["species"] == "Chinook" and x["river"] == river and x["year"] in years]
        a = np.array([p[0] for p in pts])
        b = np.array([p[1] for p in pts])
        r, pv = stats.pearsonr(a, b)
        rho, ps = stats.spearmanr(a, b)
        print(f"{river:10s} n={len(pts)} seasons  r2={r * r:.3f} p={fmt_p(pv)}  "
              f"Spearman rho={rho:.2f} p={fmt_p(ps)}")

    # Pooled main-species inter-annual TREND agreement (Section 3.5, Figure 4).
    # A raw pooled regression of annual model vs technician net counts is dominated by
    # Chinook's magnitude (r2 ~ 0.99) and measures agreement in absolute size, not trend.
    # To measure whether the model reproduces each species' year-to-year up/down pattern
    # we standardize each species/river series (z-score) so all species weigh equally,
    # then take the pooled rank (Spearman) correlation across all species-seasons.
    # All available seasons (recovered years included, as Figure 4 plots them). The
    # rank/standardized statistic is robust to the recovered years' magnitude inflation
    # because those seasons still preserve the correct run timing and year ranking.
    main = ["Chinook", "Coho", "Rainbow Trout"]
    zm, zt, panel_r2 = [], [], []
    for sp in main:
        for river in ["Ganaraska", "Credit"]:
            pts = [(float(x["model_net"]), float(x["tech_net"])) for x in rows
                   if x["species"] == sp and x["river"] == river]
            if len(pts) < 3:
                continue
            m = np.array([p[0] for p in pts])
            t = np.array([p[1] for p in pts])
            panel_r2.append(stats.pearsonr(m, t)[0] ** 2)
            zm += list((m - m.mean()) / m.std(ddof=0))
            zt += list((t - t.mean()) / t.std(ddof=0))
    rho, ps = stats.spearmanr(zm, zt)
    r_raw = stats.pearsonr(
        [float(x["model_net"]) for x in rows if x["species"] in main],
        [float(x["tech_net"]) for x in rows if x["species"] in main],
    )[0]
    print(f"\nPooled main-species trend agreement (standardized, all seasons): "
          f"Spearman rho={rho:.2f} p={fmt_p(ps)} (n={len(zm)} species-seasons)")
    print(f"  mean per-panel abundance-index r2 = {np.mean(panel_r2):.3f} ({len(panel_r2)} panels)")
    print(f"  (raw pooled Pearson r2={r_raw ** 2:.3f} -- magnitude-dominated, NOT reported)")


if __name__ == "__main__":
    main()

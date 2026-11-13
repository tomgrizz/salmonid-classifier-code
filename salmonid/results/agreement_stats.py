"""In plain terms: this file re-does the paper's "do the two counts agree?" statistics in a way
that survives a statistician's reading. The original versions made three assumptions that daily
fish counts violate. First, the Bland-Altman limits assumed the disagreement is the same size on
a 5-fish day as on a 500-fish day; it is not, so the limits are recomputed as a *percentage* of
the day's count and separately for each river. Second, the confidence intervals resampled single
days as if each day were independent of the one before it; migration runs do not work that way,
so they are recomputed by resampling contiguous blocks of days. Third, year-to-year abundance was
reported as a correlation, which a systematically inflated year can still score perfectly on, so
an agreement statistic is reported beside it.

Agreement statistics for automated vs technician fish counts, correcting three
distributional assumptions that the daily-count data does not satisfy.

Pipeline stage: **results** (data -> detect -> track -> count -> *results* -> figures).
A leaf reporting step: it consumes the already-aggregated canonical count tables and
emits agreement statistics. It loads no model and runs in the **base Python
environment** (numpy/pandas/scipy only).

What it computes
----------------
1. **Proportional / log Bland-Altman limits of agreement, per river** (Bland & Altman
   1999, sec. 3.2). Daily net counts are strongly heteroscedastic: the largest
   model-minus-technician differences land on the largest-count days, which is exactly
   the condition under which constant +-1.96 SD limits are not valid. Two remedies are
   reported, both **per river** because pooling cancels a Credit under-read against a
   Ganaraska over-read and hides both:

   * *ratio limits* -- Bland and Altman's own log-transform remedy. Differences are taken
     on the natural-log scale and back-transformed, so the limits read as "the model
     records between 0.85 and 1.18 times the technician count". Requires both counts
     positive, so zero-passage days are dropped (their number is reported).
   * *proportional limits* -- the absolute difference is regressed on the mean count, and
     the limits are drawn as +-1.96 * (fitted absolute difference) * sqrt(pi/2). This keeps
     every day including the zeros and gives limits that widen with the count.

   The constant-limits statistic the manuscript currently quotes is also recomputed per
   river so the three can be compared directly.

2. **Moving-block bootstrap CIs for Lin's CCC.** Daily passage is strongly autocorrelated
   within a run. An i.i.d. resample of days breaks that dependence and returns intervals
   that are too narrow. Blocks of consecutive days are resampled instead, drawn *within* a
   (year, river, species) season so a block never straddles two seasons or two rivers. The
   i.i.d. interval is reported alongside so the narrowing is visible.

3. **Inter-annual CCC beside the inter-annual r-squared.** r-squared measures correlation,
   not agreement: a season that reads 57% high sits perfectly on the regression line. Since
   the paper already introduces CCC to make that distinction for daily counts, the annual
   abundance index gets the same treatment. With five to six seasons the annual CCC is
   unstable, so it is reported *with* its bootstrap interval rather than as a point estimate.

4. **Spearman's rho beside Pearson's r** on the daily series. Daily counts are strongly
   right-skewed (many low days, a few very large ones), so r is dominated by the peak days.

Year-inclusion rule
-------------------
ONE rule, applied everywhere: **recovered footage-seasons are excluded from every agreement
statistic** (2019 Credit and 2020 Ganaraska, whose archived videos are more finely segmented,
so one fish can span several videos; run timing stays reliable but magnitude is inflated).
That leaves Ganaraska 2021-2025 and Credit 2020-2025. This is already what
``compute_concordance`` does for the daily statistics; applying it to the pooled
inter-annual rank correlation as well gives 33 species-years: 3 species x (5 Ganaraska
seasons + 6 Credit seasons).

Inputs : config.CANONICAL_DAILY, config.CANONICAL_SUMMARY (committed in data/).
Outputs: agreement_bland_altman.csv, agreement_ba_by_count_band.csv, agreement_ccc_block.csv,
         agreement_interannual.csv, written to --out-dir (default output/tier1/). The committed
         copies the manuscript quotes are in data/; run_tier1.py compares the two.
Feeds  : manuscript Section 3.3 (inter-annual r-squared, Spearman rho, CCC; pooled rho = 0.91,
         n = 33), Section 3.5 (per-river Bland-Altman bias; count-band median ratios; the
         moving-block intervals), the Spearman rho column of Table 5, and the Figure S3 caption.
Tier   : 1 (runs from committed data; no model, no GPU; about 20 seconds).

Reproducibility: every bootstrap is seeded (SEED). The committed tables were produced with
4,000 resamples, which is therefore the default (BOOT); with the defaults a re-run reproduces
data/agreement_*.csv exactly. A different --boot gives slightly different interval limits
(Monte Carlo error of about +-0.01 on the widest interval) but the same conclusions.

Run:
  python -m salmonid.results.agreement_stats [--out-dir <dir>] [--block 10] [--boot 4000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from salmonid import config
from salmonid.results.compute_concordance import lin_ccc, mcbride

MAIN = ["Chinook", "Coho", "Rainbow Trout"]
RIVERS = ["Ganaraska", "Credit"]
BOOT = 4000         # resamples; the value that produced the committed data/agreement_*.csv
BLOCK = 10          # days; n^(1/3) for a 400-900 day series is ~7.4-9.7, so 10 sits in the
                    # 7-14 range that is standard for a moving-block bootstrap on daily counts
SEED = 12345


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------
def load(include_recovered: bool = False, daily_path=None, summary_path=None):
    """Return (daily, summary) with the recovered seasons flagged and, by default, dropped.

    ``recovered`` lives on the summary table, so it is merged onto the daily rows by
    (year, river, species) exactly as ``compute_concordance.load`` does.

    The paths default to the committed canonical tables; pass them explicitly to score a
    different corpus build (for example a v2.7 assembly that has not yet been promoted) without
    overwriting the committed ones.
    """
    daily = pd.read_csv(daily_path or config.CANONICAL_DAILY)
    summ = pd.read_csv(summary_path or config.CANONICAL_SUMMARY)
    daily = daily.merge(summ[["year", "river", "species", "recovered"]],
                        on=["year", "river", "species"], how="left")
    if not include_recovered:
        daily = daily[daily["recovered"] == False].copy()   # noqa: E712
        summ = summ[summ["recovered"] == False].copy()      # noqa: E712
    return daily, summ


# ---------------------------------------------------------------------------------------
# 1. Bland-Altman: constant, ratio (log) and proportional limits, per river
# ---------------------------------------------------------------------------------------
def ba_constant(model, tech):
    """Classic Bland-Altman: mean difference and constant +-1.96 SD limits."""
    d = np.asarray(model, float) - np.asarray(tech, float)
    bias, sd = d.mean(), d.std(ddof=1)
    return dict(n=len(d), bias=bias, sd=sd,
                loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd)


def ba_ratio(model, tech):
    """Bland and Altman's log-transform remedy for non-constant variance.

    Differences are taken as log(model) - log(tech) and the limits back-transformed, so
    they read as multiplicative factors. Only days on which BOTH counts are positive can
    be used; ``n_dropped`` records how many zero-passage days that removes.
    """
    m = np.asarray(model, float)
    t = np.asarray(tech, float)
    keep = (m > 0) & (t > 0) & np.isfinite(m) & np.isfinite(t)
    n_dropped = int((~keep).sum())
    m, t = m[keep], t[keep]
    if len(m) < 3:
        return dict(n=len(m), n_dropped=n_dropped, ratio_bias=np.nan,
                    ratio_lo=np.nan, ratio_hi=np.nan)
    d = np.log(m) - np.log(t)
    bias, sd = d.mean(), d.std(ddof=1)
    return dict(n=len(m), n_dropped=n_dropped,
                ratio_bias=float(np.exp(bias)),
                ratio_lo=float(np.exp(bias - 1.96 * sd)),
                ratio_hi=float(np.exp(bias + 1.96 * sd)))


def ba_proportional(model, tech):
    """Limits that widen with the count (Bland & Altman 1999, sec. 3.2).

    The absolute difference is regressed on the mean of the two counts. Under normality
    E|d| = sigma * sqrt(2/pi), so the fitted absolute difference is scaled by sqrt(pi/2) to
    recover an SD and then by 1.96. ``slope_abs`` is the quantity that shows the
    heteroscedasticity (a slope of 0.08 means the limits widen by 8 fish per 100 fish/day).
    Unlike the ratio limits this keeps zero-passage days.

    Two spread fits are returned. The free-intercept fit is the diagnostic: its slope and
    p-value are the evidence that the spread depends on the count at all. It must NOT be
    used to draw the limits, because on these data the intercept is negative, so the fitted
    spread goes negative below roughly 30 fish/day and the "limits" invert. The limits are
    drawn from a through-the-origin fit (constant coefficient of variation, |d| = k * mean),
    which is well behaved over the whole observed range and is the right model for counts.
    """
    m = np.asarray(model, float)
    t = np.asarray(tech, float)
    keep = np.isfinite(m) & np.isfinite(t)
    m, t = m[keep], t[keep]
    if len(m) < 3:
        return dict(n=len(m), slope_diff=np.nan, slope_abs=np.nan, int_diff=np.nan,
                    int_abs=np.nan, k_abs=np.nan, het_r=np.nan, het_p=np.nan)
    mean = (m + t) / 2.0
    diff = m - t
    absd = np.abs(diff)
    b1, b0 = np.polyfit(mean, diff, 1)                 # bias line (free intercept)
    a1, a0 = np.polyfit(mean, absd, 1)                 # spread line, diagnostic only
    # Through-origin least squares for the spread actually used to draw the limits.
    k = float((mean @ absd) / (mean @ mean)) if (mean @ mean) else np.nan
    # Does the spread depend on the count? (the assumption the constant limits make)
    r_het, p_het = stats.pearsonr(mean, absd)
    return dict(n=len(m), int_diff=float(b0), slope_diff=float(b1),
                int_abs=float(a0), slope_abs=float(a1), k_abs=k,
                het_r=float(r_het), het_p=float(p_het))


def prop_limits_at(res, mean_count):
    """Evaluate the proportional limits at a given daily count (for reporting/plotting).

    Uses the through-origin spread fit, so the limits are +-1.96 * sqrt(pi/2) * k * mean:
    a fixed *percentage* of the day's count. The bias line keeps its free intercept.
    """
    centre = res["int_diff"] + res["slope_diff"] * mean_count
    halfwidth = 1.96 * np.sqrt(np.pi / 2) * res["k_abs"] * mean_count
    return centre - halfwidth, centre + halfwidth


def ba_by_count_band(daily, species="Chinook",
                     edges=(0, 1, 5, 20, 50, 200, np.inf)):
    """Bias and spread stratified by the size of the day, per river.

    This is the assumption-free version of the same evidence: no regression, no transform,
    just the mean and SD of the difference within bands of the day's mean count. If the
    constant limits were valid, ``sd_diff`` would be flat across the bands. Also reports the
    median ratio within each band so the relative error is visible where the absolute error
    is small, and vice versa.
    """
    rows = []
    d_sp = daily[daily.species == species]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"{int(lo)}" if hi == lo + 1 else
                      (f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f"{int(lo)}+"))
    for scope in RIVERS:
        d = d_sp[d_sp.river == scope]
        mean = (d.model_net + d.tech_net) / 2.0
        diff = (d.model_net - d.tech_net).to_numpy(float)
        for i, lab in enumerate(labels):
            sel = (mean >= edges[i]) & (mean < edges[i + 1])
            k = int(sel.sum())
            if k == 0:
                continue
            dd = diff[sel.to_numpy()]
            mm = d.model_net.to_numpy(float)[sel.to_numpy()]
            tt = d.tech_net.to_numpy(float)[sel.to_numpy()]
            pos = (mm > 0) & (tt > 0)
            rows.append(dict(
                species=species, river=scope, count_band=lab, days=k,
                mean_diff=round(float(dd.mean()), 2),
                sd_diff=round(float(dd.std(ddof=1)), 2) if k > 1 else np.nan,
                median_ratio=round(float(np.median(mm[pos] / tt[pos])), 3) if pos.sum() else np.nan,
            ))
    return pd.DataFrame(rows)


def bland_altman_table(daily, species="Chinook"):
    """Per-river (and pooled, for comparison) Bland-Altman under all three treatments."""
    rows = []
    d_sp = daily[daily.species == species]
    for scope in RIVERS + ["pooled"]:
        d = d_sp if scope == "pooled" else d_sp[d_sp.river == scope]
        if len(d) < 3:
            continue
        c = ba_constant(d.model_net, d.tech_net)
        r = ba_ratio(d.model_net, d.tech_net)
        p = ba_proportional(d.model_net, d.tech_net)
        # Report the proportional limits at two representative daily counts so the
        # widening is legible without the figure.
        lo10, hi10 = prop_limits_at(p, 10)
        lo100, hi100 = prop_limits_at(p, 100)
        rows.append(dict(
            species=species, scope=scope, river_days=c["n"],
            bias=round(c["bias"], 2), loa_lo=round(c["loa_lo"], 1), loa_hi=round(c["loa_hi"], 1),
            ratio_n=r["n"], ratio_zero_days_dropped=r["n_dropped"],
            ratio_bias=round(r["ratio_bias"], 3) if np.isfinite(r["ratio_bias"]) else np.nan,
            ratio_lo=round(r["ratio_lo"], 3) if np.isfinite(r["ratio_lo"]) else np.nan,
            ratio_hi=round(r["ratio_hi"], 3) if np.isfinite(r["ratio_hi"]) else np.nan,
            spread_slope=round(p["slope_abs"], 4), spread_r=round(p["het_r"], 3),
            spread_p=p["het_p"], spread_k=round(p["k_abs"], 4),
            prop_pct=round(100 * 1.96 * np.sqrt(np.pi / 2) * p["k_abs"], 1),
            prop_loa_at10=f"[{lo10:.1f}, {hi10:.1f}]",
            prop_loa_at100=f"[{lo100:.1f}, {hi100:.1f}]",
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# 2. Moving-block bootstrap for Lin's CCC
# ---------------------------------------------------------------------------------------
def _season_blocks(df, block, rng):
    """Resample one (year, river, species) season by moving blocks of consecutive days.

    ``df`` must already be sorted by date. Blocks start anywhere in 0..m-block (moving,
    i.e. overlapping, blocks) and are concatenated until the season's original length is
    reached. A season shorter than the block length is resampled as a single whole block,
    which is the conservative choice: it keeps the season intact rather than pretending to
    resample within it.
    """
    m = len(df)
    if m <= block:
        return df.index.to_numpy()
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, n_blocks)
    idx = np.concatenate([df.index.to_numpy()[s:s + block] for s in starts])
    return idx[:m]


def block_bootstrap_ccc(daily, species, river, block=BLOCK, B=BOOT, seed=SEED):
    """Three bootstrap CIs for the daily CCC of one species x river, to isolate one effect.

    The moving-block resample necessarily also *stratifies* by season: blocks are drawn within
    a season and each season keeps its own number of days, because a block that straddled two
    seasons would splice unrelated runs together. That is two changes at once relative to the
    published interval, which pools every day and resamples them freely. Comparing the two
    directly would confound the autocorrelation correction with the stratification, so all
    three are computed:

    * ``iid``          -- pooled i.i.d. days. This is what the manuscript currently reports.
    * ``strat``        -- i.i.d. days resampled WITHIN season, same days per season. Differs
                          from ``iid`` only by the stratification.
    * ``block``        -- moving blocks of consecutive days within season. Differs from
                          ``strat`` only by the block length.

    So ``strat`` vs ``block`` isolates the autocorrelation effect, and ``iid`` vs ``block`` is
    the practical question: how does the published interval change if both are fixed.
    """
    d = daily[(daily.species == species) & (daily.river == river)].copy()
    if len(d) < 3:
        return None
    d = d.sort_values(["year", "date"]).reset_index(drop=True)
    seasons = [g.sort_values("date") for _, g in d.groupby("year", sort=True)]
    point = lin_ccc(d.model_net, d.tech_net)

    rng = np.random.default_rng(seed)
    block_vals, iid_vals, strat_vals = [], [], []
    n = len(d)
    for _ in range(B):
        idx = np.concatenate([_season_blocks(s, block, rng) for s in seasons])
        r = d.loc[idx]
        block_vals.append(lin_ccc(r.model_net, r.tech_net)["ccc"])

        j = rng.integers(0, n, n)
        r2 = d.iloc[j]
        iid_vals.append(lin_ccc(r2.model_net, r2.tech_net)["ccc"])

        sidx = np.concatenate([s.index.to_numpy()[rng.integers(0, len(s), len(s))]
                               for s in seasons])
        r3 = d.loc[sidx]
        strat_vals.append(lin_ccc(r3.model_net, r3.tech_net)["ccc"])

    b_lo, b_hi = np.nanpercentile(np.asarray(block_vals, float), [2.5, 97.5])
    i_lo, i_hi = np.nanpercentile(np.asarray(iid_vals, float), [2.5, 97.5])
    s_lo, s_hi = np.nanpercentile(np.asarray(strat_vals, float), [2.5, 97.5])
    rho_s, p_s = stats.spearmanr(d.model_net, d.tech_net)
    return dict(
        species=species, river=river, river_days=n, seasons=len(seasons),
        pearson_r=round(point["r"], 3), spearman_rho=round(float(rho_s), 3),
        ccc=round(point["ccc"], 3), cb=round(point["cb"], 3), band=mcbride(point["ccc"]),
        iid_lo=round(i_lo, 3), iid_hi=round(i_hi, 3),
        strat_lo=round(s_lo, 3), strat_hi=round(s_hi, 3),
        block_lo=round(b_lo, 3), block_hi=round(b_hi, 3),
        block_len=block,
        # Isolates autocorrelation: block width / stratified-single-day width.
        autocorr_widening=round((b_hi - b_lo) / (s_hi - s_lo), 2) if (s_hi - s_lo) else np.nan,
        # The practical change from the published interval.
        vs_published=round((b_hi - b_lo) / (i_hi - i_lo), 2) if (i_hi - i_lo) else np.nan,
        clears_090_published=bool(i_lo >= 0.90),
        clears_090_block=bool(b_lo >= 0.90),
    )


def ccc_block_table(daily, block=BLOCK, B=BOOT, seed=SEED):
    rows = [block_bootstrap_ccc(daily, sp, rv, block, B, seed)
            for sp in MAIN for rv in RIVERS]
    return pd.DataFrame([r for r in rows if r])


# ---------------------------------------------------------------------------------------
# 3./4. Inter-annual agreement: CCC beside r-squared, and the pooled rank correlation
# ---------------------------------------------------------------------------------------
def interannual_table(summ, B=BOOT, seed=SEED):
    """Per species x river annual abundance index: r-squared, CCC with bootstrap CI, rho.

    With five to six seasons the CCC bootstrap is genuinely wide; that width is the point.
    It is reported so the reader can see how much less the annual series supports than the
    r-squared alone suggests.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for sp in MAIN:
        for rv in RIVERS:
            s = summ[(summ.species == sp) & (summ.river == rv)].sort_values("year")
            if len(s) < 3:
                continue
            m = s.model_net.to_numpy(float)
            t = s.tech_net.to_numpy(float)
            r, p = stats.pearsonr(m, t)
            rho, prho = stats.spearmanr(m, t)
            c = lin_ccc(m, t)
            n = len(m)
            vals = np.array([lin_ccc(m[i], t[i])["ccc"]
                             for i in (rng.integers(0, n, n) for _ in range(B))])
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            rows.append(dict(
                species=sp, river=rv, seasons=n,
                years=f"{int(s.year.min())}-{int(s.year.max())}",
                r2=round(r * r, 3), r2_p=p,
                spearman_rho=round(float(rho), 2), rho_p=float(prho),
                ccc=round(c["ccc"], 3), ccc_lo=round(lo, 3), ccc_hi=round(hi, 3),
                cb=round(c["cb"], 3), band=mcbride(c["ccc"]),
            ))
    return pd.DataFrame(rows)


def pooled_rank_correlation(summ):
    """The pooled standardized rank correlation across the three abundant species.

    Each species x river series is z-scored so all six panels weigh equally regardless of
    abundance, then one Spearman correlation is taken across every species-season. Reported
    here under the single year-inclusion rule (recovered seasons excluded), which gives
    33 species-years (3 species x 11 river-seasons).
    """
    zm, zt, panels = [], [], []
    for sp in MAIN:
        for rv in RIVERS:
            s = summ[(summ.species == sp) & (summ.river == rv)].sort_values("year")
            if len(s) < 3:
                continue
            m = s.model_net.to_numpy(float)
            t = s.tech_net.to_numpy(float)
            zm += list((m - m.mean()) / m.std(ddof=0))
            zt += list((t - t.mean()) / t.std(ddof=0))
            panels.append(f"{rv} {sp} {int(s.year.min())}-{int(s.year.max())} (n={len(s)})")
    rho, p = stats.spearmanr(zm, zt)
    return dict(rho=float(rho), p=float(p), n_species_years=len(zm), panels=panels)


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(config.OUTPUT_DIR / "tier1"),
                    help="Where to write the four agreement_*.csv tables (never data/ by default).")
    ap.add_argument("--block", type=int, default=BLOCK, help="Moving-block length in days.")
    ap.add_argument("--boot", type=int, default=BOOT, help="Bootstrap resamples.")
    ap.add_argument("--species", default="Chinook", help="Species for the Bland-Altman table.")
    ap.add_argument("--daily", default=None,
                    help="Override canonical_daily.csv (e.g. a rebuilt table in output/).")
    ap.add_argument("--summary", default=None, help="Override canonical_summary.csv.")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    daily, summ = load(daily_path=a.daily, summary_path=a.summary)

    print("=" * 78)
    print(f"1. BLAND-ALTMAN, daily net counts, {a.species}")
    print("=" * 78)
    ba = bland_altman_table(daily, a.species)
    print(ba.to_string(index=False))
    ba.to_csv(out / "agreement_bland_altman.csv", index=False, lineterminator="\n")
    print("\n  Read: 'ratio_bias 0.97 [0.62, 1.52]' = on a day with fish moving, the model")
    print("  records 0.97x the technician count on average, and between 0.62x and 1.52x on")
    print("  95% of days. 'spread_slope' > 0 with a small spread_p is the heteroscedasticity")
    print("  that invalidates the constant +-1.96 SD limits.")

    print()
    print("  1b. Same evidence without any model: difference stratified by the size of the day.")
    print("      If the constant limits were valid, sd_diff would be flat down each river block.")
    bb = ba_by_count_band(daily, a.species)
    print(bb.to_string(index=False))
    bb.to_csv(out / "agreement_ba_by_count_band.csv", index=False, lineterminator="\n")

    print()
    print("=" * 78)
    print(f"2. DAILY CCC, moving-block (len {a.block}) vs i.i.d. bootstrap, B = {a.boot}")
    print("   plus Spearman rho beside Pearson r")
    print("=" * 78)
    cc = ccc_block_table(daily, a.block, a.boot)
    print(cc.to_string(index=False))
    cc.to_csv(out / "agreement_ccc_block.csv", index=False, lineterminator="\n")
    print("\n  iid    = pooled single days (what the manuscript reports)")
    print("  strat  = single days resampled within season (stratification only)")
    print("  block  = moving blocks within season (stratification + autocorrelation)")
    print("  autocorr_widening isolates the autocorrelation effect (block / strat); > 1 means")
    print("  treating consecutive days as independent was genuinely too optimistic.")
    print("  vs_published is the practical change. clears_090_* re-tests the Section 3.5 claim.")

    print()
    print("=" * 78)
    print("3. INTER-ANNUAL: CCC beside r-squared")
    print("=" * 78)
    ia = interannual_table(summ, a.boot)
    print(ia.to_string(index=False))
    ia.to_csv(out / "agreement_interannual.csv", index=False, lineterminator="\n")

    print()
    print("=" * 78)
    print("4. POOLED RANK CORRELATION under the single year-inclusion rule")
    print("=" * 78)
    pr = pooled_rank_correlation(summ)
    print(f"  Spearman rho = {pr['rho']:.2f}, p = {pr['p']:.2e}, "
          f"n = {pr['n_species_years']} species-years")
    for p in pr["panels"]:
        print(f"    {p}")

    print()
    for f in ["agreement_bland_altman.csv", "agreement_ba_by_count_band.csv",
              "agreement_ccc_block.csv", "agreement_interannual.csv"]:
        print("wrote:", out / f)


if __name__ == "__main__":
    main()

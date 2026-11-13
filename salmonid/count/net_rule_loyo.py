"""In plain terms: a fish is counted as moving upstream if its track shifts at least 50 pixels
in the upstream direction, and as moving downstream only if it shifts at least 300 pixels the
other way. That 50/300 rule was chosen by trying eight options and keeping the one that best
matched the technicians, using the same seasons the paper then reports as validation. That is
circular: of course the chosen rule fits those seasons. This file re-does the choice honestly.
It hides one season, picks the best rule using only the others, and then scores that rule on
the hidden season, so the reported error is one the rule was never allowed to see.

Leave-one-year-out validation of the directional (net upstream) displacement rule.

Pipeline stage: **count** (data -> detect -> track -> *count* -> results -> figures).
Runs entirely off the committed sweep CSVs in ``data/net_rule_sweep/`` -- no GPU, no model,
no re-inference. Base Python environment (numpy only).

Why this exists
---------------
Section 2.5 states plainly that the 50/300 px thresholds "were optimized over the same years
used as multi-year validation evidence, so this margin is a fitted rather than an
out-of-sample result". That honesty is right but it costs the paper the claim. Leave-one-year-
out recovers a claim that is not fitted, at no compute cost: the sweep outputs for every
candidate rule and every season are already on disk.

What it computes
----------------
* **In-sample error** -- each candidate rule scored on all four seasons at once, the best
  one being 50/300 px at 3.56% mean absolute error. Reproduced here exactly so the
  out-of-sample figure is comparable to it.
* **Leave-one-year-out error** -- for each season in turn, the best rule is chosen using only
  the *other* seasons, then scored on the held-out season. The mean of those held-out errors
  is an out-of-sample estimate. Also reported: which rule each fold selected, since a rule
  that is selected in every fold is itself evidence that the choice is not an artifact of one
  season.
* **Rule stability** -- how often the published 50/300 rule wins its fold.

The error metric reproduces the published one exactly: model and technician net counts are
summed across **all five species and both rivers** into one annual total per season, and the
absolute percentage difference is averaged across seasons. That pooling is generous (a
Ganaraska over-read can cancel a Credit under-read), so a per-cell variant, which averages the
absolute error of each species x river x season cell separately, is reported alongside it.

Inputs : config.NET_RULE_SWEEP (48 CSVs committed in data/net_rule_sweep/:
         <year>_<rule>.csv with date, site, species, model, tech).
Outputs: net_rule_loyo.csv, written to --out-dir (default output/tier1/). The committed copy
         the manuscript quotes is data/net_rule_loyo.csv; run_tier1.py compares the two.
Feeds  : manuscript Section 3.4 (the 50/300 px rule is chosen in all four folds; mean absolute
         error 3.6% out of sample, equal to in sample; alternatives 10% to 26%).
Tier   : 1 (runs from committed data; no random component, so it reproduces exactly).

How to read the result honestly: only eight candidate rules were compared and 50/300 sits at
the corner of that grid, and because the same rule wins every fold the out-of-sample error
equals the in-sample error by construction. The result shows the choice does not depend on
any one season; it does not show that 50/300 is optimal among all possible rules.

Run:
  python -m salmonid.count.net_rule_loyo
  python -m salmonid.count.net_rule_loyo --years 2020 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

from salmonid import config

# Candidate rules -> the sweep-CSV suffix that holds that rule's per-day counts.
# "up50/dn300" is the rule the manuscript adopts; the down threshold is 300 px unless the
# suffix names another (the _u<up>_d<down> files vary both).
RULES = {
    "up50/dn300": "up50",
    "up150/dn300": "up150",
    "up250/dn300": "up250",
    "up300/dn300": "up300",
    "up50/dn150": "u50_d150",
    "up50/dn50": "u50_d50",
    "up150/dn150": "u150_d150",
    "up1/dn1": "u1_d1",
}
PUBLISHED = "up50/dn300"
DEFAULT_YEARS = [2022, 2023, 2024, 2025]


ABUNDANT = ["Chinook", "Coho", "Rainbow Trout"]


def load_cells(year, rule, sweep_dir, species=None):
    """{(site, species): [model_total, tech_total]} for one season under one rule.

    ``species`` restricts the cells; the published metric uses all five.
    """
    path = Path(sweep_dir) / f"{year}_{RULES[rule]}.csv"
    if not path.is_file():
        return None
    tot = collections.defaultdict(lambda: [0, 0])
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if species and r["species"] not in species:
                continue
            k = (r["site"], r["species"])
            tot[k][0] += int(r["model"])
            tot[k][1] += int(r["tech"])
    return tot


def season_error(cells, per_cell=False):
    """Absolute percentage error for one season.

    Default (``per_cell=False``) reproduces the published metric: everything pooled into one
    annual total. ``per_cell=True`` averages the absolute error of each species x river cell,
    which does not let over- and under-reads cancel.
    """
    if cells is None:
        return np.nan
    if per_cell:
        errs = [abs(m - t) / t * 100 for (m, t) in cells.values() if t > 0]
        return float(np.mean(errs)) if errs else np.nan
    m = sum(v[0] for v in cells.values())
    t = sum(v[1] for v in cells.values())
    return abs(m - t) / t * 100 if t > 0 else np.nan


def rule_errors(years, sweep_dir, per_cell=False, species=None):
    """{rule: {year: error}} for every candidate rule and season."""
    out = {}
    for rule in RULES:
        out[rule] = {y: season_error(load_cells(y, rule, sweep_dir, species), per_cell)
                     for y in years}
    return out


def in_sample(errs, years):
    """Best rule and its error when every season is used to choose AND to score."""
    means = {r: float(np.nanmean([errs[r][y] for y in years])) for r in errs}
    best = min(means, key=lambda r: means[r])
    return best, means


def loyo(errs, years):
    """Leave-one-year-out: choose on the other seasons, score on the held-out one."""
    folds = []
    for held in years:
        train = [y for y in years if y != held]
        means = {r: float(np.nanmean([errs[r][y] for y in train])) for r in errs}
        chosen = min(means, key=lambda r: means[r])
        folds.append(dict(held_out=held, chosen_rule=chosen,
                          train_err=means[chosen],
                          heldout_err=errs[chosen][held],
                          published_heldout_err=errs[PUBLISHED][held]))
    return folds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-dir", default=str(config.NET_RULE_SWEEP))
    ap.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--out-dir", default=str(config.OUTPUT_DIR / "tier1"),
                    help="Where to write net_rule_loyo.csv (never data/ by default).")
    a = ap.parse_args()
    years = sorted(a.years)
    all_rows = []

    # Three views: the published pooled metric, the same metric restricted to the three
    # abundant species (the ones the paper reports as validated counts), and the per-cell
    # average over all five. The per-cell/all-species view is dominated by Atlantic Salmon
    # and Credit Brown Trout, where a handful of fish sets the percentage, so it is a
    # sensitivity check rather than a competing headline.
    views = [
        (False, None, "POOLED, all five species (published metric)"),
        (False, ABUNDANT, "POOLED, three abundant species only"),
        (True, ABUNDANT, "PER-CELL (species x river averaged), abundant species only"),
        (True, None, "PER-CELL, all five species (rare-species dominated; sensitivity only)"),
    ]
    for per_cell, species, tag in views:
        errs = rule_errors(years, a.sweep_dir, per_cell, species)
        best, means = in_sample(errs, years)
        folds = loyo(errs, years)

        print("=" * 84)
        print(f"NET-RULE SWEEP, {tag} -- seasons {years[0]}-{years[-1]}")
        print("=" * 84)
        print(f"{'rule':14s}" + "".join(f"{y:>9d}" for y in years) + f"{'mean':>9s}")
        for r in sorted(means, key=lambda r: means[r]):
            print(f"{r:14s}" + "".join(f"{errs[r][y]:9.2f}" for y in years)
                  + f"{means[r]:9.2f}" + ("   <- in-sample best" if r == best else ""))

        print(f"\nIn-sample (the published construction): best rule {best}, "
              f"mean absolute error {means[best]:.2f}%")
        print("\nLeave-one-year-out (rule chosen WITHOUT the season it is scored on):")
        print(f"{'held-out':>10s}{'chosen on others':>20s}{'train err':>11s}"
              f"{'held-out err':>14s}{'published-rule err':>20s}")
        for f in folds:
            print(f"{f['held_out']:>10d}{f['chosen_rule']:>20s}{f['train_err']:>10.2f}%"
                  f"{f['heldout_err']:>13.2f}%{f['published_heldout_err']:>19.2f}%")
        oos = float(np.nanmean([f["heldout_err"] for f in folds]))
        agree = sum(f["chosen_rule"] == PUBLISHED for f in folds)
        print(f"\n  OUT-OF-SAMPLE mean absolute error: {oos:.2f}%  "
              f"(in-sample {means[best]:.2f}%)")
        print(f"  The published {PUBLISHED} rule was selected in {agree} of {len(folds)} folds.")
        print()

        view = ("per_cell" if per_cell else "pooled") + ("_abundant" if species else "_all5")
        all_rows.append((view, folds, oos, best, means[best]))
        print()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dest = out / "net_rule_loyo.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")   # LF on every platform (see classification_ci)
        w.writerow(["view", "held_out_year", "chosen_rule", "train_err_pct",
                    "heldout_err_pct", "published_rule_heldout_err_pct"])
        for view, folds, oos, best, best_err in all_rows:
            for f in folds:
                w.writerow([view, f["held_out"], f["chosen_rule"],
                            round(f["train_err"], 4), round(f["heldout_err"], 4),
                            round(f["published_heldout_err"], 4)])
            w.writerow([view, "MEAN_OUT_OF_SAMPLE", "", "", round(oos, 4), ""])
            w.writerow([view, "MEAN_IN_SAMPLE", best, "", round(best_err, 4), ""])
    print("wrote:", dest)


if __name__ == "__main__":
    main()

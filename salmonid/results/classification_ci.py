"""In plain terms: the paper reports the model's species accuracy to three decimal places on
as few as 24 example fish. Three decimals on 24 fish claims a precision the sample cannot
support. This file attaches an honest uncertainty range to each of those numbers, so a
reader can see that "0.917" on 24 fish really means "somewhere between about 0.73 and 0.98".

Confidence intervals for the held-out species-classification metrics.

Pipeline stage: **results** (data -> detect -> track -> count -> *results* -> figures).
A leaf reporting step over the frozen test set and one model's track predictions. Loads no
model; runs in the **base Python environment** (numpy/scipy only).

What it computes
----------------
* **Wilson score intervals** for per-species precision and recall. Precision and recall are
  binomial proportions (k successes out of n trials), and at the sample sizes here the
  normal-approximation interval is badly behaved: it can run past 1.0, and it collapses to
  zero width at a perfect score. Wilson does neither, which is why it is the standard choice
  for small-sample proportions (Brown, Cai & DasGupta 2001).
* **Bootstrap percentile intervals for F1 and macro-F1**, resampled over *tracks*. F1 is a
  ratio of two dependent proportions, so it has no closed-form interval; resampling the
  scored tracks propagates the uncertainty in both. Macro-F1 is resampled the same way, so
  the reported interval accounts for the fact that all five per-species scores move together.

The denominators matter and are reported explicitly. For a species, recall's denominator is
the number of true tracks of that species (n_true) and precision's is the number of tracks
the model *called* that species (n_pred). These differ, which is why one interval can be
much wider than the other for the same species.

Inputs : a predictions CSV (video_id, track_id, pred_species) and the frozen GT JSON.
         Defaults to config.PREDS_V26 + config.TESTSET_V6_GT.
Outputs: classification_ci.csv (--out-dir, default config.REPO_DATA).

Usage:
  python -m salmonid.results.classification_ci
  python -m salmonid.results.classification_ci --preds G:/Projects/deimv2_x_v27_eval/preds_0017.csv \
      --label v2.7-ckpt0017
  python -m salmonid.results.classification_ci --binomial 178 188 --label blind-review-overall
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
BOOT = 10000
SEED = 12345


# ---------------------------------------------------------------------------------------
def wilson(k, n, conf=0.95):
    """Wilson score interval for k successes in n trials.

    Chosen over the normal approximation because these denominators are small (n = 6 for the
    blind-review Atlantic Salmon, n = 24 for held-out Brown Trout under the v2.6 leak-clean
    set). The normal interval would put the upper bound above 1.0 for the near-perfect
    species and would give a zero-width interval wherever the score is exactly 1.0.
    """
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _counts(true, pred, sp):
    """(tp, fp, fn) for one species from aligned true/pred label arrays."""
    t = np.asarray(true) == sp
    p = np.asarray(pred) == sp
    return int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and np.isfinite(prec + rec) else float("nan")
    return prec, rec, f1


def score_with_ci(true, pred, species=SPECIES, B=BOOT, seed=SEED, conf=0.95):
    """Per-species precision/recall/F1 with intervals, plus macro-F1 and accuracy.

    The bootstrap resamples the scored *tracks* (the unit the test set is frozen at), so a
    resample changes both which fish are present and how many of each species there are.
    That is the right resampling unit: the 366 tracks are the sample, the species counts are
    a property of it.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = len(true)
    rng = np.random.default_rng(seed)

    # Point estimates.
    rows = []
    f1s = []
    for sp in species:
        tp, fp, fn = _counts(true, pred, sp)
        prec, rec, f1 = _prf(tp, fp, fn)
        f1s.append(f1)
        rows.append(dict(species=sp, n_true=tp + fn, n_pred=tp + fp, tp=tp, fp=fp, fn=fn,
                         precision=prec, recall=rec, f1=f1))
    macro_f1 = float(np.nanmean(f1s))
    acc = float((true == pred).mean())

    # Bootstrap over tracks for the F1s, the macro-F1 and accuracy.
    boot_f1 = {sp: [] for sp in species}
    boot_macro, boot_acc = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        bt, bp = true[idx], pred[idx]
        vals = []
        for sp in species:
            tp, fp, fn = _counts(bt, bp, sp)
            _, _, f1 = _prf(tp, fp, fn)
            boot_f1[sp].append(f1)
            vals.append(f1)
        boot_macro.append(np.nanmean(vals))
        boot_acc.append((bt == bp).mean())

    a = (1 - conf) / 2 * 100
    for r in rows:
        sp = r["species"]
        r["prec_lo"], r["prec_hi"] = wilson(r["tp"], r["n_pred"], conf)
        r["rec_lo"], r["rec_hi"] = wilson(r["tp"], r["n_true"], conf)
        lo, hi = np.nanpercentile(boot_f1[sp], [a, 100 - a])
        r["f1_lo"], r["f1_hi"] = float(lo), float(hi)

    m_lo, m_hi = np.nanpercentile(boot_macro, [a, 100 - a])
    ac_lo, ac_hi = np.nanpercentile(boot_acc, [a, 100 - a])
    summary = dict(n_tracks=n, accuracy=acc, acc_lo=float(ac_lo), acc_hi=float(ac_hi),
                   macro_f1=macro_f1, macro_lo=float(m_lo), macro_hi=float(m_hi))
    return rows, summary


# ---------------------------------------------------------------------------------------
def load_pairs(preds_path, gt_path, tier=None):
    """Align predictions to the frozen GT on (video_id, track_id). Returns (true, pred, n_missing).

    ``n_missing`` is the number of GT tracks the prediction file does not cover, which is how
    a scored n of 358 or 365 arises against a frozen 366-track set; it is reported rather
    than silently dropped.
    """
    gt = {}
    for t in json.load(open(gt_path, encoding="utf-8"))["tracks"]:
        if tier and t.get("tier") != tier:
            continue
        gt[(t["video_id"], int(t["track_id"]))] = t["species"]
    pred = {}
    with open(preds_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pred[(r["video_id"], int(r["track_id"]))] = r["pred_species"]
    true_l, pred_l = [], []
    missing = 0
    for k, sp in gt.items():
        if k in pred:
            true_l.append(sp)
            pred_l.append(pred[k])
        else:
            missing += 1
    return true_l, pred_l, missing


def print_table(rows, summary, label, missing=0):
    print(f"\n{'=' * 92}")
    print(f"{label}: held-out species classification with 95% intervals "
          f"(n = {summary['n_tracks']} tracks scored"
          + (f", {missing} GT tracks unmatched)" if missing else ")"))
    print("=" * 92)
    print(f"{'species':16s}{'n_true':>7s}{'n_pred':>7s}"
          f"{'precision [95% CI]':>26s}{'recall [95% CI]':>25s}{'F1 [95% CI]':>25s}")
    for r in rows:
        print(f"{r['species']:16s}{r['n_true']:7d}{r['n_pred']:7d}"
              f"{r['precision']:9.3f} [{r['prec_lo']:.3f}, {r['prec_hi']:.3f}]"
              f"{r['recall']:8.3f} [{r['rec_lo']:.3f}, {r['rec_hi']:.3f}]"
              f"{r['f1']:8.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]")
    print(f"{'MACRO':16s}{summary['n_tracks']:7d}{'':>7s}{'':>26s}{'':>25s}"
          f"{summary['macro_f1']:8.3f} [{summary['macro_lo']:.3f}, {summary['macro_hi']:.3f}]")
    print(f"{'accuracy':16s}{'':>7s}{'':>7s}{'':>26s}{'':>25s}"
          f"{summary['accuracy']:8.3f} [{summary['acc_lo']:.3f}, {summary['acc_hi']:.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preds", default=str(config.PREDS_V26))
    ap.add_argument("--gt", default=str(config.TESTSET_V6_GT))
    ap.add_argument("--tier", default=None, choices=[None, "pristine", "heldout"],
                    help="Restrict to one test-set tier.")
    ap.add_argument("--label", default="v2.6-ckpt0007")
    ap.add_argument("--out-dir", default=str(config.REPO_DATA))
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--binomial", nargs=2, type=int, metavar=("K", "N"),
                    help="Skip the test set; just print a Wilson interval for K of N. "
                         "Used for the blind-review per-species agreement in Section 3.2.")
    a = ap.parse_args()

    if a.binomial:
        k, n = a.binomial
        lo, hi = wilson(k, n)
        print(f"{a.label}: {k}/{n} = {k / n:.3f}, Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
        return

    true, pred, missing = load_pairs(a.preds, a.gt, a.tier)
    rows, summary = score_with_ci(true, pred, B=a.boot)
    print_table(rows, summary, a.label, missing)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    dest = out / "classification_ci.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "species", "n_true", "n_pred", "tp", "fp", "fn",
                                           "precision", "prec_lo", "prec_hi",
                                           "recall", "rec_lo", "rec_hi",
                                           "f1", "f1_lo", "f1_hi"])
        w.writeheader()
        for r in rows:
            w.writerow({"model": a.label, **{k: (round(v, 4) if isinstance(v, float) else v)
                                             for k, v in r.items()}})
        w.writerow({"model": a.label, "species": "MACRO", "n_true": summary["n_tracks"],
                    "f1": round(summary["macro_f1"], 4),
                    "f1_lo": round(summary["macro_lo"], 4),
                    "f1_hi": round(summary["macro_hi"], 4)})
    print("\nwrote:", dest)


if __name__ == "__main__":
    main()

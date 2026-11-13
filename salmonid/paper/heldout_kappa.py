"""In plain terms: on the 366 held-out fish, the model named the right species 352 times
(96.2%). This script asks how much of that agreement could be luck. If two observers each
picked species at random in the proportions they actually used, they would still agree on
some fish; Cohen's kappa removes that share and reports the agreement beyond it. Because the
held-out set is deliberately balanced across species, luck would produce agreement on only
about 22% of fish, so kappa (0.951) stays close to plain accuracy. The script also gives a
confidence interval by re-drawing the 366 fish many times.

Inputs
------
Committed in ``data/`` (Tier 1):

* ``test_set_v6_gt.json``: the species the reviewers gave each held-out track.
* ``preds_v27_ckpt0017.csv``: the species the published model gave each track.

Outputs
-------
Written to ``--out-dir`` (default ``output/paper/``):

* ``heldout_kappa.csv``: one row with accuracy, chance agreement, kappa, its bootstrap and
  large-sample intervals, and the related coefficients listed below.
* ``heldout_kappa_per_species.csv``: per-species precision, recall, F1 and a one-species-
  against-the-rest kappa.

Feeds
-----
Section 3.1: "Cohen's kappa was 0.951 (95% CI 0.926 to 0.975) ... a chance agreement of only
0.22", and the overall accuracy of 0.962 (352 of 366). Methods 2.4 describes the bootstrap
(2,000 resamples of tracks).

Run
---
::

    python -m salmonid.paper.heldout_kappa

Tier
----
1.

Provenance
----------
The statistics functions and ``analyse`` are copied unchanged from ``compute_kappa.py``
(manuscript repository, 2026-09-29 code review). That script also scored the per-video and
blind-review comparisons; those were later recomputed on the final denominators and now
live in ``pervideo_agreement`` and ``blind_review``, so only the held-out comparison is run
here. ``analyse`` reports several coefficients beside Cohen's kappa (Krippendorff's alpha,
Scott's pi, Gwet's AC1, the prevalence-adjusted kappa); the manuscript quotes kappa only, and
the others are kept because they were part of the original output and show that the
conclusion does not depend on the choice of coefficient.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
ABBR = {"Chinook": "Ch", "Coho": "Co", "Atlantic": "At", "Rainbow Trout": "RT", "Brown Trout": "BT",
        "unsure": "Un"}
B = 2000        # bootstrap resamples (Methods 2.4)
SEED = 0        # fixed so the interval is the same on every run


# ---------------------------------------------------------------------------------------------
# Core statistics on a K x K confusion matrix (rows = reference, cols = model)
# ---------------------------------------------------------------------------------------------
# kappa_from_cm returns (kappa, po, pe): po is the observed share of fish on which the two
# agree; pe is the share expected by chance, i.e. for each species the product of how often
# the reference used it and how often the model used it, summed over species;
# kappa = (po - pe) / (1 - pe).
def kappa_from_cm(cm):
    cm = np.asarray(cm, float)
    n = cm.sum()
    p = cm / n
    po = np.trace(p)
    r, c = p.sum(1), p.sum(0)
    pe = float(r @ c)
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return k, po, pe


def kappa_se_fce(cm):
    """Large-sample (non-null) SE of Cohen's kappa, Fleiss, Cohen and Everitt (1969)."""
    cm = np.asarray(cm, float)
    n = cm.sum()
    p = cm / n
    r, c = p.sum(1), p.sum(0)
    k, po, pe = kappa_from_cm(cm)
    K = len(r)
    a = sum(p[i, i] * (1 - (r[i] + c[i]) * (1 - k)) ** 2 for i in range(K))
    b = (1 - k) ** 2 * sum(p[i, j] * (c[i] + r[j]) ** 2 for i in range(K) for j in range(K) if i != j)
    cc = (k - pe * (1 - k)) ** 2
    var = (a + b - cc) / (n * (1 - pe) ** 2)
    return float(np.sqrt(max(var, 0.0)))


def kappa_se_simple(cm):
    """Cohen (1960) first-order approximation, for comparison only."""
    k, po, pe = kappa_from_cm(cm)
    n = np.asarray(cm).sum()
    return float(np.sqrt(po * (1 - po) / (n * (1 - pe) ** 2)))


def kappa_max(cm):
    """Largest kappa attainable with these two sets of marginals (Cohen 1960)."""
    cm = np.asarray(cm, float)
    p = cm / cm.sum()
    r, c = p.sum(1), p.sum(0)
    pe = float(r @ c)
    pom = float(np.minimum(r, c).sum())
    return (pom - pe) / (1 - pe)


def pabak(po, k_cat):
    """Prevalence-adjusted bias-adjusted kappa, multi-category form (k*po - 1)/(k - 1).
    Equal to Bennett, Alpert and Goldstein's S / Brennan-Prediger; a linear rescaling of po."""
    return (k_cat * po - 1) / (k_cat - 1)


def krippendorff_alpha_nominal(cm):
    """Krippendorff's alpha, nominal metric, two coders, no missing values, via the coincidence matrix."""
    cm = np.asarray(cm, float)
    o = cm + cm.T                     # each unit contributes (a,b) and (b,a)
    n_c = o.sum(1)
    n_tot = o.sum()                   # = 2N
    d_o = (n_tot - np.trace(o)) / n_tot
    d_e = (n_tot ** 2 - (n_c ** 2).sum()) / (n_tot * (n_tot - 1))
    return 1 - d_o / d_e


def scott_pi(cm):
    cm = np.asarray(cm, float)
    p = cm / cm.sum()
    m = (p.sum(1) + p.sum(0)) / 2
    pe = float((m ** 2).sum())
    return (np.trace(p) - pe) / (1 - pe)


def gwet_ac1(cm):
    cm = np.asarray(cm, float)
    p = cm / cm.sum()
    q = cm.shape[0]
    m = (p.sum(1) + p.sum(0)) / 2
    pe = float((m * (1 - m)).sum() / (q - 1))
    return (np.trace(p) - pe) / (1 - pe)


def prf(cm, labels):
    """Per-class precision/recall/F1 with the paper's convention (0 when a denominator is empty)."""
    cm = np.asarray(cm, float)
    rows = []
    for i, sp in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append(dict(species=sp, n_ref=int(tp + fn), n_model=int(tp + fp), tp=int(tp),
                         precision=prec, recall=rec, f1=f1))
    return rows


def one_vs_rest_kappa(cm, i):
    cm = np.asarray(cm, float)
    a = cm[i, i]
    b = cm[i].sum() - a
    c = cm[:, i].sum() - a
    d = cm.sum() - a - b - c
    return kappa_from_cm([[a, b], [c, d]])[0]


def landis_koch(k):
    if not np.isfinite(k):
        return ""
    if k < 0:
        return "poor"
    if k <= 0.20:
        return "slight"
    if k <= 0.40:
        return "fair"
    if k <= 0.60:
        return "moderate"
    if k <= 0.80:
        return "substantial"
    return "almost perfect"


# ---------------------------------------------------------------------------------------------
# Bootstrap (unit-level and cluster-level), 2,000 resamples, seed 0
# ---------------------------------------------------------------------------------------------
# The interval quoted in the manuscript resamples the 366 tracks with replacement and takes
# the middle 95% of the resulting kappas. The cluster version resamples whole videos (two of
# the 364 videos hold two test fish); it is reported alongside and is nearly identical.
def boot_cms(codes, K, clusters=None, B=B, seed=SEED):
    """Return a (B, K, K) array of resampled confusion matrices.

    codes: int array ref*K + pred per unit. If clusters is given, whole clusters are resampled.
    """
    rng = np.random.default_rng(seed)
    codes = np.asarray(codes)
    if clusters is None:
        n = len(codes)
        out = np.empty((B, K * K), dtype=np.int64)
        for b in range(B):
            idx = rng.integers(0, n, n)
            out[b] = np.bincount(codes[idx], minlength=K * K)
        return out.reshape(B, K, K)
    cl = pd.factorize(pd.Series(clusters))[0]
    G = cl.max() + 1
    per = np.zeros((G, K * K), dtype=np.int64)
    np.add.at(per, (cl, codes), 1)
    out = np.empty((B, K * K), dtype=np.int64)
    for b in range(B):
        idx = rng.integers(0, G, G)
        out[b] = per[idx].sum(0)
    return out.reshape(B, K, K)


def block_boot_cms(codes, K, days, L=10, B=B, seed=SEED):
    """Moving-block bootstrap over consecutive observed days (block length L), mirroring the
    moving-block resample the paper uses for the daily CCC (salmonid/results/agreement_stats.py)."""
    rng = np.random.default_rng(seed)
    days = pd.Series(days).astype(str).to_numpy()
    uniq = np.array(sorted(set(days)))
    pos = {d: i for i, d in enumerate(uniq)}
    di = np.array([pos[d] for d in days])
    D = len(uniq)
    per = np.zeros((D, K * K), dtype=np.int64)
    np.add.at(per, (di, np.asarray(codes)), 1)
    nb = int(np.ceil(D / L))
    out = np.empty((B, K * K), dtype=np.int64)
    for b in range(B):
        starts = rng.integers(0, D - L + 1, nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:D]
        out[b] = per[idx].sum(0)
    return out.reshape(B, K, K)


def pct_ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------
# One comparison -> summary row + per-class rows + confusion matrix
# ---------------------------------------------------------------------------------------------
def analyse(name, description, ref, pred, labels, source, clusters=None, unit="track", note="",
            cluster_kind="", block_days=None):
    ref = list(ref)
    pred = list(pred)
    assert len(ref) == len(pred)
    K = len(labels)
    li = {s: i for i, s in enumerate(labels)}
    bad = [(r, p) for r, p in zip(ref, pred) if r not in li or p not in li]
    assert not bad, (name, bad[:5])
    codes = np.array([li[r] * K + li[p] for r, p in zip(ref, pred)])
    cm = np.bincount(codes, minlength=K * K).reshape(K, K)
    n = int(cm.sum())

    k, po, pe = kappa_from_cm(cm)
    se = kappa_se_fce(cm)
    se1 = kappa_se_simple(cm)
    alpha = krippendorff_alpha_nominal(cm)
    pi = scott_pi(cm)
    # identity check: alpha = 1 - (1 - pi)(2N - 1)/(2N)
    assert abs(alpha - (1 - (1 - pi) * (2 * n - 1) / (2 * n))) < 1e-10
    ac1 = gwet_ac1(cm)
    kmax = kappa_max(cm)
    pab = pabak(po, K)
    pr = prf(cm, labels)
    sp5 = [r for r in pr if r["species"] in SPECIES]
    macro5 = float(np.mean([r["f1"] for r in sp5]))
    supported = [r for r in sp5 if r["n_ref"] >= 5]
    macro_sup = float(np.mean([r["f1"] for r in supported]))

    # bootstrap: unit-level (the specified CI) and, where units nest in clips/videos, cluster-level
    bcm = boot_cms(codes, K)
    bk = [kappa_from_cm(m)[0] for m in bcm]
    bacc = [np.trace(m) / m.sum() for m in bcm]
    bmacro = [np.mean([r["f1"] for r in prf(m, labels) if r["species"] in SPECIES]) for m in bcm]
    k_lo, k_hi = pct_ci(bk)
    a_lo, a_hi = pct_ci(bacc)
    f_lo, f_hi = pct_ci(bmacro)
    kc_lo = kc_hi = np.nan
    kb_lo = kb_hi = np.nan
    n_cl = np.nan
    if clusters is not None:
        n_cl = int(pd.Series(clusters).nunique())
        if n_cl < n:
            ccm = boot_cms(codes, K, clusters=clusters)
            kc_lo, kc_hi = pct_ci([kappa_from_cm(m)[0] for m in ccm])
    if block_days is not None and pd.Series(block_days).nunique() > 10:
        bbm = block_boot_cms(codes, K, block_days)
        kb_lo, kb_hi = pct_ci([kappa_from_cm(m)[0] for m in bbm])

    r_marg = cm.sum(1) / n
    c_marg = cm.sum(0) / n
    prev_ref = ", ".join(f"{ABBR[s]} {r_marg[i]:.3f}" for i, s in enumerate(labels))
    prev_mod = ", ".join(f"{ABBR[s]} {c_marg[i]:.3f}" for i, s in enumerate(labels))
    counts_ref = ", ".join(f"{ABBR[s]} {int(cm.sum(1)[i])}" for i, s in enumerate(labels))

    row = dict(
        comparison=name, description=description, unit=unit, n=n,
        n_clusters=n_cl, n_categories=K,
        n_agree=int(np.trace(cm)), accuracy=po, acc_boot_lo=a_lo, acc_boot_hi=a_hi,
        p_e=pe, kappa=k, kappa_se_asym=se, kappa_se_cohen1960=se1,
        kappa_asym_lo=k - 1.96 * se, kappa_asym_hi=min(1.0, k + 1.96 * se),
        kappa_boot_lo=k_lo, kappa_boot_hi=k_hi,
        cluster_kind=cluster_kind if clusters is not None else "",
        kappa_cluster_boot_lo=kc_lo, kappa_cluster_boot_hi=kc_hi,
        kappa_block10day_lo=kb_lo, kappa_block10day_hi=kb_hi,
        kappa_max=kmax, landis_koch=landis_koch(k),
        pabak=pab, kripp_alpha_nominal=alpha, scott_pi=pi, gwet_ac1=ac1,
        macro_f1_5class=macro5, macro_f1_boot_lo=f_lo, macro_f1_boot_hi=f_hi,
        macro_f1_classes_n_ge_5=macro_sup, n_classes_n_ge_5=len(supported),
        prevalence_reference=prev_ref, prevalence_model=prev_mod, counts_reference=counts_ref,
        max_class_prevalence=float(r_marg.max()),
        bootstrap=f"{B} percentile resamples of {unit}s, numpy default_rng seed {SEED}",
        source=source, note=note,
    )
    for i, s in enumerate(labels):
        if s in SPECIES:
            row[f"kappa_ovr_{ABBR[s]}"] = one_vs_rest_kappa(cm, i)

    # per-class long table with bootstrap CI on the one-vs-rest kappa
    per = []
    for i, r in enumerate(pr):
        s = r["species"]
        kv = one_vs_rest_kappa(cm, i)
        bvals = [one_vs_rest_kappa(m, i) for m in bcm]
        lo, hi = pct_ci(bvals) if r["n_ref"] > 0 else (np.nan, np.nan)
        per.append(dict(comparison=name, species=s, n_ref=r["n_ref"], n_model=r["n_model"], tp=r["tp"],
                        prevalence_ref=r["n_ref"] / n, precision=r["precision"], recall=r["recall"],
                        f1=r["f1"], kappa_ovr=kv, kappa_ovr_boot_lo=lo, kappa_ovr_boot_hi=hi))
    return row, per, cm


def load_heldout():
    """The 366 held-out tracks with the reviewers' species and the published model's species."""
    g = json.load(open(config.TESTSET_V6_GT, encoding="utf-8"))
    gt = pd.DataFrame(g["tracks"])
    gt = gt[gt.species.isin(SPECIES)]
    p = pd.read_csv(config.PUBLISHED_PREDS)
    m = gt.merge(p, on=["video_id", "track_id"], how="inner")
    assert len(m) == 366 and m.pred_species.isin(SPECIES).all()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    args = ap.parse_args()
    out = out_dir(args)

    h = load_heldout()
    row, per, cm = analyse(
        "heldout_v27", "Held-out test set v6: frozen GT vs published v2.7 (ckpt0017) track consensus label",
        h.species, h.pred_species, SPECIES,
        source="data/test_set_v6_gt.json + data/preds_v27_ckpt0017.csv",
        clusters=h.video_id, unit="track", cluster_kind="video",
        note="366 tracks in 364 videos; reproduces Table 1 / Figure 2 (352 of 366 correct)")
    pd.DataFrame([row]).to_csv(out / "heldout_kappa.csv", index=False, float_format="%.6f")
    pd.DataFrame(per).to_csv(out / "heldout_kappa_per_species.csv", index=False, float_format="%.6f")

    print(f"held-out tracks: n = {row['n']}, agree = {row['n_agree']} (accuracy {row['accuracy']:.3f})")
    print(f"chance agreement p_e = {row['p_e']:.3f}")
    print(f"Cohen's kappa = {row['kappa']:.3f}  track bootstrap 95% CI [{row['kappa_boot_lo']:.3f}, {row['kappa_boot_hi']:.3f}]"
          f"  video-cluster bootstrap [{row['kappa_cluster_boot_lo']:.3f}, {row['kappa_cluster_boot_hi']:.3f}]"
          f"  large-sample [{row['kappa_asym_lo']:.3f}, {row['kappa_asym_hi']:.3f}]")
    print(f"other coefficients: PABAK {row['pabak']:.3f}, Krippendorff alpha {row['kripp_alpha_nominal']:.3f}, "
          f"Gwet AC1 {row['gwet_ac1']:.3f}; macro-F1 {row['macro_f1_5class']:.3f} "
          f"[{row['macro_f1_boot_lo']:.3f}, {row['macro_f1_boot_hi']:.3f}]")
    print("confusion matrix (rows = reviewers, columns = model):")
    print(pd.DataFrame(cm, index=SPECIES, columns=SPECIES).to_string())
    print(f"\nwrote: {out / 'heldout_kappa.csv'}\nwrote: {out / 'heldout_kappa_per_species.csv'}")


if __name__ == "__main__":
    main()

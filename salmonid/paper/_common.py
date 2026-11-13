"""In plain terms: this file holds the small pieces of arithmetic that several of the
paper's agreement checks share. They answer one question in different settings: when two
observers (the model and a person) each name the species of the same fish, how often do they
agree, and how much of that agreement would have happened by luck alone? Keeping the
arithmetic in one place means the per-video comparison, the blind review and the held-out
test all use exactly the same definitions.

Inputs
------
None of its own. The functions take lists of species names (one per fish or per video).

Outputs
-------
None of its own. It is imported by ``pervideo_agreement``, ``blind_review`` and
``detection_outcomes``.

Feeds
-----
Indirectly: Cohen's kappa and its confidence intervals in Sections 3.1 and 3.2, and the
Wilson intervals on the blind-review percentages (Section 3.2).

Run
---
Not run directly. ``python -m salmonid.paper._common`` prints a two-line self-check.

Tier
----
1 (pure arithmetic; no files).

Provenance
----------
The statistics functions below are copied unchanged from the script that produced the
Section 3.2 numbers during the 2026-11-06 reconciliation of the manuscript
(``recompute_v27.py`` in the manuscript repository). Only this docstring and the comments
were added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from salmonid import config

# The five reported species, in the class order used throughout the repository.
SP = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]

# How the technicians' species words map onto the five species. "Steelhead" is the
# lake-run form of Rainbow Trout. "Unknown" and blank become "" and so never match a species.
TECHMAP = {"chinook": "Chinook", "coho": "Coho", "atlantic": "Atlantic", "rainbow": "Rainbow Trout",
           "steelhead": "Rainbow Trout", "brown trout": "Brown Trout", "brown": "Brown Trout",
           "non fish": "Non fish", "unknown": "", "": ""}

# Bootstrap settings stated in Methods 2.4: 2,000 resamples. The seed is fixed so a re-run
# gives the same interval to the last digit.
B, SEED = 2000, 0


# A video is identified by its last two path parts: the recording-session folder and the clip
# number (for example "25  09  15  14  42/244.mp4"). The technician export and the model
# output spell the leading folders differently, so only this tail is compared.
def keyof(rel):
    return "/".join(str(rel).replace("\\", "/").split("/")[-2:]).strip().lower()


# ---------------------------------------------------------------- statistics
# cm_from builds the species-by-species table ("confusion matrix"): rows are the reference
# (the person), columns are the model, each cell counts fish.
def cm_from(ref, pred, labels):
    idx = {s: i for i, s in enumerate(labels)}
    K = len(labels)
    cm = np.zeros((K, K), dtype=np.int64)
    for r, p in zip(ref, pred):
        cm[idx[r], idx[p]] += 1
    return cm


# Cohen's kappa. Plain percentage agreement (po) flatters a comparison when one species
# dominates: if 88% of videos are Chinook Salmon, two observers who both said "Chinook" most
# of the time would agree often by luck. pe is that luck: for each species, the chance both
# name it independently (row share x column share), summed over species. Kappa is the
# agreement achieved beyond luck, as a share of the agreement that was possible beyond luck:
# 1 = perfect, 0 = no better than luck. Returns (kappa, po, pe).
def kappa(cm):
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(1) * cm.sum(0)).sum() / n ** 2
    return (po - pe) / (1 - pe), po, pe


# Confidence interval for kappa on the per-video comparison. Videos from the same day are
# not independent (the same run of fish, the same water clarity, sometimes the same fish
# re-triggering the counter), so resampling single videos would give an interval that is too
# narrow. Instead whole blocks of L = 10 consecutive days are resampled, with every video
# recorded on those days carried along together.
def block_boot_kappa(ref, pred, days, labels, L=10):
    K = len(labels)
    idx = {s: i for i, s in enumerate(labels)}
    codes = np.array([idx[r] * K + idx[p] for r, p in zip(ref, pred)])
    days = pd.Series(days).astype(str).to_numpy()
    uniq = np.array(sorted(set(days)))
    pos = {d: i for i, d in enumerate(uniq)}
    di = np.array([pos[d] for d in days])
    D = len(uniq)
    per = np.zeros((D, K * K), dtype=np.int64)
    np.add.at(per, (di, codes), 1)
    rng = np.random.default_rng(SEED)
    nb = int(np.ceil(D / L))
    ks = []
    for _ in range(B):
        starts = rng.integers(0, D - L + 1, nb)
        sel = np.concatenate([np.arange(s, s + L) for s in starts])[:D]
        cm = per[sel].sum(0).reshape(K, K)
        ks.append(kappa(cm)[0])
    ks = np.array(ks)
    ks = ks[np.isfinite(ks)]
    return np.percentile(ks, [2.5, 97.5])


# Confidence interval for kappa on the blind review. Fish seen in the same clip are not
# independent, so whole clips (the "clusters") are resampled, not single fish.
def cluster_boot_kappa(ref, pred, clusters, labels):
    K = len(labels)
    idx = {s: i for i, s in enumerate(labels)}
    codes = np.array([idx[r] * K + idx[p] for r, p in zip(ref, pred)])
    cl = pd.factorize(pd.Series(clusters))[0]
    G = cl.max() + 1
    per = np.zeros((G, K * K), dtype=np.int64)
    np.add.at(per, (cl, codes), 1)
    rng = np.random.default_rng(SEED)
    ks, accs = [], []
    for _ in range(B):
        sel = rng.integers(0, G, G)
        cm = per[sel].sum(0).reshape(K, K)
        k, po, _ = kappa(cm)
        ks.append(k); accs.append(po)
    return np.percentile(ks, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])


# Wilson interval for a simple proportion (k successes out of n). Preferred to the textbook
# +/- 1.96 SE interval because it stays inside 0-100% and behaves sensibly near 100% and for
# small n, e.g. the six Atlantic Salmon of the blind review (6 of 6 -> 61% to 100%).
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------- output folder helper
def add_out_dir(ap: argparse.ArgumentParser) -> None:
    """Give a script the standard ``--out-dir`` option (default ``output/paper``)."""
    ap.add_argument("--out-dir", default=str(config.PAPER_OUTPUT),
                    help="Folder for regenerated tables (default: output/paper, git-ignored).")


def out_dir(args) -> Path:
    """Create and return the output folder chosen on the command line."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


if __name__ == "__main__":
    # Self-check on a textbook 2 x 2 example: 40 + 30 agreements out of 100.
    demo = np.array([[40, 10], [20, 30]])
    k, po, pe = kappa(demo)
    print(f"kappa self-check: po={po:.2f} pe={pe:.2f} kappa={k:.2f} (expected 0.70, 0.50, 0.40)")
    print("Wilson 6 of 6:", tuple(round(x, 2) for x in wilson(6, 6)), "(expected 0.61, 1.0)")

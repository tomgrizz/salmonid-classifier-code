"""In plain terms: a reviewer watched a sample of 2025 clips and named the species of every
fish without being shown what the model had said. This script compares the reviewer's
species with the model's for the same clips and reports how often they agree, overall and
for each species, with the uncertainty on those percentages. Because a clip can hold several
fish and the reviewer's fish are not linked one-to-one to the model's tracks, fish are paired
within each clip by species: a reviewer's Chinook Salmon is paired with a model Chinook
Salmon if there is one, and whatever is left over on both sides is paired off as
disagreements.

Inputs
------
Tier 1 (default), committed in ``data/paper/``:

* ``blind_v27_clips.csv``: one row per reviewed clip, with the reviewer's species list and
  the published model's track species list.

Tier 2 (``--from-external``), not stored in git:

* ``reference_recount.csv`` and ``reference_sample_2025.json`` in ``config.BLIND_REVIEW_DIR``:
  the reviewer's fish-by-fish record and the list of sampled clips;
* the per-track diagnostics of the published model for 2025 (see ``pervideo_agreement``).

Outputs
-------
Written to ``--out-dir`` (default ``output/paper/``):

* ``blind_v27_pairs.csv``: one row per paired fish (reviewer species, model species, clip).
* ``blind_v27_summary.csv``: the agreement figures quoted in the manuscript.
* with ``--from-external``: ``blind_v27_clips.csv`` rebuilt from the review files.

The published copy of ``blind_v27_pairs.csv`` is committed in ``data/paper/`` for comparison.

Feeds
-----
Section 3.2, blind-review paragraph: 196 of 208 paired fish (94.2%; 95% CI 90.2% to 96.7%;
kappa 0.90, 95% CI 0.84 to 0.95); 213 fish identified by the reviewer; agreement by species
(Chinook Salmon 95%, n = 125; Rainbow Trout 98%, n = 54; Coho Salmon 77%, n = 13; Atlantic
Salmon 100%, n = 6; Brown Trout 100%, n = 8); the 61% to 100% interval on the six Atlantic
Salmon. Section 4.4 repeats the 94.2%.

Run
---
::

    python -m salmonid.paper.blind_review                    # Tier 1
    python -m salmonid.paper.blind_review --from-external    # Tier 2

Tier
----
1 for every statistic. Rebuilding the per-clip table from the review files is Tier 2.

Provenance
----------
Ported from the blind-review block of ``recompute_v27.py`` (manuscript repository,
2026-11-06). The pairing, the Wilson interval and the bootstrap routine are unchanged. One
token differs from the saved script: the cluster label handed to the bootstrap (see the
"DEVIATION" note in ``score``). The saved script's version gives a zero-width kappa interval;
the version here gives the interval printed in the manuscript.
The sampling design of the review (stratified by river, run phase and crowding) is in
``salmonid.results.build_reference_sample``.
"""
from __future__ import annotations

import argparse
import collections
import json

import pandas as pd

from salmonid import config
from salmonid.paper._common import (SP, keyof, cm_from, kappa, cluster_boot_kappa, wilson,
                                    add_out_dir, out_dir)

CLIPS = config.PAPER_DATA / "blind_v27_clips.csv"

# Tier-2 inputs
V26 = config.BLIND_REVIEW_DIR        # the review files sit beside the run the sample was drawn from
V27 = config.corpus_dir(2025)        # tracks of the published model


def load_diag(river):
    d = pd.read_csv(V27 / f"{river.lower()}_track_diag_v27.csv")
    d["key"] = d.video_id.map(keyof)
    return d


def build_clips():
    """Tier 2. One row per reviewed clip: the reviewer's species and the published model's.

    The reviewer's fish are every row of the recount that was not struck out as "not a fish"
    (``removed``) and that carries a species; this includes fish the reviewer added because
    the earlier model had missed them. "unsure" is kept as its own answer. The model's fish
    are the tracks of the published (v2.7) model in the same clip. The review was first
    scored against the earlier model whose output the sample was drawn from; that older
    score (``old_ok`` / ``old_n``) is printed only as a cross-check and is not used.
    """
    config.require(V26 / "reference_recount.csv", "blind-review record", "SALMONID_BLIND_REVIEW_DIR")
    for rv in ("credit", "ganaraska"):
        config.require(V27 / f"{rv}_track_diag_v27.csv",
                       "per-track diagnostics of the published model, 2025", "SALMONID_PROJECTS_ROOT")
    sample = {c["video_id"]: c for c in json.load(open(V26 / "reference_sample_2025.json"))}
    r = pd.read_csv(V26 / "reference_recount.csv", dtype=str).fillna("")
    r = r[r.video_id.isin(sample)]
    diag = pd.concat([load_diag("Credit"), load_diag("Ganaraska")])
    diag["vid"] = diag.video_id.str.replace("\\", "/", regex=False)
    by = {k: g for k, g in diag.groupby("vid")}
    clips = []
    old_ok = old_n = 0
    for v, g in r.groupby("video_id"):
        real = g[(g.removed != "1") & (g.your_species != "")]            # kept + missed, adjudicated
        kept = g[(g.missed != "1") & (g.removed != "1") & (g.your_species != "") & (g.your_dir != "")]
        old_n += len(kept); old_ok += int((kept.model_species == kept.your_species).sum())
        truth = list(real.your_species)
        t27 = by.get(v.replace("\\", "/"))
        pred = list(t27.pred_species) if t27 is not None else []
        dom = t27.loc[t27.n_frames.idxmax()].pred_species if t27 is not None and len(t27) else None
        npair = min(len(truth), len(pred))
        inter = sum((collections.Counter(truth) & collections.Counter(pred)).values())
        clips.append(dict(video_id=v, site=sample[v]["site"], n_truth=len(truth), n_pred=len(pred), npair=npair,
                          match=min(inter, npair), truth=";".join(truth), pred=";".join(pred), dom=dom))
    c = pd.DataFrame(clips)
    print(f"  v2.6 per-track (as published): {old_ok}/{old_n} = {old_ok/old_n:.4f}")
    return c


def load_clips():
    """Tier 1. Read the committed per-clip table. Species lists are kept as text; an empty
    list (a clip in which the model returned no track) must stay an empty string."""
    c = pd.read_csv(config.require(CLIPS, "committed blind-review clip table"),
                    dtype=str, keep_default_na=False)
    for col in ("n_truth", "n_pred", "npair", "match"):
        c[col] = c[col].astype(int)
    return c


def score(c, out, as_saved=False):
    """Everything the manuscript quotes, from the per-clip table ``c``."""
    summary = []
    print(f"  clips reviewed {len(c)}; clips with no v2.7 track {int((c.n_pred == 0).sum())}; "
          f"fish identified by the reviewer {int(c.n_truth.sum())}; "
          f"clips with at least one paired fish {int((c.npair > 0).sum())}")
    # "Multiset" agreement: within a clip, count the reviewer's fish whose species the model
    # also returned (like matched with like), capped at the number of fish that can be paired.
    tot_m, tot_n = int(c.match.sum()), int(c.npair.sum())
    lo, hi = wilson(tot_m, tot_n)
    print(f"  v2.7 multiset agreement: {tot_m}/{tot_n} = {tot_m/tot_n:.4f}  Wilson [{lo:.3f},{hi:.3f}]")
    # single-fish clips: reviewer saw one real fish; v2.7 dominant track vs reviewer species
    s1 = c[(c.n_truth == 1) & (c.n_pred >= 1)].copy()
    s1["ref"] = s1.truth
    ok1 = int((s1.ref == s1.dom).sum())
    lo1, hi1 = wilson(ok1, len(s1))
    print(f"  v2.7 single-fish clips, dominant track: {ok1}/{len(s1)} = {ok1/len(s1):.4f} Wilson [{lo1:.3f},{hi1:.3f}]")
    # per-fish pairs for kappa: expand matched pairs by assigning greedy multiset pairing
    pairs = []
    for _, row in c.iterrows():
        truth = row.truth.split(";") if row.truth else []
        pred = row.pred.split(";") if row.pred else []
        if not truth or not pred:
            continue
        tc, pc = collections.Counter(truth), collections.Counter(pred)
        common = tc & pc
        rt, rp = list((tc - common).elements()), list((pc - common).elements())
        for s, k in common.items():
            pairs += [(s, s, row.video_id)] * k
        for a_, b_ in zip(rt, rp):        # leftover truths paired with leftover predictions = disagreements
            pairs.append((a_, b_, row.video_id))
    P = pd.DataFrame(pairs, columns=["ref", "pred", "clip"])
    # The reviewer could answer "unsure". Those fish stay in the comparison as a sixth
    # reference answer the model can never match, so they count as disagreements.
    labs = SP + ["unsure"]
    P = P[P.ref.isin(labs) & P.pred.isin(SP)]
    cmb = cm_from(P.ref, P.pred, labs)
    kb = kappa(cmb)
    # Fish in one clip are not independent, so the interval resamples whole clips.
    #
    # DEVIATION FROM THE SAVED SCRIPT (one token). The saved script passed ``P.clip`` as the
    # cluster label. In pandas, ``P.clip`` is the built-in DataFrame method of that name, not
    # the column called "clip", so every fish landed in a single cluster and the bootstrap
    # returned a zero-width interval (0.90 to 0.90). The manuscript's interval, 0.84 to 0.95,
    # is what the column gives, so the column ``P["clip"]`` is used here. ``--as-saved``
    # reproduces the saved script's behaviour for the record.
    clusters = P.clip if as_saved else P["clip"]
    (klo, khi), (alo, ahi) = cluster_boot_kappa(P.ref, P.pred, clusters, labs)
    n_ok = int((P.ref == P.pred).sum())
    plo, phi = wilson(n_ok, len(P))
    print(f"  v2.7 paired n={len(P)} agree={n_ok} acc={kb[1]:.4f} kappa={kb[0]:.3f} "
          f"[{klo:.2f},{khi:.2f}]  (clip bootstrap, {P['clip'].nunique()} clips)")
    summary.append(dict(group="all paired fish", n=len(P), agree=n_ok, percent=round(100 * kb[1], 1),
                        wilson_lo=round(100 * plo, 1), wilson_hi=round(100 * phi, 1),
                        kappa=round(kb[0], 3), kappa_lo=round(klo, 3), kappa_hi=round(khi, 3)))
    for s in SP:
        sub = P[P.ref == s]
        k_ = int((sub.pred == s).sum())
        w = wilson(k_, len(sub))
        print(f"    {s:14s} n={len(sub):3d} agree={k_:3d} "
              f"({100*(sub.pred == s).mean():.0f}%)  Wilson {tuple(round(float(x), 2) for x in w)}")
        summary.append(dict(group=s, n=len(sub), agree=k_, percent=round(100 * k_ / len(sub), 1),
                            wilson_lo=round(100 * w[0], 1), wilson_hi=round(100 * w[1], 1)))
    print(f"    unsure n={int((P.ref == 'unsure').sum())}")
    P.to_csv(out / "blind_v27_pairs.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "blind_v27_summary.csv", index=False)
    print(f"\nwrote: {out / 'blind_v27_pairs.csv'}\nwrote: {out / 'blind_v27_summary.csv'}")
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    ap.add_argument("--from-external", action="store_true",
                    help="Tier 2: rebuild the per-clip table from the review files and the model's tracks.")
    ap.add_argument("--as-saved", action="store_true",
                    help="Reproduce the saved script's clip bootstrap exactly (see the note in score()).")
    args = ap.parse_args()
    out = out_dir(args)
    print("\n## Blind review rescored on v2.7 tracks")
    if args.from_external:
        c = build_clips()
        c.to_csv(out / "blind_v27_clips.csv", index=False)
        c = pd.read_csv(out / "blind_v27_clips.csv", dtype=str, keep_default_na=False)
        for col in ("n_truth", "n_pred", "npair", "match"):
            c[col] = c[col].astype(int)
    else:
        c = load_clips()
    score(c, out, as_saved=args.as_saved)


if __name__ == "__main__":
    main()

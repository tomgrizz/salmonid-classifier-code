"""In plain terms: this script tests whether the kind of "visual front end" a model is built
on changes how well it tells the five species apart. Three species classifiers were trained
on the same fish images in the same way and differ only in that front end (the backbone):
an older convolutional network (ResNet-50) and two newer self-supervised networks (DINOv2,
DINOv3). Each was shown the same frames and the same boxes of every held-out fish, so any
difference in the answers comes from the backbone alone. The script counts each
classifier's mistakes by river, counts how often each called another species an Atlantic
Salmon, and tests whether the differences are larger than chance.

Inputs
------
Tier 1 (default), committed in ``data/paper/``:

* ``backbone_track_predictions.csv``: one row per test set, classifier and track, with the
  reviewers' species, the classifier's species and class probabilities, the river, year and
  recording session, and a flag saying whether the track had been used to train the
  classifiers.

Tier 2 (``--build-table``), not stored in git:

* the classifier predictions written by the inference run (``--predictions``; see
  "What is not in the deposit" below);
* the image lists the three classifiers were trained on
  (``config.CLASSIFIER_CROPS``/``train.csv`` and ``val.csv``), used only to set the flag.

Outputs
-------
Written to ``--out-dir`` (default ``output/paper/``):

* ``table_s4.csv``: the six rows of Table S4.
* ``backbone_mcnemar.csv``: the paired tests.
* with ``--build-table``: ``backbone_track_predictions.csv`` rebuilt.

Feeds
-----
* Table S4 (all cells).
* Section 3.1, last paragraph: 239 tracks; 46, 22 and 11 misidentified; discordant tracks 39
  and 4 (DINOv3) and 28 and 4 (DINOv2), exact McNemar p < 0.001; 232 tracks of other
  species, 32, 3 and 1 called Atlantic Salmon, all at the Credit River, 24 of the 32 Chinook
  Salmon from September and October; 6, 6 and 5 errors on 130 Ganaraska River tracks.
* Methods 2.3 (the comparison's design; "the same 40,638 images" is the row count of the
  Tier-2 training list, printed by ``--build-table``) and Sections 3.2 and 4.1, which cite
  the same result.

Run
---
::

    python -m salmonid.paper.backbone_comparison            # Table S4 and the tests (Tier 1)
    python -m salmonid.paper.backbone_comparison --full     # also macro-F1 by subset

Tier
----
1 for Table S4 and the tests. Tier 2 for rebuilding the committed table.

What is not in the deposit
--------------------------
The inference step that produced the predictions (three classifier checkpoints applied to
the ten highest-confidence frames of each track, mean class probability, highest wins) used
the earlier two-stage classifier code and its weights, which are not part of this
repository. The committed table is that step's output. Table S4 can therefore be recomputed
from the predictions but the predictions cannot be regenerated from this repository alone.

Provenance
----------
Ported from ``table_s4.py`` and ``analyze_backbone_comparison.py`` (manuscript repository,
2026-11-09). ``macro_f1``, ``_macro_fast`` and ``mcnemar_exact`` are copied unchanged, and
the counting in ``table_s4`` and ``full_scoring`` is unchanged. Two things were adapted for
the deposit. First, the committed table identifies a video by a short hash and carries the
river, year and recording session as columns, because some archive folder names in the full
video path contain staff first names; the original scripts read river and year from the
path. Second, the "used to train the classifiers" flag is a column of the table, set once
by ``build_table``, where the original scripts looked each track up in the external training
lists on every run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir

TABLE = config.PAPER_DATA / "backbone_track_predictions.csv"
SP = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]
IDX = {s: i for i, s in enumerate(SP)}
BACKBONES = ["resnet50", "dinov2_vitb14", "dinov3_vitb16"]
N_BOOT = 10_000
PROB_COLS = [f"p_{s.replace(' ', '_')}" for s in SP]
TABLE_COLUMNS = (["set", "model", "video_hash", "track_id", "site", "year", "session", "gt_species",
                  "tier", "n_frames", "pred_species"] + PROB_COLS + ["in_classifier_training"])


def video_hash(video_id: str) -> str:
    """Short, stable stand-in for a video's path (first 8 hex digits of its MD5). The same
    hash names the image files in the classifiers' training lists, which is what lets a test
    track be looked up there. Apply it to ``video_id`` in ``data/test_set_v6_gt.json`` to
    link a row of the committed table back to the held-out set."""
    return hashlib.md5(video_id.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- Tier 2: build the table
def crop_tracks():
    """Tracks whose images were in the classifiers' training or validation lists.

    Image files are named ``ls_<video hash>_<track id>_<frame>.jpg``; the (hash, track) pairs
    of both lists are returned. Also prints the list sizes (Methods 2.3 quotes the first).
    """
    crops = config.require(config.CLASSIFIER_CROPS, "image lists used to train the Table S4 classifiers",
                           "SALMONID_CLASSIFIER_CROPS")
    tk = set()
    for f in ("train.csv", "val.csv"):
        n = 0
        for r in csv.DictReader(open(crops / f, encoding="utf-8")):
            n += 1
            p = r["path"].split("_")
            if p[0] == "ls":
                tk.add((p[1], int(p[2])))
        print(f"  {f}: {n:,} images")
    return tk


def build_table(predictions: Path, dest: Path) -> None:
    """Tier 2. Turn the inference output into the committed table.

    For every prediction row: replace the video path by its hash; record the river (first
    path part), the year (first 20xx in the path, as the original script read it) and the
    recording-session folder (second-last path part, "YY  MM  DD  hh  mm", the start of the
    counter session the clip belongs to); and flag whether the track was in the classifiers'
    training lists.
    """
    predictions = config.require(predictions, "classifier predictions from the backbone-comparison run")
    trained = crop_tracks()
    rows = list(csv.DictReader(open(predictions, encoding="utf-8")))
    out_rows, keys_full, keys_hash = [], set(), set()
    for r in rows:
        vid = r["video_id"]
        h = video_hash(vid)
        m = re.search(r"(20\d\d)", vid)
        keys_full.add((r["set"], vid, r["track_id"]))
        keys_hash.add((r["set"], h, r["track_id"]))
        out_rows.append([r["set"], r["model"], h, r["track_id"], vid.split("/")[0],
                         int(m.group(1)) if m else "", vid.replace("\\", "/").split("/")[-2],
                         r["gt_species"], r["tier"], r["n_frames"], r["pred_species"]]
                        + [r[c] for c in PROB_COLS]
                        + [int((h, int(r["track_id"])) in trained)])
    # The hash must still tell every track apart.
    assert len(keys_full) == len(keys_hash), "video hash collision"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(TABLE_COLUMNS)
        w.writerows(out_rows)
    print(f"wrote {dest} ({len(out_rows):,} rows, {len(keys_hash):,} tracks)")


# ---------------------------------------------------------------- statistics (unchanged)
# macro_f1: per species, precision = share of the classifier's calls that were right,
# recall = share of that species' tracks it found, F1 = their harmonic mean; the macro-F1 is
# the plain average over the five species, so a rare species counts as much as a common one.
#
# mcnemar_exact: the paired test. Both classifiers saw the same tracks, so the tracks they
# BOTH got right or BOTH got wrong say nothing about which is better. Only the discordant
# tracks do: b tracks that only the first got right, c that only the second got right. If
# the two were equally good, a discordant track would favour either with probability one
# half, so the split b : c is compared with a fair coin (two-sided, exact binomial; no
# large-sample approximation, which matters with counts as small as 4).
def macro_f1(gt, pred):
    """gt, pred: int arrays. Returns (macro, acc, per-species rows)."""
    rows, f1s = [], []
    for s in range(5):
        tp = int(np.sum((gt == s) & (pred == s)))
        fp = int(np.sum((gt != s) & (pred == s)))
        fn = int(np.sum((gt == s) & (pred != s)))
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append((SP[s], tp + fn, tp, fp, fn, p, r, f1)); f1s.append(f1)
    return float(np.mean(f1s)), float(np.mean(gt == pred)), rows


def _macro_fast(gt, pred):
    f1s = []
    for s in range(5):
        tp = np.sum((gt == s) & (pred == s)); fp = np.sum((gt != s) & (pred == s)); fn = np.sum((gt == s) & (pred != s))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ---------------------------------------------------------------- loading
def load():
    """Read the committed table. Returns per-track dictionaries keyed by
    (set, video_hash, track_id): predictions per classifier, and the track's attributes."""
    pred = defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(config.require(TABLE, "committed backbone-comparison table"),
                                 encoding="utf-8")):
        k = (r["set"], r["video_hash"], int(r["track_id"]))
        pred[k][r["model"]] = r["pred_species"]
        meta[k] = {"gt": r["gt_species"], "tier": r["tier"], "site": r["site"],
                   "year": int(r["year"]) if r["year"] else None, "session": r["session"],
                   "trained": r["in_classifier_training"] == "1", "n_frames": int(r["n_frames"])}
    return pred, meta


# ---------------------------------------------------------------- Table S4
def table_s4(out: Path) -> None:
    """Numbers for Table S4 and the Section 3.1 sentences."""
    pred, meta = load()
    gt = {k: m["gt"] for k, m in meta.items()}

    def seen(k):
        return meta[k]["trained"]

    def site(k):
        return meta[k]["site"]

    def year(k):
        return meta[k]["year"]

    # The comparison uses only held-out ("bench") tracks that none of the three classifiers
    # had been trained on. 127 of the 366 had been used to train these (earlier) classifiers;
    # all three classifiers score the same on those, so they carry no information about the
    # backbone and would only dilute the comparison.
    bench = [k for k in pred if k[0] == "bench" and not seen(k)]
    rows = {
        "Benchmark, unseen, all": bench,
        "  Credit River, 2023 to 2025": [k for k in bench if site(k) == "Credit" and year(k) >= 2023],
        "  Credit River, 2020 to 2022": [k for k in bench if site(k) == "Credit" and year(k) <= 2022],
        "  Ganaraska River, all years": [k for k in bench if site(k) == "Ganaraska"],
        "Earlier test set": [k for k in pred if k[0] == "clips"],
        "Benchmark, trained on": [k for k in pred if k[0] == "bench" and seen(k)],
    }
    print("sites:", Counter(site(k) for k in pred), " years:", sorted(Counter(year(k) for k in pred).items()))
    print(f"\n{'subset':32s} {'n':>4s} {'nonAtl':>6s} | errors R50/v2/v3 | false Atlantic R50/v2/v3")
    for name, ks in rows.items():
        non = [k for k in ks if gt[k] != "Atlantic"]
        err = [sum(pred[k][m] != gt[k] for k in ks) for m in BACKBONES]
        fa = [sum(pred[k][m] == "Atlantic" for k in non) for m in BACKBONES]
        print(f"{name:32s} {len(ks):>4d} {len(non):>6d} | {err[0]:>3d} {err[1]:>3d} {err[2]:>3d}      | {fa[0]:>3d} {fa[1]:>3d} {fa[2]:>3d}")
        print(f"{'':32s} GT: {dict(Counter(gt[k] for k in ks))}")

    print("\nfalse Atlantic calls on unseen benchmark, by true species / site / year-month:")
    for m in BACKBONES:
        fa = [k for k in bench if gt[k] != "Atlantic" and pred[k][m] == "Atlantic"]
        print(" ", m, len(fa), dict(Counter(gt[k] for k in fa)), dict(Counter((site(k), gt[k]) for k in fa)))
        if m == "resnet50":
            # The session folder name starts "YY  MM", so it gives the month the clips were recorded in.
            by_month = Counter((gt[k], meta[k]["session"].split()[1]) for k in fa)
            print("      by true species and month of the recording session:", dict(sorted(by_month.items())))
            for k in sorted(fa, key=lambda k: (meta[k]["site"], meta[k]["session"], k[2])):
                print("     ", gt[k], "|", meta[k]["site"], meta[k]["year"], "session", meta[k]["session"])
    print("\nerrors that are NOT false Atlantic calls, unseen benchmark:")
    for m in BACKBONES:
        other = [k for k in bench if pred[k][m] != gt[k] and not (gt[k] != "Atlantic" and pred[k][m] == "Atlantic")]
        print(" ", m, len(other), dict(Counter((gt[k], pred[k][m]) for k in other)))
    tests = []
    print("\nexact McNemar, unseen benchmark:")
    for a, b in [("resnet50", "dinov2_vitb14"), ("resnet50", "dinov3_vitb16"), ("dinov2_vitb14", "dinov3_vitb16")]:
        ob = sum(pred[k][a] != gt[k] and pred[k][b] == gt[k] for k in bench)
        oa = sum(pred[k][a] == gt[k] and pred[k][b] != gt[k] for k in bench)
        print(f"  {b} vs {a}: {ob} against {oa}, p = {mcnemar_exact(ob, oa):.3g}")
        tests.append(["Benchmark, unseen, all", b, a, ob, oa, mcnemar_exact(ob, oa)])
    for name in ("  Ganaraska River, all years", "Earlier test set"):
        ks = rows[name]
        ob = sum(pred[k]["resnet50"] != gt[k] and pred[k]["dinov3_vitb16"] == gt[k] for k in ks)
        oa = sum(pred[k]["resnet50"] == gt[k] and pred[k]["dinov3_vitb16"] != gt[k] for k in ks)
        print(f"  {name.strip()}: dinov3 vs resnet50 {ob} against {oa}, p = {mcnemar_exact(ob, oa):.3g}")
        tests.append([name.strip(), "dinov3_vitb16", "resnet50", ob, oa, mcnemar_exact(ob, oa)])

    # ---- the table as printed in the manuscript (Credit River = both year groups together)
    credit = [k for k in bench if site(k) == "Credit"]
    gana = rows["  Ganaraska River, all years"]
    with open(out / "table_s4.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["block", "tracks", "n", "ResNet-50", "DINOv2", "DINOv3"])
        for label, ks in (("Both rivers", bench), ("Credit River", credit), ("Ganaraska River", gana)):
            w.writerow(["Misidentified, all species", label, len(ks)]
                       + [sum(pred[k][m] != gt[k] for k in ks) for m in BACKBONES])
        for label, ks in (("Both rivers", bench), ("Credit River", credit), ("Ganaraska River", gana)):
            non = [k for k in ks if gt[k] != "Atlantic"]
            w.writerow(["Other species identified as Atlantic Salmon", label, len(non)]
                       + [sum(pred[k][m] == "Atlantic" for k in non) for m in BACKBONES])
    with open(out / "backbone_mcnemar.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["tracks", "classifier_b", "classifier_a", "only_b_right", "only_a_right", "exact_p"])
        w.writerows(tests)
    print("\nTable S4 as printed:")
    print((out / "table_s4.csv").read_text(encoding="utf-8"))
    print(f"wrote: {out / 'table_s4.csv'}\nwrote: {out / 'backbone_mcnemar.csv'}")


# ---------------------------------------------------------------- full scoring (--full)
def full_scoring() -> None:
    """Macro-F1 with bootstrap intervals and the paired tests for every subset. Not quoted in
    the manuscript; kept because it was part of the original analysis record."""
    pred, meta = load()
    data = {k: {m: IDX[s] for m, s in v.items()} for k, v in pred.items()}
    keys = sorted(data)
    print(f"tracks scored: {len(keys)}  "
          f"(clips {sum(k[0]=='clips' for k in keys)}, bench {sum(k[0]=='bench' for k in keys)})")
    print("frames per track:", dict(sorted(Counter(meta[k]['n_frames'] for k in keys).items())))

    subsets = {
        "clips (180, none trained on)": [k for k in keys if k[0] == "clips"],
        "bench_all (366)": [k for k in keys if k[0] == "bench"],
        "bench_unseen (benchmark tracks not in training crops)": [k for k in keys if k[0] == "bench" and not meta[k]["trained"]],
        "bench_trained (benchmark tracks in training crops)": [k for k in keys if k[0] == "bench" and meta[k]["trained"]],
        "unseen_pooled (clips + bench_unseen)": [k for k in keys if not meta[k]["trained"]],
    }
    rng = np.random.default_rng(0)
    for title, ks in subsets.items():
        gt = np.array([IDX[meta[k]["gt"]] for k in ks])
        n = len(ks)
        print("\n" + "=" * 100)
        print(f"{title}   n = {n}   GT: " + ", ".join(f"{SP[s]} {int(np.sum(gt==s))}" for s in range(5)))
        preds = {m: np.array([data[k][m] for k in ks]) for m in BACKBONES}
        boot_idx = rng.integers(0, n, size=(N_BOOT, n))
        boots = {}
        for m in BACKBONES:
            macro, acc, prs = macro_f1(gt, preds[m])
            boots[m] = np.array([_macro_fast(gt[i], preds[m][i]) for i in boot_idx])
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            print(f"\n-- {m}: macro-F1 {macro:.3f} [{lo:.3f}, {hi:.3f}]   accuracy {acc:.3f}   "
                  f"errors {int(np.sum(gt != preds[m]))}/{n}")
            print(f"   {'species':14s} {'n':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
            for sp, nn, tp, fp, fn, p, r, f1 in prs:
                print(f"   {sp:14s} {nn:>4d} {tp:>4d} {fp:>4d} {fn:>4d} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
            cm = np.zeros((5, 5), int)
            for g, p_ in zip(gt, preds[m]):
                cm[g, p_] += 1
            print("   confusion (rows = true, cols = predicted): " + " | ".join(
                " ".join(f"{v:3d}" for v in row) for row in cm))
        print("\n-- paired comparisons (same tracks)")
        for a, b in [("resnet50", "dinov2_vitb14"), ("resnet50", "dinov3_vitb16"), ("dinov2_vitb14", "dinov3_vitb16")]:
            ca, cb = preds[a] == gt, preds[b] == gt
            only_b = int(np.sum(~ca & cb)); only_a = int(np.sum(ca & ~cb))
            diff = boots[b] - boots[a]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            print(f"   {b} minus {a}: macro-F1 difference {np.mean(diff):+.3f} [{lo:+.3f}, {hi:+.3f}]; "
                  f"tracks only {b} got right {only_b}, only {a} got right {only_a}; "
                  f"McNemar exact p = {mcnemar_exact(only_b, only_a):.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    ap.add_argument("--full", action="store_true", help="Also print macro-F1 and paired tests by subset.")
    ap.add_argument("--build-table", action="store_true",
                    help="Tier 2: rebuild backbone_track_predictions.csv in --out-dir from --predictions.")
    ap.add_argument("--predictions", default=None,
                    help="With --build-table: the inference run's track_predictions.csv.")
    args = ap.parse_args()
    out = out_dir(args)
    if args.build_table:
        if not args.predictions:
            raise SystemExit("--build-table needs --predictions <track_predictions.csv>")
        build_table(Path(args.predictions), out / TABLE.name)
        return
    table_s4(out)
    if args.full:
        full_scoring()


if __name__ == "__main__":
    main()

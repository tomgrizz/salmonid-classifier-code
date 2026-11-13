"""In plain terms: the counter can trigger more than once for the same fish, so the clip
recorded just before or just after a held-out clip may show the same individual. The
held-out clips themselves were kept out of training, but their neighbours were not. This
script finds the held-out fish whose neighbouring clip contributed a training image of the
same species (76 of the 366), sets them aside, and re-scores the model on the other 290. If
the score barely moves, the benchmark was not being flattered by near-duplicate fish.

Inputs
------
Tier 1 (default), committed:

* ``data/paper/heldout_adjacent_clip_tracks.csv``: the 366 held-out tracks with a 0/1 flag
  for "a same-species training image exists in the clip numbered one before or one after".
* ``data/test_set_v6_gt.json`` and ``data/preds_v27_ckpt0017.csv``: reviewers' and model's
  species for each held-out track.

Tier 2 (``--build-table``), not stored in git:

* the published model's training list (``config.COCO_V27``/``train.json``).

Outputs
-------
Written to ``--out-dir`` (default ``output/paper/``):

* ``heldout_adjacent_clip_sensitivity.csv``: macro-F1, accuracy and per-species F1 on all 366
  tracks and on the 290 that remain.
* with ``--build-table``: ``heldout_adjacent_clip_tracks.csv`` rebuilt.

Feeds
-----
* Methods 2.4: "for 76 of the tracks the immediately adjacent clip contributed training
  images of the same species".
* Section 3.1: macro-averaged F1 from 0.961 to 0.950, Brown Trout 0.966 to 0.935.
* Section 4.1: Atlantic Salmon and Brown Trout 0.953 and 0.935 without those tracks.
* Section 4.5: "the one-point drop in macro-averaged F1".

Run
---
::

    python -m salmonid.paper.heldout_adjacent_clips                  # Tier 1
    python -m salmonid.paper.heldout_adjacent_clips --build-table    # Tier 2

Tier
----
1 for the scores. Flagging the tracks needs the training list (Tier 2).

Provenance
----------
WRITTEN FOR THE DEPOSIT. The 76 tracks and the 0.950 were found during the 2026-09-29 code
review of the deposit, in a scratch session whose script was not kept. The flag is rebuilt
here with the neighbour rule of ``salmonid.data.audit_coco_holdout_leak --report-adjacent``
(same recording folder, clip number one lower or one higher, neighbour not itself held out,
a training box of the held-out fish's species), and the scoring uses the deposit's own
``salmonid.eval.score_testset_v6.species_f1``. An adjacent clip need not show the same fish,
so this is an upper bound on the exposure, not a count of duplicated fish.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from salmonid import config
from salmonid.eval.score_testset_v6 import species_f1
from salmonid.paper._common import add_out_dir, out_dir

TABLE = config.PAPER_DATA / "heldout_adjacent_clip_tracks.csv"


def video_hash(video_id: str) -> str:
    """First 8 hex digits of the MD5 of the video path. The committed table identifies a
    video by this hash because some archive folder names in the full path contain staff
    first names; the same hash is used in ``backbone_track_predictions.csv``."""
    return hashlib.md5(video_id.encode()).hexdigest()[:8]


def build_table(dest: Path) -> None:
    """Tier 2. Flag every held-out track whose adjacent clip gave a same-species training box."""
    # Imported here so the Tier-1 path does not depend on the training-data audit module.
    from salmonid.data.audit_coco_holdout_leak import _clip_neighbours, _image_video
    from salmonid.holdout import _read_holdout_file

    coco_path = config.require(config.COCO_V27 / "train.json", "training list of the published model",
                               "SALMONID_COCO_CACHE")
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    species_by_img = collections.defaultdict(set)
    for an in coco["annotations"]:
        species_by_img[an["image_id"]].add(cats.get(an["category_id"]))
    imgs_by_video = collections.defaultdict(list)
    for im in coco["images"]:
        vid = _image_video(im)
        if vid:
            imgs_by_video[vid].append(im)
    # A neighbour that is itself a held-out clip was excluded from training already.
    _, held_videos = _read_holdout_file(config.TESTSET_V6_HOLDOUT)

    gt = json.loads(Path(config.TESTSET_V6_GT).read_text(encoding="utf-8"))["tracks"]
    rows = []
    for t in gt:
        n_imgs = n_same = 0
        for nb in _clip_neighbours(t["video_id"]):
            if nb in held_videos:
                continue
            for im in imgs_by_video.get(nb, []):
                n_imgs += 1
                n_same += int(t["species"] in species_by_img.get(im["id"], ()))
        rows.append(dict(video_hash=video_hash(t["video_id"]), track_id=t["track_id"], species=t["species"],
                         tier=t.get("tier", ""), site=t.get("site", ""),
                         adjacent_training_images=n_imgs, adjacent_same_species_images=n_same,
                         adjacent_same_species=int(n_same > 0)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False, lineterminator="\n")
    print(f"wrote {dest}: {sum(r['adjacent_same_species'] for r in rows)} of {len(rows)} tracks flagged")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    ap.add_argument("--build-table", action="store_true",
                    help="Tier 2: rebuild heldout_adjacent_clip_tracks.csv in --out-dir from the training list.")
    args = ap.parse_args()
    out = out_dir(args)
    if args.build_table:
        build_table(out / TABLE.name)
        return

    flags = pd.read_csv(config.require(TABLE, "committed adjacent-clip flag table"))
    flagged = {(r.video_hash, int(r.track_id)) for r in flags.itertuples() if r.adjacent_same_species == 1}
    by_sp = flags.groupby("species").adjacent_same_species.agg(["sum", "size"])
    print(f"held-out tracks with a same-species training image in an adjacent clip: {len(flagged)} of {len(flags)}")
    print("  " + ", ".join(f"{sp} {int(r['sum'])}/{int(r['size'])}" for sp, r in by_sp.iterrows()))

    gt = {(t["video_id"], int(t["track_id"])): t["species"]
          for t in json.loads(Path(config.TESTSET_V6_GT).read_text(encoding="utf-8"))["tracks"]}
    preds = {(r["video_id"], int(r["track_id"])): r["pred_species"]
             for r in csv.DictReader(open(config.PUBLISHED_PREDS, encoding="utf-8"))}
    subsets = {
        "all 366 tracks": gt,
        "excluding flagged tracks": {k: v for k, v in gt.items() if (video_hash(k[0]), k[1]) not in flagged},
        "flagged tracks only": {k: v for k, v in gt.items() if (video_hash(k[0]), k[1]) in flagged},
    }
    rows = []
    for name, g in subsets.items():
        # Same scorer as the headline result: per-species F1 from the confusion matrix, and
        # the macro-F1 as the unweighted mean of the five.
        macro, acc, per, _, n = species_f1(g, preds)
        row = dict(subset=name, n=n, macro_f1=round(macro, 4), accuracy=round(acc, 4))
        row.update({f"f1_{sp}": round(f1, 3) for sp, *_, f1 in per})
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(out / "heldout_adjacent_clip_sensitivity.csv", index=False)
    print(f"\nwrote: {out / 'heldout_adjacent_clip_sensitivity.csv'}")


if __name__ == "__main__":
    main()

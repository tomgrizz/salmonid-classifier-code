"""In plain terms: this script is the check that a set of training pictures contains none of
the "exam" fish used to grade the model. It reads the list of training pictures, looks up
which video clip each one came from, and counts how many came from a clip that belongs to the
held-out test set. If it finds any, it says so and (with ``--strict``) refuses to let training
start. It can also write a cleaned copy of the list with the offending pictures removed,
which is how the training set of the published model (v2.7) was produced from the previous,
leaky one (v2.6).

Inputs
------
* ``--coco``: the list of training (or validation) pictures to check, in COCO JSON format.
  For the published model: ``<SALMONID_COCO_CACHE>/deimv2_fish_6class_masked_v26_clean/train.json``.
* The held-out sets, loaded by ``salmonid.holdout`` (the published test set is committed in
  ``data/test_set_v6_holdout.json``).
* ``--synth-coco`` (optional): the records of the computer-composited pictures, which say
  which real fish each composite was cut from.
* ``--deep`` (optional): the stored detections of the run the test set was drawn from
  (``<SALMONID_PROJECTS_ROOT>/model_outputs/run_20260428/detections``) and the committed test
  answers (``data/test_set_v6_gt.json``). Only read for clips that already failed the VIDEO check.

Outputs
-------
* A printed table of how many pictures fail each check, and a verdict (CLEAN / LEAKING).
* Exit status 1 with ``--strict`` if anything fails (this is what blocks training).
* ``--out PATH``: a cleaned copy of the list. ``--emit-leaked-tracks CSV``: the held-out fish
  that were trained on. Neither is written unless asked for.

Feeds
-----
The manuscript's statement that the held-out test set was excluded from training at the
video level, and so the held-out score of the published model (macro-F1 0.961, 366 fish).
It also reproduces the reason the previous model's 0.966 is not reported: run on the v2.6
training list with ``--deep`` it finds 62 of the 366 test fish in the training pictures.

Run
---
::

    # gate the published training list (exit 1 if anything leaks)
    python -m salmonid.data.audit_coco_holdout_leak --coco <COCO_V27>/train.json --strict
    # the same, plus the two report-only sections described below
    python -m salmonid.data.audit_coco_holdout_leak --coco <COCO_V27>/train.json --report-adjacent
    # show the v2.6 leak and write the leak-clean copy
    python -m salmonid.data.audit_coco_holdout_leak --coco <COCO_V26>/train.json --deep \
        --synth-coco <PROJECTS_ROOT>/sam2_synth_partial/synth_coco.json ... --out <somewhere>/train.json

``<COCO_V27>`` etc. are the paths in ``salmonid/config.py``. ``salmonid.train.train_v27`` runs
the first form automatically before training.

Tier
----
2: the training lists are external downloads (see ``docs/DATA_AVAILABILITY.md``). The second
held-out file is also external; with only the committed test set available, set
``SALMONID_ALLOW_MISSING_HOLDOUT=1`` (see ``salmonid/holdout.py``).

The four checks (unchanged from the research repository, where this was
``scripts/_audit_coco_holdout_leak.py``)
------------------------------------------------------------------------
::

  TRACK   the picture's (video_id, track_id) is a held-out pair          [the old guard]
  VIDEO   the picture's video_id is a held-out clip, whatever its track   [the hole that leaked]
  SYNTH   a composite whose source cut-out came from a held-out clip      [via srcs / --synth-coco]
  DEEP    (--deep) the picture's box overlaps a held-out fish's own box in that same frame.
          This is the strongest statement: "this exact fish was trained on with this label".

Why TRACK alone was not enough is explained in ``salmonid/holdout.py``.

What these checks cannot see, and what was added here to report it
-------------------------------------------------------------------
The checks above look up a picture by its ``video_id``. Two groups of pictures are therefore
outside them. Both are reported below the table but do **not** change the verdict or the exit
status, because the published v2.7 model was trained on a list that passed exactly the four
checks above; making these sections fail the gate would misrepresent what was run.

* **UNKEYED (always printed).** Pictures that carry no ``video_id``. In the published list
  these are the 1,854 frames from the CVAT annotation tool, which record an absolute
  ``video_path`` instead. This section rebuilds the clip name from that path and counts the
  ones that fall in a held-out clip. On the published v2.7 training list it finds 25 frames
  from the clip of one fish of the published test set (an Atlantic Salmon, track 353168), and
  560 frames from 10 clips of the older evaluation set that the guard also covers but the
  manuscript does not use. This is a known limitation of the published model, not something
  this script repairs.
* **ADJACENT (``--report-adjacent``).** The camera can record one fish in several consecutive
  clips. This section counts training pictures from the clip numbered one before or one after
  a held-out clip in the same recording folder, and how many held-out fish have a
  same-species training box there. It cannot tell whether the neighbouring clip shows the same
  individual; it shows how many held-out fish that question applies to.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from salmonid import config
from salmonid.holdout import _read_holdout_file, load_holdout_pairs


def _synth_provenance(paths) -> dict:
    """file_name -> [srcs] for composited images (the merged COCO keeps the pixels, drops the srcs)."""
    prov: dict = {}
    for p in paths or []:
        sc = json.loads(Path(p).read_text(encoding="utf-8"))
        for im in sc.get("images", []):
            prov[im["file_name"]] = im.get("srcs", [])
    return prov


def _iou(a, b) -> float:
    """Overlap of two boxes (x1, y1, x2, y2) as intersection area / union area, 0..1."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _deep_hits(coco, imgs_by_video, gt_file, det_dir, iou_thresh):
    """Held-out GT tracks whose OWN box appears as a training annotation in the same video+frame.

    Returns (hits, image_ids, missing): ``hits`` is a set of (video_id, track_id, tier, species),
    one per held-out fish; ``missing`` lists clips whose stored detections could not be found.
    """
    import pyarrow.parquet as pq
    from salmonid.util.schema import video_id_to_parquet_stem

    anns_by_img = collections.defaultdict(list)
    for an in coco["annotations"]:
        anns_by_img[an["image_id"]].append(an)
    gt_by_video = collections.defaultdict(list)
    for t in json.loads(Path(gt_file).read_text(encoding="utf-8"))["tracks"]:
        gt_by_video[t["video_id"]].append(t)

    hits, img_ids = set(), set()          # hits are UNIQUE GT tracks, not (track, image) pairs
    missing = []
    # Only clips that are BOTH in the test set and in the training list are opened, so a clean
    # list reads no detection files at all.
    for vid in sorted(set(gt_by_video) & set(imgs_by_video)):
        pf = Path(det_dir) / f"{video_id_to_parquet_stem(vid)}.parquet"
        if not pf.is_file():
            # The original skipped this silently. It cannot hide a leak from the verdict (the
            # clip is already counted at VIDEO level), but it would under-count the DEEP level,
            # so it is collected and reported.
            missing.append(vid)
            continue
        df = pq.read_table(str(pf), columns=["track_id", "frame_number",
                                             "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]).to_pandas()
        for t in gt_by_video[vid]:
            g = df[df.track_id == int(t["track_id"])]
            gbox = {int(r.frame_number): [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2] for r in g.itertuples()}
            for im in imgs_by_video[vid]:
                if im.get("frame_number") not in gbox:
                    continue
                for an in anns_by_img.get(im["id"], []):
                    x, y, w, h = an["bbox"]
                    if _iou([x, y, x + w, y + h], gbox[im["frame_number"]]) >= iou_thresh:
                        hits.add((vid, int(t["track_id"]), t.get("tier"), t.get("species")))
                        img_ids.add(im["id"])
                        break
    return hits, img_ids, missing


# ------------------------------------------------------------------------------------------
# Report-only additions (not in the research repository's script; they never change the
# verdict). See the module docstring, "What these checks cannot see".
# ------------------------------------------------------------------------------------------
_SITES = tuple(config.video_roots())     # ("Ganaraska", "Credit")


def _video_id_from_path(video_path: str) -> str | None:
    """Rebuild a ``<site>/.../<clip>.mp4`` video_id from an absolute footage path.

    CVAT pictures store e.g. ``<footage root>/Credit/2023/<dates>/<time>/140.mp4``. The
    video_id used everywhere else is the same path from the river name onward, so everything
    before the first folder named after a study river is dropped. Returns None if no such
    folder is present.
    """
    parts = str(video_path).replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in _SITES:
            return "/".join(parts[i:])
    return None


def _image_video(im) -> str | None:
    """The clip a picture came from: its video_id, else the one rebuilt from its video_path."""
    return im.get("video_id") or (_video_id_from_path(im["video_path"]) if im.get("video_path") else None)


def _report_unkeyed(imgs, hold, synth_ids) -> set:
    """Print the UNKEYED section; return the ids of unkeyed pictures that sit in a held-out clip."""
    # Which held-out clips belong to the PUBLISHED test set (as opposed to the older evaluation
    # set that is also guarded). Only the published set bears on a manuscript number.
    _, published_videos = _read_holdout_file(Path(config.TESTSET_V6_HOLDOUT))
    unkeyed = [im for im in imgs if not im.get("video_id") and im["id"] not in synth_ids]
    by_src = collections.Counter(im.get("source") or im.get("synth_source") or "(no source tag)"
                                 for im in unkeyed)
    hit_ids, hit_videos, unresolved = set(), collections.Counter(), 0
    for im in unkeyed:
        vid = _image_video(im)
        if vid is None:
            unresolved += 1
        elif vid in hold.videos:
            hit_ids.add(im["id"])
            hit_videos[vid] += 1
    pub = {v: n for v, n in hit_videos.items() if v in published_videos}
    other = {v: n for v, n in hit_videos.items() if v not in published_videos}
    print()
    print(f"UNKEYED (report only; not part of the verdict): {len(unkeyed)} pictures carry no "
          f"video_id and were NOT examined by the checks above  {dict(by_src)}")
    print(f"  their clip was rebuilt from video_path ({unresolved} could not be rebuilt):")
    print(f"  in the PUBLISHED test set (test_set_v6): {sum(pub.values())} pictures "
          f"from {len(pub)} held-out clip(s)")
    for vid, n in sorted(pub.items()):
        print(f"    {n:4d} pictures  {vid}")
    print(f"  in another guarded evaluation set (not used in the manuscript): "
          f"{sum(other.values())} pictures from {len(other)} clip(s)")
    for vid, n in sorted(other.items()):
        print(f"    {n:4d} pictures  {vid}")
    if hit_ids:
        print("  ^ KNOWN LIMITATION of the published model: these frames were not excluded because "
              "they have no video_id for the guard to test. See salmonid/holdout.py.")
    return hit_ids


def _clip_neighbours(video_id: str) -> list[str]:
    """The clips numbered one below and one above ``video_id`` in the same recording folder."""
    folder, _, name = video_id.rpartition("/")
    stem, dot, ext = name.rpartition(".")
    if not stem.isdigit():
        return []
    n = int(stem)
    return [f"{folder}/{m}{dot}{ext}" for m in (n - 1, n + 1) if m >= 0]


def _report_adjacent(coco, gt_file, hold) -> None:
    """Print the ADJACENT section: pictures from clips next to a published-test-set clip."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    species_by_img = collections.defaultdict(set)
    for an in coco["annotations"]:
        species_by_img[an["image_id"]].add(cats.get(an["category_id"]))
    imgs_by_video = collections.defaultdict(list)
    for im in coco["images"]:
        vid = _image_video(im)
        if vid:
            imgs_by_video[vid].append(im)

    gt = json.loads(Path(gt_file).read_text(encoding="utf-8"))["tracks"]
    near_imgs, near_same, near_held_videos = set(), set(), set()
    flagged = collections.Counter()
    totals = collections.Counter(t["species"] for t in gt)
    for t in gt:
        same_species = False
        for nb in _clip_neighbours(t["video_id"]):
            if nb in hold.videos:        # a neighbour that is itself held out is already excluded
                continue
            for im in imgs_by_video.get(nb, []):
                near_imgs.add(im["id"])
                near_held_videos.add(t["video_id"])
                if t["species"] in species_by_img.get(im["id"], ()):
                    near_same.add(im["id"])
                    same_species = True
        if same_species:
            flagged[t["species"]] += 1
    print()
    print(f"ADJACENT (report only; not part of the verdict): {len(near_imgs)} pictures in this list "
          f"come from the clip numbered just before or after {len(near_held_videos)} test_set_v6 clips "
          f"({len(near_same)} carry a box of that held-out fish's species)")
    print(f"  held-out fish with a same-species box in an adjacent clip: "
          f"{sum(flagged.values())} of {len(gt)}")
    for sp in sorted(totals, key=lambda s: -flagged[s]):
        print(f"    {sp:14s} {flagged[sp]:3d} / {totals[sp]}")
    print("  An adjacent clip need not show the same individual; this counts how many held-out "
          "fish that question applies to.")


def main() -> None:
    a = _parse_args()
    if a.out and Path(a.out).resolve() == Path(a.coco).resolve():
        raise SystemExit("--out is the same file as --coco: refusing to overwrite the list being audited.")
    hold = load_holdout_pairs()
    print(f"guard: {len(hold)} held-out tracks across {len(hold.videos)} held-out videos")

    coco = json.loads(Path(a.coco).read_text(encoding="utf-8"))
    imgs, anns = coco["images"], coco["annotations"]
    print(f"coco: {Path(a.coco).name} — {len(imgs)} images, {len(anns)} anns")

    prov = _synth_provenance(a.synth_coco)
    if prov:
        print(f"synth provenance: {len(prov)} composited images resolved from {len(a.synth_coco)} synth COCO(s)")

    pairs = set(hold)                      # the pure pair set, so TRACK and VIDEO stay distinguishable
    leak_track, leak_video, leak_synth = set(), set(), set()
    synth_ids = set()                      # composites (they legitimately have no video_id)
    imgs_by_video = collections.defaultdict(list)
    for im in imgs:
        vid, tid = im.get("video_id"), im.get("track_id")
        if vid:
            imgs_by_video[vid].append(im)
            if vid in hold.videos:
                leak_video.add(im["id"])
            if tid is not None and (vid, int(tid)) in pairs:
                leak_track.add(im["id"])
        # A composite is checked through the real fish it was cut from. The merged training
        # list does not keep that record, so it is looked up by file name in --synth-coco.
        srcs = im.get("srcs") or prov.get(im.get("file_name"), [])
        if im.get("synth_source") or srcs:
            synth_ids.add(im["id"])
        if any(hold.held(s["video_id"], s.get("track_id")) for s in srcs):
            leak_synth.add(im["id"])

    deep_ids: set = set()
    if a.deep:
        hits, deep_ids, missing = _deep_hits(coco, imgs_by_video, a.gt_file, a.detections_dir, a.iou_thresh)
        by_sp = collections.Counter(h[3] for h in hits)
        by_tier = collections.Counter(h[2] for h in hits)
        print(f"\nDEEP: {len(hits)} held-out GT tracks trained on with a box on the SAME fish "
              f"(IoU>={a.iou_thresh})  tiers={dict(by_tier)}  species={dict(by_sp)}")
        if missing:
            print(f"  WARNING: stored detections not found for {len(missing)} leaked clip(s) under "
                  f"{a.detections_dir}; the DEEP count is a lower bound (VIDEO still counts them).")
        if a.emit_leaked_tracks:
            import csv as _csv
            Path(a.emit_leaked_tracks).parent.mkdir(parents=True, exist_ok=True)
            with open(a.emit_leaked_tracks, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f); w.writerow(["video_id", "track_id", "tier", "species"])
                w.writerows(sorted(hits))
            print(f"  leaked track list -> {a.emit_leaked_tracks} "
                  f"(feed to salmonid.eval.score_testset_v6 --drop-tracks for the honest score)")

    bad = leak_track | leak_video | leak_synth | deep_ids
    n_leak_videos = len({im["video_id"] for im in imgs if im["id"] in leak_video})
    print(f"\n{'level':8s} {'images':>8s}")
    print(f"{'TRACK':8s} {len(leak_track):>8d}")
    print(f"{'VIDEO':8s} {len(leak_video):>8d}   ({len(leak_video - leak_track)} beyond TRACK; "
          f"from {n_leak_videos} held-out clips)")
    print(f"{'SYNTH':8s} {len(leak_synth):>8d}")
    if a.deep:
        print(f"{'DEEP':8s} {len(deep_ids):>8d}   ({len(hits)} distinct GT test tracks)")
    print(f"{'TOTAL':8s} {len(bad):>8d}  ({100.0 * len(bad) / max(1, len(imgs)):.2f}% of images)")

    # Report-only sections. They are printed after the table and never added to `bad`.
    _report_unkeyed(imgs, hold, synth_ids)
    if a.report_adjacent:
        _report_adjacent(coco, a.gt_file, hold)

    if a.out:
        keep_imgs = [im for im in imgs if im["id"] not in bad]
        keep_ids = {im["id"] for im in keep_imgs}
        keep_anns = [an for an in anns if an["image_id"] in keep_ids]
        out = dict(coco)
        out["images"], out["annotations"] = keep_imgs, keep_anns
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out), encoding="utf-8")
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        tally = collections.Counter(cats.get(an["category_id"], an["category_id"]) for an in keep_anns)
        print(f"\nwrote leak-clean COCO -> {a.out}")
        print(f"  images {len(imgs)} -> {len(keep_imgs)} (-{len(imgs) - len(keep_imgs)});  "
              f"anns {len(anns)} -> {len(keep_anns)} (-{len(anns) - len(keep_anns)})")
        print(f"  boxes per class: {dict(tally)}")
        print("\nVERDICT: CLEAN (re-run without --out to confirm the written file)")
        return
    print(f"\nVERDICT: {'CLEAN' if not bad else 'LEAKING'}  "
          f"(TRACK / VIDEO / SYNTH{' / DEEP' if a.deep else ''} levels; UNKEYED is reported above, not judged)")
    if bad and a.strict:
        raise SystemExit(f"LEAK: {len(bad)} images carry held-out eval fish — refuse to train on {a.coco}")


def _parse_args():
    p = argparse.ArgumentParser(description="Audit (and optionally clean) a training COCO for eval leakage")
    p.add_argument("--coco", required=True, help="COCO json to audit (train.json / val.json).")
    p.add_argument("--out", default=None, help="Write a leak-clean copy here (drops leaked images + anns).")
    p.add_argument("--synth-coco", nargs="*", default=[], metavar="PATH",
                   help="synth_coco.json file(s) supplying `srcs` provenance for composited images.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero if anything leaks.")
    p.add_argument("--deep", action="store_true",
                   help="Also match training boxes against the GT test tracks' own boxes (needs parquets).")
    p.add_argument("--emit-leaked-tracks", default=None, metavar="CSV",
                   help="With --deep: write the leaked (video_id, track_id, tier, species) rows so the "
                        "honest score can be reproduced via salmonid.eval.score_testset_v6 --drop-tracks.")
    p.add_argument("--report-adjacent", action="store_true",
                   help="Report only: count training pictures from clips numbered +/-1 next to a "
                        "held-out clip (does not change the verdict).")
    # The committed copy of the test answers has the same content as the one the original
    # default pointed at on the project data drive (provenance/v2.7/CHECKSUMS.md), so the
    # committed file is the default here.
    p.add_argument("--gt-file", default=str(config.TESTSET_V6_GT))
    p.add_argument("--detections-dir",
                   default=str(config.MODEL_OUTPUTS / "run_20260428" / "detections"))
    p.add_argument("--iou-thresh", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())

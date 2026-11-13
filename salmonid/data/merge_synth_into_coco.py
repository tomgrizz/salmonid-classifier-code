"""In plain terms: this script adds computer-composited ("synthetic") fish pictures to the list
of real labelled pictures the model learns from. The composites give the model extra practice on
two hard situations: a fish only partly in view, and two fish overlapping. While merging, it
removes every real picture that comes from a held-out test clip, and every composite that was
made from a fish cut out of such a clip, so that no exam fish reaches training by either route.
It caps how many composites are added so they stay a minority of the training set.

Inputs
------
* ``--base``: the list of real training pictures (COCO JSON, six classes). Its class list is
  the authority for the merge.
* ``--synth`` (repeatable): the record written by ``salmonid.data.sam2_composite`` for one batch
  of composites. Each composite lists the real fish it was cut from (``srcs``).
* The held-out sets. The two canonical files are always loaded through ``salmonid.holdout`` and
  cannot be switched off; ``--holdout-keys`` only ADDS further files. Two older evaluation lists
  (``--op-testset``, ``--ls-test-queue``) are applied to the composites as well.

Outputs
-------
``--out``: the merged list = base pictures (minus held-out ones) + the capped composites.
Pictures are referenced by absolute file name; no image is copied.

Feeds
-----
No manuscript number directly. The composites it folds in make up 7,391 of the 45,249 pictures
and 11,644 of the 52,444 fish boxes the published v2.7 model was trained on (Methods,
training-set composition: "11,644 were synthetic").

Run
---
::

    python -m salmonid.data.merge_synth_into_coco --base <base>/train.json
        --synth <PROJECTS_ROOT>/sam2_synth_partial/synth_coco.json
        --synth <PROJECTS_ROOT>/sam2_synth_proximity/synth_coco.json
        --out <merged>/train.json --max-synth 3300

(one command; shown on four lines)

Tier
----
2. The base list, the composites and two of the holdout lists are external. The published
training list is a frozen file whose checksum is in ``provenance/v2.7/``; this script is
deposited so the merge can be read, not so the list can be regenerated from a clone.

How this differs from what built the published data
---------------------------------------------------
The guard here is the corrected, video-level one (research repository commit ``d40b869``). The
merges that built the v2.5 and v2.6 lists ran the earlier version, which re-listed its own
holdout files and matched on exact (video, track) pairs. The published v2.7 list is the v2.6
list cleaned afterwards by ``salmonid.data.audit_coco_holdout_leak --out`` (which removed 311
real pictures and 23 composites). A fresh merge with this version would not admit them.

Known limitations (left as they were, because the published model was trained on their output)
------------------------------------------------------------------------------------------------
* The merged record of a composite keeps its file name but drops ``srcs``. The frozen training
  list therefore cannot be checked for composite provenance on its own; the separate
  ``synth_coco.json`` files are needed (``audit_coco_holdout_leak --synth-coco``).
* Base pictures with no ``video_id`` (the CVAT frames) are not tested against the holdout, here
  or anywhere else in the pipeline. See ``salmonid/holdout.py``.
* The exact commands that merged the four targeted v2.6 composite batches were not recorded in
  any repository, and the step that blacks out the 40-pixel header strip on real frames is not
  part of this deposit. The composites are shuffled with a fixed seed (42) before the cap.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from salmonid import config


def _short_rel(video_id: str) -> str:
    """Return the last two components of a video_id (site/clip key) used for op-video holdout matching."""
    return "/".join(video_id.split("/")[-2:])


def _load_holdout(op_path, ls_path):
    """Load the operational-video and label-studio holdout key sets used to reject synth images.

    Args:
        op_path: op_testset JSON (``{"videos": [{site, video_rel}, ...]}``), or None/missing.
        ls_path: ls_test_set JSON (``{"items": [{video_id, track_id}, ...]}``), or None/missing.

    Returns:
        Tuple ``(op_keys, ls_keys)`` — a set of ``(site, short_rel)`` video keys and a set of
        ``(video_id, track_id)`` track keys. Either set is empty when its path is absent.
    """
    op_keys, ls_keys = set(), set()
    for label, path in (("--op-testset", op_path), ("--ls-test-queue", ls_path)):
        # The original skipped a missing file without a word. These two lists are secondary to
        # the canonical guard (which is fatal when its own files are missing), so a missing one
        # is still not an error here, but it is now said out loud.
        if path and not Path(path).is_file():
            print(f"  WARNING: {label} file not found, so it excludes nothing: {path}")
    if op_path and Path(op_path).is_file():
        for v in json.loads(Path(op_path).read_text(encoding="utf-8")).get("videos", []):
            op_keys.add((v["site"], v["video_rel"].replace("\\", "/")))
    if ls_path and Path(ls_path).is_file():
        for it in json.loads(Path(ls_path).read_text(encoding="utf-8")).get("items", []):
            ls_keys.add((it["video_id"], int(it["track_id"])))
    return op_keys, ls_keys


def _img_is_holdout(img, op_keys, ls_keys) -> bool:
    """True if any source track of a synth image falls in the op-video or ls-track holdout sets."""
    for s in img.get("srcs", []):
        vid = s["video_id"]; site = vid.split("/", 1)[0]
        if (site, _short_rel(vid)) in op_keys or (vid, int(s["track_id"])) in ls_keys:
            return True
    return False


def _load_track_keys(extra_paths):
    """The canonical eval holdout (salmonid.holdout), plus any --holdout-keys files ADDED to it.

    Returns a video-aware HoldoutKeys: `(video_id, track_id) in keys` is True for a held-out pair OR any
    track in a held-out video. Dropped from BOTH the base (by its per-image video_id/track_id) and the
    synth (by srcs). Importing CANONICAL_HOLDOUT_FILES rather than re-listing it here is what makes the
    project rule true: add a frozen eval set once and every builder honors it.
    """
    from salmonid.holdout import load_holdout_pairs
    return load_holdout_pairs(extra_files=extra_paths)


def main() -> None:
    """Merge base + capped synth COCO, drop holdout images from both, and write the merged train.json.

    Side effects: reads the ``--base`` / ``--synth`` / holdout JSONs; writes the merged COCO to ``--out``
    (creating parent dirs); prints a summary (image/annotation counts, synth percentage of train, and the
    per-synth-mode tally). Aborts if a synth category_id is not defined by the base categories.
    """
    a = _parse_args()
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    base_imgs, base_anns = base["images"], base["annotations"]
    cats = {c["id"]: c["name"] for c in base["categories"]}
    print(f"base: {len(base_imgs)} images, {len(base_anns)} anns, categories {cats}")

    # Eval holdout (test_set_v6 + ls_test_v5): drop matching BASE images + anns. Video-aware, so an image
    # from a held-out clip is dropped even when its track id is -1/unknown or from another detection run.
    hk = _load_track_keys(a.holdout_keys)
    if hk or hk.videos:
        def _held(im):
            # Keyed on video_id. A base picture WITHOUT a video_id (the CVAT frames) is kept
            # unexamined: a known gap, documented in the module docstring, not closed here.
            return im.get("video_id") is not None and hk.held(im["video_id"], im.get("track_id"))
        kept_imgs = [im for im in base_imgs if not _held(im)]
        kept_ids = {im["id"] for im in kept_imgs}
        kept_anns = [an for an in base_anns if an["image_id"] in kept_ids]
        print(f"  holdout ({len(hk)} eval tracks / {len(hk.videos)} eval videos): dropped "
              f"{len(base_imgs)-len(kept_imgs)} base images / {len(base_anns)-len(kept_anns)} anns")
        base_imgs, base_anns = kept_imgs, kept_anns

    max_img_id = max((im["id"] for im in base_imgs), default=0)
    max_ann_id = max((an["id"] for an in base_anns), default=0)
    op_keys, ls_keys = _load_holdout(a.op_testset, a.ls_test_queue)
    rng = random.Random(a.seed)

    # Gather all synth images (with their anns) across the passed synth files.
    pool = []  # [(img, [anns], mode_tag)]
    for sp in a.synth:
        sc = json.loads(Path(sp).read_text(encoding="utf-8"))
        anns_by_img = {}
        for an in sc["annotations"]:
            anns_by_img.setdefault(an["image_id"], []).append(an)
        n_hold = 0
        for im in sc["images"]:
            # A composite is held out if ANY fish pasted into it was cut from a held-out clip.
            src_held = any(hk.held(s["video_id"], s.get("track_id")) for s in im.get("srcs", []))
            if src_held or _img_is_holdout(im, op_keys, ls_keys):
                n_hold += 1
                continue
            pool.append((im, anns_by_img.get(im["id"], []), im.get("synth_source", "synth")))
        print(f"synth {Path(sp).parent.name}: {len(sc['images'])} imgs "
              f"({n_hold} holdout-dropped here)")

    # Seeded shuffle, then cap: which composites survive the cap is fixed by --seed.
    rng.shuffle(pool)
    if a.max_synth and len(pool) > a.max_synth:
        pool = pool[:a.max_synth]

    # Verify category ids align (synth uses 0-4; base must define them).
    for cid in {an["category_id"] for _, anns, _ in pool for an in anns}:
        if cid not in cats:
            raise SystemExit(f"synth category_id {cid} not in base categories {cats} — aborting")

    new_imgs, new_anns = list(base_imgs), list(base_anns)
    iid, aid = max_img_id, max_ann_id
    src_tally = Counter()
    for im, anns, tag in pool:
        iid += 1
        # Only these five fields are carried over. `srcs` (which real fish the composite was cut
        # from) is NOT, which is why a later audit needs the original synth_coco.json files.
        rec = {"id": iid, "file_name": im["file_name"], "width": im["width"],
               "height": im["height"], "synth_source": tag}
        new_imgs.append(rec)
        for an in anns:
            aid += 1
            new_anns.append({"id": aid, "image_id": iid, "category_id": an["category_id"],
                             "bbox": an["bbox"], "area": an.get("area", an["bbox"][2] * an["bbox"][3]),
                             "iscrowd": 0})
        src_tally[tag] += 1

    out = {"categories": base["categories"], "images": new_imgs, "annotations": new_anns}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    added = len(new_imgs) - len(base_imgs)
    pct = 100.0 * added / max(1, len(new_imgs))
    print(f"\nmerged -> {a.out}")
    print(f"  base {len(base_imgs)} + synth {added} = {len(new_imgs)} images "
          f"(synth = {pct:.1f}% of train)")
    print(f"  synth by mode: {dict(src_tally)}")
    print(f"  synth boxes added: {len(new_anns) - len(base_anns)}")
    print("  NEXT: composites made without --mask-header-px 40 still show the header strip; black it "
          "out before training (that step is not part of this deposit).")


def _parse_args():
    """Build and parse the CLI (base/synth/out, the ``--max-synth`` cap, and the holdout-file defaults)."""
    p = argparse.ArgumentParser(description="Fold SAM2 synth COCO into a DEIMv2 base train COCO")
    p.add_argument("--base", required=True, help="Base DEIMv2 train.json (val stays untouched).")
    p.add_argument("--synth", action="append", required=True, metavar="PATH",
                   help="synth_coco.json from salmonid.data.sam2_composite (repeatable).")
    p.add_argument("--out", required=True)
    p.add_argument("--max-synth", type=int, default=None,
                   help="Cap total synth images (keep it a minority — e.g. ~10%% of the base).")
    p.add_argument("--holdout-keys", nargs="*", default=None, metavar="PATH",
                   help="EXTRA holdout file(s) to add on top of salmonid.holdout's canonical frozen "
                        "eval sets (which are always applied — they cannot be turned off from the CLI). "
                        "Dropped from BOTH the base (by per-image video_id/track_id) and the synth (by srcs).")
    p.add_argument("--op-testset",
                   default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    p.add_argument("--ls-test-queue",
                   default=str(config.LABEL_BATCHES / "ls_test_set_v5.json"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()

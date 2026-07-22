"""
In plain terms: this file assembles the collection of training pictures used to teach the fish-detection
model. It mixes in extra computer-generated ("synthetic") fish images alongside the real ones to give the
model more practice, while carefully removing any picture that belongs to the separate set of fish the model
is graded on — so the model can never secretly study its own exam. It does not itself report a number in the
paper, but it prepares the training data behind the published detection model (v2.6).

Fold synthetic COCO into the masked base, cap synth, and drop holdout images (builds masked_v26/train.json).

Fold SAM2 synth COCO (scripts/_sam2_composite.py) into a DEIMv2 base COCO's TRAIN split — the one-variable
augmentation fold for Branch A's v2.5/v2.6 experiment (single-stage). Val is NEVER touched; synth goes to
train only, tagged `synth_source` so it is trivially ablatable, and every synth image is re-checked against
the eval holdouts before it lands (defense-in-depth on top of the compositor's own filter).

Pipeline position
-----------------
DATA-prep stage (data -> **here** -> detect/train -> track -> eval/count -> results -> figures). This
produces the training COCO that the DEIMv2 single-stage detector is trained on; it does not itself compute
a manuscript number, but it is upstream of the published v2.6 model.

Inputs
------
* ``--base`` train.json — a DEIMv2 6-class COCO; its ``categories`` are the authority for the merge.
* one or more ``--synth`` synth_coco.json from _sam2_composite.py (synth uses category ids 0-4).
* the frozen track-level / operational / label-studio holdout files (test_set_v6, ls_test_set_v5,
  op_testset) — used to *reject* any synth image and *drop* any base image touching held-out tracks.

Output
------
A merged train.json (e.g. ``masked_v26/train.json``) = base + capped synth images/annotations with all
holdout images removed. Images are referenced by absolute ``file_name``, so nothing is copied.

Environment
-----------
Base Python (pure ``json``/``pathlib``; no torch / DEIMv2). The DEIMv2 train that consumes the output runs
later in the isolated ``.venv_deim`` environment.

Example
-------
  python scripts/_merge_synth_into_coco.py \
      --base   G:/Projects/deimv2_fish_6class_v25/train.json \
      --synth  G:/Projects/sam2_synth_partial/synth_coco.json \
      --synth  G:/Projects/sam2_synth_proximity/synth_coco.json \
      --out    G:/Projects/deimv2_fish_6class_v25/train_plus_synth.json \
      --max-synth 3300

Then run Branch A's header-mask step over the merged COCO (so synth frames get the same 40px mask as real
ones) and train v2.5/v2.6 with ONLY this variable changed. Categories come from --base (6-class safe: synth
uses ids 0-4 which align). Images are referenced by absolute file_name, so nothing is copied.
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
    if op_path and Path(op_path).is_file():
        for v in json.loads(Path(op_path).read_text()).get("videos", []):
            op_keys.add((v["site"], v["video_rel"].replace("\\", "/")))
    if ls_path and Path(ls_path).is_file():
        for it in json.loads(Path(ls_path).read_text()).get("items", []):
            ls_keys.add((it["video_id"], int(it["track_id"])))
    return op_keys, ls_keys


def _img_is_holdout(img, op_keys, ls_keys) -> bool:
    """True if any source track of a synth image falls in the op-video or ls-track holdout sets."""
    for s in img.get("srcs", []):
        vid = s["video_id"]; site = vid.split("/", 1)[0]
        if (site, _short_rel(vid)) in op_keys or (vid, int(s["track_id"])) in ls_keys:
            return True
    return False


def _load_track_keys(paths) -> set:
    """Load (video_id, track_id) holdout keys from files with a 'tracks' or 'items' list (e.g.
    test_set_v6_holdout.json). These are dropped from BOTH the base (by its per-image video_id/track_id)
    and the synth (by srcs) — the leak fix that was missing when v2.5 first built."""
    keys = set()
    for p in paths or []:
        d = json.loads(Path(p).read_text())
        for t in d.get("tracks", []):
            keys.add((t[0], int(t[1])))
        for it in d.get("items", []):
            keys.add((it["video_id"], int(it["track_id"])))
    return keys


def main() -> None:
    """Merge base + capped synth COCO, drop holdout images from both, and write the merged train.json.

    Side effects: reads the ``--base`` / ``--synth`` / holdout JSONs; writes the merged COCO to ``--out``
    (creating parent dirs); prints a summary (image/annotation counts, synth percentage of train, and the
    per-synth-mode tally). Aborts if a synth category_id is not defined by the base categories.
    """
    a = _parse_args()
    base = json.loads(Path(a.base).read_text())
    base_imgs, base_anns = base["images"], base["annotations"]
    cats = {c["id"]: c["name"] for c in base["categories"]}
    print(f"base: {len(base_imgs)} images, {len(base_anns)} anns, categories {cats}")

    # Track-level holdout (test_set_v6): drop matching BASE images (they carry video_id/track_id) + anns.
    hk = _load_track_keys(a.holdout_keys)
    if hk:
        def _held(im):
            return im.get("track_id") is not None and (im.get("video_id"), int(im["track_id"])) in hk
        kept_imgs = [im for im in base_imgs if not _held(im)]
        kept_ids = {im["id"] for im in kept_imgs}
        kept_anns = [an for an in base_anns if an["image_id"] in kept_ids]
        print(f"  holdout ({len(hk)} test_set_v6 tracks): dropped {len(base_imgs)-len(kept_imgs)} base "
              f"images / {len(base_anns)-len(kept_anns)} anns")
        base_imgs, base_anns = kept_imgs, kept_anns

    max_img_id = max((im["id"] for im in base_imgs), default=0)
    max_ann_id = max((an["id"] for an in base_anns), default=0)
    op_keys, ls_keys = _load_holdout(a.op_testset, a.ls_test_queue)
    rng = random.Random(a.seed)

    # Gather all synth images (with their anns) across the passed synth files.
    pool = []  # [(img, [anns], mode_tag)]
    for sp in a.synth:
        sc = json.loads(Path(sp).read_text())
        anns_by_img = {}
        for an in sc["annotations"]:
            anns_by_img.setdefault(an["image_id"], []).append(an)
        n_hold = 0
        for im in sc["images"]:
            src_held = hk and any((s["video_id"], int(s["track_id"])) in hk for s in im.get("srcs", []))
            if src_held or _img_is_holdout(im, op_keys, ls_keys):
                n_hold += 1
                continue
            pool.append((im, anns_by_img.get(im["id"], []), im.get("synth_source", "synth")))
        print(f"synth {Path(sp).parent.name}: {len(sc['images'])} imgs "
              f"({n_hold} holdout-dropped here)")

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
    Path(a.out).write_text(json.dumps(out))
    added = len(new_imgs) - len(base_imgs)
    pct = 100.0 * added / max(1, len(new_imgs))
    print(f"\nmerged -> {a.out}")
    print(f"  base {len(base_imgs)} + synth {added} = {len(new_imgs)} images "
          f"(synth = {pct:.1f}% of train)")
    print(f"  synth by mode: {dict(src_tally)}")
    print(f"  synth boxes added: {len(new_anns) - len(base_anns)}")
    print("  NEXT: run the header-mask step over the merged COCO, then train v2.5 (one variable).")


def _parse_args():
    """Build and parse the CLI (base/synth/out, the ``--max-synth`` cap, and the holdout-file defaults)."""
    p = argparse.ArgumentParser(description="Fold SAM2 synth COCO into a DEIMv2 base train COCO")
    p.add_argument("--base", required=True, help="Base DEIMv2 train.json (val stays untouched).")
    p.add_argument("--synth", action="append", required=True, metavar="PATH",
                   help="synth_coco.json from _sam2_composite.py (repeatable).")
    p.add_argument("--out", required=True)
    p.add_argument("--max-synth", type=int, default=None,
                   help="Cap total synth images (keep it a minority — e.g. ~10%% of the base).")
    p.add_argument("--holdout-keys", nargs="*",
                   default=[str(config.LABEL_BATCHES / "test_set_v6_holdout.json"),
                            str(config.LABEL_BATCHES / "ls_test_set_v5.json")],
                   help="Track-level holdout file(s) — dropped from BOTH the base (by per-image "
                        "video_id/track_id) and the synth (by srcs). Defaults to the canonical frozen "
                        "eval sets so leakage can't happen by forgetting the flag.")
    p.add_argument("--op-testset",
                   default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    p.add_argument("--ls-test-queue",
                   default=str(config.LABEL_BATCHES / "ls_test_set_v5.json"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()

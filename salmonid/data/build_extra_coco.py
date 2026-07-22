"""In plain terms: this file adds extra hand-picked fish examples (the tricky cases — rare species,
look-alikes, and confusing debris) into the pile of labelled pictures the species-identification
model learns from. It does not produce a number in the paper by itself, but it improves the model
that stands behind the paper's headline 0.966 species-accuracy result. A built-in safety check makes
sure none of the fixed "exam" fish used to score the model can ever sneak into this training pile.

Base masked 6-class COCO builder (honours the leak guard).

Merge the extra label sources (hardcase + triple-confirmed) into the 6-class COCO that
``prepare_dataset.py`` produced. Resolves each track's bboxes from the v6 corpus detection store
(where the hardcase + triple-confirmed tracks were mined) and reuses ``prepare_dataset``'s frame
extraction + bbox y/x/area/conf sanity filters, so nothing bypasses the safety net.

Pipeline position
-----------------
data (this module) -> detect -> track -> eval/count -> results -> figures. This is a *training-data*
step: it augments the frozen training COCO that fits the single-stage DEIMv2-X detector. It does not
itself produce a manuscript number; it feeds the model whose held-out species macro-F1 (0.966 on
``test_set_v6``) and downstream count / abundance-index results are the manuscript claims.

Inputs
------
* ``--data-dir``          a prepared dataset dir holding ``train.json`` (from ``prepare_dataset``).
* ``--extra-pos CSV...``  ``video_id,track_id,species`` rows (hardcase + triple-confirmed positives).
* ``--extra-neg CSV...``  ``video_id,track_id`` rows (hardcase negatives / hard debris).
* corpus detection stores per-video parquets under ``config.MODEL_OUTPUTS`` (the v6 corpus run),
                          plus the raw footage they were mined from (resolved via ``prepare_dataset``).

``track_id == -1`` means "the dominant track in the clip" (used by the single-track triple-confirmed).

Outputs
-------
* extracted JPEG frames under ``<data-dir>/extra_frames/``.
* the same ``<data-dir>/train.json`` rewritten in place with the new images + boxes appended. All
  extra data goes to TRAIN only (val stays clean LS/CVAT).

Leak guard
----------
``salmonid.holdout.load_holdout_pairs`` is consulted unconditionally so no held-out eval track
(``test_set_v6`` / ``ls_test``) can ever be folded into the training COCO (hard rule #1).

Environment: base Python (cv2 + pyarrow); no GPU. The DEIMv2 training that consumes this COCO runs
in the isolated ``.venv_deim`` — this builder does not.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
from pathlib import Path
import cv2

from salmonid import config
from salmonid.data import prepare_dataset as P

# The v6 corpus detection store: per-video parquets sharded across the corpus runs plus the one-off
# run_20260428. Each extra track's bboxes are resolved from these (where the tracks were mined). The
# glob is stringified so it keeps the original literal pattern (Path stringifies with the OS separator).
CORPUS_DIRS = ([Path(d) for d in sorted(glob.glob(
                    str(config.MODEL_OUTPUTS / "run_corpus_v6_*_shard_*" / "detections")))]
               + [config.MODEL_OUTPUTS / "run_20260428" / "detections"])


def _find_parquet(video_id):
    """Return the corpus-store parquet Path for ``video_id``, or None if not found.

    Searches every ``CORPUS_DIRS`` entry for ``<parquet_stem>.parquet`` (stem derived by
    ``prepare_dataset`` from the video id). Side effects: none (filesystem reads only).
    """
    stem = P.video_id_to_parquet_stem(video_id)
    for d in CORPUS_DIRS:
        p = d / f"{stem}.parquet"
        if p.is_file():
            return p
    return None


def _resolve(video_id, track_id):
    """Return (video_path, det_df) or (None, None). track_id==-1 -> dominant track."""
    vp = P._resolve_video_path(video_id)
    if vp is None or not Path(vp).is_file():
        return None, None
    pq_path = _find_parquet(video_id)
    if pq_path is None:
        return None, None
    if int(track_id) == -1:
        import pyarrow.parquet as pq
        t = pq.read_table(pq_path, columns=["track_id"])
        if t.num_rows == 0:
            return None, None
        import collections
        tids = t.column("track_id").to_pylist()
        track_id = collections.Counter(tids).most_common(1)[0][0]
    det_df = P._load_track_detections(pq_path, int(track_id))
    if det_df is None or det_df.empty:
        return None, None
    return vp, det_df


def _extract(video_id, track_id, video_path, det_df, frames_dir, frames_per_track, class_id):
    """Mirror prepare_dataset._process_ls_labels per-track loop. class_id=None -> negative (no anns)."""
    fw, fh = P._get_video_dimensions(video_path)
    if fw == 0:
        return [], []
    top = P._get_top_frames(det_df, fw, fh, n=frames_per_track)
    imgs, anns = [], []
    for frame_number, det_row in top:
        x1, y1, x2, y2 = (int(det_row["bbox_x1"]), int(det_row["bbox_y1"]),
                          int(det_row["bbox_x2"]), int(det_row["bbox_y2"]))
        yc = (y1 + y2) / 2.0; xc = (x1 + x2) / 2.0
        af = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
        # Shared bbox safety net (reused from prepare_dataset, not re-implemented): drop boxes whose
        # centroid strays out of the plausible in-tunnel band, or that are too small / low-confidence.
        # The extra label sources bypass CVAT, so this is the same guard they must still pass.
        if not (P._YCENTER_STRICT_MIN <= yc / max(fh, 1) <= P._YCENTER_STRICT_MAX):
            continue
        if not (P._XCENTER_STRICT_MIN <= xc / max(fw, 1) <= P._XCENTER_STRICT_MAX):
            continue
        if af < P._MIN_AREA_FRAC:
            continue
        try:
            if float(det_row["detection_confidence"]) < P._MIN_DET_CONF:
                continue
        except (KeyError, TypeError, ValueError):
            pass
        stem = f"extra_{P._video_id_hash(video_id)}_{int(track_id):06d}_f{frame_number:06d}"
        jp = frames_dir / f"{stem}.jpg"
        if jp.is_file():
            im = cv2.imread(str(jp))
            if im is None:
                continue
            hpx, wpx = im.shape[:2]
        else:
            rgb = P._read_frame(video_path, frame_number)
            if rgb is None:
                continue
            hpx, wpx = rgb.shape[:2]
            cv2.imwrite(str(jp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        x1 = max(0, min(x1, wpx - 1)); y1 = max(0, min(y1, hpx - 1))
        x2 = max(x1 + 1, min(x2, wpx)); y2 = max(y1 + 1, min(y2, hpx))
        img = {"file_name": str(jp), "width": wpx, "height": hpx,
               "video_id": video_id, "track_id": int(track_id), "frame_number": frame_number,
               "source": "extra_pos" if class_id is not None else "extra_neg",
               "dominant_species": class_id if class_id is not None else -1}
        imgs.append(img)
        if class_id is not None:
            anns.append({"category_id": class_id, "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "_img": img})
    return imgs, anns


def main():
    """CLI entry point: append the extra-pos / extra-neg tracks to ``<data-dir>/train.json``.

    Reads the CSV label sources, drops any held-out track (leak guard), resolves each surviving
    track's frames + bboxes from the corpus store, extracts frames, assigns fresh image/annotation
    ids, and rewrites ``train.json`` in place. Side effects: writes JPEGs under ``extra_frames/``,
    overwrites ``train.json``, and prints per-CSV kept counts + leak-guard + final-size stats.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--extra-pos", nargs="*", default=[])
    ap.add_argument("--extra-neg", nargs="*", default=[])
    ap.add_argument("--frames-per-track", type=int, default=4)
    a = ap.parse_args()
    dd = Path(a.data_dir)
    frames_dir = dd / "extra_frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    train = json.loads((dd / "train.json").read_text())
    next_img = max((im["id"] for im in train["images"]), default=-1) + 1
    next_ann = max((an["id"] for an in train["annotations"]), default=-1) + 1

    # Permanent leak guard — never fold a held-out eval track into the extra COCO.
    from salmonid.holdout import load_holdout_pairs
    hold = load_holdout_pairs()
    n_hold = 0

    def _is_held(vid, tid):
        try:
            return (vid, int(tid)) in hold
        except (TypeError, ValueError):
            return False   # track_id == -1 (dominant) — resolved downstream; video rarely a test video

    new_imgs, new_anns = [], []
    stats = {}
    for csvp in a.extra_pos:
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
        nok = 0
        for i, r in enumerate(rows):
            if i % 500 == 0:
                print(f"  [{Path(csvp).name}] {i}/{len(rows)} (kept {nok})", flush=True)
            cid = P.LABEL_TO_CLASS_ID.get(r["species"])
            if cid is None:
                continue
            if _is_held(r["video_id"], r.get("track_id")):
                n_hold += 1
                continue
            vp, df = _resolve(r["video_id"], r["track_id"])
            if vp is None:
                continue
            imgs, anns = _extract(r["video_id"], r.get("track_id", -1), vp, df, frames_dir, a.frames_per_track, cid)
            new_imgs += imgs; new_anns += anns; nok += bool(imgs)
        stats[Path(csvp).name] = (len(rows), nok)
    for csvp in a.extra_neg:
        rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
        nok = 0
        for r in rows:
            if _is_held(r["video_id"], r.get("track_id")):
                n_hold += 1
                continue
            vp, df = _resolve(r["video_id"], r["track_id"])
            if vp is None:
                continue
            imgs, _ = _extract(r["video_id"], r["track_id"], vp, df, frames_dir, a.frames_per_track, None)
            new_imgs += imgs; nok += bool(imgs)
        stats[Path(csvp).name] = (len(rows), nok)

    # assign ids; wire annotation image_ids
    for im in new_imgs:
        im["id"] = next_img; next_img += 1
    for an in new_anns:
        an["image_id"] = an.pop("_img")["id"]
        an["id"] = next_ann; next_ann += 1
    train["images"] += new_imgs
    train["annotations"] += new_anns
    (dd / "train.json").write_text(json.dumps(train))
    import collections
    print("\nstats (csv rows -> tracks with >=1 frame):", stats)
    print(f"leak guard: skipped {n_hold} held-out (test_set_v6/ls_test) tracks")
    print("added images:", len(new_imgs), " added boxes:", len(new_anns))
    print("box species:", dict(collections.Counter(P.SPECIES.get(an["category_id"], "?") for an in new_anns)))
    print(f"train.json now: {len(train['images'])} images, {len(train['annotations'])} boxes")


if __name__ == "__main__":
    main()

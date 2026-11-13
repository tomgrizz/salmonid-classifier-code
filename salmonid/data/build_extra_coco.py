"""In plain terms: this script adds extra hand-picked fish examples (rare species, look-alike
species, and confusing debris) to the pile of labelled pictures the model learns from. For each
listed fish it finds the video clip, pulls out a few of the clearest frames, and records the box
around the fish and its species. Before adding anything it asks the leak guard whether the clip
belongs to the held-out test set, and skips it if so. This is the script through which most of
the 62 test fish reached the previous model's training set, so the guard call below is the single
most important line in it.

Inputs
------
* ``--data-dir``: a prepared dataset folder holding ``train.json`` (from ``prepare_dataset``).
* ``--extra-pos CSV...``: rows of ``video_id,track_id,species`` (the hand-picked fish).
* ``--extra-neg CSV...``: rows of ``video_id,track_id`` (debris that must not be called a fish).
* Stored detections of the earlier runs, under ``<SALMONID_PROJECTS_ROOT>/model_outputs``, and
  the raw footage the clips came from.

``track_id == -1`` in a CSV row means "the main fish in this clip" (the track with the most
detections). This shorthand is why a guard keyed on track numbers could not work: the row never
names the fish's real track number, so it could never match a held-out (video, track) pair.

Outputs
-------
* JPEG frames under ``<data-dir>/extra_frames/``.
* ``<data-dir>/train.json`` rewritten in place with the new pictures and boxes appended. All of
  it goes to the training split; the validation split is left alone.

Feeds
-----
No manuscript number directly. It supplied 14,013 pictures of positives (``extra_pos``) and 397
of negatives (``extra_neg``) of the 45,249 pictures the published v2.7 model was trained on
(Methods, training-set composition), and so stands behind the held-out species score (macro-F1
0.961) and every count the model produces.

Run
---
::

    python -m salmonid.data.build_extra_coco --data-dir <dataset dir>
        --extra-pos <hardcase_pos.csv> <triple_confirmed_pos.csv>
        --extra-neg <hardcase_neg.csv>

(one command; shown on three lines)

Tier
----
2. It needs the raw footage, the stored detections and the label CSVs, none of which are in
git. The published training list is a frozen file (see ``provenance/v2.7/``); this script is
deposited so the way that file was assembled can be read, not so it can be re-run from a clone.

How this differs from what built the published data
---------------------------------------------------
The published v2.7 training list was not rebuilt from scratch with a corrected guard. It is the
v2.6 list, which an earlier pair-keyed version of this script helped build, with every picture
from a held-out clip removed afterwards by ``salmonid.data.audit_coco_holdout_leak --out``. The
guard below is the corrected one (research repository commit ``d40b869``): a fresh build with it
would not let those pictures in to begin with.

Known limitations (left as they were, because the published model was trained on their output)
------------------------------------------------------------------------------------------------
* The ``track_id`` written to each picture record is the raw CSV value, so it is ``-1`` for
  11,103 of the 14,013 ``extra_pos`` pictures in the published list. Those pictures can be traced
  to a clip but not to a track.
* ``train.json`` is overwritten in place, so running the script twice appends the same pictures
  twice. It was run once per build.
* The default of 4 frames per fish differs from ``prepare_dataset``'s default of 3.
* Negatives are frames of tracks labelled "not a fish". Nothing checks that no other fish is
  visible in the same frame.
* The scripts that produced the input CSVs are not part of this deposit.
* A row whose clip, stored detections or frames cannot be found is skipped without a message
  (the per-CSV "rows -> kept" tally printed at the end is the only trace).
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
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
    train = json.loads((dd / "train.json").read_text(encoding="utf-8"))
    next_img = max((im["id"] for im in train["images"]), default=-1) + 1
    next_ann = max((an["id"] for an in train["annotations"]), default=-1) + 1

    # Permanent leak guard — never fold a held-out eval track into the extra COCO.
    from salmonid.holdout import load_holdout_pairs
    hold = load_holdout_pairs()
    n_hold = 0

    def _is_held(vid, tid):
        # VIDEO-level first: `track_id == -1` means "the dominant track in this clip" (_resolve above),
        # which is normally the very fish the test set holds out, and mined rows carry the track id of
        # whichever detection run produced them. Pair-matching alone missed both — it put 62 test_set_v6
        # fish into the v2.6 COCO with their own label (audit 2026-07-24).
        #
        # The version this replaces read `(vid, int(tid)) in hold` against a plain set of pairs and
        # returned False when the id could not be parsed. For a `-1` row that test could never be
        # True, so the guard was silently inert for most of this script's input.
        return hold.held(vid, tid)

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
    (dd / "train.json").write_text(json.dumps(train), encoding="utf-8")
    import collections
    print("\nstats (csv rows -> tracks with >=1 frame):", stats)
    print(f"leak guard: skipped {n_hold} held-out (test_set_v6/ls_test) tracks")
    print("added images:", len(new_imgs), " added boxes:", len(new_anns))
    print("box species:", dict(collections.Counter(P.SPECIES.get(an["category_id"], "?") for an in new_anns)))
    print(f"train.json now: {len(train['images'])} images, {len(train['annotations'])} boxes")


if __name__ == "__main__":
    main()

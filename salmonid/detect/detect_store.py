"""In plain terms: this script watches every video once with the fish-recognition model and
writes down what it saw, so that no later step has to run the model again. For each video it
saves one small file listing, frame by frame, the box around each fish and the model's score
for each species. That saved file is called a "detect-store". Running the model is by far the
slowest part of the whole analysis (days of GPU time for six seasons), so it is done once
here and everything afterwards (following fish between frames, counting, trying different
counting rules) works from the saved files in minutes.

Pass A: run the detector over every video in a list and save the detections.

Pipeline stage: data -> **detect (this module, "Pass A")** -> track ("Pass B",
``track_from_store.py``) -> count (``salmonid.count.corpus_countr_from_store``) -> results.

Inputs : * a video list ("manifest"), a JSON file ``{"videos": [{"site", "video_rel",
           "video_id", ...}]}`` under ``config.LABEL_BATCHES``, e.g. ``corpus_2024_credit.json``;
         * the raw footage under one or more ``--footage-root`` folders;
         * the model: ``config.INFER_CONFIG`` + ``config.CHECKPOINT_V27``.
Outputs: one pickle file per video in ``--out-dir``:
           {"site", "rel", "video_id", "fps", "n_frames",
            "frames": {frame_number: {"boxes": N x 4, "probs": N x 6}}}
         Only frames with at least one detection are listed. Boxes are x1, y1, x2, y2 in
         original 1280 x 960 pixels; "probs" holds the six class scores per box.
Feeds  : all counts (Tables 2-5, Figures 4-7) through Pass B, and the held-out species score
         (Table 1, Figure 2) through ``salmonid.eval.predict_testset_v6_from_store``.
Tier   : 2 (needs the raw footage, the model weights, the DEIMv2 environment and a GPU).

Run (the two settings that produced published numbers; run from the repository root in the
DEIMv2 environment):

  # (a) one season at one river, as used for the 2020-2025 counts
  python -m salmonid.detect.detect_store \\
      --op-json <label_batches>/corpus_2024_credit.json \\
      --out-dir <projects>/deimv2_x_v27_corpus2024/store \\
      --key-mode video_id --thresh 0.3 --mask-header-px 40 --full-dist --fp16 \\
      --frame-batch-size 32 --frame-skip 3 --shard 0 --num-shards 2

  # (b) the 364 held-out test videos, as used for the species benchmark
  python -m salmonid.detect.detect_store \\
      --op-json <label_batches>/test_set_v6_videos.json \\
      --out-dir <projects>/deimv2_x_v27_eval/store_0017 \\
      --key-mode rel --thresh 0.3 --mask-header-px 40 --full-dist

``python -m salmonid.count.run_corpus --steps detect --dry-run`` prints command (a) for every
season with all paths filled in.

What the settings mean, and why
-------------------------------
* ``--frame-skip 3``: look at every third frame only (frames 0, 3, 6, ...), which takes a
  third of the time. How many looks per second this gives depends on the season, because the
  recording frame rate changed: the 2023-2025 videos are about 10 frames per second (about
  3 looks per second), the 2020-2022 videos are 25 (Ganaraska) or 30 (Credit) frames per
  second (8 to 10 looks per second). Each store file records its video's frame rate
  ("fps"). The held-out benchmark (b) looked at every frame instead.
* ``--thresh 0.3``: keep a candidate box only if its best class score is above 0.30. Lower
  scores are mostly reflections, debris and noise.
* ``--mask-header-px 40``: black out the date/river text at the top of the frame, as in
  training. The script refuses to run without it unless ``--allow-unmasked`` is given.
* ``--full-dist``: save all six class scores for each box instead of only the winning class.
  The later steps average these scores along a fish's track, which is more reliable than
  counting per-frame "votes".
* ``--fp16``: half-precision arithmetic, about twice as fast, used for the season-long runs.
  It changes scores in the third decimal place; the benchmark run (b) did not use it.
* ``--key-mode``: how a video's file is named in the store. ``video_id`` names it by the
  video's full path, which is unique across years, and was used for all seasons. ``rel`` is
  an older scheme (river + last folder + file name) kept only because the held-out benchmark
  store is read back by that name.
* ``--shard i --num-shards N``: split the video list into N parts so N copies can run side by
  side on one GPU. The published runs used 2.

Safety checks ported from the research repository
-------------------------------------------------
* **Resume by validity, not by existence.** The script can be stopped and restarted; videos
  already done are skipped. A crash while a file is being written can leave a damaged file.
  Before skipping, the script now checks that the existing file is non-empty and does not
  begin with a zero byte (the signature of an interrupted write) and re-does the video if the
  check fails. Without this, one damaged file out of 47,997 silently removed 679 videos from
  the 2025 Credit River season during the original run.
* **More than one footage folder.** ``--footage-root`` takes several folders, searched in
  order, so a fast copy can be listed before the full archive. The fast copy used for the
  original run turned out to be missing clips; with a single root those videos were silently
  left out. Videos found under none of the roots are counted and reported as "unresolved".

Which research-repository version this is: the two fixes above were made on two separate
development branches of the research repository, and this file contains both. According to
the run logs, the 2025, 2024 and 2023 seasons and Ganaraska 2022 were detected on 27-30 July
2026 with the two-root version; Credit 2022, 2021, 2020 and the recovered 2020 Ganaraska
footage on 1-3 August 2026 with a single root. The validity check was added on 5 August 2026
after the 2025 Credit loss and was used to re-detect that one damaged file. Every run
reported 0 unresolved videos. For any one video the detection arithmetic is the same in all
of these versions.

Known limits, kept as they were in the published run
----------------------------------------------------
* A video file that cannot be opened is stored as "no detections" and is not retried.
* The store file does not record the settings it was made with (frame skip, threshold,
  mask, precision). The settings above are therefore the record; Pass B must be given the
  same ``--frame-skip``.
* Files are written in place. The validity check above catches an interrupted write on the
  next run.
"""
import argparse, json, glob, pickle, hashlib, sys
from pathlib import Path
import numpy as np, cv2

from salmonid import config
from salmonid.detect import deimv2_infer as di  # the detector wrapper (loads nothing external on import)


def build_index(footage_roots, sites=("Ganaraska", "Credit")):
    """List every video under the footage roots, keyed by (river, "<last folder>/<file>").

    Only used with ``--key-mode rel`` (the held-out benchmark store). The roots are searched
    in the order given and the FIRST file found for a key is kept, so a fast copy can be
    listed ahead of the full archive.
    """
    idx = {}
    for root in footage_roots:
        for site in sites:
            for f in glob.glob(f"{root}/{site}/**/*.mp4", recursive=True):
                p = Path(f); idx.setdefault((site, f"{p.parent.name}/{p.name}"), f)
    return idx


def store_key(site, rel):
    """Store file name in ``rel`` mode: ``<river>__<12-character hash of "folder/file">.pkl``.

    Kept because the held-out benchmark store was written, and is read back, under this name.
    The same "folder/file" can occur in more than one date-range folder within a season, so
    this scheme is not safe for whole seasons; those use ``store_key_vid``.
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Store file name in ``video_id`` mode: a 16-character hash of the video's full path.

    The full path is unique across rivers, years and date-range folders, so two different
    videos cannot end up sharing a store file.
    """
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def resolve_vid(footage_roots, video_id):
    """Find the video file for a ``video_id`` (``<river>/<year folder>/<dates>/<session>/<file>.mp4``).

    Tries each footage root in order and returns the first that holds the file, or None.
    """
    for root in footage_roots:
        p = Path(root) / video_id
        if p.is_file():
            return str(p)
    return None


def main():
    """Run the detector over every video in the list and write one store file per video."""
    ap = argparse.ArgumentParser(description="Pass A: save the model's detections for every video in a list.")
    ap.add_argument("--config", default=str(config.INFER_CONFIG),
                    help="Model architecture file (default: config.INFER_CONFIG).")
    ap.add_argument("--ckpt", default=str(config.CHECKPOINT_V27),
                    help="Trained weights (default: the published model, config.CHECKPOINT_V27).")
    ap.add_argument("--op-json", required=True,
                    help="Video list (manifest JSON with a 'videos' array).")
    ap.add_argument("--thresh", type=float, default=0.3,
                    help="Keep a candidate box if its best class score is above this (published: 0.3).")
    ap.add_argument("--out-dir", default=str(config.OUTPUT_DIR / "detect_store"),
                    help="Folder for the per-video store files.")
    ap.add_argument("--footage-root", nargs="+",
                    default=[str(config.FOOTAGE_ROOT_FAST), str(config.FOOTAGE_ROOT)],
                    help="Folders holding <river> subfolders, in priority order (default: the fast copy, "
                         "then the full archive). A video found under none of them is reported as unresolved.")
    ap.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                    help="video_id (default) = one store file per unique video path (all seasons). "
                         "rel = older naming, used only for the held-out benchmark store.")
    ap.add_argument("--mask-header-px", type=int, default=0,
                    help="Black out the top N rows before detection. The published model was trained with "
                         "40, so pass 40 (the script stops otherwise, unless --allow-unmasked).")
    ap.add_argument("--allow-unmasked", action="store_true",
                    help="Allow --mask-header-px 0 (only for old models trained without the mask).")
    ap.add_argument("--channels-last", action="store_true",
                    help="Memory-layout hint (little or no gain for this model; not used for published runs).")
    ap.add_argument("--fp16", action="store_true",
                    help="Half precision, about twice as fast; scores change in the third decimal place. "
                         "Used for the season runs, not for the held-out benchmark.")
    ap.add_argument("--max-videos", type=int, default=0, help="Stop after N videos (0 = all); for testing.")
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Look at every Nth frame (1 = every frame). Season runs used 3.")
    ap.add_argument("--frame-batch-size", type=int, default=32,
                    help="Frames sent through the model at once (published: 32).")
    ap.add_argument("--decode-backend", choices=["cv2", "decord"], default="cv2",
                    help="Library used to read video frames. The published runs used cv2.")
    ap.add_argument("--shard", type=int, default=0, help="Which part of the list this copy handles (0-based).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split the list into N parts for N copies running side by side.")
    ap.add_argument("--full-dist", action="store_true",
                    help="Save all six class scores per box ('probs') instead of only the winning class. "
                         "Required for the published counting and benchmark steps.")
    args = ap.parse_args()

    # The published model learned from frames whose top 40 rows were blacked out. Showing it the
    # date/river text at inference would be a different input from the one it was trained on.
    if args.mask_header_px <= 0 and not args.allow_unmasked:
        sys.exit("ERROR: --mask-header-px is 0, but the published model was trained with the top 40 rows "
                 "blacked out. Pass --mask-header-px 40, or --allow-unmasked for an old unmasked model.")

    config.require(args.op_json, "the video list (manifest JSON) given as --op-json", "SALMONID_PROJECTS_ROOT")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    op = json.load(open(args.op_json))["videos"]
    if args.max_videos:
        op = op[:args.max_videos]
    if args.num_shards > 1:
        # Every Nth video goes to this copy, so the parts are the same size and never overlap.
        op = [v for i, v in enumerate(op) if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(op)} videos", flush=True)

    idx = None
    if args.key_mode == "rel":
        print(f"indexing footage (rel mode) from {args.footage_root}...", flush=True)
        idx = build_index(args.footage_root)
        print(f"indexed {len(idx)} clips; op videos: {len(op)}", flush=True)

    det = di.DEIMv2Detector(args.config, args.ckpt, mask_header_px=args.mask_header_px,
                            channels_last=args.channels_last, fp16=args.fp16)
    bs = max(1, args.frame_batch_size)
    n_done = n_miss = n_skip = n_recovered = 0
    for vi, v in enumerate(op):
        site = v["site"]; rel = v["video_rel"].replace("\\", "/"); vid = v.get("video_id")
        if args.key_mode == "video_id" and vid:
            kp = out / store_key_vid(vid); vp = resolve_vid(args.footage_root, vid)
        else:
            kp = out / store_key(site, rel); vp = idx.get((site, rel)) if idx is not None else None
        # Resume by VALIDITY, not merely existence. An interrupted write leaves a file that
        # would otherwise count as "already done" forever: this pass would never redo it and
        # the tracking pass would fail on it. In the original run one such 752-byte file cost
        # 679 videos of the 2025 Credit River season. The check reads one byte, so it costs
        # nothing even over hundreds of thousands of files; a bad file is deleted and redone.
        if kp.is_file():
            try:
                ok = kp.stat().st_size > 0
                if ok:
                    with open(kp, "rb") as _f:
                        ok = _f.read(1)[:1] != b"\x00"
            except OSError:
                ok = False
            if ok:
                n_skip += 1; n_done += 1; continue
            print(f"  CORRUPT store file, re-detecting: {kp.name}", flush=True)
            n_recovered += 1
            try:
                kp.unlink()
            except OSError:
                pass
        if vp is None:
            # The video is in the list but its file was not found under any footage root.
            # It gets no store file, and is therefore absent from all later counts.
            n_miss += 1; continue

        # Read the video. cv2 reads frames one after another; decord can jump straight to
        # every Nth frame. The published runs used cv2.
        if args.decode_backend == "decord":
            from decord import VideoReader, cpu as _dcpu
            vr = VideoReader(vp, ctx=_dcpu(0))
            fps = float(vr.get_avg_fps() or 15); n_total = len(vr)
            _idxs = range(0, n_total, args.frame_skip) if args.frame_skip > 1 else range(n_total)

            def _frames():
                for fi in _idxs:
                    yield fi, vr[fi].asnumpy()          # RGB, 8-bit, height x width x 3
        else:
            cap = cv2.VideoCapture(vp)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 15); n_total = None

            def _frames():
                fi = 0
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if args.frame_skip > 1 and fi % args.frame_skip != 0:
                        fi += 1; continue               # frame numbers stay those of the full video
                    yield fi, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    fi += 1
                cap.release()

        frames = {}; last_fi = -1
        buf_fi, buf_rgb = [], []

        def _flush():
            """Send the frames collected so far through the model and keep the detections."""
            if not buf_rgb:
                return
            if args.full_dist:
                for j, (boxes, probs) in enumerate(det.predict_full_batch(buf_rgb, args.thresh)):
                    if len(boxes):                      # frames with no fish are simply not listed
                        frames[buf_fi[j]] = {"boxes": boxes.astype(np.float32),
                                             "probs": probs.astype(np.float32)}
            else:
                for j, rgb in enumerate(buf_rgb):
                    boxes, scores, labels = det.predict(rgb, args.thresh)
                    if len(boxes):
                        frames[buf_fi[j]] = {"boxes": boxes.astype(np.float32),
                                             "scores": scores.astype(np.float32),
                                             "labels": labels.astype(np.int16)}
            buf_fi.clear(); buf_rgb.clear()

        for fi, rgb in _frames():
            last_fi = fi
            buf_fi.append(fi); buf_rgb.append(rgb)
            if len(buf_rgb) >= bs:
                _flush()
        _flush()

        n_frames = n_total if n_total is not None else last_fi + 1
        with open(kp, "wb") as f:
            pickle.dump({"site": site, "rel": rel, "video_id": vid, "fps": float(fps),
                         "n_frames": n_frames, "frames": frames}, f)
        n_done += 1
        if (vi + 1) % 20 == 0:
            print(f"  {vi+1}/{len(op)}  (done {n_done}, skip {n_skip}, miss {n_miss})", flush=True)
    print(f"\nDONE: stored {n_done} videos ({n_skip} pre-existing, {n_miss} unresolved"
          + (f", {n_recovered} CORRUPT re-detected" if n_recovered else "") + f") -> {out}")


if __name__ == "__main__":
    main()

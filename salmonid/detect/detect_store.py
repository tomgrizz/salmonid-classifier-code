"""
In plain terms: this script is the first step of the fish-counting pipeline. For every video it runs the
detector (the part that finds fish and draws a box around each one) over the frames and saves those boxes
to a file on disk, so a later step can follow each fish across the frames and tally the counts. It does not
itself produce a headline number in the paper, but its output feeds the multi-year counts behind the paper's
abundance-index results (the annual abundance-index r2 of 0.96 and 0.98) and the daily-count figures.

Pass A of the DEIMv2 tracked-op / corpus eval (runs in .venv_deim): run the DEIMv2 single-model
detector over the frames of every video in a manifest and write a per-video detection store (pickle).
Pass B (_track_op_from_store.py, main venv) reads these + re-decodes frames for BoTSORT, so the tracker
exactly matches the two-stage baselines. Decoupled because .venv_deim's numpy 2.x can't host boxmot.

Store: one pickle per video, {frame_idx: {"boxes": Nx4 float32 (xyxy pixel), "probs": NxC}} (full-dist)
or {..., "scores", "labels"} (legacy one-hot) for frames with >=1 detection, plus meta. Resumable
(skips existing pickles). Shardable (run N workers, each --shard i --num-shards N).

Throughput (numerically SET-identical to the per-frame path -- the detection set is unchanged; retained
box/prob values differ only by GPU batch-kernel rounding ~0.5px / ~4e-4, which does not change counts):
  --frame-batch-size 32   ONE forward over a stack of frames (removes the per-frame .cpu() sync); ~1.6x.
  --footage-root F:/RiverWatcher   read from the NVMe mirror, not the G: HDD I/O ceiling.
  --channels-last         memory-layout hint (small/no gain on this ViT; off by default).

Multi-year correctness: key + resolve on the manifest's UNIQUE video_id (--key-mode video_id, default),
NOT <session>/<file> (which collides across date-ranges: Ganaraska 2023 = 1,425 colliding rels). The
2025 store + op eval were built with the old key -> pass --key-mode rel to re-read those.

  .venv_deim/Scripts/python.exe scripts/_deimv2_detect_store.py \
      --config <abs path>/single_stage/configs/deimv2/deimv2_fish_x_v25.yml \
      --ckpt <abs>/DEIMv2/outputs/deimv2_fish_x_v26/checkpoint0007.pth \
      --op-json G:/Projects/label_batches/corpus_2023_ganaraska.json \
      --thresh 0.3 --mask-header-px 40 --full-dist --frame-batch-size 32 --frame-skip 3 \
      --footage-root F:/RiverWatcher --out-dir G:/Projects/deimv2_x_v26_corpus2023/store
"""
import argparse, json, glob, pickle, hashlib, sys
from pathlib import Path
import numpy as np, cv2

from salmonid import config
from salmonid.detect import deimv2_infer as di  # single-model DEIMv2 detector wrapper (the Pass A engine)


def build_index(roots):
    """Legacy (--key-mode rel): glob footage keyed by (site, <session>/<file>). Kept for op eval."""
    idx = {}
    for site, root in roots.items():
        for f in glob.glob(f"{root}/**/*.mp4", recursive=True):
            p = Path(f); idx[(site, f"{p.parent.name}/{p.name}")] = f
    return idx


def store_key(site, rel):
    """Legacy (--key-mode rel) store filename: ``<site>__md5(session/file)[:12].pkl``.

    Kept only to re-read the 2025 store + op eval, which were built with this key before the
    multi-year video_id key existed (rel collides across date-ranges; see module docstring).
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Unique per video across all years (full path hash) -> no cross-date-range/cross-year collision."""
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def resolve_vid(footage_root, video_id):
    """video_id = '<site>/<yearfolder>/<daterange>/<session>/<file>.mp4'; footage_root holds <site> subdirs."""
    p = Path(footage_root) / video_id
    return str(p) if p.is_file() else None


def main():
    """Parse CLI args and run Pass A: detect over every video in --op-json, one pickle each.

    For each manifest entry: resolve the video path (via --key-mode video_id or the legacy rel
    index), skip if its store pickle already exists (resumable), decode frames (cv2 or decord),
    run the DEIMv2 detector in frame-batches, and pickle the per-frame boxes/probs plus meta into
    --out-dir. Side effects: GPU inference, file writes, progress prints. No return value.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--op-json", default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--out-dir", default=str(config.PROJECTS_ROOT / "deimv2_op_detstore"))
    ap.add_argument("--footage-root", default=str(config.FOOTAGE_ROOT_FAST),
                    help="Root holding <site> subdirs (F: NVMe mirror is faster than the G: HDD).")
    ap.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                    help="video_id (default) = unique per-video key (multi-year safe). rel = legacy "
                         "<site>__md5(session/file) for the 2025 store + op eval.")
    ap.add_argument("--mask-header-px", type=int, default=0,
                    help="Black out top N rows before detection. The v2.6 model was TRAINED with 40 -> "
                         "you MUST pass 40 (guarded below) unless --allow-unmasked.")
    ap.add_argument("--allow-unmasked", action="store_true",
                    help="Escape hatch to run with mask-header-px 0 (unmasked legacy models only).")
    ap.add_argument("--channels-last", action="store_true",
                    help="channels_last memory format (small/no gain on the DINOv3 ViT; off by default).")
    ap.add_argument("--fp16", action="store_true",
                    help="autocast the forward (~1.5-2.5x). Perturbs scores ~1e-3 -> corpus COUNT runs only "
                         "(count-r is scale-robust), NOT exact eval-AP/F1. Stored probs cast back to fp32.")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--frame-skip", type=int, default=1,
                    help="Detect only every Nth frame (1=every frame). 3 = the two-stage counting cadence.")
    ap.add_argument("--frame-batch-size", type=int, default=32,
                    help="Frames per forward. >1 removes the per-frame host sync (~1.6x). Detection SET is "
                         "identical to batch-1; retained values differ only by GPU batch-kernel rounding.")
    ap.add_argument("--decode-backend", choices=["cv2", "decord"], default="cv2",
                    help="decord-cpu is ~2x faster than cv2 on decode-bound (every-frame) runs.")
    ap.add_argument("--shard", type=int, default=0, help="This worker's shard index (0-based).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total shards -- run N parallel workers, each --shard i, to split the video list.")
    ap.add_argument("--full-dist", action="store_true",
                    help="store the FULL per-detection class distribution (NxC) as 'probs' instead of "
                         "the one-hot top-1 (recovers Coho buried under more-numerous Chinook one-hots).")
    args = ap.parse_args()

    # Guard: the v2.6 model is header-masked-trained. Refuse to silently run it on unmasked frames.
    if args.mask_header_px <= 0 and not args.allow_unmasked:
        sys.exit("ERROR: --mask-header-px is 0 but the v2.6 model was trained with a 40px header mask. "
                 "Pass --mask-header-px 40 (recommended) or --allow-unmasked to override.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    op = json.load(open(args.op_json))["videos"]
    if args.max_videos:
        op = op[:args.max_videos]
    if args.num_shards > 1:
        op = [v for i, v in enumerate(op) if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(op)} videos", flush=True)

    idx = None
    if args.key_mode == "rel":
        roots = {"Ganaraska": f"{args.footage_root}/Ganaraska", "Credit": f"{args.footage_root}/Credit"}
        print("indexing footage (rel mode)...", flush=True); idx = build_index(roots)
        print(f"indexed {len(idx)} clips; op videos: {len(op)}", flush=True)

    det = di.DEIMv2Detector(args.config, args.ckpt, mask_header_px=args.mask_header_px,
                            channels_last=args.channels_last, fp16=args.fp16)
    bs = max(1, args.frame_batch_size)
    n_done = n_miss = n_skip = 0
    for vi, v in enumerate(op):
        site = v["site"]; rel = v["video_rel"].replace("\\", "/"); vid = v.get("video_id")
        if args.key_mode == "video_id" and vid:
            kp = out / store_key_vid(vid); vp = resolve_vid(args.footage_root, vid)
        else:
            kp = out / store_key(site, rel); vp = idx.get((site, rel)) if idx is not None else None
        if kp.is_file():
            n_skip += 1; n_done += 1; continue
        if vp is None:
            n_miss += 1; continue

        # decode: cv2 (default) or decord-cpu (~2x faster; use on decode-bound every-frame runs)
        if args.decode_backend == "decord":
            from decord import VideoReader, cpu as _dcpu
            vr = VideoReader(vp, ctx=_dcpu(0))
            fps = float(vr.get_avg_fps() or 15); n_total = len(vr)
            _idxs = range(0, n_total, args.frame_skip) if args.frame_skip > 1 else range(n_total)

            def _frames():
                for fi in _idxs:
                    yield fi, vr[fi].asnumpy()          # RGB uint8 HxWx3
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
                        fi += 1; continue
                    yield fi, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    fi += 1
                cap.release()

        frames = {}; last_fi = -1
        buf_fi, buf_rgb = [], []

        def _flush():
            if not buf_rgb:
                return
            if args.full_dist:
                for j, (boxes, probs) in enumerate(det.predict_full_batch(buf_rgb, args.thresh)):
                    if len(boxes):
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
    print(f"\nDONE: stored {n_done} videos ({n_skip} pre-existing, {n_miss} unresolved) -> {out}")


if __name__ == "__main__":
    main()

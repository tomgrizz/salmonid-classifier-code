"""
In plain terms: this file takes the boxes drawn around fish in each video frame (from the
earlier detection step) and links them over time into one "track" per fish -- the step that
follows a single fish across the video frames -- so each real fish gets counted once instead of
many times. The per-fish tracks it produces are what the paper's multi-year fish-count results
are built on (the yearly abundance-index agreement of r-squared 0.96 and 0.98, and the Chinook
count totals).

Pass B: re-track a detect-store with BoT-SORT + OSNet re-ID -> per-track aggregates.

Pipeline position
-----------------
    data -> detect (Pass A) -> **track (this module, Pass B)** -> eval/count -> results -> figures

This is the *producer* half of the DEIMv2 tracked eval / corpus absolute-count path. Pass A
(_deimv2_detect_store.py, run in the DEIMv2 .venv_deim) writes a per-video *detection store*
(boxes + per-frame class prob-vecs, one pickle per video). This module runs in the BASE env
(boxmot 10.0.52 + OSNet, the same tracker as the two-stage op-presence-F1 0.894 baseline). It
reads that store, re-decodes the video frames so OSNet can compute re-ID appearance features,
runs BoT-SORT to stitch detections into tracks, and DUMPS per-track aggregates (boxes +
per-frame 5-class prob-vecs + the technician tech_set) to a single pickle. Scoring / counting
happens OFFLINE downstream: _score_tracked.py (op presence-F1) or
_corpus_countr_from_store.py --agg (absolute counts).

Why a second decode pass at all
-------------------------------
This is the GOLD-STANDARD absolute-count path: BoT-SORT gives ~1.02 tracks/fish vs the
store-only IoU tracker's ~1.48 (fixes the dense-Ganaraska over-count). OSNet re-ID needs the
actual pixels, so this re-decodes video -- a second decode pass beyond Pass A. It is optimized
by (a) reading from the F: NVMe mirror, (b) decord-fetching ONLY the stored detection frames
(skips the ~2/3 empty frames of a frame-skip-3 store), (c) optional fp16 re-ID.

Inputs:  a detect-store dir (one pickle/video), a video manifest JSON (site / video_rel /
         video_id / tech_species per video), and the raw footage under --footage-root.
Outputs: one aggregate pickle keyed by store-key -> {site, fw, fh, tech_set, tracks:
         {track_id: {boxes, pvs}}}. Feeds the manuscript multi-year absolute-count claim.
Env:     base Python (boxmot / OSNet / torch) -- NOT the DEIMv2 .venv_deim.
Re-ID weight: resolved from config.REID_WEIGHTS (boxmot auto-downloads if absent).

  # op eval (legacy key, cv2):
  python -m salmonid.detect.track_from_store --store-dir G:/Projects/deimv2_op_detstore \
      --op-json G:/Projects/label_batches/op_testset_v1.json --key-mode rel \
      --agg-out G:/Projects/deimv2_op_detstore/_agg.pkl --new-track-thresh 0.5
  # corpus year (video_id key, decord, fp16 re-ID):
  python -m salmonid.detect.track_from_store --store-dir G:/Projects/deimv2_x_v26_corpus2023/store \
      --op-json G:/Projects/label_batches/corpus_2023_ganaraska.json --key-mode video_id \
      --footage-root F:/RiverWatcher --decode-backend decord --reid-fp16 --frame-skip 3 \
      --agg-out G:/Projects/deimv2_x_v26_corpus2023/agg_gan2023.pkl
"""
import argparse, json, glob, pickle, hashlib
from pathlib import Path
import numpy as np, cv2, torch

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
TECH_NORM = {"chinook": "Chinook", "coho": "Coho", "atlantic": "Atlantic",
             "atlantic salmon": "Atlantic", "rainbow": "Rainbow Trout",
             "rainbow trout": "Rainbow Trout", "brown": "Brown Trout", "brown trout": "Brown Trout"}
# BoT-SORT OSNet re-ID weights (boxmot auto-downloads the pretrained model on first use if absent).
REID = str(config.REID_WEIGHTS)


def build_index(roots):
    """Map (site, "<parent>/<name>.mp4") -> absolute path for every mp4 under each site root.

    Used only in legacy ``--key-mode rel``: the op-testset manifest keys videos by a relative
    "<parent>/<name>" path, so we glob the footage once and build a lookup. Args: ``roots`` =
    {site: footage_dir}. Returns the lookup dict.
    """
    idx = {}
    for site, root in roots.items():
        for f in glob.glob(f"{root}/**/*.mp4", recursive=True):
            p = Path(f); idx[(site, f"{p.parent.name}/{p.name}")] = f
    return idx


def store_key(site, rel):
    """Legacy (``rel`` mode) store filename: ``<site>__<md5(rel)[:12]>.pkl`` (must match Pass A)."""
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Multi-year (``video_id`` mode) store filename: ``<md5(video_id)[:16]>.pkl``.

    Keying on the full video_id (not site+rel) avoids the cross-site rel-path collisions that
    corrupted the earlier corpus run; must match the key Pass A wrote the store under.
    """
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def resolve_vid(footage_root, video_id):
    """Return the absolute footage path for ``video_id`` under ``footage_root``, or None if missing."""
    p = Path(footage_root) / video_id
    return str(p) if p.is_file() else None


def _dets_from_df(df):
    """Build the boxmot detection matrix + parallel real 5-class prob vectors for one frame.

    ``df`` is one frame's slice of the detection store (or None for an empty frame). Returns
    ``(dets, probvecs)`` where ``dets`` is an ``[N, 6]`` float array of ``[x1,y1,x2,y2,score,cls]``
    rows (boxmot's expected input) and ``probvecs`` is the parallel list of the true 5-class
    probability vectors (kept separately because boxmot only carries a single winning ``cls``,
    but downstream counting votes over the full per-frame distribution). Two store formats:
    the full-distribution store (``probs`` present, argmax over the 5 species gives cls/score)
    and the legacy one-hot store (``labels``/``scores``, rebuilt into a one-hot prob vector).
    """
    dets, probvecs = [], []
    if df is not None and "probs" in df:                    # full-distribution store
        probs = df["probs"]
        for k in range(len(df["boxes"])):
            b = df["boxes"][k]; full = probs[k]
            spi = int(np.argmax(full[:5])); sc = float(full[spi])
            dets.append([b[0], b[1], b[2], b[3], sc, spi])
            probvecs.append(np.asarray(full[:5], np.float32))
    elif df is not None:                                    # legacy one-hot store
        for k in range(len(df["boxes"])):
            b = df["boxes"][k]; spi = int(df["labels"][k]); sc = float(df["scores"][k])
            if not (0 <= spi < 5):
                continue
            dets.append([b[0], b[1], b[2], b[3], sc, spi])
            pv = np.zeros(5, np.float32); pv[spi] = sc; probvecs.append(pv)
    return (np.array(dets, dtype=float) if dets else np.empty((0, 6))), probvecs


def main():
    """CLI entry point: track every video in the manifest and dump the aggregate pickle.

    Resolves store keys per ``--key-mode``, resumes from a partial ``--agg-out`` (skipping
    already-tracked videos, flushing every 300), runs BoT-SORT over the stored detections of
    each video, and writes ``{store_key: aggregate}`` to ``--agg-out``. Side effects: reads the
    store dir / footage / manifest; writes (and periodically re-writes) the output pickle.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", default=str(config.PROJECTS_ROOT / "deimv2_op_detstore"))
    ap.add_argument("--op-json", default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    ap.add_argument("--agg-out", required=True)
    ap.add_argument("--footage-root", default=str(config.FOOTAGE_ROOT_FAST))
    ap.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                    help="video_id (default, multi-year safe) or rel (legacy 2025 store + op eval).")
    ap.add_argument("--decode-backend", choices=["cv2", "decord"], default="cv2",
                    help="decord fetches ONLY the stored detection frames (skips empty frames of a "
                         "frame-skip store) -- big decode win at corpus scale. cv2 = sequential (legacy).")
    ap.add_argument("--reid-fp16", action="store_true",
                    help="fp16 OSNet re-ID (matches the two-stage v6 --fp16 corpus). ~identical tracks.")
    ap.add_argument("--new-track-thresh", type=float, default=0.5)
    ap.add_argument("--track-high-thresh", type=float, default=0.5)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--det-frames-only", action="store_true",
                    help="Only step BoTSORT on frames that HAVE detections (compressed timeline). REQUIRED "
                         "for a frame-skip store (implied by --decode-backend decord).")
    ap.add_argument("--frame-skip", type=int, default=3, help="the store's frame-skip (for the det-frames rate)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, help="run N parallel workers, each --shard i")
    args = ap.parse_args()

    from boxmot import BoTSORT
    store = Path(args.store_dir)
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
    dev = torch.device("cuda")
    use_decord = args.decode_backend == "decord"
    det_only = args.det_frames_only or use_decord
    if use_decord:
        from decord import VideoReader, cpu as _dcpu

    # resume: reuse a partial agg (per-shard file), skip videos already tracked, flush periodically
    agg = {}
    if Path(args.agg_out).is_file():
        try:
            agg = pickle.load(open(args.agg_out, "rb")); print(f"resume: {len(agg)} videos in {args.agg_out}", flush=True)
        except Exception:
            agg = {}
    n_done = n_miss = 0
    for vi, v in enumerate(op):
        site = v["site"]; rel = v["video_rel"].replace("\\", "/"); vid = v.get("video_id")
        if args.key_mode == "video_id" and vid:
            k = store_key_vid(vid); vp = resolve_vid(args.footage_root, vid)
        else:
            k = store_key(site, rel); vp = idx.get((site, rel)) if idx is not None else None
        if k in agg:
            n_done += 1; continue
        kp = store / k
        if not kp.is_file() or vp is None:
            n_miss += 1; continue
        with open(kp, "rb") as f:
            sd = pickle.load(f)
        det_frames = sd["frames"]; fps = sd.get("fps", 15)
        # BoT-SORT's frame_rate scales its track_buffer (how long a lost track survives). When we
        # only step on detection frames of a frame-skip-N store, the effective rate is fps/N.
        fr_rate = max(1, round(fps / args.frame_skip)) if det_only else max(1, round(fps))
        # Fresh tracker per video (no track leakage across clips); same thresholds as the two-stage
        # v6 corpus tracker so the counts are directly comparable.
        tracker = BoTSORT(model_weights=Path(REID), device=dev, fp16=args.reid_fp16,
                          track_high_thresh=args.track_high_thresh, track_low_thresh=0.3,
                          new_track_thresh=args.new_track_thresh,
                          track_buffer=25, match_thresh=0.9, frame_rate=fr_rate)
        tracks = {}; fw = fh = 0

        def _step(rgb, fi):
            """Feed one RGB frame + its stored detections to the tracker; append box+probvec per live track.

            Records frame size into the enclosing fw/fh; ``det_ind`` maps each tracker output back
            to its source detection so the true 5-class prob vector travels with the box.
            """
            nonlocal fw, fh
            fh, fw = rgb.shape[:2]
            dets, probvecs = _dets_from_df(det_frames.get(fi))
            for trk in tracker.update(dets, rgb):
                x1, y1, x2, y2, tid, conf, cls, det_ind = trk
                tid = int(tid); det_ind = int(det_ind)
                pv = probvecs[det_ind] if 0 <= det_ind < len(probvecs) else np.zeros(5, np.float32)
                t = tracks.setdefault(tid, {"boxes": [], "pvs": []})
                t["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
                t["pvs"].append(pv.tolist())

        if use_decord:                                      # fetch ONLY detection frames, in order
            det_fis = sorted(det_frames.keys())
            if det_fis:
                vr = VideoReader(vp, ctx=_dcpu(0))
                batch = vr.get_batch(det_fis).asnumpy()     # [K, H, W, 3] RGB
                for j, fi in enumerate(det_fis):
                    _step(batch[j], fi)
        else:                                               # cv2 sequential (legacy)
            cap = cv2.VideoCapture(vp); fi = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if det_only and det_frames.get(fi) is None:
                    fi += 1; continue
                _step(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), fi)
                fi += 1
            cap.release()

        tech_set = {TECH_NORM.get(s.strip().lower(), s) for s in v.get("tech_species", [])}
        tech_set = {s for s in tech_set if s in SPECIES}
        agg[k] = {
            "site": site, "fw": fw, "fh": fh, "tech_set": sorted(tech_set),
            "tracks": {tid: {"boxes": np.array(t["boxes"], np.float32),
                             "pvs": np.array(t["pvs"], np.float32)}
                       for tid, t in tracks.items()}}
        n_done += 1
        if (vi + 1) % 40 == 0:
            print(f"  {vi+1}/{len(op)} videos", flush=True)
        if (vi + 1) % 300 == 0:
            Path(args.agg_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.agg_out, "wb") as f:
                pickle.dump(agg, f)

    Path(args.agg_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.agg_out, "wb") as f:
        pickle.dump(agg, f)
    print(f"\ntracked {n_done} videos ({n_miss} unresolved) -> {args.agg_out}")
    print("score with: python scripts/_score_tracked.py --agg", args.agg_out)


if __name__ == "__main__":
    main()

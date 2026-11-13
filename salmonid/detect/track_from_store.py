"""In plain terms: the detection step leaves a list of boxes for each video frame, but it does
not know that the box in one frame and the box in the next are the same fish. This script
joins the boxes over time into "tracks", one track per fish, so that a fish seen in forty
frames is counted once and not forty times. It follows each fish by predicting where it will
be in the next frame and by comparing what it looks like. It saves, for every video, the list
of tracks with each track's positions and species scores. Counting comes afterwards.

Pass B: link stored detections into one track per fish (BoT-SORT), and save the tracks.

Pipeline stage: data -> detect ("Pass A", ``detect_store.py``) -> **track (this module,
"Pass B")** -> count (``salmonid.count.corpus_countr_from_store``) -> results -> figures.

Inputs : * a detect-store folder written by Pass A (one file per video);
         * the same video list (manifest JSON) that Pass A used;
         * the raw footage under ``--footage-root`` (needed to compare what fish look like;
           not needed with ``--tracker bytetrack``);
         * the appearance-model weights ``osnet_x0_25_msmt17.pt`` (``config.REID_WEIGHTS``).
Outputs: one "aggregate" pickle (``--agg-out``):
           {store_file_name: {"site", "fw", "fh", "tech_set",
                              "tracks": {track_id: {"boxes": M x 4, "pvs": M x 5}}}}
         "boxes" are the track's positions (x1, y1, x2, y2 in original-frame pixels), one row
         per processed frame; "pvs" are the five species scores of the detection matched at
         each of those frames.
Feeds  : every count in the manuscript (Tables 2-5, Figures 4-7; Sections 3.2-3.6) and, with
         ``--tracker bytetrack``, the tracker comparison in Section 3.4.
Tier   : 2 (needs the detect-stores, the raw footage and a GPU).

Environment: the BASE Python environment (requirements.txt), not the DEIMv2 environment. Every
published track was made with Python 3.11, boxmot 10.0.52 and numpy 2.4.4. The boxmot version
matters: this script uses the 10.x interface (``BoTSORT``, ``BYTETracker``), which later
boxmot releases replaced, and a different release may also link detections differently.

Run (from the repository root; this is the setting used for every published season):

  python -m salmonid.detect.track_from_store \\
      --store-dir <projects>/deimv2_x_v27_corpus2024/store \\
      --op-json <label_batches>/corpus_2024_credit.json --key-mode video_id \\
      --decode-backend decord --reid-fp16 --frame-skip 3 \\
      --agg-out <projects>/deimv2_x_v27_corpus2024/agg_credit_shard0.pkl --shard 0 --num-shards 2

  # tracker comparison of Section 3.4 (motion only, no video needed, no GPU, about 5 minutes).
  # This exact command reproduces the stored comparison aggregate track for track:
  python -m salmonid.detect.track_from_store --tracker bytetrack \\
      --store-dir <projects>/deimv2_x_v27_corpus2025/store \\
      --op-json <label_batches>/corpus_2025_ganaraska_full.json --key-mode video_id \\
      --frame-skip 3 --agg-out output/ablation/agg_bytetrack_2025_ganaraska.pkl

``python -m salmonid.count.run_corpus --steps track --dry-run`` prints the first command for
every season with all paths filled in.

How a track is formed (the settings behind the published counts)
----------------------------------------------------------------
The tracker is BoT-SORT as implemented in the boxmot library, version 10.0.52. Each detection
gets one confidence value here: its highest score among the FIVE species (the "Small Salmonid"
score is ignored from this point on).

* A NEW track can only start from a detection scoring above 0.50, and it is kept only if
  another detection above 0.50 matches it in the very next processed frame. One isolated
  detection never becomes a track. This is what stops a flash of light or a leaf from being
  counted as a fish.
* Detections scoring between 0.30 and 0.50 cannot start a track. They can only extend a
  track that is already being followed. A fish that turns sideways or passes behind another
  for a moment is therefore not lost.
* A track that finds no matching detection is kept "alive" for a short while in case the
  fish reappears, then closed. The time is set from the video's frame rate, and the frame
  rate differs between seasons: 2 processed frames for the 2023-2025 videos (about 10
  frames per second), 6 for 2020-2022 Ganaraska (25) and 8 for 2020-2022 Credit (30). In
  seconds this is roughly the same everywhere (0.6 to 0.8 s).
* Matching uses two cues: predicted position (a fish keeps moving roughly as it was) and
  appearance (a small network, OSNet, that summarises what the boxed image looks like; it
  was trained on photographs of people and is used unchanged).
* Only frames that contain at least one detection are given to the tracker, in order. Empty
  stretches are skipped, so the tracker's clock counts frames with fish, not seconds.
* A fresh tracker is started for every video. Tracks never continue from one video file to
  the next. Where the recorder cut one passage into several short files (the recovered 2020
  Ganaraska footage), one fish can therefore produce more than one track.

Two details that matter for the counting step
---------------------------------------------
* The positions saved for a track are the tracker's smoothed estimates of where the fish
  was, not the raw detection boxes.
* The tracker does not report a track's first detection (it waits for the confirmation in
  the next frame). A saved track therefore starts at the fish's second detection, and a track
  with three saved positions means the fish was detected in at least four processed frames.
  The one exception is a fish already in view in the first processed frame of a video: its
  track is reported from that first frame.

The tracker comparison (``--tracker bytetrack``)
------------------------------------------------
ByteTrack follows fish by predicted position only; it never looks at the picture. Running
it on the same stored detections shows what the appearance comparison adds. It is given the
same start threshold (0.50) and matching threshold as BoT-SORT. Two settings differ:

* boxmot's ByteTrack lets detections scoring down to 0.10 extend a track, where BoT-SORT
  stops at 0.30. The stored detections all scored above 0.30 on their best of six classes,
  so this only affects boxes whose best *species* score is between 0.10 and 0.30 (boxes kept
  because of their "Small Salmonid" score).
* How the published comparison was computed: it was run without ``--decode-backend decord``.
  In that case the script takes the tracker's frame rate to be the video's full rate
  (10 per second in 2025) instead of one third of it, so a lost track was kept alive for
  8 processed frames, against 2 in the BoT-SORT run. Adding ``--decode-backend decord`` gives ByteTrack
  the same 2 frames. Checked for the deposit on the 2025 Ganaraska detections (35,238
  videos; Chinook net count against the technician's 17,638):
      BoT-SORT (published counts)            48,721 tracks   18,267   +3.6%
      ByteTrack as published (8 frames)      48,850 tracks   18,247   +3.5%
      ByteTrack with matched 2 frames        49,951 tracks   18,234   +3.4%
  The net count is almost the same in all three. The near-equal number of tracks holds only
  for the longer keep-alive time.

Safety checks ported from the research repository
-------------------------------------------------
* A store file that cannot be read is skipped, counted and reported. Before this check one
  damaged file stopped a whole run part-way and silently dropped 679 videos from the 2025
  Credit River season.
* The appearance-model weights are looked up as an absolute path and the run stops with a
  clear message if they are missing. Before, a missing file made the library try to download
  it, fail, and end the run in seconds while the log still said "done"; three finished
  seasons lost their tracking pass that way and had to be re-tracked.

Which research-repository version this is: the lineage that carries these checks and the
ByteTrack option (the "v2.7 corpus pipeline" commit of 5 August 2026). The other development
branch has the same BoT-SORT code without them; for a readable store the two give the same
tracks.
"""
import argparse, hashlib, glob, json, pickle
from pathlib import Path
import numpy as np, cv2, torch

from salmonid import config

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # same order as salmonid.config.SPECIES
TECH_NORM = {"chinook": "Chinook", "coho": "Coho", "atlantic": "Atlantic",
             "atlantic salmon": "Atlantic", "rainbow": "Rainbow Trout",
             "rainbow trout": "Rainbow Trout", "brown": "Brown Trout", "brown trout": "Brown Trout"}


def reid_weights():
    """Absolute path of the OSNet appearance-model weights; stops clearly if the file is missing.

    A relative ``config.REID_WEIGHTS`` is taken relative to the repository root, never the
    working directory, so the result does not depend on where the command is launched from.
    If the file's fingerprint differs from the one used for the published run, a warning is
    printed (a different file would change which detections are joined into tracks).
    """
    p = Path(config.REID_WEIGHTS)
    if not p.is_absolute():
        p = config.REPO_ROOT / p
    p = config.require(p, "the OSNet appearance-model weights used by BoT-SORT (osnet_x0_25_msmt17.pt)",
                       "SALMONID_REID_WEIGHTS")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != config.REID_WEIGHTS_SHA256:
        print(f"WARNING: {p.name} is not the file used for the published run "
              f"(sha256 {digest[:12]}..., expected {config.REID_WEIGHTS_SHA256[:12]}...). "
              "Tracks may differ from the published ones.", flush=True)
    return p


def build_index(roots):
    """List every video under each river folder, keyed by (river, "<last folder>/<file>").

    Only used with ``--key-mode rel`` (old naming). If the same "<last folder>/<file>" occurs
    twice within a river, the last one found is kept.
    """
    idx = {}
    for site, root in roots.items():
        for f in glob.glob(f"{root}/**/*.mp4", recursive=True):
            p = Path(f); idx[(site, f"{p.parent.name}/{p.name}")] = f
    return idx


def store_key(site, rel):
    """Store file name in ``rel`` mode (must match Pass A): ``<river>__<hash of "folder/file">.pkl``."""
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Store file name in ``video_id`` mode (must match Pass A): a hash of the video's full path."""
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def resolve_vid(footage_root, video_id):
    """Return the video file for ``video_id`` under ``footage_root``, or None if it is not there."""
    p = Path(footage_root) / video_id
    return str(p) if p.is_file() else None


def _dets_from_df(df):
    """Turn one frame's stored detections into the table the tracker expects.

    Returns ``(dets, probvecs)``. ``dets`` has one row per detection: x1, y1, x2, y2, a single
    confidence, and a class number. The confidence is the detection's highest score among the
    five species and the class is that species. ``probvecs`` keeps the full five species
    scores of each detection, because the tracker itself carries only one class and the
    counting step needs all five.

    How the published numbers were computed: the sixth score ("Small Salmonid") is dropped
    here (``[:5]``). A juvenile fish that the model scored mainly as "Small Salmonid" is not
    removed; it is carried forward under whichever adult species had its next-highest score,
    provided that score is high enough to form a track (see the module notes).
    """
    dets, probvecs = [], []
    if df is not None and "probs" in df:                    # store written with --full-dist (all published runs)
        probs = df["probs"]
        for k in range(len(df["boxes"])):
            b = df["boxes"][k]; full = probs[k]
            spi = int(np.argmax(full[:5])); sc = float(full[spi])
            dets.append([b[0], b[1], b[2], b[3], sc, spi])
            probvecs.append(np.asarray(full[:5], np.float32))
    elif df is not None:                                    # older store holding only the winning class
        for k in range(len(df["boxes"])):
            b = df["boxes"][k]; spi = int(df["labels"][k]); sc = float(df["scores"][k])
            if not (0 <= spi < 5):
                continue
            dets.append([b[0], b[1], b[2], b[3], sc, spi])
            pv = np.zeros(5, np.float32); pv[spi] = sc; probvecs.append(pv)
    return (np.array(dets, dtype=float) if dets else np.empty((0, 6))), probvecs


def main():
    """Track every video in the list and write the aggregate file."""
    ap = argparse.ArgumentParser(description="Pass B: link stored detections into one track per fish.")
    ap.add_argument("--store-dir", required=True, help="Detect-store folder written by Pass A.")
    ap.add_argument("--op-json", required=True, help="The video list (manifest JSON) Pass A used.")
    ap.add_argument("--agg-out", required=True, help="Output aggregate pickle (one per shard).")
    ap.add_argument("--footage-root", default=str(config.FOOTAGE_ROOT_FAST),
                    help="Folder holding <river> subfolders of raw video (default: the fast copy).")
    ap.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                    help="How store files are named; must match Pass A (all seasons: video_id).")
    ap.add_argument("--decode-backend", choices=["cv2", "decord"], default="cv2",
                    help="decord reads only the frames that have detections (used for all published "
                         "seasons). cv2 reads every frame in turn.")
    ap.add_argument("--reid-fp16", action="store_true",
                    help="Run the appearance model in half precision (used for all published seasons).")
    ap.add_argument("--new-track-thresh", type=float, default=0.5,
                    help="Minimum confidence for a detection to start a new track (published: 0.5).")
    ap.add_argument("--track-high-thresh", type=float, default=0.5,
                    help="Detections above this are matched first; lower ones (down to 0.3) can only "
                         "extend an existing track (published: 0.5).")
    ap.add_argument("--max-videos", type=int, default=0, help="Stop after N videos (0 = all); for testing.")
    ap.add_argument("--det-frames-only", action="store_true",
                    help="Give the tracker only frames that have detections. Always on with "
                         "--decode-backend decord. Needed for a store made with --frame-skip > 1.")
    ap.add_argument("--frame-skip", type=int, default=3,
                    help="The --frame-skip that Pass A used for this store (published seasons: 3). "
                         "It sets how long a lost track is kept alive.")
    ap.add_argument("--shard", type=int, default=0, help="Which part of the list this copy handles (0-based).")
    ap.add_argument("--num-shards", type=int, default=1, help="Split the list into N parts for N copies.")
    ap.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort",
                    help="botsort (default; position + appearance; used for every count) or bytetrack "
                         "(position only; the comparison in manuscript Section 3.4). ByteTrack never "
                         "opens the video and needs no GPU.")
    ap.add_argument("--frame-size", nargs=2, type=int, default=[1280, 960], metavar=("W", "H"),
                    help="Frame width and height written into each aggregate record. Only used with "
                         "--tracker bytetrack, which never opens the video. Not used for counting.")
    args = ap.parse_args()

    bytetrack = args.tracker == "bytetrack"
    config.require(args.store_dir, "the detect-store folder (--store-dir) written by Pass A",
                   "SALMONID_PROJECTS_ROOT")
    config.require(args.op_json, "the video list (manifest JSON) given as --op-json", "SALMONID_PROJECTS_ROOT")
    REID = None if bytetrack else reid_weights()
    from boxmot import BoTSORT, BYTETracker
    store = Path(args.store_dir)
    op = json.load(open(args.op_json))["videos"]
    if args.max_videos:
        op = op[:args.max_videos]
    if args.num_shards > 1:
        op = [v for i, v in enumerate(op) if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: {len(op)} videos", flush=True)

    idx = None
    if args.key_mode == "rel" and not bytetrack:
        roots = {"Ganaraska": f"{args.footage_root}/Ganaraska", "Credit": f"{args.footage_root}/Credit"}
        print("indexing footage (rel mode)...", flush=True); idx = build_index(roots)
    dev = torch.device("cuda") if not bytetrack else None
    use_decord = args.decode_backend == "decord" and not bytetrack
    # "det_only" = the tracker is stepped only on frames that have detections.
    det_only = args.det_frames_only or args.decode_backend == "decord"
    if use_decord:
        from decord import VideoReader, cpu as _dcpu
    if bytetrack:
        print("tracker=bytetrack: motion-only, no re-ID -> frames are never decoded", flush=True)

    # Resume: if the output file already exists, keep the videos it holds and only track the rest.
    agg = {}
    if Path(args.agg_out).is_file():
        try:
            agg = pickle.load(open(args.agg_out, "rb")); print(f"resume: {len(agg)} videos in {args.agg_out}", flush=True)
        except Exception:
            # An unreadable partial output is discarded and the shard starts again from nothing.
            print(f"WARNING: could not read existing {args.agg_out}; starting this shard from scratch", flush=True)
            agg = {}
    n_done = n_miss = n_bad = 0
    for vi, v in enumerate(op):
        site = v["site"]; rel = v["video_rel"].replace("\\", "/"); vid = v.get("video_id")
        if args.key_mode == "video_id" and vid:
            k = store_key_vid(vid); vp = resolve_vid(args.footage_root, vid)
        else:
            k = store_key(site, rel); vp = idx.get((site, rel)) if idx is not None else None
        if k in agg:
            n_done += 1; continue
        kp = store / k
        # A video with no store file, or whose footage cannot be found, is counted as
        # "unresolved" and left out of the aggregate (and so out of every later count).
        # ByteTrack never opens the video, so missing footage is not a problem for it.
        if not kp.is_file() or (vp is None and not bytetrack):
            n_miss += 1; continue
        # One unreadable store file used to stop the whole shard at that point, with no error
        # visible downstream because the aggregate simply ended early (679 videos of the 2025
        # Credit River season were lost that way in the original run). Skip it, count it, and
        # report it at the end instead.
        try:
            with open(kp, "rb") as f:
                sd = pickle.load(f)
        except Exception as e:                                      # noqa: BLE001
            n_bad += 1
            print(f"  CORRUPT STORE {kp.name}: {type(e).__name__}: {e} -- skipped", flush=True)
            continue
        det_frames = sd["frames"]; fps = sd.get("fps", 15)
        # The tracker keeps a lost track alive for track_buffer * frame_rate / 30 frames. Because
        # it only sees every 3rd frame, its frame rate is the video's rate divided by 3: at
        # 10 frames per second that is round(10/3) = 3, giving int(3/30 * 25) = 2 processed frames
        # (25 frames per second: 8 -> 6 frames; 30 frames per second: 10 -> 8 frames).
        fr_rate = max(1, round(fps / args.frame_skip)) if det_only else max(1, round(fps))
        if bytetrack:
            # Same start threshold, matching threshold and keep-alive time as the BoT-SORT run,
            # so the main difference between the two runs is the appearance comparison.
            tracker = BYTETracker(track_thresh=args.track_high_thresh, match_thresh=0.9,
                                  track_buffer=25, frame_rate=fr_rate)
        else:
            # A fresh tracker per video: tracks never carry over from one file to the next.
            # track_low_thresh 0.3  -> detections of 0.30-0.50 may only extend a live track
            # track_high/new 0.5    -> only detections above 0.50 are matched first / start a track
            # match_thresh 0.9      -> a loose position match is accepted (boxes need overlap only slightly)
            tracker = BoTSORT(model_weights=Path(REID), device=dev, fp16=args.reid_fp16,
                              track_high_thresh=args.track_high_thresh, track_low_thresh=0.3,
                              new_track_thresh=args.new_track_thresh,
                              track_buffer=25, match_thresh=0.9, frame_rate=fr_rate)
        tracks = {}; fw = fh = 0

        def _step(rgb, fi):
            """Give the tracker one frame and its stored detections; record each live track's box and scores."""
            nonlocal fw, fh
            fh, fw = rgb.shape[:2]
            dets, probvecs = _dets_from_df(det_frames.get(fi))
            for trk in tracker.update(dets, rgb):
                x1, y1, x2, y2, tid, conf, cls, det_ind = trk
                tid = int(tid); det_ind = int(det_ind)
                # det_ind says which detection this track was matched to in this frame, so the
                # five species scores of that detection travel with the track.
                pv = probvecs[det_ind] if 0 <= det_ind < len(probvecs) else np.zeros(5, np.float32)
                t = tracks.setdefault(tid, {"boxes": [], "pvs": []})
                t["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
                t["pvs"].append(pv.tolist())

        if bytetrack:                       # position only: step through the stored detections, no pictures
            fw, fh = args.frame_size
            for fi in sorted(det_frames.keys()):
                dets, probvecs = _dets_from_df(det_frames.get(fi))
                for trk in tracker.update(dets, None):
                    x1, y1, x2, y2, tid, conf, cls, det_ind = trk
                    tid = int(tid); det_ind = int(det_ind)
                    pv = probvecs[det_ind] if 0 <= det_ind < len(probvecs) else np.zeros(5, np.float32)
                    t = tracks.setdefault(tid, {"boxes": [], "pvs": []})
                    t["boxes"].append([float(x1), float(y1), float(x2), float(y2)])
                    t["pvs"].append(pv.tolist())
        elif use_decord:                                    # read ONLY the frames that have detections, in order
            det_fis = sorted(det_frames.keys())
            if det_fis:
                vr = VideoReader(vp, ctx=_dcpu(0))
                batch = vr.get_batch(det_fis).asnumpy()     # [K, height, width, 3] RGB
                for j, fi in enumerate(det_fis):
                    _step(batch[j], fi)
        else:                                               # cv2: read every frame in turn
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

        # Species the technician recorded for this video, if the manifest carries them
        # (used by a separate per-video check, not by the counts).
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
        if (vi + 1) % 300 == 0:                             # save progress so an interrupted run can resume
            Path(args.agg_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.agg_out, "wb") as f:
                pickle.dump(agg, f)

    Path(args.agg_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.agg_out, "wb") as f:
        pickle.dump(agg, f)
    print(f"\ntracked {n_done} videos ({n_miss} unresolved, {n_bad} corrupt-store) -> {args.agg_out}")
    if n_bad:
        print(f"WARNING: {n_bad} store file(s) unreadable and skipped. Re-run detect-store for those "
              f"videos, or this cell's counts are short by that many.")
    if n_miss:
        print(f"WARNING: {n_miss} video(s) had no store file or no footage and are NOT in the aggregate. "
              f"Check them with: python -m salmonid.detect.audit_store_integrity")
    print("count with: python -m salmonid.count.corpus_countr_from_store --agg", args.agg_out)


if __name__ == "__main__":
    main()

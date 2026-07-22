"""
In plain terms: this script tallies how many fish of each species passed the camera each day and
which way they were swimming (upstream vs downstream), then lines those daily totals up against the
counts a human technician recorded for the same days. It works from detections the model already
saved earlier -- the boxes it drew around each fish -- so it follows each fish across the video
frames (to avoid counting the same fish twice) without re-watching any video. The result is the
daily model-vs-technician comparison table behind the paper's day-by-day agreement numbers (daily
correlation r about 0.94-0.99) and the count-reliability check used to decide the model is trustworthy
enough to deploy.

SINGLE-STAGE counting adapter: detect-store -> daily model-vs-tech CSV.

Pipeline position
-----------------
    data -> detect (Pass A) -> **track + count (this module)** -> eval/count -> results -> figures

count-r deploy gate for a SINGLE-STAGE DEIMv2 detect-store (v2.6). Turns a corpus detect-store
into the daily model-vs-tech CSV that the count-r gate consumes, using the SAME counting method
as the two-stage full-year path (salmonid.count.corpus_fullyear_vs_tech): min_frames=3, an
ASYMMETRIC net rule (up>=50 / down>=300 px |delta_x|), file-mtime dating (optionally overridden by
a true-capture-date map), and the validated ``load_tech`` technician-truth loader.

The detect-store carries per-frame boxes+probs but NO track_id, so this module TRACKS here -- a
lightweight greedy-IoU tracker (or an optional ByteTrack) over the stored detection frames, with
NO video re-decode (the frames were already run through the model in Pass A). count-r is a
correlation, robust to a constant tracking-scale offset, so store-only tracking is sufficient for
the gate; gold-standard ABSOLUTE counts instead pass pre-computed BoT-SORT tracks via --agg.

Inputs:  a detect-store dir (one pickle/video: {frames: {fi: {boxes, probs}}, fps}); a video
         manifest JSON (site / video_rel / video_id per video); the year's technician truth.
Outputs: a long daily CSV (date, site, species, model, tech) -- one of the manuscript's daily
         model-vs-tech tables -- and, optionally, a per-track diagnostic CSV.
Env:     base Python (numpy only) -- NOT the DEIMv2 .venv_deim; runs offline on the stored dets.
Claim:   feeds the daily model-vs-tech agreement (daily-r) and the count-r deploy gate.

  python -m salmonid.count.corpus_countr_from_store \
      --store-dir <detect-store>/store \
      --videos-json <label_batches>/corpus_2025_both.json \
      --out-csv <out-dir>/2025_daily_model_vs_tech.csv
  # then gate the CSV: --daily-csv <that csv> --min-days 8 --r-gate 0.90
"""
from __future__ import annotations

import argparse, collections, csv, datetime, glob, hashlib, json, os, pickle
from pathlib import Path
import numpy as np

from salmonid import config
from salmonid.count.corpus_fullyear_vs_tech import load_tech  # reuse the validated tech loader

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}          # +delta_x = upstream; mirrors salmonid.config.UP_SIGN
VIDEO_ROOTS = config.video_roots()                    # {site: <FOOTAGE_ROOT>/<site>} for mtime-based dating
DATE_MAP = {}                                          # optional video_id -> true capture date (overrides mtime)


def store_key(site, rel):
    """Legacy store-key: ``<site>__<md5(rel)[:12]>.pkl`` (2025 store / op eval; --key-mode rel).

    Args:
        site: river name (``Ganaraska`` / ``Credit``).
        rel:  video path relative to the site root (forward-slash normalised).
    Returns: the pickle filename the detect-store used for this video.
    """
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Unique per-video key (full path hash) -- multi-year safe. Matches salmonid.detect.detect_store."""
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def _load_tech_for_year(year, tech_dir):
    """Year-aware technician truth -> (act_d, net_d, act_t, net_t, uncl), keyed (site,date)->{sp_idx:cnt}.
    2025 = per-clip export app (corpus_fullyear_vs_tech, activity+net); 2024 = Vaki AbsDb + Ganny_Total
    (corpus_2024_vs_tech, activity+net); 2020-2023 = daily-NET Vaki only (corpus_historical_vs_tech,
    net only -> activity tech is empty). Reflects that exact per-clip matching exists only for 2025."""
    if year == 2024:
        from salmonid.count.corpus_2024_vs_tech import load_tech as _lt
        return _lt()
    if 2018 <= year <= 2023:                          # FINAL_CREDIT_<year> + Ganny_Total cover 2018-2023
        from salmonid.count.corpus_historical_vs_tech import load_tech as _lt
        net_d, net_t, uncl, act_d, act_t = _lt(year, with_activity=True)   # activity = Σ|CNT| (tech down <1%)
        return act_d, net_d, act_t, net_t, uncl
    return load_tech(Path(tech_dir))                      # 2025 (default)


def _iou(b, boxes):
    """Vectorised IoU of one box ``b`` (x1,y1,x2,y2) against an ``Nx4`` array ``boxes``.

    Returns a length-N array of IoUs (empty array if ``boxes`` is empty). Used by the greedy
    store tracker to associate each detection with the most-overlapping active track.
    """
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(b[0], boxes[:, 0]); y1 = np.maximum(b[1], boxes[:, 1])
    x2 = np.minimum(b[2], boxes[:, 2]); y2 = np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ar = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + ar - inter, 1e-6, None)


def track_store(frames, iou_thr=0.3, buffer=8):
    """Greedy IoU tracker over stored detection frames. Returns [ {boxes:[Mx4], pvs:[Mx5], fis:[M]} ] per track.
    `pvs` keeps the FULL per-detection distribution (5 species) so voting + confidence/margin diagnostics use real
    probabilities; `fis` keeps each detection's frame index so proximity (nearest co-detection) is recoverable."""
    tracks = []          # each: {"boxes":[], "pvs":[], "fis":[], "last_fi":int}
    for fi in sorted(frames):
        d = frames[fi]
        boxes = np.asarray(d["boxes"], np.float32); probs = np.asarray(d["probs"], np.float32)
        if len(boxes) == 0:
            continue
        active = [t for t in tracks if fi - t["last_fi"] <= buffer]
        used = set()
        for k in range(len(boxes)):
            b = boxes[k]
            cand = [(t, _iou(b, np.array([t["boxes"][-1]]))[0]) for t in active if id(t) not in used]
            cand = [(t, v) for t, v in cand if v >= iou_thr]
            if cand:
                t = max(cand, key=lambda x: x[1])[0]; used.add(id(t))
            else:
                t = {"boxes": [], "pvs": [], "fis": [], "last_fi": fi}; tracks.append(t)
            t["boxes"].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            t["pvs"].append(np.asarray(probs[k][:5], np.float32)); t["fis"].append(int(fi)); t["last_fi"] = fi
    return tracks


_BYTETRACK = {}   # fps -> reusable tracker is unsafe (stateful); build fresh per video


def track_store_bytetrack(frames, fps, buffer_frames=30, act_thr=0.25):
    """ByteTrack (Kalman motion + two-stage association) over the stored boxes -- NO video re-decode. Fixes the
    greedy-IoU fragmentation (1.48 tracks/fish on dense Ganaraska) by predicting motion across the frame-skip gap
    and holding lost tracks in a buffer. Returns the same {boxes, pvs, fis} per-track structure as track_store()."""
    import supervision as sv
    tr = sv.ByteTrack(frame_rate=max(1, int(round(fps or 10))),
                      lost_track_buffer=buffer_frames, track_activation_threshold=act_thr)
    acc = {}
    for fi in sorted(frames):
        d = frames[fi]
        boxes = np.asarray(d["boxes"], np.float32); probs = np.asarray(d["probs"], np.float32)
        if len(boxes) == 0:
            continue
        sp5 = probs[:, :5]
        det = sv.Detections(xyxy=boxes.copy(), confidence=sp5.max(1).astype(np.float32),
                            class_id=sp5.argmax(1).astype(int), data={"oi": np.arange(len(boxes))})
        det = tr.update_with_detections(det)
        for k in range(len(det)):
            tid = det.tracker_id[k]
            if tid is None:
                continue
            oi = int(det.data["oi"][k]); b = det.xyxy[k]
            t = acc.setdefault(int(tid), {"boxes": [], "pvs": [], "fis": []})
            t["boxes"].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            t["pvs"].append(np.asarray(probs[oi][:5], np.float32)); t["fis"].append(int(fi))
    return list(acc.values())


FRAME_W, FRAME_H = 1280.0, 960.0          # RiverWatcher tunnel frame (boxes stored in this pixel space)


def _track_diag(t, frames):
    """Per-track diagnostic row from a tracked detection sequence (no video re-decode). Captures the confidence /
    runner-up margin / geometry / proximity a post-hoc review needs (chrome-confusion + partial + overlap analysis)."""
    boxes = np.asarray(t["boxes"], np.float32); pvs = np.asarray(t["pvs"], np.float32); fis = t["fis"]
    mean_pv = pvs.mean(0); order = np.argsort(mean_pv)[::-1]
    sp = int(order[0]); margin = float(mean_pv[order[0]] - mean_pv[order[1]])
    per_frame_conf = pvs.max(1)                                   # winning-class score each frame
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (FRAME_W * FRAME_H)   # frac of frame
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0; cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    # proximity: min centre-distance to ANY other simultaneous detection over the track's frames (px); and the
    # max number of fish co-present in any of its frames. small min_neighbor_px = overlapping/co-migrating fish.
    min_neigh = np.inf; max_cooccur = 1
    for i, fi in enumerate(fis):
        ob = np.asarray(frames[fi]["boxes"], np.float32)
        max_cooccur = max(max_cooccur, len(ob))
        if len(ob) < 2:
            continue
        ocx = (ob[:, 0] + ob[:, 2]) / 2.0; ocy = (ob[:, 1] + ob[:, 3]) / 2.0
        dist = np.sqrt((ocx - cx[i]) ** 2 + (ocy - cy[i]) ** 2)
        dist[dist < 1e-3] = np.inf                               # exclude self (distance 0)
        if np.isfinite(dist.min()):
            min_neigh = min(min_neigh, float(dist.min()))
    return {
        "n_frames": len(boxes), "pred_idx": sp, "pred_species": SPECIES[sp],
        "conf_top": round(float(mean_pv[sp]), 4), "conf2_species": SPECIES[int(order[1])],
        "conf2": round(float(mean_pv[order[1]]), 4), "margin": round(margin, 4),
        "conf_frame_mean": round(float(per_frame_conf.mean()), 4),
        "conf_frame_max": round(float(per_frame_conf.max()), 4),
        "area_mean": round(float(area.mean()), 5), "area_max": round(float(area.max()), 5),
        "cy_mean": round(float(cy.mean() / FRAME_H), 4),
        "min_neighbor_px": (round(min_neigh, 1) if np.isfinite(min_neigh) else ""),
        "max_cooccur": int(max_cooccur),
    }


def _resolve(video_id):
    """Map a ``<site>/<relpath>`` video_id back to an on-disk footage path, or None.

    Tries each configured river root (VIDEO_ROOTS) and returns the first existing file. Used
    only for mtime-based dating when a video is not covered by an explicit --date-map.
    """
    for site, root in VIDEO_ROOTS.items():
        if video_id.startswith(site + "/"):
            p = Path(root) / video_id[len(site) + 1:]
            if p.is_file():
                return str(p)
    return None


def main():
    """Build the daily model-vs-tech CSV from a detect-store (the module entry point).

    Reads the video manifest and store, tracks each video's stored detections (IoU / ByteTrack,
    or reuses BoT-SORT tracks via --agg), dates each video (--date-map else file mtime), applies
    the min_frames + asymmetric-net counting rule, joins the year's technician truth, and writes
    the long (date, site, species, model, tech) CSV (+ optional per-track diagnostic CSV).
    Side effects: prints a summary and writes --out-csv (and --diag-csv if given).
    """
    global DATE_MAP
    a = _parse_args()
    if a.date_map:
        DATE_MAP = json.load(open(a.date_map))
        print(f"loaded date-map: {len(DATE_MAP)} video_id -> capture date (overrides mtime dating)")
    vids = json.load(open(a.videos_json))["videos"]
    if a.key_mode == "video_id":
        k2vid = {store_key_vid(v["video_id"]): v["video_id"] for v in vids if v.get("video_id")}
    else:                                                     # legacy: 2025 store + op eval
        k2vid = {store_key(v["site"], v["video_rel"].replace("\\", "/")): v.get("video_id") for v in vids}
    store = Path(a.store_dir)

    # --agg: use pre-computed BoTSORT tracks (from salmonid.detect.track_from_store shards) instead of IoU-tracking the store.
    agg = {}
    if a.agg:
        for gf in glob.glob(a.agg):
            try:
                for k, rec in pickle.load(open(gf, "rb")).items():
                    agg[k] = rec.get("tracks", {})
            except Exception:
                pass
        print(f"loaded {len(agg)} tracked videos from agg '{a.agg}'")

    # model daily: (site,date) -> species -> activity count / net
    m_act = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net = collections.defaultdict(lambda: collections.defaultdict(int))
    diag_rows = []                                   # per-track diagnostics (only if --diag-csv; store mode only)
    n_ok = n_nostore = n_nodate = 0
    for key, vid in k2vid.items():
        frames = None
        if a.agg:
            rec = agg.get(key)
            if rec is None:
                n_nostore += 1; continue
            tracks = [{"boxes": np.asarray(t["boxes"], float), "pvs": np.asarray(t["pvs"], float),
                       "fis": list(range(len(t["boxes"])))} for t in rec.values()]
        else:
            kp = store / key
            if not kp.is_file():
                n_nostore += 1; continue
            try:
                sd = pickle.load(open(kp, "rb")); frames = sd.get("frames", {})
            except Exception:
                n_nostore += 1; continue
            if a.tracker == "bytetrack":
                tracks = list(track_store_bytetrack(frames, sd.get("fps", 10)))
            else:
                tracks = list(track_store(frames))
        if DATE_MAP and vid in DATE_MAP:
            date = DATE_MAP[vid]                          # true capture date (e.g. RiverWatcher .log), not mtime
        else:
            vp = _resolve(vid) if vid else None
            if not vp:
                n_nodate += 1; continue
            date = datetime.date.fromtimestamp(os.path.getmtime(vp)).isoformat()
        site = vid.split("/", 1)[0]
        for ti, t in enumerate(tracks):
            if len(t["boxes"]) < a.min_frames:            # min_frames=3: drop 1-2 frame flickers (re-derived, not tracker-filtered)
                continue
            pvs = np.asarray(t["pvs"]); sp = int(np.argmax(pvs.mean(0)[:5]))   # species = argmax of mean 5-class prob-vec (Small Salmonid excluded)
            m_act[(site, date)][sp] += 1                  # activity = one count per surviving track
            cx = (np.asarray(t["boxes"])[:, 0] + np.asarray(t["boxes"])[:, 2]) / 2.0
            # delta_x = box-centroid drift, mean of last-w minus first-w frames (w<=3); sign * UP_SIGN[site] gives travel direction.
            w = min(3, len(cx)); dx = float(cx[-w:].mean() - cx[:w].mean())
            sgn = dx * UP_SIGN[site]
            # ASYMMETRIC net rule (the family optimum; matches the technicians' convention): liberal upstream
            # (|dx| >= up_thresh, default 50px) vs strict downstream (|dx| >= down_thresh, default 300px, kills milling).
            nc = 1 if (sgn > 0 and abs(dx) >= a.up_thresh) else (-1 if (sgn < 0 and abs(dx) >= a.down_thresh) else 0)
            m_net[(site, date)][sp] += nc
            if a.diag_csv and frames is not None:
                row = {"video_id": vid, "date": date, "site": site, "track_id": ti,
                       "dx": round(dx, 1), "net": nc}
                row.update(_track_diag(t, frames))
                diag_rows.append(row)
        n_ok += 1

    t_act_d, t_net_d, *_ = _load_tech_for_year(a.year, a.tech_dir)
    # write long CSV: date,site,species,model,tech  (activity; a --net variant writes net)
    rows = []
    src_d = m_net if a.metric == "net" else m_act
    tech_d = t_net_d if a.metric == "net" else t_act_d
    keys = set(src_d) | set(k for k in tech_d)
    for (site, date) in sorted(keys):
        for si, sp in enumerate(SPECIES):
            mv = src_d.get((site, date), {}).get(si, 0)
            tv = tech_d.get((site, date), {}).get(si, 0)   # load_tech keys tech by species INDEX (0-4), not name
            if mv == 0 and tv == 0:
                continue
            rows.append({"date": date, "site": site, "species": sp, "model": mv, "tech": tv})
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "site", "species", "model", "tech"]); w.writeheader(); w.writerows(rows)
    print(f"tracked {n_ok} videos ({n_nostore} no-store, {n_nodate} unresolved); {len(rows)} daily rows "
          f"({a.metric}) -> {a.out_csv}")

    if a.diag_csv and diag_rows:
        cols = ["video_id", "date", "site", "track_id", "pred_species", "conf_top", "conf2_species", "conf2",
                "margin", "conf_frame_mean", "conf_frame_max", "n_frames", "area_mean", "area_max", "cy_mean",
                "min_neighbor_px", "max_cooccur", "dx", "net", "pred_idx"]
        Path(a.diag_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.diag_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(diag_rows)
        print(f"per-track diagnostics: {len(diag_rows)} tracks -> {a.diag_csv}")


def _parse_args():
    """Parse the CLI. Returns the argparse namespace consumed by main()."""
    p = argparse.ArgumentParser(description="count-r daily CSV from a single-stage detect-store")
    p.add_argument("--store-dir", required=True)
    p.add_argument("--agg", default="", help="glob of salmonid.detect.track_from_store agg pickles (BoTSORT tracks); when set, "
                                             "counts use those tracks instead of IoU-tracking the store.")
    p.add_argument("--tracker", choices=["iou", "bytetrack"], default="iou",
                   help="store-only tracker when --agg is NOT given: 'iou' (greedy, ~1.48 tracks/fish) or "
                        "'bytetrack' (Kalman motion, no re-decode, ~1.1 tracks/fish -- fast preview of the "
                        "BoT-SORT correction). Gold-standard absolute counts still use --agg (BoT-SORT).")
    p.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                   help="video_id (default, multi-year safe) or rel (legacy 2025 store + op eval).")
    p.add_argument("--videos-json", required=True)
    p.add_argument("--year", type=int, default=2025,
                   help="selects the tech loader: 2025 (per-clip export), 2024 (AbsDb+Ganny), "
                        "2020-2023 (daily-NET Vaki, net only).")
    p.add_argument("--tech-dir", default="docs/Riverwatcher_2025")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--diag-csv", default="",
                   help="Also write a per-track diagnostic table (species, confidence, runner-up margin, box "
                        "geometry, proximity/co-occurrence) for post-hoc review. No extra GPU — from the same store.")
    p.add_argument("--metric", choices=["activity", "net"], default="activity")
    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--up-thresh", type=float, default=50.0)
    p.add_argument("--down-thresh", type=float, default=300.0)
    p.add_argument("--date-map", default="", help="JSON {video_id: 'YYYY-MM-DD'} to date videos by true capture "
                                                 "time (e.g. RiverWatcher .log) instead of file mtime.")
    return p.parse_args()


if __name__ == "__main__":
    main()

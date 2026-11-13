"""In plain terms: this script turns the fish tracks from one season into daily counts and
puts them beside the technicians' counts for the same days. For every track it decides three
things: which species it is, which day it passed, and whether it went upstream, went
downstream, or stayed put. It then adds the tracks up by river, day and species, and writes a
table with one row per river, day and species holding the model's count and the technician's
count. It does not run the model or look at any video; it works from the tracks saved earlier.
This is the script that produced every count in the paper.

Count step: tracks (or stored detections) -> daily model-versus-technician table for one season.

Pipeline stage: data -> detect -> track -> **count (this module)** -> results
(``salmonid.results.build_canonical_results``) -> figures.

Inputs : * ``--agg``: the track files written by ``salmonid.detect.track_from_store``
           (``<projects>/deimv2_x_v27_corpus<year>/agg_*.pkl``). Every published count uses these.
         * ``--videos-json``: the season's video list (manifest) for both rivers.
         * the date of each video: the video file's last-modified time under
           ``config.FOOTAGE_ROOT``, or a ``--date-map`` JSON file.
         * the technician record for ``--year`` (three different sources, see below).
Outputs: ``--out-csv`` with columns ``date, site, species, model, tech``; with
         ``--dump-date-map`` also the video-to-date table that was used.
Feeds  : through ``build_canonical_results``: Tables 2-5 and Figures 4-7 (Sections 3.2-3.6).
         Directly: the eight-rule comparison of Section 3.4 (``data/net_rule_sweep/*.csv``,
         one file per season and rule, written by this script with different thresholds).
Run    : python -m salmonid.count.corpus_countr_from_store \\
             --store-dir <projects>/deimv2_x_v27_corpus2024/store \\
             --agg "<projects>/deimv2_x_v27_corpus2024/agg_*.pkl" \\
             --videos-json <label_batches>/corpus_2024_both.json --key-mode video_id \\
             --year 2024 --metric net --up-thresh 50 --down-thresh 300 \\
             --out-csv output/corpus_v27/net_rule_sweep/2024_up50.csv
         ``python -m salmonid.count.run_corpus`` runs this for every season, rule and metric.
Tier   : 2 (needs the track aggregates, the technician files and either the raw footage or a
         date map; base Python environment, no GPU, a few seconds per season).

How the published numbers were computed
---------------------------------------
Each point below is what the code does. Several are simplifications a reader should know
about; none was changed for the deposit, because changing any of them would change published
numbers.

1. **A track needs at least 3 saved positions** (``--min-frames 3``). Shorter tracks are
   dropped. One or two positions are most often a reflection, a piece of debris, or a
   fragment split off another fish's track. Positions are saved for every third video frame,
   and the tracker does not save a track's first detection, so three saved positions mean the
   fish was detected in at least four of the frames examined. In time that is about one
   second for the 2023-2025 videos (about 10 frames per second) but only about 0.4 second
   for the 2020-2022 videos (25 or 30 frames per second): the rule is in frames, not seconds,
   so it is less strict for the earlier seasons. The same holds for the three-position
   averages used for direction (point 3).

2. **Species** is decided once per track: the five species scores are averaged over all of
   the track's positions and the species with the highest average wins. No minimum
   confidence is applied. Every surviving track is assigned to one of the five species;
   there is no "unknown". The detector's sixth class, "Small Salmonid" (juveniles), was
   dropped at the tracking step, so a juvenile's track is assigned to whichever adult
   species scored highest for it.

3. **Direction** comes from how far the centre of the box moved sideways:
   delta_x = (mean x-centre of the LAST three positions) - (mean x-centre of the FIRST three),
   in pixels of the 1280-pixel-wide frame. The sign that means "upstream" is set per river
   (``UP_SIGN``: to the right at the Ganaraska, to the left at the Credit).
     - upstream   (+1): the fish moved at least 50 px in the upstream direction;
     - downstream (-1): the fish moved at least 300 px in the downstream direction;
     - holding     (0): anything else.
   Why 50 but 300: fish often hold station in front of the camera and drift back and forth.
   Upstream is the direction of the run, so modest upstream progress is accepted as a
   passage, which is also how the technicians score it. A fish is only counted as having gone
   back down when it clearly travels downstream, about a quarter of the frame width. With the
   same small threshold in both directions, drifting fish would be scored as many false
   downstream passages and the net count would come out too low.
   Consequences of using three-position means: a track with exactly three positions has the
   same first and last three, so its delta_x is 0 and it is always "holding"; a track with
   four positions must travel 150 px end to end to register 50 px.

4. **Two counts are produced.** ``--metric activity`` counts every surviving track once,
   whatever its direction. ``--metric net`` adds +1 for upstream, -1 for downstream and 0
   for holding; this "net" count is what the monitoring programme reports.

5. **Date** is the day on which the video file was last modified, read in the time zone of
   the computer running this script (the original run: Eastern Time). The counter saves a
   short video file for each passage event, so this is the passage date as long as the files
   keep their original timestamps. Copying or downloading footage usually resets those timestamps, so a deposit
   reader should pass ``--date-map`` instead (``--dump-date-map`` writes one). The recovered
   2020 Ganaraska footage had lost its timestamps and was dated from the counter's log file
   through ``--date-map`` (``config.RECOVERED_2020_DATEMAP``). A video that cannot be dated
   is left out and reported as "unresolved".

6. **Technician counts** are read by year, from three different kinds of record:
     - 2020-2023: the official daily net counts (Ganaraska ``Ganny_Total.csv``; Credit
       ``FINAL_CREDIT_<year>.csv``). "Activity" for these years is the sum of the absolute
       values of the daily net records, because nothing finer exists.
     - 2024: Ganaraska as above (net only); Credit from the reviewed counter database export,
       one row per fish with its direction.
     - 2025: per-fish review exports for both rivers (``config.TECH_2025_DIR``).
   Technician records without a species ("Salmonid", "Unknown") are not included in any
   species' count.

7. **Which rows are written.** One row per river, day and species where the model count or
   the technician count is not zero. A day on which the technician recorded fish but the
   model has no videos (or no tracks) appears with model = 0; it is scored as the model
   missing those fish, not excluded. A day on which both counts are zero is not written. For
   the net count this includes days where fish passed but ups and downs cancelled exactly.
   The daily correlation reported in the paper is computed over exactly the rows written.

Other trackers in this file (not used for published counts)
-----------------------------------------------------------
Without ``--agg`` the script links the stored detections itself, either with a simple
overlap rule (``--tracker iou``) or with the ``supervision`` library's ByteTrack
(``--tracker bytetrack``). These need no video and were used for quick checks, for the
per-track diagnostic table (``--diag-csv``) and for the statement in Section 3.4 that linking
by overlap alone over-counts. The ByteTrack comparison reported in Section 3.4 is a
different thing: it is ``salmonid.detect.track_from_store --tracker bytetrack`` followed by
this script with ``--agg``.

Which research-repository version this is: the counting code is identical on both development
branches of the research repository (they differ by one comment). Additions for the deposit:
paths come from ``salmonid.config``, missing inputs stop with a clear message, skipped
videos and unreadable track files are reported instead of passed over silently, and
``--dump-date-map`` was added. None of these changes a count.
"""
from __future__ import annotations

import argparse, collections, csv, datetime, glob, hashlib, json, os, pickle
from pathlib import Path
import numpy as np

from salmonid import config
from salmonid.count.corpus_fullyear_vs_tech import load_tech  # the 2025 technician-record reader

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # same order as salmonid.config.SPECIES
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}          # sign of delta_x that means "upstream" at each river
VIDEO_ROOTS = config.video_roots()                    # {river: <FOOTAGE_ROOT>/<river>}, where file dates are read
DATE_MAP = {}                                          # optional {video_id: "YYYY-MM-DD"}; used instead of the file date


def store_key(site, rel):
    """Store file name in the old ``rel`` naming: ``<river>__<hash of "folder/file">.pkl``."""
    return f"{site}__{hashlib.md5(rel.encode()).hexdigest()[:12]}.pkl"


def store_key_vid(video_id):
    """Store file name for a video: a hash of its full path (same as the detect and track steps)."""
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def _load_tech_for_year(year, tech_dir):
    """Read the technician record for one season.

    Returns ``(act_d, net_d, act_t, net_t, uncl)``. ``act_d`` and ``net_d`` map
    ``(river, date)`` to ``{species number: count}`` for activity and net; the rest are season
    totals. Which reader is used depends on the year, because the technician record changed
    form over time (see point 6 of the module notes).
    """
    if not 2018 <= year <= 2025:
        raise SystemExit(f"No technician record reader for {year}: this script covers 2018-2025. "
                         "(Any other year would silently be compared with the 2025 record.)")
    if year == 2024:
        from salmonid.count import corpus_2024_vs_tech as _m
        config.require(_m.GANA_TECH, "the Ganaraska official daily counts (Ganny_Total.csv)",
                       "SALMONID_LOMU_INTERVIEW_DIR")
        config.require(_m.CREDIT_TECH, "the 2024 Credit River reviewed counter exports (two .xlsx files)",
                       "SALMONID_MODEL_PRO_DIR")
        return _m.load_tech()
    if 2018 <= year <= 2023:
        from salmonid.count import corpus_historical_vs_tech as _m
        config.require(_m.GANA_TECH, "the Ganaraska official daily counts (Ganny_Total.csv)",
                       "SALMONID_LOMU_INTERVIEW_DIR")
        config.require(_m.CREDIT_HIST / str(year) / f"FINAL_CREDIT_{year}.csv",
                       f"the {year} Credit River official daily counts", "SALMONID_MODEL_PRO_DIR")
        net_d, net_t, uncl, act_d, act_t = _m.load_tech(year, with_activity=True)   # activity = sum of |daily net|
        return act_d, net_d, act_t, net_t, uncl
    # 2025. The reader returns an empty record, without complaint, if its files are not
    # there, so check for them first: an empty record would look like "technicians saw nothing".
    rw = Path(tech_dir)
    config.require(rw / "Ganaraska_Counter_Analysis_2025" / "all_counts_gan.csv",
                   "the 2025 Ganaraska technician review export", "SALMONID_TECH_2025_DIR")
    config.require(rw / "Credit_Counter_Analysis_2025" / "data" / "raw_data",
                   "the 2025 Credit River technician exports folder", "SALMONID_TECH_2025_DIR")
    return load_tech(rw)


def _iou(b, boxes):
    """Overlap of one box with each of several boxes (0 = none, 1 = identical)."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(b[0], boxes[:, 0]); y1 = np.maximum(b[1], boxes[:, 1])
    x2 = np.minimum(b[2], boxes[:, 2]); y2 = np.minimum(b[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[2] - b[0]) * (b[3] - b[1]); ar = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + ar - inter, 1e-6, None)


def track_store(frames, iou_thr=0.3, buffer=8):
    """Simple overlap tracker (``--tracker iou``); NOT used for published counts.

    A detection joins the existing track whose last box it overlaps most (at least 0.3), if
    that track was seen within the last 8 frames; otherwise it starts a new track. It does not
    predict motion, so a fast fish breaks into several tracks and is counted more than once.
    Returns one ``{"boxes", "pvs", "fis"}`` record per track (positions, species scores, frame numbers).
    """
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


_BYTETRACK = {}   # unused; a tracker holds state and must be built fresh for every video


def track_store_bytetrack(frames, fps, buffer_frames=30, act_thr=0.25):
    """Motion-only tracker from the ``supervision`` library (``--tracker bytetrack``); NOT used
    for published counts and NOT the ByteTrack comparison of manuscript Section 3.4 (that one is
    ``salmonid.detect.track_from_store --tracker bytetrack``, with different settings).
    Needs ``pip install supervision``. Returns the same per-track records as ``track_store``.
    """
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


FRAME_W, FRAME_H = 1280.0, 960.0          # size of a video frame in pixels (boxes are stored in these units)


def _track_diag(t, frames):
    """One row of the optional per-track diagnostic table (``--diag-csv``).

    Records how confident the species call was, the runner-up species, the size and height of
    the box in the frame, and how close other fish were. Only available when this script does
    its own tracking (no ``--agg``), because the saved BoT-SORT tracks do not keep frame numbers.
    """
    boxes = np.asarray(t["boxes"], np.float32); pvs = np.asarray(t["pvs"], np.float32); fis = t["fis"]
    mean_pv = pvs.mean(0); order = np.argsort(mean_pv)[::-1]
    sp = int(order[0]); margin = float(mean_pv[order[0]] - mean_pv[order[1]])
    per_frame_conf = pvs.max(1)                                   # best species score in each frame
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (FRAME_W * FRAME_H)   # share of the frame
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0; cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    # Nearest other fish in any of the track's frames (pixels between box centres), and the
    # largest number of fish in view at once. A small distance means overlapping fish.
    min_neigh = np.inf; max_cooccur = 1
    for i, fi in enumerate(fis):
        ob = np.asarray(frames[fi]["boxes"], np.float32)
        max_cooccur = max(max_cooccur, len(ob))
        if len(ob) < 2:
            continue
        ocx = (ob[:, 0] + ob[:, 2]) / 2.0; ocy = (ob[:, 1] + ob[:, 3]) / 2.0
        dist = np.sqrt((ocx - cx[i]) ** 2 + (ocy - cy[i]) ** 2)
        dist[dist < 1e-3] = np.inf                               # ignore the fish itself
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
    """Find the video file for a ``video_id`` under ``config.FOOTAGE_ROOT``, or None.

    Used only to read the file's last-modified date when no ``--date-map`` entry exists.
    """
    for site, root in VIDEO_ROOTS.items():
        if video_id.startswith(site + "/"):
            p = Path(root) / video_id[len(site) + 1:]
            if p.is_file():
                return str(p)
    return None


def main():
    """Build the daily model-versus-technician table for one season and write it as a CSV."""
    global DATE_MAP
    a = _parse_args()
    config.require(a.videos_json, "the season's video list (manifest JSON) given as --videos-json",
                   "SALMONID_PROJECTS_ROOT")
    if a.date_map:
        config.require(a.date_map, "the video-to-date table given as --date-map")
        DATE_MAP = json.load(open(a.date_map))
        print(f"loaded date-map: {len(DATE_MAP)} video_id -> capture date (overrides mtime dating)")
    vids = json.load(open(a.videos_json))["videos"]
    # Map each store/aggregate key back to its video. A video listed twice appears once here.
    if a.key_mode == "video_id":
        k2vid = {store_key_vid(v["video_id"]): v["video_id"] for v in vids if v.get("video_id")}
    else:                                                     # old naming (not used for any published season)
        k2vid = {store_key(v["site"], v["video_rel"].replace("\\", "/")): v.get("video_id") for v in vids}
    store = Path(a.store_dir)

    # --agg: read the tracks made by salmonid.detect.track_from_store (BoT-SORT). This is the
    # path behind every published count. Several files (one per river and shard) are merged.
    agg = {}
    if a.agg:
        agg_files = glob.glob(a.agg)
        if not agg_files:
            config.require(a.agg, "the track aggregate files matching --agg (written by "
                                  "salmonid.detect.track_from_store)", "SALMONID_PROJECTS_ROOT")
        for gf in agg_files:
            try:
                for k, rec in pickle.load(open(gf, "rb")).items():
                    agg[k] = rec.get("tracks", {})
            except Exception as e:                                  # noqa: BLE001
                # Same as the original (the file is skipped), but now said out loud: every
                # video in an unreadable file would otherwise just be missing from the counts.
                print(f"WARNING: could not read track file {gf} ({type(e).__name__}); its videos are skipped")
        print(f"loaded {len(agg)} tracked videos from agg '{a.agg}'")

    # Model daily totals: (river, date) -> species number -> count.
    m_act = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net = collections.defaultdict(lambda: collections.defaultdict(int))
    diag_rows = []                                   # per-track diagnostics (only with --diag-csv and no --agg)
    used_dates = {}                                  # video_id -> date actually used (for --dump-date-map)
    n_ok = n_nostore = n_nodate = 0
    for key, vid in k2vid.items():
        frames = None
        if a.agg:
            rec = agg.get(key)
            if rec is None:
                n_nostore += 1; continue             # video is on the list but has no tracks file entry
            tracks = [{"boxes": np.asarray(t["boxes"], float), "pvs": np.asarray(t["pvs"], float),
                       "fis": list(range(len(t["boxes"])))} for t in rec.values()]
            # --diag-csv produces nothing in this mode: the saved BoT-SORT tracks hold positions
            # and species scores but not the frame number of each position, which the diagnostics
            # need. This is why the per-video accuracy in the paper's Table 2 is computed with
            # the simple overlap tracker while the counts in the same table come from BoT-SORT.
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
            date = DATE_MAP[vid]                          # date supplied by the caller (e.g. from the counter log)
        else:
            vp = _resolve(vid) if vid else None
            if not vp:
                n_nodate += 1; continue                   # cannot be dated -> left out of the counts
            # The day the file was last modified, in this computer's local time zone.
            date = datetime.date.fromtimestamp(os.path.getmtime(vp)).isoformat()
        used_dates[vid] = date
        site = vid.split("/", 1)[0]
        for ti, t in enumerate(tracks):
            if len(t["boxes"]) < a.min_frames:            # fewer than 3 saved positions: dropped (see note 1)
                continue
            # Species: average the five species scores over the track, take the highest (note 2).
            pvs = np.asarray(t["pvs"]); sp = int(np.argmax(pvs.mean(0)[:5]))
            m_act[(site, date)][sp] += 1                  # activity: every surviving track counts once
            cx = (np.asarray(t["boxes"])[:, 0] + np.asarray(t["boxes"])[:, 2]) / 2.0   # x-centre at each position
            # delta_x: mean of the last three x-centres minus mean of the first three (note 3).
            # With exactly three positions the two means are the same numbers, so dx = 0.
            w = min(3, len(cx)); dx = float(cx[-w:].mean() - cx[:w].mean())
            sgn = dx * UP_SIGN[site]                      # positive = moved in the upstream direction
            # Upstream needs >= up_thresh (50 px); downstream needs >= down_thresh (300 px); else holding.
            nc = 1 if (sgn > 0 and abs(dx) >= a.up_thresh) else (-1 if (sgn < 0 and abs(dx) >= a.down_thresh) else 0)
            m_net[(site, date)][sp] += nc
            if a.diag_csv and frames is not None:
                row = {"video_id": vid, "date": date, "site": site, "track_id": ti,
                       "dx": round(dx, 1), "net": nc}
                row.update(_track_diag(t, frames))
                diag_rows.append(row)
        n_ok += 1

    t_act_d, t_net_d, *_ = _load_tech_for_year(a.year, a.tech_dir)
    # One row per (river, date, species) where the model or the technician count is not zero.
    # The key set is the union of model days and technician days, so a technician day without
    # any model video is written with model = 0 (note 7).
    rows = []
    src_d = m_net if a.metric == "net" else m_act
    tech_d = t_net_d if a.metric == "net" else t_act_d
    keys = set(src_d) | set(k for k in tech_d)
    for (site, date) in sorted(keys):
        for si, sp in enumerate(SPECIES):
            mv = src_d.get((site, date), {}).get(si, 0)
            tv = tech_d.get((site, date), {}).get(si, 0)   # the technician readers key species by number 0-4
            if mv == 0 and tv == 0:
                continue
            rows.append({"date": date, "site": site, "species": sp, "model": mv, "tech": tv})
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "site", "species", "model", "tech"]); w.writeheader(); w.writerows(rows)
    print(f"tracked {n_ok} videos ({n_nostore} no-store, {n_nodate} unresolved); {len(rows)} daily rows "
          f"({a.metric}) -> {a.out_csv}")
    if n_nostore or n_nodate:
        print(f"WARNING: {n_nostore} listed video(s) have no tracks and {n_nodate} could not be dated; "
              "they are NOT in the model counts above. The published runs had 0 of each.")

    if a.dump_date_map:
        Path(a.dump_date_map).parent.mkdir(parents=True, exist_ok=True)
        with open(a.dump_date_map, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(used_dates.items())), f, indent=0)
        print(f"date map: {len(used_dates)} videos -> {a.dump_date_map}")

    if a.diag_csv and diag_rows:
        cols = ["video_id", "date", "site", "track_id", "pred_species", "conf_top", "conf2_species", "conf2",
                "margin", "conf_frame_mean", "conf_frame_max", "n_frames", "area_mean", "area_max", "cy_mean",
                "min_neighbor_px", "max_cooccur", "dx", "net", "pred_idx"]
        Path(a.diag_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.diag_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(diag_rows)
        print(f"per-track diagnostics: {len(diag_rows)} tracks -> {a.diag_csv}")


def _parse_args():
    """Read the command-line options."""
    p = argparse.ArgumentParser(description="Daily model-versus-technician counts for one season, from saved tracks.")
    p.add_argument("--store-dir", required=True,
                   help="The season's detect-store folder. Only read when --agg is not given.")
    p.add_argument("--agg", default="",
                   help="File pattern of the track files from salmonid.detect.track_from_store "
                        "(e.g. '<season folder>/agg_*.pkl'). Used for every published count.")
    p.add_argument("--tracker", choices=["iou", "bytetrack"], default="iou",
                   help="Only when --agg is NOT given: link the stored detections here with a simple "
                        "overlap rule (iou) or supervision's ByteTrack. Quick checks only.")
    p.add_argument("--key-mode", choices=["video_id", "rel"], default="video_id",
                   help="How store files are named; must match the detect step (all seasons: video_id).")
    p.add_argument("--videos-json", required=True, help="The season's video list (manifest JSON).")
    p.add_argument("--year", type=int, default=2025,
                   help="Season; selects the technician record: 2025 (per-fish exports), 2024, or "
                        "2018-2023 (official daily counts).")
    p.add_argument("--tech-dir", default=str(config.TECH_2025_DIR),
                   help="Folder of the 2025 technician exports (default: config.TECH_2025_DIR). Only used for 2025.")
    p.add_argument("--out-csv", required=True, help="Output table (date, site, species, model, tech).")
    p.add_argument("--diag-csv", default="",
                   help="Also write a per-track diagnostic table. Only works without --agg.")
    p.add_argument("--metric", choices=["activity", "net"], default="activity",
                   help="activity = every track counts once; net = upstream minus downstream.")
    p.add_argument("--min-frames", type=int, default=3,
                   help="Minimum saved positions for a track to count (published: 3).")
    p.add_argument("--up-thresh", type=float, default=50.0,
                   help="Pixels of upstream movement needed to count as upstream (published: 50).")
    p.add_argument("--down-thresh", type=float, default=300.0,
                   help="Pixels of downstream movement needed to count as downstream (published: 300).")
    p.add_argument("--date-map", default="",
                   help="JSON {video_id: 'YYYY-MM-DD'}. Videos listed there are dated from it instead of "
                        "from the file's last-modified time.")
    p.add_argument("--dump-date-map", default="",
                   help="Write the date used for every counted video to this JSON file, so the counts can "
                        "be reproduced later without the raw footage (pass it back as --date-map).")
    return p.parse_args()


if __name__ == "__main__":
    main()

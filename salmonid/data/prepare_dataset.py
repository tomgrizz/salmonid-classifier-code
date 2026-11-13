"""In plain terms: this script gathers the fish that people have already labelled (each with
its species and a box drawn around it) and packages them into the list of pictures the detector
learns from. For every labelled fish it finds the video clip, pulls out a few of the clearest
frames, and writes down where the fish is and what species it is. It also adds frames labelled
"not a fish" so the model learns to stay quiet on debris. Before using any fish it asks the leak
guard whether the clip belongs to the held-out test set and leaves it out if so. Finally it
splits the pictures into a large training part and a small validation part.

Inputs
------
* ``--labels-db``: the review database of species labels (``config.LABELS_DB``).
* ``--tracks-db`` and ``--detections-dir``: the stored detections of an earlier run, which
  supply each labelled fish's boxes frame by frame.
* ``--cvat-json``: frames hand-annotated in the CVAT tool (the images already exist on disk).
* ``--test-manifest``: an older, frame-level test set whose CVAT frames are kept out.
* The held-out sets, loaded unconditionally from ``salmonid.holdout``; ``--ls-test-queue`` adds more.
* Optional sources (pseudo-labels, CVAT clips, technician count CSVs). None of them
  contributed a picture to the published training set; see "What the published set contains".
* The raw footage, under ``config.FOOTAGE_ROOT``.

Outputs
-------
``<--output-dir>/train.json`` and ``val.json`` (COCO format), plus the extracted frames under
``ls_frames/``, ``hard_neg_frames/`` (and ``pseudo_frames/``, ``tech_frames/`` if those sources
are used).

Feeds
-----
No manuscript number directly. It produced the human-labelled core of the published v2.7
training set (Methods, training-set composition): 17,121 pictures from review labels, 1,854
from CVAT and 567 "not a fish" frames, of 45,249 in total.

Run
---
::

    python -m salmonid.data.prepare_dataset
        --labels-db       <config.LABELS_DB>
        --tracks-db       <config.MODEL_OUTPUTS>/run_20260428/tracks.sqlite
        --detections-dir  <config.MODEL_OUTPUTS>/run_20260428/detections
        --cvat-json       <config.PROJECTS_ROOT>/cvat_dataset/annotations.json
        --test-manifest   <path to test_set_v1>/manifest.json
        --output-dir      <config.PROJECTS_ROOT>/training_data
        [--frames-per-track 3] [--hard-neg-max 500] [--val-fraction 0.1] [--seed 42]

(one command; shown on several lines). The exact arguments of the build behind the published
set were not recorded in any repository.

Tier
----
2. It needs the raw footage, the label database and the stored detections. The published
training list is a frozen file (checksums in ``provenance/v2.7/``); this script is deposited so
the way that file was assembled can be read, not so it can be re-run from a clone.

The six label sources
---------------------
1. Review labels from the label database: one species per track, top-N frames per track.
   Eligible: one of the five species or "Small Salmonid" (not "Not a fish", "Multiple fish",
   "Unsure"), with no unresolved reviewer disagreement.
2. CVAT annotations: frame-level, images already on disk.
3. (Optional) Pseudo-labels: high-confidence model predictions from a newer run.
4. (Optional) Hard negatives: tracks labelled "Not a fish", written as frames with no box.
5. (Optional) Pre-extracted CVAT training-clip frames.
6. (Optional) Technician per-video count CSVs matched to model tracks.

What the published set contains
-------------------------------
Only sources 1, 2 and 4 are present in the published training list. No picture in it carries
the ``pseudo_label``, ``tech_reviewed`` or CVAT-clip source tag. In particular the seasonal
plausibility table in this file (``_SEASONAL_VALID``) is applied only to pseudo-labels, so it
filtered nothing in the published data. It should not be read as validated phenology: applied
to the verified held-out fish it would reject 58 of 60 Brown Trout and 51 of 60 Atlantic Salmon.

The leak guard, and how this differs from what built the published data
------------------------------------------------------------------------
``salmonid.holdout.load_holdout_pairs()`` is loaded UNCONDITIONALLY and returns a video-aware
key set, so every ``(video_id, track_id) in ls_test_pairs`` test in this file excludes the
whole held-out CLIP, not only the numbered fish. This is the corrected guard (research
repository commit ``d40b869``). The build behind the published set ran the earlier, pair-keyed
version; the pictures it let through were removed afterwards by
``salmonid.data.audit_coco_holdout_leak --out``.

Known limitations of the published training data (documented, deliberately NOT fixed here,
because the published model was trained on exactly this output)
------------------------------------------------------------------------------------------
* **CVAT frames are not tested against the held-out test set.** ``_process_cvat`` drops only
  the frames of the older ``--test-manifest`` set. CVAT records have a ``video_path`` and no
  ``video_id``, so the guard above never sees them, and neither did the cleaning audit. The
  published training set therefore contains 25 CVAT frames (the validation set 2) from the
  clip of one held-out Atlantic Salmon (test track 353168).
* **Pseudo-labels are not tested against the holdout either** (``_process_pseudo_labels`` only
  skips tracks that already have a review label). No pseudo-label is in the published set.
* **Clips next to a held-out clip are not excluded.** See ``salmonid/holdout.py``.
* **The train/validation split is by picture, not by fish or clip** (``_stratified_split``).
  About 98% of the validation tracks also have frames in training, so validation accuracy
  measures memory, not generalisation. No manuscript number uses it, and the published
  checkpoint was not chosen on it (see ``provenance/v2.7/CHECKPOINT_SELECTION.md``).
* **One box per real frame.** Each picture from sources 1 and 2 carries exactly one fish box,
  because pictures are emitted per (track, frame). Any other fish visible in that frame is
  taught to the model as background.
* **Duplicated frames with conflicting labels.** The same video frame can enter through more
  than one source. In the published training set, 184 frames have the same box (overlap 0.9 or
  more) labelled as different species in different copies. Separately, 45 CVAT file names occur
  twice within training and 11 occur in both training and validation, each copy carrying a
  different single fish (the CVAT source lists one image record per fish and they are not merged).
* **"Not a fish" frames are not checked for other fish** in the same frame.
* **Latent bugs not exercised by the published build:** the image-id offsets of sources 4, 5 and
  6 can collide if source 5 or 6 is switched on; the bounding-box sanity check runs after the
  files are written, so its "ABORT" leaves them on disk; CVAT boxes are copied without clamping
  to the frame (one published box starts at x = -3.73); an out-of-range frame number is clamped
  to the last frame instead of being refused (``_read_frame``).
* The header strip (top 40 pixels) is blacked out on the published pictures by a later step
  that is not part of this deposit; this script writes unmasked frames.

Environment: base Python 3.11 (cv2 / pandas / pyarrow / decord). No GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import csv
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from salmonid import config
from salmonid.util.schema import video_id_to_parquet_stem
# external dependency (not vendored in this repo): scoring.compute_best_id_frame is a helper
# from the original research repo's inference/ package, not part of this published
# reproduction. It is NOT exercised by any code path in this module, so the import is left
# commented out (activating it would raise ImportError on a clean checkout). Reinstate it
# only if you also vendor the two-stage `scoring` module.
# from scoring import compute_best_id_frame  # noqa: F401

SPECIES: dict[int, str] = {
    0: "Chinook",
    1: "Coho",
    2: "Atlantic",
    3: "Rainbow Trout",
    4: "Brown Trout",
    5: "Small Salmonid",
}
LABEL_TO_CLASS_ID: dict[str, int] = {v: k for k, v in SPECIES.items()}

TRAINING_POSITIVE_LABELS = frozenset(SPECIES.values())
EXCLUDED_RESOLUTIONS = frozenset(["pending", "senior_overruled"])

# {site: footage_dir} for the two study rivers. config.video_roots() resolves each site to
# str(config.FOOTAGE_ROOT / site) — byte-identical to the original hard-coded RiverWatcher
# per-site paths on the source workstation, but env-overridable via SALMONID_FOOTAGE_ROOT.
VIDEO_ROOTS: dict[str, str] = config.video_roots()

# Audit CSV statuses to drop (from the stored-detection box audit; its CSV is an optional input).
_AUDIT_DROP_STATUSES = frozenset({"wrong_y_too_high", "wrong_y_too_low", "ambiguous"})

# CVAT annotation audit statuses to drop (from the CVAT annotation audit; optional input CSV).
_CVAT_DROP_STATUSES = frozenset({"frame_empty", "manual_blank"})


def _load_cvat_phantom_anns(audit_csv: str | None) -> frozenset[int]:
    """Return CVAT annotation ids on blank frames (phantom interpolated boxes)."""
    if not audit_csv:
        return frozenset()
    p = Path(audit_csv)
    if not p.is_file():
        print(f"WARNING: --cvat-audit-csv not found: {audit_csv} (no filter applied)")
        return frozenset()
    drop: set[int] = set()
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") in _CVAT_DROP_STATUSES:
                try:
                    drop.add(int(r["ann_id"]))
                except (KeyError, ValueError):
                    pass
    return frozenset(drop)

# Strict per-bbox y-sanity bounds for parquet-sourced detections.
# Pre-2026-05-26 y-bug shifts y_center/H to ~0.40–0.50 on 960px-tall frames.
# Correct bboxes fall in ~0.50–0.75. 0.30/0.85 catches the bug while
# tolerating fish near the top/bottom of the viewing tunnel.
_YCENTER_STRICT_MIN = 0.30
_YCENTER_STRICT_MAX = 0.85
# Strict per-bbox x-sanity + min-area (added 2026-06-03): catches detector over-detections
# on frame edges/corners (vignette/substrate) — mar10 over-detects there at high confidence,
# so the audit/y-band/conf can't catch them. A real fish box sits centrally and isn't tiny.
_XCENTER_STRICT_MIN = 0.13
_XCENTER_STRICT_MAX = 0.87
_MIN_AREA_FRAC = 0.01
# Min detection confidence floor (belt-and-suspenders; drops weak detections). NOTE: conf
# alone does NOT separate fish from debris — mar10 over-detects debris at ~0.94 — geometry
# above does that work; this just removes the low-conf tail.
_MIN_DET_CONF = 0.60


def _load_bad_parquet_stems(audit_csvs: list[str] | str | None) -> frozenset[str]:
    """Return parquet stems (no .parquet) flagged for skip across one or more audit CSVs."""
    if not audit_csvs:
        return frozenset()
    if isinstance(audit_csvs, str):
        audit_csvs = [audit_csvs]
    bad: set[str] = set()
    for audit_csv in audit_csvs:
        p = Path(audit_csv)
        if not p.is_file():
            print(f"WARNING: --bbox-audit-csv not found: {audit_csv} (skipped)")
            continue
        n_before = len(bad)
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") in _AUDIT_DROP_STATUSES:
                    name = r.get("parquet", "")
                    if name.endswith(".parquet"):
                        name = name[: -len(".parquet")]
                    if name:
                        bad.add(name)
        print(f"  audit {p.name}: +{len(bad) - n_before:,} stems")
    return frozenset(bad)


def main() -> None:
    """Build train.json/val.json (COCO) from all requested label sources.

    Parses CLI args, loads the leak guard (frozen eval tracks AND their whole video clips
    are excluded), assembles LS + CVAT + optional pseudo/clip/hard-neg/tech sources into
    one image+annotation list, stratifies into train/val, and writes both COCO files plus a
    bbox y-sanity report. Side effects: extracts frame JPEGs to <output-dir> subfolders and
    writes train.json / val.json.
    """
    args = _parse_args()

    output_dir = Path(args.output_dir)
    ls_frames_dir = output_dir / "ls_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    ls_frames_dir.mkdir(exist_ok=True)

    test_manifest_path = Path(args.test_manifest)
    if not test_manifest_path.is_file():
        sys.exit(
            f"ERROR: Test manifest not found: {test_manifest_path}\n"
            "It is the older frame-level CVAT test set (an external input; see the module docstring)."
        )

    test_annotations_path = test_manifest_path.parent / "annotations.json"
    with open(test_annotations_path, encoding="utf-8") as f:
        test_coco = json.load(f)
    # Use file_name (not id) for exclusion — CVAT annotations.json was regenerated
    # after the frame-offset bug fix in May 2026, which reassigned some image IDs.
    # The frozen test set retains the old IDs, so ID-based filtering misses ~14 images.
    test_image_files = {img["file_name"] for img in test_coco["images"]}
    print(f"Reserving {len(test_image_files)} CVAT frames for test set (matched by file_name)")

    # Load LS test queue exclusions — full (video_id, track_id) pairs, not batch_id,
    # because labels for these tracks may have been saved under a different batch_id.
    ls_test_pairs: set[tuple[str, int]] = set()
    if args.ls_test_queue:
        for queue_path in args.ls_test_queue:
            with open(queue_path, encoding="utf-8") as f:
                q = json.load(f)
            items = q.get("items", [])
            for item in items:
                ls_test_pairs.add((item["video_id"], int(item["track_id"])))
        print(f"Reserving {len(ls_test_pairs)} LS test queue (video_id, track_id) pairs")
    # Permanent leak guard — ALWAYS exclude the canonical frozen eval sets, flag or no flag.
    # Assign (don't |=): the guard is a video-aware HoldoutKeys set, and `plain_set |= keys` would
    # flatten it back to pair-only matching — the hole that leaked test_set_v6 into v2.6.
    # From here on `(video_id, track_id) in ls_test_pairs` is True for ANY track in a held-out
    # clip, which is what every source below that receives `ls_test_pairs` relies on.
    from salmonid.holdout import load_holdout_pairs
    ls_test_pairs = load_holdout_pairs(extra_pairs=ls_test_pairs)
    print(f"Reserving {len(ls_test_pairs)} held-out (video_id, track_id) pairs "
          f"+ {len(ls_test_pairs.videos)} held-out videos after guard")

    bad_parquet_stems = _load_bad_parquet_stems(args.bbox_audit_csv)
    if bad_parquet_stems:
        print(f"Bbox audit filter active: {len(bad_parquet_stems):,} parquet stems "
              f"will be skipped (from {args.bbox_audit_csv})")

    # --- Part 1: Label Studio labels ---
    print("\n--- Part 1: Label Studio labels ---")
    ls_images, ls_annotations = _process_ls_labels(
        labels_db=Path(args.labels_db),
        tracks_db=Path(args.tracks_db),
        detections_dir=Path(args.detections_dir),
        ls_frames_dir=ls_frames_dir,
        frames_per_track=args.frames_per_track,
        ls_test_pairs=ls_test_pairs,
        limit=args.limit,
        extra_detections_dirs=[Path(d) for d in args.extra_detections_dir] if args.extra_detections_dir else [],
        bad_parquet_stems=bad_parquet_stems,
    )
    print(f"LS: {len(ls_images)} images extracted")
    _print_species_breakdown(ls_annotations, "LS training data")

    # --- Part 2: CVAT training portion ---
    print("\n--- Part 2: CVAT annotations (non-test) ---")
    cvat_images, cvat_annotations = _process_cvat(
        cvat_json=Path(args.cvat_json),
        test_image_files=test_image_files,
        id_offset_images=len(ls_images),
        id_offset_annotations=len(ls_annotations),
        cvat_drop_anns=_load_cvat_phantom_anns(args.cvat_audit_csv),
    )
    print(f"CVAT: {len(cvat_images)} training images")
    _print_species_breakdown(cvat_annotations, "CVAT training data")

    # --- Part 3: Pseudo-labels from newer inference run (optional) ---
    pseudo_images: list = []
    pseudo_annotations: list = []
    if args.pseudo_tracks_db and args.pseudo_detections:
        print("\n--- Part 3: Pseudo-labels from newer inference run ---")
        pseudo_classes = set(args.pseudo_classes) if args.pseudo_classes else None
        if pseudo_classes is not None:
            names = [SPECIES[c] for c in sorted(pseudo_classes) if c in SPECIES]
            print(f"  Pseudo-label classes restricted to: {names}")
        pseudo_frames_dir = output_dir / "pseudo_frames"
        pseudo_frames_dir.mkdir(exist_ok=True)
        pseudo_images, pseudo_annotations = _process_pseudo_labels(
            pseudo_tracks_db=Path(args.pseudo_tracks_db),
            pseudo_detections_dir=Path(args.pseudo_detections),
            labels_db=Path(args.labels_db),
            pseudo_frames_dir=pseudo_frames_dir,
            min_conf=args.pseudo_min_conf,
            max_per_species=args.pseudo_max_per_species,
            frames_per_track=args.frames_per_track,
            id_offset_images=len(ls_images) + len(cvat_images),
            id_offset_annotations=len(ls_annotations) + len(cvat_annotations),
            allowed_classes=pseudo_classes,
            bad_parquet_stems=bad_parquet_stems,
        )
        print(f"Pseudo-labels: {len(pseudo_images)} images added")
        _print_species_breakdown(pseudo_annotations, "Pseudo-label data")

    # --- Part 5: CVAT clips (optional) ---
    cvat_clip_images: list = []
    cvat_clip_annotations: list = []
    if args.cvat_clips_db:
        print("\n--- Part 5: CVAT training clips ---")
        id_off_img = len(ls_images) + len(cvat_images) + len(pseudo_images)
        id_off_ann = len(ls_annotations) + len(cvat_annotations) + len(pseudo_annotations)
        cvat_clip_images, cvat_clip_annotations = _process_cvat_clips(
            cvat_clips_db=Path(args.cvat_clips_db),
            ls_test_pairs=ls_test_pairs,
            id_offset_images=id_off_img,
            id_offset_annotations=id_off_ann,
        )
        print(f"CVAT clips: {len(cvat_clip_images)} images")
        _print_species_breakdown(cvat_clip_annotations, "CVAT clip data")

    # --- Part 4: Hard negatives — "Not a fish" clips as empty-annotation frames ---
    hard_neg_images: list = []
    if args.hard_neg_max > 0:
        print("\n--- Part 4: Hard negatives (not-a-fish labels) ---")
        hard_neg_frames_dir = output_dir / "hard_neg_frames"
        hard_neg_frames_dir.mkdir(exist_ok=True)
        hard_neg_images = _process_hard_negatives(
            labels_db=Path(args.labels_db),
            tracks_db=Path(args.tracks_db),
            detections_dir=Path(args.detections_dir),
            hard_neg_frames_dir=hard_neg_frames_dir,
            frames_per_track=args.frames_per_track,
            ls_test_pairs=ls_test_pairs,
            max_tracks=args.hard_neg_max,
            id_offset=len(ls_images) + len(cvat_images) + len(pseudo_images),
            bad_parquet_stems=bad_parquet_stems,
        )
        print(f"Hard negatives: {len(hard_neg_images)} images (no annotations)")

    # --- Part 6: Tech-reviewed per-video labels (optional) ---
    tech_images: list = []
    tech_annotations: list = []
    if args.tech_labels_dir:
        print("\n--- Part 6: Tech-reviewed per-video labels ---")
        tech_frames_dir = output_dir / "tech_frames"
        tech_frames_dir.mkdir(exist_ok=True)
        tech_tracks_db = Path(args.tech_tracks_db) if args.tech_tracks_db else Path(args.tracks_db)
        tech_det_dir = Path(args.tech_detections_dir) if args.tech_detections_dir else Path(args.detections_dir)
        id_off_img = len(ls_images) + len(cvat_images) + len(pseudo_images) + len(hard_neg_images)
        id_off_ann = len(ls_annotations) + len(cvat_annotations) + len(pseudo_annotations)
        tech_classes = set(args.tech_classes) if args.tech_classes else None
        tech_class_limits: dict[int, int] | None = None
        if args.tech_max_per_class:
            tech_class_limits = {}
            for pair in args.tech_max_per_class:
                cls_str, max_str = pair.split(":")
                tech_class_limits[int(cls_str)] = int(max_str)
        tech_images, tech_annotations = _process_tech_labels(
            tech_labels_dir=Path(args.tech_labels_dir),
            tracks_db=tech_tracks_db,
            detections_dir=tech_det_dir,
            tech_frames_dir=tech_frames_dir,
            frames_per_track=args.frames_per_track,
            ls_test_pairs=ls_test_pairs,
            max_per_species=args.tech_max_per_species,
            allowed_classes=tech_classes,
            class_limits=tech_class_limits,
            id_offset_images=id_off_img,
            id_offset_annotations=id_off_ann,
            dry_run=False,
            bad_parquet_stems=bad_parquet_stems,
        )
        print(f"Tech-reviewed: {len(tech_images)} images added")
        _print_species_breakdown(tech_annotations, "Tech-reviewed data")

    # --- Part 6b: Second tech source (optional) ---
    tech2_images: list = []
    tech2_annotations: list = []
    if args.tech_labels_dir and args.tech_tracks_db2:
        print("\n--- Part 6b: Tech-reviewed per-video labels (2nd source) ---")
        tech_frames_dir2 = output_dir / "tech_frames2"
        tech_frames_dir2.mkdir(exist_ok=True)
        tech_tracks_db2 = Path(args.tech_tracks_db2)
        tech_det_dir2 = Path(args.tech_detections_dir2) if args.tech_detections_dir2 else Path(args.detections_dir)
        id_off_img2 = len(ls_images) + len(cvat_images) + len(pseudo_images) + len(hard_neg_images) + len(tech_images)
        id_off_ann2 = len(ls_annotations) + len(cvat_annotations) + len(pseudo_annotations) + len(tech_annotations)
        tech_classes2 = set(args.tech_classes_2) if args.tech_classes_2 else None
        tech_max2 = args.tech_max_per_species_2 if args.tech_max_per_species_2 is not None else args.tech_max_per_species
        tech_class_limits2: dict[int, int] | None = None
        if args.tech_max_per_class_2:
            tech_class_limits2 = {}
            for pair in args.tech_max_per_class_2:
                cls_str, max_str = pair.split(":")
                tech_class_limits2[int(cls_str)] = int(max_str)
        tech2_images, tech2_annotations = _process_tech_labels(
            tech_labels_dir=Path(args.tech_labels_dir),
            tracks_db=tech_tracks_db2,
            detections_dir=tech_det_dir2,
            tech_frames_dir=tech_frames_dir2,
            frames_per_track=args.frames_per_track,
            ls_test_pairs=ls_test_pairs,
            max_per_species=tech_max2,
            allowed_classes=tech_classes2,
            class_limits=tech_class_limits2,
            id_offset_images=id_off_img2,
            id_offset_annotations=id_off_ann2,
            dry_run=False,
            bad_parquet_stems=bad_parquet_stems,
        )
        print(f"Tech-reviewed (2nd source): {len(tech2_images)} images added")
        _print_species_breakdown(tech2_annotations, "Tech-reviewed data (2nd source)")

    # --- Combine and split ---
    all_images = ls_images + cvat_images + cvat_clip_images + pseudo_images + hard_neg_images + tech_images + tech2_images
    all_annotations = ls_annotations + cvat_annotations + cvat_clip_annotations + pseudo_annotations + tech_annotations + tech2_annotations
    # hard_neg_images have no annotations — their image IDs simply won't appear in
    # all_annotations, so dataset.py returns empty boxes for them (correct behaviour)

    print(f"\nCombined: {len(all_images)} images, {len(all_annotations)} annotations")
    _print_species_breakdown(all_annotations, "Combined")
    if hard_neg_images:
        print(f"  Hard negatives (no-box frames): {len(hard_neg_images)}")

    train_images, val_images, train_anns, val_anns = _stratified_split(
        all_images, all_annotations, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"\nSplit: {len(train_images)} train / {len(val_images)} val images")

    categories = [
        {"id": k, "name": v, "supercategory": "fish"} for k, v in SPECIES.items()
    ]
    coco_info = {
        "description": "Lake Ontario Fish CV — combined LS + CVAT training data",
        "source_ls_tracks": len(ls_images),
        "source_cvat_frames": len(cvat_images),
    }

    def _write_coco(images: list, annotations: list, path: Path) -> None:
        coco = {
            "info": coco_info,
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)
        print(f"  Written: {path}  ({len(images)} images, {len(annotations)} annotations)")

    _write_coco(train_images, train_anns, output_dir / "train.json")
    _write_coco(val_images, val_anns, output_dir / "val.json")
    _bbox_sanity_check(ls_annotations, ls_images)
    print("\nDone. Next: salmonid.data.build_extra_coco, then merge_synth_into_coco (see docs/REPRODUCE.md).")


def _bbox_sanity_check(ls_annotations: list, ls_images: list) -> None:
    """Warn if LS bbox y-centers are suspiciously high (pre-fix signature).

    For 1280x960 Riverwatcher footage, correct bboxes have y_center/H in 0.35-0.65.
    The pre-fix bug produces y_center/H ~0.20-0.45 (shifted ~25% upward).
    Aborts if ratio is dangerously low (<0.28), warns otherwise.
    """
    import statistics
    if not ls_annotations or not ls_images:
        return
    height_by_id = {img["id"]: img["height"] for img in ls_images if "height" in img}
    ratios = []
    for ann in ls_annotations:
        img_h = height_by_id.get(ann["image_id"])
        if not img_h:
            continue
        x, y, w, h = ann["bbox"]  # COCO: x1, y1, width, height
        ratios.append((y + h / 2.0) / img_h)
    if not ratios:
        return
    median_ratio = statistics.median(ratios)
    if 0.35 <= median_ratio <= 0.65:
        print("Bbox sanity [OK]: LS median y_center/H = %.3f (expected 0.35-0.65)" % median_ratio)
    elif median_ratio < 0.28:
        print("Bbox sanity [ABORT]: LS median y_center/H = %.3f -- severely too high." % median_ratio)
        print("  Pre-fix bbox signature. Training will corrupt the model.")
        print("  The stored detections predate the box-coordinate fix; regenerate them and rerun.")
        raise SystemExit(1)
    else:
        print("Bbox sanity [WARNING]: LS median y_center/H = %.3f (expected 0.35-0.65)." % median_ratio)
        print("  Bboxes may be partially wrong. Inspect a sample of frames before training.")


# ---------------------------------------------------------------------------
# Part 1: Label Studio
# ---------------------------------------------------------------------------

def _process_ls_labels(
    labels_db: Path,
    tracks_db: Path,
    detections_dir: Path,
    ls_frames_dir: Path,
    frames_per_track: int,
    ls_test_pairs: set[tuple[str, int]],
    limit: int | None,
    extra_detections_dirs: list[Path] | None = None,
    bad_parquet_stems: frozenset[str] = frozenset(),
) -> tuple[list, list]:
    """Extract training frames + boxes from eligible Label Studio track labels.

    For each eligible (video_id, track_id) label, locates the track's detections parquet
    (primary dir then any extra dirs), picks the top-N frames, applies the per-bbox geometry
    sanity filters, caches a JPEG per frame, and emits COCO image+annotation dicts.

    Returns (images, annotations). Side effect: writes JPEGs to ls_frames_dir.
    """
    conn_labels = sqlite3.connect(str(labels_db))
    conn_labels.row_factory = sqlite3.Row
    conn_tracks = sqlite3.connect(str(tracks_db))
    conn_tracks.row_factory = sqlite3.Row

    # One eligible label per (video_id, track_id):
    #   - single-reviewed: disagreement_resolution IS NULL
    #   - agreed multi-reviewed: disagreement_resolution IS NULL, same label
    #   - senior-confirmed: disagreement_resolution = 'senior'
    # Pending and senior_overruled rows are excluded by the WHERE clause.
    # GROUP BY de-duplicates agreed multi-reviewed pairs (both have NULL, same label).
    query = """
        SELECT video_id, track_id, final_label
        FROM labels
        WHERE final_label IN ('Chinook','Coho','Atlantic','Rainbow Trout','Brown Trout','Small Salmonid')
          AND (disagreement_resolution IS NULL
               OR disagreement_resolution = 'senior')
        GROUP BY video_id, track_id
        HAVING COUNT(DISTINCT final_label) = 1
        ORDER BY video_id, track_id
    """
    rows = conn_labels.execute(query).fetchall()
    if limit:
        rows = rows[:limit]

    if ls_test_pairs:
        n_before = len(rows)
        rows = [r for r in rows if (r["video_id"], int(r["track_id"])) not in ls_test_pairs]
        print(f"  Eligible LS labels: {len(rows)} ({n_before - len(rows)} excluded — in LS test queue)")
    else:
        print(f"  Eligible LS labels: {len(rows)}")

    images: list[dict] = []
    annotations: list[dict] = []
    n_skip = 0
    n_extra = 0
    n_strict_drop = 0

    for i, row in enumerate(rows):
        if i > 0 and i % 100 == 0:
            print(f"  {i}/{len(rows)} processed ({n_skip} skipped)...", end="\r")

        video_id = row["video_id"]
        track_id = int(row["track_id"])
        class_id = LABEL_TO_CLASS_ID[row["final_label"]]

        video_path = _resolve_video_path(video_id)
        if video_path is None or not Path(video_path).is_file():
            n_skip += 1
            continue

        parquet_stem = video_id_to_parquet_stem(video_id)
        if parquet_stem in bad_parquet_stems:
            n_skip += 1
            continue
        # Try primary dir first, then each extra dir in order.
        # We search until we find a parquet that has data for this specific track_id,
        # because the same video may appear in multiple runs with different track_ids.
        det_df = None
        all_dirs = [detections_dir] + (extra_detections_dirs or [])
        for dir_idx, det_dir in enumerate(all_dirs):
            candidate = det_dir / f"{parquet_stem}.parquet"
            if not candidate.is_file():
                continue
            df = _load_track_detections(candidate, track_id)
            if df is not None and not df.empty:
                det_df = df
                if dir_idx > 0:
                    n_extra += 1
                break
        if det_df is None:
            n_skip += 1
            continue

        fw, fh = _get_video_dimensions(video_path)
        if fw == 0:
            n_skip += 1
            continue

        top_frames = _get_top_frames(det_df, fw, fh, n=frames_per_track)
        if not top_frames:
            n_skip += 1
            continue

        track_added = 0
        for frame_number, det_row in top_frames:
            x1 = int(det_row["bbox_x1"])
            y1 = int(det_row["bbox_y1"])
            x2 = int(det_row["bbox_x2"])
            y2 = int(det_row["bbox_y2"])

            # Strict per-bbox y-sanity: catch pre-2026-05-26 y-bug stragglers
            # that slipped past the audit-CSV filter (e.g. parquets without an
            # audit row, or runs we haven't audited yet).
            y_center = (y1 + y2) / 2.0
            x_center = (x1 + x2) / 2.0
            area_frac = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
            if not (_YCENTER_STRICT_MIN <= y_center / max(fh, 1) <= _YCENTER_STRICT_MAX):
                n_strict_drop += 1
                continue
            # x-sanity + min-area: drop edge/corner over-detections (mar10 vignette FPs)
            if not (_XCENTER_STRICT_MIN <= x_center / max(fw, 1) <= _XCENTER_STRICT_MAX):
                n_strict_drop += 1
                continue
            if area_frac < _MIN_AREA_FRAC:
                n_strict_drop += 1
                continue
            try:
                if float(det_row["detection_confidence"]) < _MIN_DET_CONF:
                    n_strict_drop += 1
                    continue
            except (KeyError, TypeError, ValueError):
                pass  # CVAT/hand sources have no conf — keep

            frame_stem = f"{_video_id_hash(video_id)}_{track_id:06d}_f{frame_number:06d}"
            jpeg_path = ls_frames_dir / f"{frame_stem}.jpg"

            if jpeg_path.is_file():
                img_check = cv2.imread(str(jpeg_path))
                if img_check is None:
                    continue
                fh_px, fw_px = img_check.shape[:2]
            else:
                frame_rgb = _read_frame(video_path, frame_number)
                if frame_rgb is None:
                    continue
                fh_px, fw_px = frame_rgb.shape[:2]
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(jpeg_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            x1 = max(0, min(x1, fw_px - 1))
            y1 = max(0, min(y1, fh_px - 1))
            x2 = max(x1 + 1, min(x2, fw_px))
            y2 = max(y1 + 1, min(y2, fh_px))
            w = x2 - x1
            h = y2 - y1

            img_id = len(images)
            images.append({
                "id": img_id,
                "file_name": str(jpeg_path),
                "width": fw_px,
                "height": fh_px,
                "source": "label_studio",
                "video_id": video_id,
                "track_id": track_id,
                "frame_number": frame_number,
                "dominant_species": class_id,
            })
            annotations.append({
                "id": len(annotations),
                "image_id": img_id,
                "category_id": class_id,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            track_added += 1

        if track_added == 0:
            n_skip += 1

    extra_msg = f", {n_extra} from extra-detections-dir" if n_extra else ""
    strict_msg = f", {n_strict_drop} bboxes dropped by strict y-sanity 0.30–0.85" if n_strict_drop else ""
    print(f"\n  Done: {len(images)} extracted, {n_skip} skipped{extra_msg}{strict_msg}")
    conn_labels.close()
    conn_tracks.close()
    return images, annotations


# ---------------------------------------------------------------------------
# Part 2: CVAT
# ---------------------------------------------------------------------------

def _process_cvat(
    cvat_json: Path,
    test_image_files: set[str],
    id_offset_images: int,
    id_offset_annotations: int,
    cvat_drop_anns: frozenset[int] = frozenset(),
) -> tuple[list, list]:
    """Load the non-test CVAT frames from a COCO annotations.json.

    Skips any image whose file_name is reserved for the test set, drops phantom
    (blank-frame) annotations, renumbers image/annotation ids by the given offsets so they
    concatenate cleanly with the LS block, and tags each image with its dominant species.

    Returns (images, annotations); reads from disk only, no frame extraction.
    """
    with open(cvat_json, encoding="utf-8") as f:
        coco = json.load(f)

    n_phantom = 0
    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["id"] in cvat_drop_anns:
            n_phantom += 1
            continue
        ann_by_image[ann["image_id"]].append(ann)
    if cvat_drop_anns:
        print(f"  CVAT phantom filter: dropped {n_phantom} blank-frame annotations")

    images: list[dict] = []
    annotations: list[dict] = []

    for img in coco["images"]:
        # KNOWN LIMITATION (deliberately unchanged): this is the ONLY exclusion on the CVAT path,
        # and it covers the older frame-level test set, not the held-out test set of the
        # manuscript. CVAT records carry `video_path`, not `video_id`, so the video-level guard
        # never applies to them. 25 frames from one held-out clip are in the published training
        # set as a result (module docstring, "Known limitations").
        if img["file_name"] in test_image_files:
            continue

        anns = ann_by_image.get(img["id"], [])
        if not anns:
            continue

        dominant = Counter(a["category_id"] for a in anns).most_common(1)[0][0]
        new_img_id = id_offset_images + len(images)

        images.append({
            **img,
            "id": new_img_id,
            "source": "cvat",
            "dominant_species": dominant,
        })
        for ann in anns:
            annotations.append({
                **ann,
                "id": id_offset_annotations + len(annotations),
                "image_id": new_img_id,
            })

    return images, annotations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import re as _re

# ---------------------------------------------------------------------------
# Seasonal validity for pseudo-labels
# Lake Ontario tributary fish runs (Credit & Ganaraska rivers)
# Key:   0=Chinook  1=Coho  2=Atlantic  3=Rainbow Trout  4=Brown Trout
#
# Update with tech-reviewed seasonal distributions when available.
# None months use all species (unknown season = no filter).
# ---------------------------------------------------------------------------
_SEASONAL_VALID: dict[int, frozenset] = {
    1:  frozenset([3, 4]),          # Jan  — Rainbow, Brown Trout
    2:  frozenset([3, 4]),          # Feb  — Rainbow, Brown Trout
    3:  frozenset([3]),             # Mar  — Rainbow Trout peak
    4:  frozenset([3]),             # Apr  — Rainbow Trout peak
    5:  frozenset([3]),             # May  — Rainbow Trout (tapering)
    6:  frozenset([3]),             # Jun  — Rainbow Trout
    7:  frozenset([3]),             # Jul  — Light activity, Rainbow
    8:  frozenset([3]),             # Aug  — Light activity, Rainbow
    9:  frozenset([0, 1, 3]),       # Sep  — Chinook starting, Rainbow
    10: frozenset([0, 1, 3]),       # Oct  — Chinook, Coho peak; Rainbow returning
    11: frozenset([0, 1, 2, 3, 4]),# Nov  — All species possible
    12: frozenset([3, 4]),          # Dec  — Rainbow, Brown Trout
}


def _extract_month(video_id: str) -> int | None:
    """Extract calendar month from a video_id path.

    Handles both date-range folders (MMDDYYYY-MMDDYYYY) and timestamp
    folders (YY  MM  DD  HH  MM).
    """
    # Date range: MMDDYYYY at start of a path component
    m = _re.search(r'/(\d{2})\d{2}\d{4}-\d{8}', video_id)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month
    # Timestamp folder: YY  MM  DD ...
    m = _re.search(r'/\d{2}\s{1,3}(\d{2})\s{1,3}\d{2}\s', video_id)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month
    return None


def _is_seasonally_valid(video_id: str, predicted_class: int) -> bool:
    """Return False if the species is implausible for the video's time of year."""
    month = _extract_month(video_id)
    if month is None:
        return True  # can't determine month — don't reject
    valid = _SEASONAL_VALID.get(month, frozenset(range(5)))
    return predicted_class in valid


def _process_pseudo_labels(
    pseudo_tracks_db: Path,
    pseudo_detections_dir: Path,
    labels_db: Path,
    pseudo_frames_dir: Path,
    min_conf: float,
    max_per_species: int,
    frames_per_track: int,
    id_offset_images: int,
    id_offset_annotations: int,
    allowed_classes: set | None = None,
    bad_parquet_stems: frozenset[str] = frozenset(),
) -> tuple[list, list]:
    """Extract high-confidence tracks from a newer inference run as pseudo-labels.

    Only uses tracks NOT already human-labeled (so labels.sqlite takes priority).
    Caps per-species to avoid overwhelming the human-labeled training signal.
    """
    conn_pseudo = sqlite3.connect(str(pseudo_tracks_db))
    conn_pseudo.row_factory = sqlite3.Row
    conn_labels = sqlite3.connect(str(labels_db))

    already_labeled = {
        (r[0], r[1]) for r in conn_labels.execute(
            "SELECT DISTINCT video_id, track_id FROM labels"
        ).fetchall()
    }
    conn_labels.close()

    from collections import Counter as _Counter
    per_species: _Counter = _Counter()
    images: list = []
    annotations: list = []
    n_skip = 0
    n_strict_drop = 0

    rows = conn_pseudo.execute("""
        SELECT video_id, track_id, predicted_class, mean_detection_confidence
        FROM tracks
        WHERE predicted_class_6 != 5
          AND mean_detection_confidence >= ?
        ORDER BY mean_detection_confidence DESC
    """, (min_conf,)).fetchall()
    conn_pseudo.close()

    for row in rows:
        cls_id = row["predicted_class"]
        if cls_id not in SPECIES:
            continue
        if allowed_classes is not None and cls_id not in allowed_classes:
            continue
        if per_species[cls_id] >= max_per_species:
            continue
        if (row["video_id"], row["track_id"]) in already_labeled:
            continue

        # Seasonal validity check — reject ecologically implausible predictions
        if not _is_seasonally_valid(row["video_id"], cls_id):
            continue

        parquet_stem = video_id_to_parquet_stem(row["video_id"])
        if parquet_stem in bad_parquet_stems:
            n_skip += 1
            continue
        parquet_file = pseudo_detections_dir / f"{parquet_stem}.parquet"
        if not parquet_file.is_file():
            n_skip += 1
            continue

        det_df = _load_track_detections(parquet_file, row["track_id"])
        if det_df is None or det_df.empty:
            n_skip += 1
            continue

        video_path = _resolve_video_path(row["video_id"])
        if video_path is None or not Path(video_path).is_file():
            n_skip += 1
            continue

        fw, fh = _get_video_dimensions(video_path)
        if fw == 0:
            n_skip += 1
            continue

        top_frames = _get_top_frames(det_df, fw, fh, n=frames_per_track)
        if not top_frames:
            n_skip += 1
            continue

        track_added = 0
        for frame_number, det_row in top_frames:
            x1 = int(det_row["bbox_x1"]); y1 = int(det_row["bbox_y1"])
            x2 = int(det_row["bbox_x2"]); y2 = int(det_row["bbox_y2"])

            # Strict per-bbox y-sanity safety net (see top of file).
            if not (_YCENTER_STRICT_MIN <= ((y1 + y2) / 2.0) / max(fh, 1) <= _YCENTER_STRICT_MAX):
                n_strict_drop += 1
                continue

            frame_stem = f"pseudo_{_video_id_hash(row['video_id'])}_{row['track_id']:06d}_f{frame_number:06d}"
            jpeg_path = pseudo_frames_dir / f"{frame_stem}.jpg"

            if jpeg_path.is_file():
                img_check = cv2.imread(str(jpeg_path))
                if img_check is None:
                    continue
                fh_px, fw_px = img_check.shape[:2]
            else:
                frame_rgb = _read_frame(video_path, frame_number)
                if frame_rgb is None:
                    continue
                fh_px, fw_px = frame_rgb.shape[:2]
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(jpeg_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            x1 = max(0, min(x1, fw_px - 1)); y1 = max(0, min(y1, fh_px - 1))
            x2 = max(x1 + 1, min(x2, fw_px)); y2 = max(y1 + 1, min(y2, fh_px))
            w = x2 - x1; h = y2 - y1

            img_id = id_offset_images + len(images)
            images.append({
                "id": img_id, "file_name": str(jpeg_path),
                "width": fw_px, "height": fh_px,
                "source": "pseudo_label",
                "video_id": row["video_id"], "track_id": row["track_id"],
                "frame_number": frame_number, "dominant_species": cls_id,
            })
            annotations.append({
                "id": id_offset_annotations + len(annotations),
                "image_id": img_id, "category_id": cls_id,
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
            })
            track_added += 1

        if track_added > 0:
            per_species[cls_id] += 1

    msg = f"  Pseudo-label tracks used: {sum(per_species.values()):,}  (skipped: {n_skip})"
    if n_strict_drop:
        msg += f"  (strict y-sanity dropped: {n_strict_drop})"
    print(msg)
    for sp_id, name in SPECIES.items():
        print(f"    {name}: {per_species[sp_id]:,}")
    return images, annotations


# ---------------------------------------------------------------------------
# Part 5: CVAT clips
# ---------------------------------------------------------------------------

def _process_cvat_clips(
    cvat_clips_db: Path,
    ls_test_pairs: set[tuple[str, int]],
    id_offset_images: int,
    id_offset_annotations: int,
) -> tuple[list, list]:
    """Load pre-extracted CVAT training clip frames from cvat_clips_tracks.sqlite.

    Each row has a best_id_frame JPEG path and bbox coords stored during extraction.
    These are treated identically to LS frames in training — one image + one annotation.
    Tracks whose (video_id, track_id) appear in ls_test_pairs are excluded.
    """
    conn = sqlite3.connect(str(cvat_clips_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT video_id, track_id, best_id_frame, species, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2 FROM tracks"
    ).fetchall()
    conn.close()

    images: list = []
    annotations: list = []
    n_skip_test = n_skip_missing = 0

    for row in rows:
        if (row["video_id"], row["track_id"]) in ls_test_pairs:
            n_skip_test += 1
            continue
        frame_path = Path(row["best_id_frame"])
        if not frame_path.is_file():
            n_skip_missing += 1
            continue

        cls_id = LABEL_TO_CLASS_ID.get(row["species"])
        if cls_id is None:
            continue

        import cv2 as _cv2
        img_check = _cv2.imread(str(frame_path))
        if img_check is None:
            n_skip_missing += 1
            continue
        fh_px, fw_px = img_check.shape[:2]

        x1 = max(0, row["bbox_x1"]); y1 = max(0, row["bbox_y1"])
        x2 = min(fw_px, row["bbox_x2"]); y2 = min(fh_px, row["bbox_y2"])
        w = max(1, x2 - x1); h = max(1, y2 - y1)

        img_id = id_offset_images + len(images)
        images.append({
            "id": img_id,
            "file_name": str(frame_path),
            "width": fw_px,
            "height": fh_px,
            "source": "cvat_clip",
            "video_id": row["video_id"],
            "track_id": row["track_id"],
            "dominant_species": cls_id,
        })
        annotations.append({
            "id": id_offset_annotations + len(annotations),
            "image_id": img_id,
            "category_id": cls_id,
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    if n_skip_test:
        print(f"  Skipped {n_skip_test} tracks in LS test queue")
    if n_skip_missing:
        print(f"  Skipped {n_skip_missing} tracks with missing frame files")
    return images, annotations


# ---------------------------------------------------------------------------
# Part 4: Hard negatives
# ---------------------------------------------------------------------------

def _process_hard_negatives(
    labels_db: Path,
    tracks_db: Path,
    detections_dir: Path,
    hard_neg_frames_dir: Path,
    frames_per_track: int,
    ls_test_pairs: set[tuple[str, int]],
    max_tracks: int,
    id_offset: int,
    bad_parquet_stems: frozenset[str] = frozenset(),
) -> list:
    """Return COCO image entries (no annotations) for not-a-fish labeled tracks.

    Frames are chosen by highest detection confidence — these are the clips where
    the model was most wrong, giving the strongest training signal.
    dominant_species=5 (background) keeps them in their own stratification bucket.
    """
    conn = sqlite3.connect(str(labels_db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT DISTINCT video_id, track_id
        FROM labels
        WHERE label_action = 'not_a_fish'
          AND (disagreement_resolution IS NULL OR disagreement_resolution = 'senior')
        ORDER BY video_id, track_id
    """).fetchall()
    conn.close()

    if ls_test_pairs:
        rows = [r for r in rows if (r["video_id"], int(r["track_id"])) not in ls_test_pairs]

    print(f"  Eligible not-a-fish tracks: {len(rows)}, capping at {max_tracks}")

    images: list = []
    n_tracks = 0
    n_skip = 0

    for row in rows:
        if n_tracks >= max_tracks:
            break

        video_id = row["video_id"]
        track_id = int(row["track_id"])

        video_path = _resolve_video_path(video_id)
        if video_path is None or not Path(video_path).is_file():
            n_skip += 1
            continue

        parquet_stem = video_id_to_parquet_stem(video_id)
        if parquet_stem in bad_parquet_stems:
            n_skip += 1
            continue
        parquet_file = detections_dir / f"{parquet_stem}.parquet"
        if not parquet_file.is_file():
            n_skip += 1
            continue

        det_df = _load_track_detections(parquet_file, track_id)
        if det_df is None or det_df.empty:
            n_skip += 1
            continue

        fw, fh = _get_video_dimensions(video_path)
        if fw == 0:
            n_skip += 1
            continue

        top_frames = _get_top_frames(det_df, fw, fh, n=frames_per_track)
        if not top_frames:
            n_skip += 1
            continue

        track_added = 0
        for frame_number, _ in top_frames:
            frame_stem = f"hardneg_{_video_id_hash(video_id)}_{track_id:06d}_f{frame_number:06d}"
            jpeg_path = hard_neg_frames_dir / f"{frame_stem}.jpg"

            if jpeg_path.is_file():
                img_check = cv2.imread(str(jpeg_path))
                if img_check is None:
                    continue
                fh_px, fw_px = img_check.shape[:2]
            else:
                frame_rgb = _read_frame(video_path, frame_number)
                if frame_rgb is None:
                    continue
                fh_px, fw_px = frame_rgb.shape[:2]
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(jpeg_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

            images.append({
                "id": id_offset + len(images),
                "file_name": str(jpeg_path),
                "width": fw_px,
                "height": fh_px,
                "source": "hard_negative",
                "video_id": video_id,
                "track_id": track_id,
                "frame_number": frame_number,
                "dominant_species": -1,  # background — no annotations (was 5; 5 is now Small Salmonid)
            })
            track_added += 1

        if track_added > 0:
            n_tracks += 1
        else:
            n_skip += 1

    print(f"  Hard negative tracks used: {n_tracks}  (skipped: {n_skip})")
    return images


# ---------------------------------------------------------------------------
# Part 6: Tech-reviewed per-video labels
# ---------------------------------------------------------------------------

# Riverwatcher species name → model class id.
# Handles slight variations in the tech-reviewer CSV exports.
_TECH_SPECIES_MAP: dict[str, int] = {
    "chinook": 0,
    "chinook salmon": 0,
    "coho": 1,
    "coho salmon": 1,
    "atlantic": 2,
    "atlantic salmon": 2,
    "rainbow": 3,
    "rainbow trout": 3,
    "brown": 4,
    "brown trout": 4,
}

# Infer site name from a path containing the site directory.
_SITE_NAMES = ("Ganaraska", "Credit")


def _infer_site(csv_path: Path) -> str | None:
    """Return 'Ganaraska'/'Credit' if either site name appears in the path, else None."""
    for part in csv_path.parts:
        for site in _SITE_NAMES:
            if site.lower() in part.lower():
                return site
    return None


def _build_tech_reverse_index(
    tracks_db: Path,
) -> dict[str, list[tuple[str, int, float]]]:
    """Build a lookup from (site/timestamp_folder/filename) → [(video_id, track_id, conf)].

    The Riverwatcher folder structure nests videos under year and date-range
    subdirectories that the tech-reviewed CSVs don't include.  For example:
      video_id  = "Ganaraska/Ganaraska 2025/10022025-10092025/25  10  02  09  51/1.mp4"
      video_rel = "25  10  02  09  51\\1.mp4"   (from CSV)

    The reverse index key is built from the last two path components of video_id
    (timestamp_folder/filename), prefixed with the site name:
      key = "Ganaraska/25  10  02  09  51/1.mp4"

    Values are sorted by mean_detection_confidence DESC so callers can take the
    top-N tracks without a second sort.
    """
    conn = sqlite3.connect(str(tracks_db))
    rows = conn.execute(
        "SELECT video_id, track_id, mean_detection_confidence FROM tracks"
    ).fetchall()
    conn.close()

    index: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for video_id, track_id, conf in rows:
        parts = video_id.split("/")
        if len(parts) < 3:
            continue
        # key: site + last two path components (timestamp_folder + filename)
        key = parts[0] + "/" + "/".join(parts[-2:])
        index[key].append((video_id, int(track_id), float(conf or 0)))

    # Sort each list by confidence DESC so callers can slice [:N] for top-N
    for key in index:
        index[key].sort(key=lambda t: t[2], reverse=True)

    return dict(index)


def _process_tech_labels(
    tech_labels_dir: Path,
    tracks_db: Path,
    detections_dir: Path,
    tech_frames_dir: Path,
    frames_per_track: int,
    ls_test_pairs: set[tuple[str, int]],
    max_per_species: int,
    id_offset_images: int,
    id_offset_annotations: int,
    allowed_classes: set[int] | None = None,
    class_limits: dict[int, int] | None = None,
    dry_run: bool = False,
    bad_parquet_stems: frozenset[str] = frozenset(),
) -> tuple[list, list]:
    """Extract training frames from tech-reviewed per-video fish count CSVs.

    Reads all fish_counts_export_*.csv files under tech_labels_dir, matches each
    event to inference tracks via a reverse index (site/timestamp_folder/filename),
    and extracts frames exactly like _process_ls_labels does for LS labels.

    Assignment logic:
      - For each tech-reviewed event (one row = one event in one video):
        * Skip: false_trigger=1, empty video_rel, species="Non fish"
        * Look up all tracks for that video using the reverse index
        * If 0 tracks → skip (video not in inference run)
        * If 1 track → use it, with tech species as ground truth
        * If N tracks and count ≤ N → take top-count tracks by detection_confidence
        * If count > N (more fish than detected) → skip (ambiguous)

    Tech species always overrides model's predicted_class.
    """
    csv_files = list(tech_labels_dir.rglob("fish_counts_export_*.csv"))
    if not csv_files:
        print(f"  WARNING: No fish_counts_export_*.csv found under {tech_labels_dir}")
        return [], []
    print(f"  Found {len(csv_files)} tech-reviewed CSV files")

    # Pre-load reverse index to avoid per-row DB queries
    print("  Building reverse index from tracks.sqlite...")
    reverse_index = _build_tech_reverse_index(tracks_db)
    print(f"  Reverse index: {len(reverse_index):,} unique (site/timestamp/file) keys")

    per_species: Counter = Counter()
    images: list = []
    annotations: list = []
    n_skip_filter = n_skip_no_tracks = n_skip_parquet = n_skip_video = n_skip_ambiguous = 0
    n_skip_test = 0
    n_strict_drop = 0

    for csv_path in sorted(csv_files):
        site = _infer_site(csv_path)
        if site is None:
            print(f"  WARNING: Cannot infer site from path: {csv_path} — skipping")
            continue

        try:
            df = pd.read_csv(str(csv_path))
        except Exception as e:
            print(f"  WARNING: Could not read {csv_path.name}: {e}")
            continue

        required = {"video_rel", "false_trigger", "species", "count"}
        if not required.issubset(df.columns):
            print(f"  WARNING: {csv_path.name} missing columns {required - set(df.columns)} — skipping")
            continue

        for _, row in df.iterrows():
            # --- filter ---
            if row.get("false_trigger", 0) == 1:
                n_skip_filter += 1
                continue
            video_rel = str(row.get("video_rel", "")).strip()
            if not video_rel or video_rel == "nan":
                n_skip_filter += 1
                continue
            species_str = str(row.get("species", "")).strip().lower()
            cls_id = _TECH_SPECIES_MAP.get(species_str)
            if cls_id is None:
                n_skip_filter += 1
                continue
            if allowed_classes is not None and cls_id not in allowed_classes:
                n_skip_filter += 1
                continue
            cap = class_limits.get(cls_id, max_per_species) if class_limits else max_per_species
            if per_species[cls_id] >= cap:
                continue

            try:
                tech_count = int(float(row.get("count", 1)))
            except (ValueError, TypeError):
                tech_count = 1
            tech_count = max(1, tech_count)

            # Reverse index key: "Site/timestamp_folder/filename"
            rel_normalized = video_rel.replace("\\", "/")
            lookup_key = site + "/" + rel_normalized

            track_entries = reverse_index.get(lookup_key)
            if not track_entries:
                n_skip_no_tracks += 1
                continue

            n_tracks = len(track_entries)
            if tech_count > n_tracks:
                n_skip_ambiguous += 1
                continue

            # top-count entries are already sorted by confidence DESC
            selected = track_entries[:tech_count]

            for (video_id, track_id, _conf) in selected:
                if ls_test_pairs and (video_id, track_id) in ls_test_pairs:
                    n_skip_test += 1
                    continue

                parquet_stem = video_id_to_parquet_stem(video_id)
                if parquet_stem in bad_parquet_stems:
                    n_skip_parquet += 1
                    continue
                parquet_file = detections_dir / f"{parquet_stem}.parquet"
                if not parquet_file.is_file():
                    n_skip_parquet += 1
                    continue

                det_df = _load_track_detections(parquet_file, track_id)
                if det_df is None or det_df.empty:
                    n_skip_parquet += 1
                    continue

                video_path = _resolve_video_path(video_id)
                if video_path is None or not Path(video_path).is_file():
                    n_skip_video += 1
                    continue

                if dry_run:
                    per_species[cls_id] += 1
                    continue

                fw, fh = _get_video_dimensions(video_path)
                if fw == 0:
                    n_skip_video += 1
                    continue

                top_frames = _get_top_frames(det_df, fw, fh, n=frames_per_track)
                if not top_frames:
                    n_skip_video += 1
                    continue

                track_added = 0
                for frame_number, det_row in top_frames:
                    x1 = int(det_row["bbox_x1"]); y1 = int(det_row["bbox_y1"])
                    x2 = int(det_row["bbox_x2"]); y2 = int(det_row["bbox_y2"])

                    # Strict per-bbox y-sanity safety net (see top of file).
                    if not (_YCENTER_STRICT_MIN <= ((y1 + y2) / 2.0) / max(fh, 1) <= _YCENTER_STRICT_MAX):
                        n_strict_drop += 1
                        continue

                    frame_stem = f"tech_{_video_id_hash(video_id)}_{track_id:06d}_f{frame_number:06d}"
                    jpeg_path = tech_frames_dir / f"{frame_stem}.jpg"

                    if jpeg_path.is_file():
                        img_check = cv2.imread(str(jpeg_path))
                        if img_check is None:
                            continue
                        fh_px, fw_px = img_check.shape[:2]
                    else:
                        frame_rgb = _read_frame(video_path, frame_number)
                        if frame_rgb is None:
                            continue
                        fh_px, fw_px = frame_rgb.shape[:2]
                        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(jpeg_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

                    x1 = max(0, min(x1, fw_px - 1)); y1 = max(0, min(y1, fh_px - 1))
                    x2 = max(x1 + 1, min(x2, fw_px)); y2 = max(y1 + 1, min(y2, fh_px))
                    w = x2 - x1; h = y2 - y1

                    img_id = id_offset_images + len(images)
                    images.append({
                        "id": img_id, "file_name": str(jpeg_path),
                        "width": fw_px, "height": fh_px,
                        "source": "tech_reviewed",
                        "video_id": video_id, "track_id": track_id,
                        "frame_number": frame_number, "dominant_species": cls_id,
                    })
                    annotations.append({
                        "id": id_offset_annotations + len(annotations),
                        "image_id": img_id, "category_id": cls_id,
                        "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                    })
                    track_added += 1

                if track_added > 0:
                    per_species[cls_id] += 1

    print(f"  Skipped: {n_skip_filter} (filtered/non-fish), {n_skip_no_tracks} (no inference tracks), "
          f"{n_skip_ambiguous} (ambiguous count), {n_skip_parquet} (missing parquet), "
          f"{n_skip_video} (video unavailable), {n_skip_test} (in test queue), "
          f"{n_strict_drop} (strict y-sanity 0.30–0.85)")
    if dry_run:
        print("  DRY RUN — no frames extracted. Match counts by species:")
    for sp_id, name in SPECIES.items():
        print(f"    {name}: {per_species[sp_id]:,}")
    return images, annotations


def _stratified_split(
    all_images: list,
    all_annotations: list,
    val_fraction: float,
    seed: int,
) -> tuple[list, list, list, list]:
    """Split images into train/val stratified by dominant_species (seeded, deterministic).

    Each species bucket is shuffled independently and its first ``val_fraction`` (>=1 image)
    goes to val; annotations follow their image via image_id. Returns
    (train_images, val_images, train_annotations, val_annotations).
    """
    rng = random.Random(seed)
    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in all_annotations:
        ann_by_image[ann["image_id"]].append(ann)

    by_species: dict[int, list] = defaultdict(list)
    for img in all_images:
        by_species[img["dominant_species"]].append(img)

    train_imgs: list = []
    val_imgs: list = []
    for sp, imgs in by_species.items():
        imgs = list(imgs)
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * val_fraction))
        val_imgs.extend(imgs[:n_val])
        train_imgs.extend(imgs[n_val:])

    val_ids = {img["id"] for img in val_imgs}
    train_anns = [a for a in all_annotations if a["image_id"] not in val_ids]
    val_anns = [a for a in all_annotations if a["image_id"] in val_ids]
    return train_imgs, val_imgs, train_anns, val_anns


def _get_top_frames(
    det_df: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    n: int,
) -> list[tuple[int, pd.Series]]:
    """Return top-N frames by best_id_score as (frame_number, detection_row) pairs."""
    if det_df.empty:
        return []
    fw, fh = float(frame_width), float(frame_height)
    cx = (det_df["bbox_x1"] + det_df["bbox_x2"]) / 2.0
    cy = (det_df["bbox_y1"] + det_df["bbox_y2"]) / 2.0
    bw = det_df["bbox_x2"] - det_df["bbox_x1"]
    bh = det_df["bbox_y2"] - det_df["bbox_y1"]
    bbox_area_norm = (bw * bh) / (fw * fh)
    dist = np.sqrt((cx - fw / 2) ** 2 + (cy - fh / 2) ** 2)
    half_diag = np.sqrt((fw / 2) ** 2 + (fh / 2) ** 2)
    dist_norm = dist / half_diag if half_diag > 0 else 0.0
    # "best_id_score" is re-derived here (not read from the DB): favour frames that are
    # confident AND show a large, well-centred fish (biggest, most-central = best ID view).
    scores = (
        det_df["detection_confidence"].values
        * bbox_area_norm.values
        * (1.0 - dist_norm.values)
    )
    top_indices = np.argsort(scores)[::-1][:n]
    return [
        (int(det_df.iloc[int(idx)]["frame_number"]), det_df.iloc[int(idx)])
        for idx in top_indices
    ]


def _get_best_id_frame(conn: sqlite3.Connection, video_id: str, track_id: int) -> int | None:
    """Return the stored best_id_frame index for a track, or None if absent/error."""
    try:
        row = conn.execute(
            "SELECT best_id_frame FROM tracks WHERE video_id=? AND track_id=?",
            (video_id, track_id),
        ).fetchone()
        if row and row["best_id_frame"] is not None:
            return int(row["best_id_frame"])
    except Exception:
        pass
    return None


def _load_track_detections(parquet_file: Path, track_id: int) -> pd.DataFrame | None:
    """Return the per-frame detection rows for one track_id from a parquet, or None."""
    try:
        tbl = pq.read_table(str(parquet_file), filters=[("track_id", "=", track_id)])
        return tbl.to_pandas() if len(tbl) > 0 else None
    except Exception as exc:  # noqa: BLE001
        # The caller counts a None as an ordinary "no detections" skip. The original returned
        # None here without a word, so a corrupt or re-formatted file dropped its fish silently.
        # The outcome is unchanged (still skipped); the reason is now printed.
        print(f"  WARNING: could not read {Path(parquet_file).name} for track {track_id}: "
              f"{type(exc).__name__}: {exc}")
        return None


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Return (frame_width, frame_height) for a video, or (0, 0) if it cannot be opened."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fw, fh


def _read_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Return frame as RGB numpy array. Tries decord first, falls back to cv2."""
    try:
        from decord import VideoReader, cpu  # type: ignore
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_idx = max(0, min(frame_idx, len(vr) - 1))
        return vr[frame_idx].asnumpy()
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resolve_video_path(video_id: str) -> str | None:
    """Map a 'site/...tail' video_id to its absolute footage path via VIDEO_ROOTS.

    Returns None if the leading site component is not a known footage root.
    """
    parts = video_id.split("/")
    if not parts:
        return None
    site = parts[0]
    if site not in VIDEO_ROOTS:
        return None
    tail = "/".join(parts[1:])
    return str(Path(VIDEO_ROOTS[site]) / tail)


def _video_id_hash(video_id: str) -> str:
    """Return a short (12-char) stable hash of a video_id, used in cached JPEG filenames."""
    return hashlib.sha1(video_id.encode()).hexdigest()[:12]


def _print_species_breakdown(annotations: list, label: str) -> None:
    """Print per-species annotation counts and percentages for a labelled block."""
    counts = Counter(a["category_id"] for a in annotations)
    total = sum(counts.values())
    print(f"\n  {label} annotation breakdown:")
    for sp_id, name in SPECIES.items():
        n = counts.get(sp_id, 0)
        pct = 100 * n / total if total else 0
        print(f"    {name}: {n} ({pct:.1f}%)")


def _parse_args() -> argparse.Namespace:
    """Define and parse the CLI. All path args are required and take no defaults."""
    p = argparse.ArgumentParser(description="Prepare DETR training dataset from LS + CVAT")
    p.add_argument("--labels-db", required=True, help="Path to labels.sqlite")
    p.add_argument("--tracks-db", required=True, help="Path to tracks.sqlite")
    p.add_argument("--detections-dir", required=True, help="Directory of detection Parquet files")
    p.add_argument(
        "--extra-detections-dir", action="append", default=None, metavar="PATH",
        help="Additional detections directory to search for track data (repeatable). "
             "Each dir is tried in order until the specific track_id is found. "
             "Use for labels from different inference runs (e.g. run_coho_v15, run_v10). "
             "Example: --extra-detections-dir .../run_coho_v15/detections "
             "         --extra-detections-dir .../run_v10/detections"
    )
    p.add_argument("--cvat-json", required=True, help="Path to CVAT annotations.json")
    p.add_argument("--test-manifest", required=True, help="Path to test_set_v1/manifest.json")
    p.add_argument("--output-dir", required=True, help="Output directory for training data")
    p.add_argument(
        "--frames-per-track", type=int, default=3,
        help="Number of frames to extract per LS track (default: 3)"
    )
    p.add_argument(
        "--ls-test-queue", action="append", default=None, metavar="PATH",
        help="Path to an LS test queue JSON; tracks in it are excluded from training "
             "by (video_id, track_id). Repeat for multiple queues.",
    )
    p.add_argument("--pseudo-tracks-db",  default=None,
                   help="tracks.sqlite from a newer inference run for pseudo-labels")
    p.add_argument("--pseudo-detections", default=None,
                   help="Detections directory for the pseudo-label inference run")
    p.add_argument("--pseudo-min-conf",   type=float, default=0.975,
                   help="Minimum confidence for pseudo-label tracks (default: 0.975)")
    p.add_argument(
        "--pseudo-classes", type=int, nargs="+", default=None, metavar="CLS",
        help="Restrict pseudo-labels to these class IDs "
             "(0=Chinook,1=Coho,2=Atlantic,3=Rainbow,4=Brown). "
             "Default: all classes. E.g. --pseudo-classes 0 1 for Chinook+Coho only."
    )
    p.add_argument("--pseudo-max-per-species", type=int, default=3000,
                   help="Max pseudo-label tracks per species (default: 3000)")
    p.add_argument("--hard-neg-max", type=int, default=500,
                   help="Max not-a-fish tracks to include as hard negatives (default: 500, 0=off)")
    p.add_argument(
        "--cvat-clips-db", default=None,
        help="Path to cvat_clips_tracks.sqlite from extract_cvat_training_clips.py "
             "(Part 5 — adds pre-extracted CVAT clip best-frames to training)",
    )
    p.add_argument(
        "--tech-labels-dir", default=None, metavar="PATH",
        help="Root directory containing tech-reviewed fish_counts_export_*.csv files "
             "(e.g. docs/Riverwatcher_2025). Subdirectories are searched recursively. "
             "Only files named fish_counts_export_*.csv are used (Vaki exports ignored)."
    )
    p.add_argument(
        "--tech-tracks-db", default=None, metavar="PATH",
        help="tracks.sqlite to match tech-reviewed video IDs against. "
             "Defaults to --tracks-db if not specified."
    )
    p.add_argument(
        "--tech-detections-dir", default=None, metavar="PATH",
        help="Detections directory for tech-reviewed frame extraction. "
             "Defaults to --detections-dir if not specified."
    )
    p.add_argument(
        "--tech-max-per-species", type=int, default=1000,
        help="Max tech-reviewed tracks per species (default: 1000)"
    )
    p.add_argument(
        "--tech-max-per-class", type=str, nargs="+", default=None, metavar="CLS:MAX",
        help="Per-class cap overrides for tech labels, e.g. --tech-max-per-class 0:1500 1:3000. "
             "Overrides --tech-max-per-species for the specified class IDs."
    )
    p.add_argument(
        "--tech-classes", type=int, nargs="+", default=None, metavar="CLS",
        help="Whitelist of class IDs to include from tech-reviewed labels "
             "(default: all). E.g. --tech-classes 0 1 3 4 excludes Atlantic (2)."
    )
    # Second tech source (optional) — same matching logic, separate tracks.sqlite + detections dir
    p.add_argument(
        "--tech-tracks-db2", default=None, metavar="PATH",
        help="Second inference tracks.sqlite for tech-reviewed label matching "
             "(e.g. run_rainbow_v15/tracks.sqlite). Uses same --tech-labels-dir CSVs."
    )
    p.add_argument(
        "--tech-detections-dir2", default=None, metavar="PATH",
        help="Detections directory for the second tech source. "
             "Defaults to --detections-dir if not specified."
    )
    p.add_argument(
        "--tech-max-per-species-2", type=int, default=None,
        help="Max tech-reviewed tracks per species for the second source "
             "(default: same as --tech-max-per-species)."
    )
    p.add_argument(
        "--tech-max-per-class-2", type=str, nargs="+", default=None, metavar="CLS:MAX",
        help="Per-class cap overrides for the second tech source, e.g. --tech-max-per-class-2 3:800."
    )
    p.add_argument(
        "--tech-classes-2", type=int, nargs="+", default=None, metavar="CLS",
        help="Whitelist of class IDs for the second tech source "
             "(default: all). E.g. --tech-classes-2 3 for Rainbow only."
    )
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only N LS tracks (for testing)"
    )
    p.add_argument(
        "--bbox-audit-csv", action="append", default=None, metavar="PATH",
        help="Path to parquet_audit.csv (the stored-detection box audit; optional). "
             "Tracks whose source parquet was flagged wrong_y_too_high/low or ambiguous "
             "are skipped across LS, pseudo, tech, and hard-negative collection. "
             "Repeatable — pass once per inference run."
    )
    p.add_argument(
        "--cvat-audit-csv", default=None, metavar="PATH",
        help="Path to cvat_annotation_audit.csv (the CVAT annotation audit; optional). "
             "CVAT annotations on blank frames (status frame_empty/manual_blank) are "
             "dropped — prevents the detector learning to fire on empty water."
    )
    return p.parse_args()


if __name__ == "__main__":
    main()

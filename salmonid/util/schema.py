"""
In plain terms: this file is the "blank forms" for the fish counter's output. It
defines exactly what gets recorded for every fish the model finds — the species
scores, the box drawn around the fish in each video frame, which way it swam, and
the database tables that hold one row per fish-track. It does not produce a number
in the paper by itself; instead it fixes the layout that every count, movement, and
species result is read from, and it stamps each file with a version tag so later
steps know the fish-position coordinates were computed correctly.

Output schemas for inference pipeline v2.

SCHEMA_VERSION, dataclasses, PyArrow schema for per-detection Parquet,
and SQLite DDL for tracks and processing-log databases.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pyarrow as pa

SCHEMA_VERSION: int = 2

# Bbox pipeline version — bump WHENEVER a coordinate-relevant change lands in
# inference/pipeline.py (post-processing, target_sizes, letterbox, padding).
#
# History:
#   "pre_2026-05-26"        — old pipeline (target_sizes=[[max_dim, max_dim]]).
#                             Y was under-scaled by 0.75 on landscape frames.
#                             DO NOT WRITE THIS VALUE — only legacy parquets
#                             carry it (when re-stamped by fix_bbox_y.py).
#   "fixed_2026-05-26"      — pipeline.py uses target_sizes=[[H, W]] (the
#                             aspect-preserving post-process scale). This is
#                             what every parquet written from the corrected
#                             pipeline must declare.
#
# Inference writes the current value into parquet metadata under the
# ``bbox_pipeline_version`` key. fix_bbox_y.py, when post-hoc-correcting an
# old file, stamps "corrected_pre_2026-05-26" so downstream prep code can
# distinguish "born clean" from "rescued by the fix script."
#
# Downstream prepare_* scripts read this key and refuse parquets whose
# version is in BBOX_KNOWN_BAD_PIPELINE_VERSIONS unless an audit CSV
# explicitly vouches for the file, or --allow-unknown-pipeline-version is set.
BBOX_PIPELINE_VERSION: str = "fixed_2026-05-26"
BBOX_KNOWN_BAD_PIPELINE_VERSIONS: frozenset[str] = frozenset({"pre_2026-05-26"})

# Class labels matching model/mar10_2026_detr-resnet50/config.json id2label.
# Index 5 is the DETR no-object class, stored as "Background" throughout.
FISH_CLASSES: dict[int, str] = {
    0: "Chinook",
    1: "Coho",
    2: "Atlantic",
    3: "Rainbow Trout",
    4: "Brown Trout",
}
BACKGROUND_CLASS_IDX: int = 5
ALL_CLASSES: dict[int, str] = {**FISH_CLASSES, BACKGROUND_CLASS_IDX: "Background"}
NUM_CLASSES: int = len(ALL_CLASSES)  # 6


@dataclass
class DetectionRecord:
    video_id: str
    frame_number: int
    timestamp_seconds: float
    track_id: int
    detection_id: int        # 0-based row index within this video's Parquet
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    detection_confidence: float
    prob_chinook: float
    prob_coho: float
    prob_atlantic: float
    prob_rainbow: float
    prob_brown: float
    prob_background: float
    predicted_class: int     # argmax of fish-only probs (0–4, excludes Background)
    predicted_class_6: int   # argmax across all 6 classes (0–5, includes Background)


@dataclass
class TrackSummary:
    video_id: str
    track_id: int
    start_frame: int
    end_frame: int
    start_timestamp_seconds: float
    end_timestamp_seconds: float
    n_frames: int
    mean_prob_chinook: float
    mean_prob_coho: float
    mean_prob_atlantic: float
    mean_prob_rainbow: float
    mean_prob_brown: float
    mean_prob_background: float
    predicted_class: int     # argmax of mean fish-only probs (0–4)
    predicted_class_6: int   # argmax of mean probs across all 6 classes
    mean_detection_confidence: float
    max_detection_confidence: float  # peak confidence across all track frames
    direction: str           # "Left" | "Right" | "Unknown"
    entrance_side: str       # first side fish appeared on: "Left" | "Right" | "None"
    exit_side: str           # last side fish appeared on: "Left" | "Right" | "None"
    representative_frame: int
    # Spatial summary — centre_x mean of first/last 3 boxes; delta_x = end_x - start_x
    start_x: float
    end_x: float
    delta_x: float
    # Site name (first component of video_id, e.g. "Ganaraska" or "Credit")
    site: str
    # 1 = moving upstream, 0 = downstream, None = direction unknown
    upstream: int | None
    # Foreign key to run_config.run_id for the inference run that produced this track
    run_id: str


# ---------------------------------------------------------------------------
# PyArrow schema for per-detection Parquet files.
# schema_version is stored as file-level metadata so every file is self-describing.
#
# ``bbox_pipeline_version`` is the authoritative key for "was this file
# written by a coordinate-correct pipeline?" — bumped via BBOX_PIPELINE_VERSION
# above whenever post-processing geometry changes. Downstream training-prep
# scripts gate on it. fix_bbox_y.py overwrites it with "corrected_<src>" when
# it post-hoc-fixes a legacy buggy file.
#
# ``y_coord_fix_applied`` is retained for backwards compatibility with the
# original fix_bbox_y.py stamp convention. New code should prefer the
# version key; this field stays so older audit / fix runs remain idempotent.
# ---------------------------------------------------------------------------
DETECTION_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("video_id", pa.string()),
        pa.field("frame_number", pa.int32()),
        pa.field("timestamp_seconds", pa.float32()),
        pa.field("track_id", pa.int32()),
        pa.field("detection_id", pa.int32()),
        pa.field("bbox_x1", pa.int16()),
        pa.field("bbox_y1", pa.int16()),
        pa.field("bbox_x2", pa.int16()),
        pa.field("bbox_y2", pa.int16()),
        pa.field("detection_confidence", pa.float32()),
        pa.field("prob_chinook", pa.float32()),
        pa.field("prob_coho", pa.float32()),
        pa.field("prob_atlantic", pa.float32()),
        pa.field("prob_rainbow", pa.float32()),
        pa.field("prob_brown", pa.float32()),
        pa.field("prob_background", pa.float32()),
        pa.field("predicted_class", pa.int8()),
        pa.field("predicted_class_6", pa.int8()),
    ],
    metadata={
        b"schema_version":         str(SCHEMA_VERSION).encode(),
        b"bbox_pipeline_version":  BBOX_PIPELINE_VERSION.encode(),
        # legacy compatibility (older code reads this; do NOT trust it as
        # evidence of correctness — see CLAUDE.md "bbox bug" section)
        b"y_coord_fix_applied":    b"baseline_post_fix",
    },
)


# ---------------------------------------------------------------------------
# SQLite DDL
# ---------------------------------------------------------------------------
TRACKS_DDL: str = """
CREATE TABLE IF NOT EXISTS tracks (
    video_id                    TEXT    NOT NULL,
    track_id                    INTEGER NOT NULL,
    start_frame                 INTEGER NOT NULL,
    end_frame                   INTEGER NOT NULL,
    start_timestamp_seconds     REAL,
    end_timestamp_seconds       REAL,
    n_frames                    INTEGER NOT NULL,
    mean_prob_chinook           REAL,
    mean_prob_coho              REAL,
    mean_prob_atlantic          REAL,
    mean_prob_rainbow           REAL,
    mean_prob_brown             REAL,
    mean_prob_background        REAL,
    predicted_class             INTEGER,
    predicted_class_6           INTEGER,
    mean_detection_confidence   REAL,
    max_detection_confidence    REAL,
    direction                   TEXT,
    entrance_side               TEXT,
    exit_side                   TEXT,
    representative_frame        INTEGER,
    start_x                     REAL,
    end_x                       REAL,
    delta_x                     REAL,
    site                        TEXT,
    upstream                    INTEGER,
    run_id                      TEXT,
    PRIMARY KEY (video_id, track_id)
)
"""

RUN_CONFIG_DDL: str = """
CREATE TABLE IF NOT EXISTS run_config (
    run_id                  TEXT PRIMARY KEY,
    started_at              TEXT NOT NULL,
    model_checkpoint        TEXT,
    model_name              TEXT,
    box_score_thresh        REAL,
    per_species_thresh      TEXT,
    frame_skip              INTEGER,
    frame_batch_size        INTEGER,
    tracker                 TEXT,
    fp16                    INTEGER,
    botsort_track_high_thresh  REAL,
    botsort_track_low_thresh   REAL,
    botsort_new_track_thresh   REAL,
    botsort_track_buffer       INTEGER,
    botsort_match_thresh       REAL,
    min_frames_for_track    INTEGER,
    min_displacement_frac   REAL,
    min_displacement_px     REAL,
    schema_version          INTEGER
)
"""

DETECTIONS_DDL: str = """
CREATE TABLE IF NOT EXISTS detections (
    video_id                TEXT    NOT NULL,
    frame_number            INTEGER NOT NULL,
    track_id                INTEGER NOT NULL,
    bbox_x1                 INTEGER,
    bbox_y1                 INTEGER,
    bbox_x2                 INTEGER,
    bbox_y2                 INTEGER,
    detection_confidence    REAL,
    prob_chinook            REAL,
    prob_coho               REAL,
    prob_atlantic           REAL,
    prob_rainbow            REAL,
    prob_brown              REAL,
    prob_background         REAL,
    predicted_class         INTEGER,
    predicted_class_6       INTEGER,
    PRIMARY KEY (video_id, frame_number, track_id)
)
"""

PROCESSING_LOG_DDL: str = """
CREATE TABLE IF NOT EXISTS processing_log (
    video_id                        TEXT    PRIMARY KEY,
    status                          TEXT    NOT NULL,
    processed_at                    TEXT    NOT NULL,
    processing_duration_seconds     REAL,
    n_detections                    INTEGER,
    n_tracks                        INTEGER,
    error_message                   TEXT
)
"""

# Both tracks.sqlite and processing_log.sqlite include a meta table for
# schema_version and any future key/value metadata.
META_DDL: str = """
CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
)
"""

_META_INSERT_VERSION: str = (
    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)"
)


def init_tracks_db(conn) -> None:
    """Create tracks, run_config, detections, and meta tables. Idempotent."""
    conn.execute(TRACKS_DDL)
    conn.execute(RUN_CONFIG_DDL)
    conn.execute(DETECTIONS_DDL)
    conn.execute(META_DDL)
    conn.execute(_META_INSERT_VERSION, (str(SCHEMA_VERSION),))
    # Indexes for common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_site ON tracks (site)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_predicted_class ON tracks (predicted_class)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_run_id ON tracks (run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_video ON detections (video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_track ON detections (video_id, track_id)")
    conn.commit()


def init_processing_log_db(conn) -> None:
    """Create processing_log and meta tables; stamp schema_version. Idempotent."""
    conn.execute(PROCESSING_LOG_DDL)
    conn.execute(META_DDL)
    conn.execute(_META_INSERT_VERSION, (str(SCHEMA_VERSION),))
    conn.commit()


# ---------------------------------------------------------------------------
# Parquet filename derivation
# ---------------------------------------------------------------------------
def video_id_to_parquet_stem(video_id: str) -> str:
    """Return the Parquet filename stem for a given video_id.

    Uses the first 12 hex chars of SHA1(video_id). 12 chars gives ~4.8e14
    possible values; at 244k videos the birthday-collision probability is
    effectively zero (~1e-7). The user-facing example used 8 chars, but 8
    chars (~4.3e9 values) has near-certain collision at this scale.

    Example: "Ganaraska/Ganaraska 2020/1.mp4" -> "det_a0b1c2d3e4f5"
    """
    digest = hashlib.sha1(video_id.encode()).hexdigest()
    return f"det_{digest[:12]}"

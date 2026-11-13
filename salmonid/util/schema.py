"""In plain terms: an earlier version of the fish counter saved, for every video, one small
table listing each box it drew around a fish in each frame. Those tables are still used in
two places: they hold the frame-by-frame boxes of the 366 held-out test fish, and they were
one source of training pictures. This file contains the single rule that says what such a
table is called on disk, so that every script looks for the same file name. It also records,
in one place, which number stands for which species.

Inputs
------
None. ``video_id_to_parquet_stem`` takes a video identifier (text) and returns a file name.

Outputs
-------
None. Nothing is read or written here.

Feeds
-----
No manuscript number directly. The naming rule is used by the scripts that read the
reference detection tables: ``salmonid.eval.predict_testset_v6_from_store`` (the boxes of
the held-out test fish, behind Table 1 and Figure 2), ``salmonid.eval.build_holdout_testset_queue``
(test-set construction), ``salmonid.data.prepare_dataset`` and
``salmonid.data.audit_coco_holdout_leak`` (training-set construction and its leak audit).

Run
---
Not run directly; imported by the modules above.

Tier
----
Not applicable (a helper with no data of its own; standard library only).

Status
------
Published path (a helper used by published-path scripts). The rest of the original module
was removed, see "What was removed" below.

Class numbers used by the published model
-----------------------------------------
The published model (v2.7) is a single network that both finds each fish and scores its
species. It gives every box six scores, in this fixed order:

    0 Chinook Salmon    1 Coho Salmon    2 Atlantic Salmon
    3 Rainbow Trout     4 Brown Trout    5 Small Salmonid

Classes 0 to 4 are the five species the manuscript reports (``salmonid.config.SPECIES``).
Class 5, "Small Salmonid", is a catch-all for juveniles too small to identify. The model
detects it, but it is excluded from the reported counts and never appears in the technician
record. The six scores are independent values between 0 and 1; they are not shares that add
up to 1, and there is no "background" class. ``salmonid.config.NUM_DETECT_CLASSES`` (6) and
``salmonid.detect.deimv2_infer.SPECIES`` hold the same list for the code.

What was removed
----------------
Until this cleanup the module also carried the output layout of the retired two-stage
pipeline (a detector followed by a separate species classifier): record classes, a table
layout for the per-frame files, and database table definitions. None of it was imported
anywhere in this package. It also labelled class 5 "Background", which was true of the
retired detector and is wrong for the published model. The full original is in the git
history of this file.

Two facts from the removed text are kept because other docstrings refer to them:

* In the retired pipeline's track tables, ``delta_x`` is the mean horizontal centre of a
  track's last three boxes minus that of its first three, in pixels. The published counting
  code uses the same definition (see ``docs/COUNTING_METHOD.md``).
* The reference detection tables are stored one file per video under the name returned by
  ``video_id_to_parquet_stem``, in Parquet format (a compact column-oriented table file).
"""
from __future__ import annotations

import hashlib


def video_id_to_parquet_stem(video_id: str) -> str:
    """Return the file-name stem of the per-video detection table for ``video_id``.

    A video is identified by its path below the footage folder, for example
    ``"Ganaraska/Ganaraska 2020/.../1.mp4"``. That text contains slashes and spaces, so it
    cannot be used as a file name. Instead the name is ``det_`` followed by the first 12
    hexadecimal characters of the SHA-1 fingerprint of the identifier, e.g.
    ``det_a0b1c2d3e4f5``; the table itself is ``<stem>.parquet``.

    Why 12 characters: 12 hexadecimal characters give 16**12 (about 2.8e14) possible names.
    With the roughly 244,000 videos of the archive, the chance that any two videos share a
    name is about 1 in 10,000. The function must not be changed: the existing tables were
    written under these names.
    """
    digest = hashlib.sha1(video_id.encode()).hexdigest()
    return f"det_{digest[:12]}"

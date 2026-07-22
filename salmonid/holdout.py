"""In plain terms: this file keeps the fixed set of "exam" fish — the ones used to
grade the model — out of the pile of fish the model learns from, so it can never
be trained on the very fish it is later tested on. That separation is what makes
the paper's headline accuracy honest: the held-out test-set score (the 0.966
figure) only means something because these fish were provably never seen during
training.

Canonical eval-holdout GUARD — the permanent leak fix.

WHY THIS EXISTS (the bug it prevents)
-------------------------------------
An earlier iteration of this project leaked evaluation fish into the *training* set: one
data-prep script honoured an ``--exclude`` flag, another forgot it, so a subset of the
operational test videos ended up as training crops and inflated the reported accuracy.
The fix is architectural, not a flag: **every** training-prep entry point
(``data.build_extra_coco``, ``data.merge_synth_into_coco``, ``data.sam2_composite``,
``data.prepare_dataset``) imports :func:`load_holdout_pairs` and drops these
``(video_id, track_id)`` pairs *unconditionally* — there is no CLI switch to forget.

Add a new frozen eval set to :data:`CANONICAL_HOLDOUT_FILES` **once** and every builder
honours it automatically. This is the single most important guard for an honest number:
the manuscript's held-out ``test_set_v6`` score is only meaningful because these tracks
were provably never seen during training.

FILE FORMAT
-----------
Each holdout JSON lists tracks either as ``{"tracks": [[video_id, track_id], ...]}`` or
``{"items": [{"video_id":..., "track_id":...}, ...]}``; both are unioned.
"""
from __future__ import annotations

import json
from pathlib import Path

from salmonid import config

# Frozen eval sets, as track-level (video_id, track_id) holdouts.
#   * test_set_v6_holdout.json  — the frozen, publishable held-out test set (committed to this
#     repo under data/, so the guard is self-contained on a fresh clone).
#   * ls_test_set_v5.json       — the label-studio clip-average eval set (external; if you have
#     not downloaded it the guard simply warns and continues — it is only relevant when you
#     rebuild the training COCO, not for scoring).
CANONICAL_HOLDOUT_FILES = [
    config.TESTSET_V6_HOLDOUT,
    config.LABEL_BATCHES / "ls_test_set_v5.json",
]


def load_holdout_pairs(extra_files=None, verbose: bool = True) -> set:
    """Return the union of (video_id, int track_id) pairs to exclude from ALL training.

    Args:
        extra_files: optional additional holdout JSONs to union in (same format).
        verbose: print a per-file tally of how many pairs each contributed.

    Missing files are skipped with a warning (never a hard error) so that rebuilding the
    training data does not require every external eval set to be present — but the committed
    ``test_set_v6_holdout.json`` is always available, so the publishable test set is always
    excluded.
    """
    pairs: set = set()
    for f in list(CANONICAL_HOLDOUT_FILES) + list(extra_files or []):
        p = Path(f)
        if not p.is_file():
            if verbose:
                print(f"  holdout guard: WARNING missing {p.name}")
            continue
        d = json.loads(p.read_text())
        n0 = len(pairs)
        # accept either the "tracks" (list-of-pairs) or "items" (list-of-dicts) layout
        for t in d.get("tracks", []):
            pairs.add((t[0], int(t[1])))
        for it in d.get("items", []):
            pairs.add((it["video_id"], int(it["track_id"])))
        if verbose:
            print(f"  holdout guard: +{len(pairs) - n0} pairs from {p.name}")
    return pairs

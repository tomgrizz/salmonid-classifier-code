"""In plain terms: this file keeps the fixed set of "exam" fish (the 366 fish used to grade
the model) out of the pile of fish the model learns from. A model that has already seen an
exam fish during training will score better on it than it deserves, so the paper's headline
accuracy only means something if this separation really held. Every script that prepares
training pictures asks this file "is this clip an exam clip?" and drops the clip if the
answer is yes. The answer is given for the whole VIDEO CLIP, not for one numbered fish inside
it, because an earlier version that asked about numbered fish let 62 exam fish slip through.

Inputs
------
* ``data/test_set_v6_holdout.json`` (committed): the published held-out test set, 366 fish
  in 364 video clips. It lists the clips (``video_ids``) and the fish (``tracks`` / ``items``).
* ``<SALMONID_PROJECTS_ROOT>/label_batches/ls_test_set_v5.json`` (external, 180 fish): an
  older evaluation set that the original builds also kept out of training. It is not used for
  any number in the manuscript, but the published training set was cleaned against it too.

Outputs
-------
Nothing on disk. :func:`load_holdout_pairs` returns the set of held-out fish and clips that
the data builders and the leak audit test against.

Feeds
-----
The manuscript's claim (Methods, model training) that the held-out test set was excluded
from every data-preparation path at the video level, and therefore the held-out species
score of the published v2.7 model (macro-F1 0.961 on 366 fish). It is the guard behind
``salmonid.data.audit_coco_holdout_leak``, which is the check that the training launcher
(``salmonid.train.train_v27``) runs before any training.

Run
---
Not run on its own. To see what it loads::

    python -c "from salmonid.holdout import load_holdout_pairs as f; k = f(); print(len(k), len(k.videos))"

Tier
----
1 for the published test set (its file is committed). The second file is Tier 2; without it
the guard stops with an error unless ``SALMONID_ALLOW_MISSING_HOLDOUT=1`` is set (see below).

Why the guard is by video clip, not by fish number
--------------------------------------------------
The first guard kept a list of ``(video_id, track_id)`` pairs and dropped a training picture
only when both matched. A ``track_id`` is a number the tracking software hands out, and it
does not identify a fish:

* the numbers are assigned afresh every time the detector is re-run, so the same fish carries
  one number in the run the test set was drawn from and another in the run a training label
  was mined from;
* one of the data builders (``data.build_extra_coco``) uses ``track_id == -1`` as shorthand
  for "the main fish in this clip", which is usually the very fish the test set holds out.

Matching on pairs therefore passed 62 of the 366 test fish into the training set of the
previous model (v2.6) with their own species label: same clip, same frame, the same box.
That model scored 0.966, but the number was inflated and is not used in the manuscript.

The fix, ported here unchanged from the research repository (commit ``d40b869``), is the
:class:`HoldoutKeys` set. It still holds the pairs, but asking ``(video_id, track_id) in keys``
now also answers True whenever ``video_id`` is a held-out clip, whatever the track number is.
Every existing ``(vid, tid) in holdout`` test in the builders therefore became a video-level
test without its own logic changing.

What the guard still does NOT cover (known limitations of the published model)
-------------------------------------------------------------------------------
These are documented, not fixed, because fixing them would change the training data that the
published v2.7 model was trained on.

* **Pictures with no ``video_id``.** Frames that came from the CVAT annotation tool record an
  absolute ``video_path`` and no ``video_id``, so neither this guard nor the audit keyed on
  them. The published training set therefore still contains 25 CVAT frames (and the
  validation set 2) from the clip of one fish of the published test set, an Atlantic Salmon
  (track 353168). It also contains 560 CVAT frames (validation: 65) from 10 clips of the
  older ``ls_test_set_v5`` evaluation set, which no manuscript number uses.
  ``salmonid.data.audit_coco_holdout_leak`` reports both groups under "UNKEYED".
* **Neighbouring clips.** The camera can record the same fish in several consecutive clips.
  A video-level guard removes the held-out clip but not the clip recorded just before or
  after it. 76 of the 366 held-out fish have a training picture of the same species from the
  immediately adjacent clip. Whether those are the same individuals is not known.
  ``audit_coco_holdout_leak --report-adjacent`` counts them.

Fail-closed behaviour
---------------------
A guard that cannot find its list of exam fish and carries on with an empty list is the
failure this module exists to prevent, so a missing file is an error. To rebuild or audit
with only the committed test set (the second file is external), set the environment variable
``SALMONID_ALLOW_MISSING_HOLDOUT=1``: the missing file is then reported loudly and skipped.
(The research repository called this switch ``FISHCV_ALLOW_MISSING_HOLDOUT``; that name is
still honoured so the original runners behave as they did.)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from salmonid import config

# Frozen evaluation sets. Each file lists held-out fish as (video_id, track_id) pairs and,
# for the published test set, the list of video clips they live in.
#   * test_set_v6_holdout.json : the published held-out test set. Committed under data/, and
#     identical (apart from line endings) to the copy the training data was built against;
#     see provenance/v2.7/CHECKSUMS.md.
#   * ls_test_set_v5.json      : an older evaluation set (180 fish). External. It has no
#     `video_ids` list, so its clips are taken from its pairs (see _read_holdout_file).
CANONICAL_HOLDOUT_FILES = [
    config.TESTSET_V6_HOLDOUT,
    config.LABEL_BATCHES / "ls_test_set_v5.json",
]

# Environment switches that downgrade "holdout file missing" from an error to a warning.
# The first is the deposit's name; the second is the research repository's original name.
_ALLOW_MISSING_ENV = ("SALMONID_ALLOW_MISSING_HOLDOUT", "FISHCV_ALLOW_MISSING_HOLDOUT")


class HoldoutKeys(set):
    """Held-out (video_id, track_id) pairs that ALSO answer True for any track in a held-out video.

    ``(vid, tid) in keys`` is True when the exact pair is held out OR when ``vid`` is a
    held-out video. The second case is what catches the ``track_id == -1`` shorthand and track
    numbers from a different detector run. ``.videos`` exposes the set of held-out clips for
    builders that filter pictures rather than tracks.

    It subclasses ``set`` on purpose: the builders already wrote ``(vid, tid) in holdout``,
    and changing what ``in`` means fixed all of them at once. The price is one trap, noted at
    each call site: ``plain_set |= keys`` copies only the pairs and silently loses the video
    test. Callers must keep the ``HoldoutKeys`` object itself.
    """

    def __init__(self, pairs=(), videos=()):
        super().__init__(pairs)
        self.videos: set = set(videos)

    def __contains__(self, key) -> bool:  # noqa: D105
        # Video first: a 2-tuple whose first element is a held-out clip is held out, whatever
        # its second element is. `type(key) is tuple` (not isinstance) keeps this to the plain
        # (video_id, track_id) form the builders use.
        if type(key) is tuple and len(key) == 2 and key[0] in self.videos:
            return True
        return super().__contains__(key)

    def held(self, video_id: str, track_id=None) -> bool:
        """Explicit form. A held-out video is held out whatever (or whether) the track id is."""
        if video_id in self.videos:
            return True
        try:
            # set.__contains__ (not `in self`) tests the exact pair only; the video test
            # already ran on the line above.
            return set.__contains__(self, (video_id, int(track_id)))
        except (TypeError, ValueError):
            # The track id is missing or not a number. The clip itself was checked above and
            # is not held out, and a pair with no usable track id cannot match any held-out
            # pair, so "not held" is the correct answer here, not a swallowed error.
            return False


def _read_holdout_file(path: Path) -> tuple[set, set]:
    """-> (pairs, videos). Videos come from the file's ``video_ids`` list, else from its pairs.

    Both layouts are accepted and unioned: ``{"tracks": [[video_id, track_id], ...]}`` and
    ``{"items": [{"video_id": ..., "track_id": ...}, ...]}``.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    pairs = {(t[0], int(t[1])) for t in d.get("tracks", [])}
    pairs |= {(it["video_id"], int(it["track_id"])) for it in d.get("items", [])}
    videos = set(d.get("video_ids", []))
    if not videos:
        # Older files list fish only. Every clip that holds a held-out fish is a held-out clip.
        videos = {v for v, _ in pairs}
    return pairs, videos


def _load(extra_files=None, verbose: bool = True) -> HoldoutKeys:
    pairs: set = set()
    videos: set = set()
    for f in list(CANONICAL_HOLDOUT_FILES) + list(extra_files or []):
        p = Path(f)
        if not p.is_file():
            msg = (f"holdout guard: MISSING {p} — training data cannot be verified leak-free. "
                   f"Set {_ALLOW_MISSING_ENV[0]}=1 to continue with the files that are present.")
            if any(os.environ.get(name) == "1" for name in _ALLOW_MISSING_ENV):
                # Printed even when verbose=False: skipping part of the guard must never be quiet.
                print(f"  WARNING {msg}")
                continue
            raise FileNotFoundError(msg)
        p_new, v_new = _read_holdout_file(p)
        n0, m0 = len(pairs), len(videos)
        pairs |= p_new
        videos |= v_new
        if verbose:
            print(f"  holdout guard: +{len(pairs) - n0} tracks / +{len(videos) - m0} videos from {p.name}")
    return HoldoutKeys(pairs, videos)


def load_holdout_pairs(extra_files=None, extra_pairs=None, verbose: bool = True) -> HoldoutKeys:
    """The union of held-out tracks to exclude from ALL training, as a video-aware key set.

    Args:
        extra_files: more holdout JSON files to add on top of ``CANONICAL_HOLDOUT_FILES``.
        extra_pairs: caller-supplied (video_id, track_id) pairs to fold in (for example a
            ``--ls-test-queue`` list). They are added INTO the ``HoldoutKeys`` so the result
            keeps its video test; the caller must not do ``plain_set |= keys`` instead.
        verbose: print how many fish and clips each file contributed.

    The canonical files cannot be switched off from here or from any command line.
    """
    keys = _load(extra_files, verbose)
    for pair in extra_pairs or ():
        # set.add stores the pair only. A caller-supplied pair does not make its whole clip
        # held out; only the files' own video lists do. (Unchanged from the original.)
        set.add(keys, (pair[0], int(pair[1])))
    return keys


def load_holdout_videos(extra_files=None, verbose: bool = False) -> set:
    """Just the held-out video_ids, for builders that filter pictures rather than tracks."""
    return _load(extra_files, verbose).videos

# How a fish in the video becomes a count

This page states the counting method exactly as the code implements it for the published
results. Where the implementation is simpler or different from a one-sentence description, it
says so. Nothing here was changed when the deposit was prepared: changing any of these rules
would change published numbers.

The code is in three places:

| Step | Code | Run in |
|---|---|---|
| 1. Detect fish in frames ("Pass A") | `salmonid/detect/detect_store.py` | DEIMv2 (GPU) environment |
| 2. Link detections into tracks ("Pass B") | `salmonid/detect/track_from_store.py` | base environment |
| 3. Turn tracks into daily counts | `salmonid/count/corpus_countr_from_store.py` | base environment |

`python -m salmonid.count.run_corpus` runs the three in order for the seasons named.

## 1. Detection

- **Frames examined.** Every third frame of each video. The cameras recorded at about 10 frames
  per second in 2023 to 2025 and at 25 (Ganaraska) or 30 (Credit) in 2020 to 2022, so "every
  third frame" is about 3 per second in the later seasons and 8 to 10 per second in the
  earlier ones.
- **Image preparation.** The top 40 pixel rows, which carry the camera's date and site text,
  are blacked out, as they were in training. The 1280 × 960 frame is then resized to
  960 × 960, which squeezes it horizontally by a quarter; it is not padded to keep its shape.
  Box positions are converted back to the original frame's pixels.
- **What is kept.** Every box whose highest class score is at least 0.30, with its scores for
  all six classes (five species and "Small Salmonid"). One file per video (a "detect-store")
  holds these, so tracking and counting can be redone without running the model again.

## 2. Tracking

BoT-SORT as implemented in BoxMOT 10.0.52, restarted for every video file.

- A detection's confidence is its highest score among the **five species**. The
  "Small Salmonid" score is ignored from here on.
- A **new track** starts only from a detection above 0.50 that is matched by another
  detection above 0.50 in the next processed frame. A single isolated detection never becomes
  a track.
- Detections between 0.30 and 0.50 can **extend** a track but not start one.
- A track with no matching detection is kept for about 0.6 to 0.8 seconds, then closed.
- Matching uses predicted position and appearance (an OSNet re-identification network, used
  as distributed, trained on images of people).
- Tracks never continue from one video file to the next.
- The tracker does not report a track's first detection, so a saved track starts at the
  fish's second detection.

## 3. Counting

For each track:

1. **Minimum length.** A track needs at least **3 saved positions**, which means the fish was
   detected in at least 4 of the frames examined. Shorter tracks are dropped. The rule is in
   frames, not seconds: about 1 second in 2023 to 2025 and about 0.4 second in 2020 to 2022.
2. **Species.** The five species scores are averaged over all of the track's positions and
   the highest average wins. No minimum confidence is applied and there is no "unknown":
   every surviving track is assigned one of the five species. A juvenile's track is therefore
   assigned to whichever adult species scored highest.
3. **Direction.** Δx = (mean horizontal centre of the last three positions) − (mean of the
   first three), in pixels of the 1280-pixel-wide frame. Upstream is to the right at the
   Ganaraska River and to the left at the Credit River.
   - **upstream** if the fish moved at least **50 px** upstream;
   - **downstream** if it moved at least **300 px** downstream;
   - **holding** otherwise.

   Because three-position means are used, a track with exactly three positions has Δx = 0 and
   is always "holding", and a track with four positions must travel 150 px end to end to
   register 50 px.
4. **Counts.** *Activity* counts every surviving track once. *Net* adds +1 for upstream, −1 for
   downstream and 0 for holding. The manuscript's counts are net counts.
5. **Date.** The day the video file was last modified, in the time zone of the computer that
   ran the count (Eastern Time). The counter writes one short file per passage event, so this
   is the passage date while the files keep their original timestamps. The recovered 2020
   Ganaraska files had lost theirs and were dated from the counter's log.

### Why 50 pixels up but 300 down

Fish often hold in front of the camera and drift back and forth. Upstream is the direction of
the run, so modest upstream progress is accepted as a passage, which is how technicians score
it. A fish is counted as having gone back down only when it clearly travels downstream, about
a quarter of the frame width. With one small threshold in both directions, drifting fish are
scored as many false downstream passages and the net count comes out too low.

The rule was chosen from eight candidates by comparing each with the technician counts
(`data/net_rule_sweep/`), using the same seasons later reported. `salmonid/count/net_rule_loyo.py`
re-does the choice with each season left out in turn: the 50/300 rule is chosen in all four
folds, with 3.6% mean absolute error. Read that result for what it is: only eight rules were
compared, 50/300 is at the corner of that grid, and because the same rule wins every fold the
out-of-sample error equals the in-sample error by construction. Among the seven alternatives
the error ranges from 10.5% to 33.6%.

## The technician counts the model is compared with

| Seasons | Source | Notes |
|---|---|---|
| 2020 to 2023 | Official daily net counts for each river | "Activity" for these seasons is the sum of the absolute daily net values, because nothing finer exists. |
| 2024 | Ganaraska: official daily net counts. Credit: the reviewed counter database export, one row per fish with direction. | |
| 2025 | Per-fish review exports for both rivers | The 2025 Credit total from this record (10,533 Chinook Salmon) is higher than the published annual total (9,748). |

Technician records with no species ("Salmonid", "Unknown") are not included in any species'
count.

## Which days are compared

One row is written per river, day and species on which **either** the model count or the
technician count is not zero. A day on which the technician recorded fish but the model has no
video appears with a model count of zero: it is scored as fish the model missed, not left out.
A day on which both are zero is not written. Daily correlations are computed over exactly
these rows.

## The 2020 Ganaraska season

This season's video was recovered from a separate archive in which one passage is often split
across several short files. Since tracks do not continue between files, one fish can yield
more than one track and the model over-counts (+59% for Chinook Salmon). The season is shown
in the figures but flagged `recovered` in `data/canonical_summary.csv` and excluded from
every agreement statistic.

## Earlier development notes

`docs/history/counting_and_movement_development_notes.md` records how these rules were worked
out on an earlier pipeline. It is not a description of the published method.

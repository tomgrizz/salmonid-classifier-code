# `salmonid/detect/`: from video to fish tracks (stage D)

Two passes, run in two different environments, that hand results to each other through files.

| Order | File | What it does | Environment |
|---|---|---|---|
| 1 | `deimv2_infer.py` | Loads the trained model and runs it on frames. Used by the next script; not run on its own. | DEIMv2 (GPU) |
| 2 | `detect_store.py` | **Pass A.** Runs the model once over every third frame of each video and saves everything it marked, one file per video (a "detect-store"). Nothing later needs the model again. | DEIMv2 (GPU) |
| 3 | `audit_store_integrity.py` | A stock-take: every video in the season's list has a readable stored file, and every stored file made it into the track files. | base |
| 4 | `track_from_store.py` | **Pass B.** Links the stored marks into one track per fish with the BoT-SORT tracker, so a fish seen in forty frames is counted once. `--tracker bytetrack` is the comparison tracker of Section 3.4. | base (boxmot 10.0.52) |

All **Tier 2**: they need the video, the model weights and a graphics card. You do not run
these one by one; `python -m salmonid.count.run_corpus --steps all --dry-run` prints the
exact command for every season, and without `--dry-run` runs them.

The exact thresholds and rules are in `docs/COUNTING_METHOD.md`. Every published count went
through these four files. What was and was not re-run when the deposit was prepared is stated
in `docs/REPRODUCE.md`, stage D.

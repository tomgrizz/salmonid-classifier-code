# Training records of the published model (v2.7)

**In plain terms.** Every model-derived number in the manuscript comes from one trained model,
called v2.7. This folder holds the records that were written when that model was trained: what
code, settings and pictures went in, and the evidence for which saved copy of the model was
published. None of these files is needed to run anything. They are here so that a reader can
check, without trusting the authors' description, what the published model was made from.

## What each file is and what it shows

| File | What it is | What it shows |
|---|---|---|
| `run_manifest.json` | The "receipt" written immediately before training started (2026-07-24 18:45 UTC) | Which code, settings, training pictures, seed and software were used. See "Reading the receipt". |
| `config_snapshot/` | The settings file and the six files it includes, copied at the same moment | The exact training recipe. `deimv2_fish_x_v27.yml` is identical to `configs/deimv2/deimv2_fish_x_v27.yml`; the other six are identical to the files in `configs/`. |
| `deimv2_worktree.patch` | The local changes to the external DEIMv2 code at that moment | That the only change to the detector code was registering one colour augmentation and renaming three functions for a newer library version. |
| `launch_log_excerpt.txt` | The first lines of the launch log of the run | That the held-out leak check ran on the training and validation lists immediately before training and found nothing, and that training started from the public COCO weights. |
| `CHECKSUMS.md` | Fingerprints (sha256) and sizes of the weights, the picture lists and the test set | Lets anyone confirm that a downloaded file is the one used for the paper. |
| `CHECKPOINT_SELECTION.md` | The scores of all seven candidate checkpoints | Why checkpoint 0017 is the published one, and what that choice does and does not show. |
| `candidate_preds/` | The species predictions of each candidate on the 366 held-out fish | Lets the table in `CHECKPOINT_SELECTION.md` be recomputed. `preds_0017.csv` is the same file as `data/preds_v27_ckpt0017.csv`. |

## Reading the receipt (`run_manifest.json`)

**What it establishes.**

- *The training pictures.* `data.train` and `data.val` give the sha256, size, picture count,
  box count and boxes per species of the two lists the trainer read. They match the files
  described in `CHECKSUMS.md` exactly: 45,249 pictures and 52,444 fish boxes for training
  (Rainbow Trout 18,341; Coho Salmon 12,954; Chinook Salmon 12,177; Brown Trout 6,327;
  Atlantic Salmon 2,067; small unidentified salmonids 578), and 2,160 pictures with 2,097
  boxes for validation. These are the numbers in the manuscript's Methods.
- *The starting point.* `command` shows `-t ckpts/deimv2_x_coco.pth`: training started from
  the public COCO-pretrained detector, not from an earlier fish model. This matters because the
  earlier fish models had been trained on some held-out test fish, and starting from their
  weights would have carried that over.
- *The seed and the software.* Seed 0; Python 3.11.9, PyTorch 2.7.0 with CUDA 12.8, one
  NVIDIA RTX 5090.
- *The detector code.* DEIMv2 at commit `0fff8d4` plus the patch in this folder.

**What it shows honestly, and what it cannot establish.**

- *The research code had uncommitted changes.* `repos.fish_cv.dirty` is `true`. The run was
  launched from commit `51bf64d` of the research repository with thirteen changed or new files
  on top, listed in `dirty_files`. Eight of them are the leak fix itself (the video-level guard,
  the four data builders that call it, the leak audit, the v2.7 settings file and launch
  script); the rest are two documentation files, a crop builder for an earlier two-stage
  model, the previous model's settings file (not yet committed at the time), and an unrelated
  data folder. The leak fix was
  committed three days later, on 2026-07-27, as `d40b869`. The receipt therefore does not pin the exact text of the
  leak-fix code at the moment of training. What ties the committed code to the run instead:
  1. the leak audit deposited here (`salmonid/data/audit_coco_holdout_leak.py`), run today on
     the previous model's picture lists with `--out`, reproduces the two published lists
     **byte for byte** (same sha256 as in the receipt);
  2. run on the published lists, it prints the same counts as the launch log of 2026-07-24
     (546 held-out fish in 544 clips; 45,249 pictures; 7,500 composites resolved; zero at
     every level);
  3. the note and command strings in the receipt are the ones in the committed launch script.
- *Two file names have lost their first character.* `dirty_files` lists `LAUDE.md` and
  `ngine/data/transforms/_transforms.py`. The real names are `CLAUDE.md` and
  `engine/data/transforms/_transforms.py`. This was a bug in the script that wrote the receipt
  (it trimmed a leading space that was part of the data). It affects only the first entry of
  each list. The bug is fixed in `salmonid/provenance/record_run_manifest.py`; the receipt is
  kept as it was written.
- *The DEIMv2 code is listed as "dirty" too.* One entry is the patch described above. The other
  twenty are untracked files: scratch scripts and the project's own settings files, which
  DEIMv2's repository does not contain. None of them is detector code.
- *The saved patch cannot be applied as it is.* It was written with Windows line endings and
  without its final newline, and `git apply` rejects it. `docs/deimv2_worktree.patch` has the
  same content in a form that applies. This too is fixed in the recorder.
- *The paths are the original workstation's.* They are recorded as they were.
- *"Seeded" is not "repeatable bit for bit".* The receipt's `determinism` field says so:
  the seed is fixed, but the GPU library was not run in deterministic mode and mixed-precision
  arithmetic was used. Retraining gives a model that is close to, not identical to, the
  published one. The published weights are identified by their checksum.

## How the published training pictures were made, and what the leak check does not cover

The published lists are the previous model's lists (v2.6) with every picture from a held-out
clip removed. On the training list the audit removed 334 of 45,583 pictures: 311 real frames
from 75 held-out clips and 23 composites that contained a fish cut from a held-out clip. In
those pictures, 62 of the 366 test fish appeared with their own species label, which is why
the v2.6 score of 0.966 is not reported. To repeat the removal and compare fingerprints:

```
python -m salmonid.train.train_v27 --dry-run --verify-derivation
```

Three limits of the leak check are known. The manuscript working copy of 2026-11-10 discloses
the second; `docs/REPRODUCIBILITY_REVIEW.md` (items L2 to L4) lists all three:

1. **Frames with no clip name.** 1,854 training frames from the CVAT annotation tool record a
   file path instead of a clip name, and the check did not examine them. 25 of them come from
   the clip of one fish of the test set (an Atlantic Salmon, test track 353168); 2 more are in
   the validation list. The audit now reports this group under the heading "UNKEYED".
   (A further 560 such frames come from 10 clips of an older evaluation set that the guard
   also covers but that no manuscript number uses.)
2. **Neighbouring clips.** The camera can record one fish in consecutive clips. 76 of the 366
   test fish have a same-species training picture from the clip numbered just before or after
   theirs. Whether these are the same individuals is not known.
   `python -m salmonid.data.audit_coco_holdout_leak --coco <train.json> --report-adjacent`
   reproduces the count.
3. **Model selection.** The published checkpoint was chosen using the test set
   (`CHECKPOINT_SELECTION.md`).

## What is not in this folder

- The model weights (0.8 GB) and the picture lists and images: external downloads, see
  `docs/DATA_AVAILABILITY.md`. Their fingerprints are in `CHECKSUMS.md`.
- The full training log and the scoring logs of the run. The numbers taken from them are in
  `CHECKPOINT_SELECTION.md`, with the predictions needed to recompute the species scores.
- A record of the commands that built the v2.6 picture lists from the raw labels. No
  repository has one. The lists are therefore deposited as frozen files identified by
  checksum, and the builder scripts in `salmonid/data/` are deposited so the method can be read.

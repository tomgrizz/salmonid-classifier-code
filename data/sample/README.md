# Tier-2 demo sample

This folder holds a **tiny** bundle so a reviewer can watch the inference→track→count path
*execute* without downloading the multi-TB footage or the 819 MB checkpoint.

It is a demonstration of the code path, **not** a source of any manuscript number — those come
from the full corpus (see `docs/DATA_AVAILABILITY.md`). Contents (populated during repo build):

- `store/` — one or two per-video DEIMv2 detection-store pickles (already-run detections).
- `videos.json` — a manifest naming those videos.
- an example command, e.g.:

  ```bash
  python -m salmonid.count.corpus_countr_from_store \
      --store-dir data/sample/store --videos-json data/sample/videos.json \
      --out-csv data/sample/sample_daily.csv
  ```

If this folder contains only this README, the demo bundle was not attached to the checkout; the
full pipeline still runs against your own detect-stores (set `SALMONID_*` per `.env.example`).

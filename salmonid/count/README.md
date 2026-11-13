# `salmonid/count/`: from tracks to daily counts (stage E)

A track becomes a count through three decisions: its species, the day it passed, and whether it
moved upstream, downstream or neither. `docs/COUNTING_METHOD.md` states each rule exactly.

## The published path

| Order | File | What it does | Tier |
|---|---|---|---|
| 1 | `run_corpus.py` | Runs the whole chain for the seasons named: (detect, track,) audit, count, assemble, compare with `data/`. The default steps need only the stored tracks. | 2 |
| 2 | `corpus_countr_from_store.py` | One season's tracks → a table of daily model and technician counts per river and species. **The script that produced every count in the paper**, and the files in `data/net_rule_sweep/` and `data/activity_sweep/`. | 2 |
| 3 | `net_rule_loyo.py` | Re-chooses the upstream/downstream rule with each season left out in turn (Section 3.4). | **1** |

The technician counts are read by three helper functions, one per kind of record:
`load_tech()` in `corpus_historical_vs_tech.py` (2020 to 2023), `corpus_2024_vs_tech.py`
(2024) and `corpus_fullyear_vs_tech.py` (2025).

## Kept for the record, not used for any manuscript number

These belong to an earlier two-stage pipeline whose per-frame output files are not deposited.
Each begins with a STATUS note saying so. They are here because they document how the
counting rules were arrived at.

| File | What it was for |
|---|---|
| `corpus_fullyear_vs_tech.py`, `corpus_2024_vs_tech.py`, `corpus_historical_vs_tech.py` (everything except `load_tech()`) | The earlier pipeline's counting engine and per-season comparisons |
| `corpus_multiyear_vs_tech.py` | The earlier multi-year scorecard |
| `net_threshold_sweep.py`, `net_direction_decomp.py` | The analyses that led to the asymmetric 50/300 pixel rule |
| `track_reaggregate.py` | The analysis that led to the three-position minimum track length |

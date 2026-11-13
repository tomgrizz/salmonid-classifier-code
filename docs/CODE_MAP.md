# Code map: how the pieces connect

A one-page picture of the pipeline: what feeds what, where each stage's code lives, and where
its result is stored. `REPRODUCE.md` gives the commands; `PAPER_TO_CODE.md` gives the
manuscript item for every number.

```
 TIER 2 (needs external files)                                   stored result
 ─────────────────────────────                                   ─────────────
 raw fishway video (two rivers, 2020 to 2025)
        │
   A    ├─ choose and label 366 test fish ───────────────────►  data/test_set_v6_gt.json
        │    eval/build_holdout_testset_queue.py                 data/test_set_v6_holdout.json
        │    eval/finalize_testset_v6.py
        │
   B    ├─ build the training list, minus every test video
        │    holdout.py  (the exclusion rule)
        │    data/prepare_dataset.py → build_extra_coco.py
        │         → sam2_composite.py → merge_synth_into_coco.py
        │    data/audit_coco_holdout_leak.py  (the check) ───►  provenance/v2.7/ (checksums)
        │
   C    ├─ train ───────────────────────────────────────────►  model weights (external)
        │    train/train_v27.py                                  provenance/v2.7/ (run record,
        │    configs/deimv2/deimv2_fish_x_v27.yml                checkpoint choice)
        │    provenance/record_run_manifest.py
        │
   D    ├─ Pass A: detect fish in every third frame
        │    detect/deimv2_infer.py, detect/detect_store.py ─►  detect-stores (external)
        │    detect/audit_store_integrity.py
        │  Pass B: link detections into one track per fish
        │    detect/track_from_store.py ────────────────────►  track files (external)
        │
        ├─ species call for each test fish
        │    eval/predict_testset_v6_from_store.py ─────────►  data/preds_v27_ckpt0017.csv
        │
   E    └─ species, date and direction for each track;
           daily totals beside the technician totals
             count/corpus_countr_from_store.py ─────────────►  data/net_rule_sweep/
             count/run_corpus.py  (runs D and E in order)       data/activity_sweep/
             technician records ← count/corpus_*_vs_tech.py     data/recovered_2020/
                                   (load_tech only)

 TIER 1 (runs from data/; python run_tier1.py)
 ─────────────────────────────────────────────
   F    classification
          eval/score_testset_v6.py ─────────────────────────►  Table 1, Section 3.1
          results/classification_ci.py ─────────────────────►  Table 1 intervals
          paper/heldout_kappa.py, heldout_adjacent_clips.py ─►  Sections 3.1, 4.1, 4.5
          paper/backbone_comparison.py ─────────────────────►  Table S4

        counts
          results/build_canonical_results.py ───────────────►  data/canonical_daily.csv
                                                               data/canonical_summary.csv
          results/compute_concordance.py ───────────────────►  Table 5, Figures 6, S3
          results/agreement_stats.py ───────────────────────►  Sections 3.3, 3.5
          count/net_rule_loyo.py ───────────────────────────►  Section 3.4
          paper/count_tables.py ────────────────────────────►  Tables 2, 3, 4
          paper/daily_totals_and_peak_days.py ──────────────►  Abstract, Conclusions

        2025 video-by-video comparison (from data/paper/)
          paper/pervideo_agreement.py ──────────────────────►  Section 3.2, Table 2
          paper/detection_outcomes.py ──────────────────────►  Section 3.2
          paper/blind_review.py ────────────────────────────►  Section 3.2

        figures
          figures/render_manuscript_figures.py ─────────────►  Figures 2, S2
          figures/render_hero_regulation.py ────────────────►  Figures 4, 5, 7
          figures/render_sensitivity_figure.py ─────────────►  supporting figure
          (Tier 2: figures/fig_species_morphology.py → Figure 3;
                   figures/fig_bbox_grid.py → Figure S1;
                   paper/fig1_study_map.py → Figure 1)
```

## Things that are easy to get wrong

- **`salmonid/config.py` is the only place a path or a study constant is defined.** Other
  modules import from it. To move data, set an environment variable; do not edit code.
- **`data/` is read-only.** Regenerated tables go to `output/`, figures to `figures/`.
- **Detection and tracking run in different Python environments** and talk through files
  (`environment/README.md`).
- **`salmonid/count/` contains two generations of code.** The published path is
  `run_corpus.py`, `corpus_countr_from_store.py` and `net_rule_loyo.py`, plus the three
  `load_tech()` functions. The other files belong to an earlier pipeline and say so at the top.
- **The scripts under `salmonid/eval/` other than the scorer and the test-set builders are
  development checks**, not manuscript results (`salmonid/eval/README.md`).
- **Some rules are simpler in the code than in a one-line description.** `COUNTING_METHOD.md`
  states each one as implemented.

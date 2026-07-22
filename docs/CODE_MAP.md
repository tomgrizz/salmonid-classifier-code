# Code map — how the modules chain together

This is the reader's guide to the `salmonid/` package: the end-to-end pipeline, which module does
what, and which manuscript number or figure each stage produces. Every path a module touches comes
from `salmonid/config.py`; every eval track is protected from training by `salmonid/holdout.py`.

## Pipeline overview

```
                    raw footage (Ganaraska / Credit .mp4)
                                   │
        [ data/ ]  training-data prep            [ detect/ ]  inference
   ┌───────────────────────────────┐        ┌──────────────────────────────────┐
   │ sam2_composite.py  synth fish │        │ deimv2_infer.py   DEIMv2 decode   │  .venv_deim
   │ build_extra_coco.py  base COCO│        │        (orig_target_sizes=[[W,H]])│
   │ merge_synth_into_coco.py ─────┼──┐     │ detect_store.py   Pass A: per-video│  .venv_deim
   │ prepare_dataset.py            │  │     │        detection-store pickle      │
   └───────────────────────────────┘  │     │ track_from_store.py Pass B: BoT-  │  base env
        holdout.py drops eval tracks ◄─┘     │        SORT+OSNet -> track aggs    │
                                            └───────────────┬──────────────────┘
   train (external DEIMv2 train.py,                         │
   runner: runners/run_deimv2_fish_x_v26.ps1) ──► checkpoint0007.pth (v2.6)
                                                            │
                    ┌───────────────────────────────────────┴───────────────────┐
       [ eval/ ]  held-out scoring + gates                       [ count/ ]  counting & movement
   ┌────────────────────────────────────────┐          ┌────────────────────────────────────────┐
   │ predict_testset_v6_from_store.py        │          │ corpus_countr_from_store.py (single-    │
   │   detect-store -> per-track species     │          │   stage adapter -> daily model-vs-tech) │
   │ score_testset_v6.py  ⇒ macro-F1 0.966   │  Fig 2   │ corpus_fullyear_vs_tech.py (engine +    │
   │ score_tracked.py  (ckpt selection)      │          │   load_tech); *_2024_/*_historical_/    │
   │ bbox_regression_gate.py / deimv2_bbox_  │          │   *_multiyear_vs_tech.py (year loaders  │
   │   gate.py   (IoU>=0.80, cy +-0.10)      │          │   + scorecard)                          │
   │ count_r_gate.py / corpus_year_gate.py   │          │ net_threshold_sweep / net_direction_    │
   │ finalize_testset_v6.py / build_holdout_ │          │   decomp / track_reaggregate (rule      │
   │   testset_queue.py  (freeze the test set)│          │   justification)                        │
   └────────────────────────────────────────┘          └──────────────────┬─────────────────────┘
                                                                           │ per-year daily CSVs
                                              [ results/ ] ◄───────────────┘
                              ┌───────────────────────────────────────────────────────┐
                              │ build_canonical_results.py -> canonical_daily/summary  │
                              │ compute_concordance.py -> Lin's CCC + Bland-Altman      │  Fig 7 / S4
                              │ compile_manuscript_results.py -> the results doc        │
                              │ build_reference_sample.py / score_reference_sample.py   │  94.7% blind
                              └───────────────────────────────────────────────────────┘
                                              [ figures/ ]  manuscript_style.py + render_*
```

## Stage-by-stage

**`data/`** builds the training COCO. `build_extra_coco.py` makes the masked 6-class base;
`sam2_composite.py` composites SAM2 synthetic fish (partial exposures + Coho/Chinook proximity
pairs), header-masked at 40 px; `merge_synth_into_coco.py` folds synth in, caps it, and drops
held-out images; `prepare_dataset.py` does the stratified split. All import `holdout.py`.

**`detect/`** runs the single-stage model. `deimv2_infer.py` is the decode wrapper — note the
`orig_target_sizes=[[W, H]]` (width-first) convention and direct 0–5 class mapping.
`detect_store.py` (Pass A, `.venv_deim`) writes a per-video detection store; `track_from_store.py`
(Pass B, base env) re-tracks it with BoT-SORT + OSNet re-ID for gold-standard counts.

**`eval/`** defines the honest number. `predict_testset_v6_from_store.py` turns a detect-store into
per-track predictions; `score_testset_v6.py` scores them → species F1, the confusion matrix, and
presence-F1 (**macro-F1 0.966**, Fig 2). `score_tracked.py` selects the deploy checkpoint on the
leak-clean operational split. Two **mandatory gates** run before any model is trusted: the bbox
gate (IoU ≥ 0.80, y-centre within ±0.10) and the count-r gate (Pearson r vs technician counts).

**`count/`** turns detections into fish counts and net upstream movement, re-deriving tracks from
the detections at `min_frames = 3` and applying the **asymmetric net rule** (upstream if
`|Δx| ≥ 50 px`, downstream if `≥ 300 px`). `corpus_fullyear_vs_tech.py` is the reference engine and
owns `load_tech()`; the `*_2024_/*_historical_/*_multiyear_vs_tech.py` modules add the per-year
technician sources and the multi-year scorecard. See `counting_and_movement.md` for the method.

**`results/`** assembles the canonical CSVs from the per-year sweeps
(`build_canonical_results.py`), computes Lin's CCC + Bland–Altman agreement
(`compute_concordance.py`, Fig 7 / S4), regenerates the results document so text and numbers can't
drift (`compile_manuscript_results.py`), and runs the stratified reference-standard blind accuracy
sample (`build_/score_reference_sample.py`, the 94.7% blind-human number).

**`figures/`** renders the manuscript figures. `manuscript_style.py` is the shared CJFAS style
(≥7 pt embedded fonts, Okabe-Ito grayscale-safe palette, true physical size); each `render_*` /
`fig_*` script produces one figure group and writes PNG + PDF/TIFF into `figures/`.

**`provenance/record_run_manifest.py`** pins a run: both repo SHAs, the DEIMv2 patch, the resolved
config chain, train/val COCO sha256 + category counts, seed, environment, and command.

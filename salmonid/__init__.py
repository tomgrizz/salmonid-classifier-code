"""salmonid — reproduction code for the single-stage DEIMv2-X salmonid classifier (v2.6).

Sub-packages mirror the pipeline stages:
    data/       training-data preparation (masked COCO build, SAM2 synthetic composites)
    detect/     single-stage inference (DEIMv2 decode, detect-store, BoT-SORT tracking)
    eval/       held-out evaluation and the mandatory pre-deploy gates
    count/      counting & net-movement, model-vs-technician daily aggregation
    results/    canonical results assembly + concordance (Lin's CCC) statistics
    figures/    manuscript figure generators
    provenance/ per-run reproducibility manifest
    util/       shared helpers (parquet schema, notifications)

Paths and constants come from ``salmonid.config``. See the top-level README for the two
reproducibility tiers and ``docs/CODE_MAP.md`` for how the modules chain together.
"""

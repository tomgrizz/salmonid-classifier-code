# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role: tracking (Pass B) for the SUPERSEDED v2.6 model. No number in the paper comes from v2.6
# (62 of its 366 test fish had been used in training; see the README). The published v2.7
# run used the run_corpus_v27_*.ps1 scripts in this folder.
#
# Source: this repository's first commit (extracted from the research repository, July 2026).
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps all --dry-run     (published model, v2.7)
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Multi-year BoT-SORT re-track -- Pass B of the manuscript corpus (GOLD-STANDARD absolute counts).
# Reads each year's DEIMv2 v2.6 detect-store, re-decodes ONLY the stored detection frames (decord) for
# OSNet re-ID, and dumps per-track BoTSORT aggregates -> ~1.02 tracks/fish (vs the store-only IoU 1.48).
# Runs in the MAIN venv (boxmot needs numpy<2). Resumable per shard (agg skips tracked videos).
# Run AFTER the detect-store year lands. Feed the aggs to count-r with:
#   python scripts/_corpus_countr_from_store.py --store-dir <store> --agg "<...>/agg_*.pkl" \
#       --videos-json <both-rivers manifest> --key-mode video_id --metric net --out-csv <year>_net.csv
#
# Usage:
#   ./run_corpus_v26_botsort_by_year.ps1                       # all years, 3 shards
#   ./run_corpus_v26_botsort_by_year.ps1 -Years 2024 -Shards 3
param(
    [int[]]$Years     = @(2024, 2023, 2022, 2021, 2020),
    [int]  $Shards    = 3,
    [int]  $FrameSkip = 3,
    [string]$KeyMode  = "video_id"        # use "rel" for the pre-campaign 2025 store (rel-keyed)
)

$py     = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"   # MAIN venv (boxmot)
$repo   = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$footage= "F:\RiverWatcher"
$labels = "G:\Projects\label_batches"
$ntfy   = "$repo\scripts\_ntfy_send.py"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

foreach ($yr in $Years) {
    $store = "G:\Projects\deimv2_x_v26_corpus${yr}\store"
    if (-not (Test-Path $store)) { "[$(Get-Date)] MISSING store $store -- skip (run detect-store first)" | Tee-Object "$repo\logs\corpus_v26_botsort.log" -Append; continue }
    foreach ($site in @("credit", "ganaraska")) {
        $manifest = "$labels\corpus_${yr}_${site}.json"
        if (-not (Test-Path $manifest)) { continue }
        $procs = @()
        for ($i = 0; $i -lt $Shards; $i++) {
            $agg = "G:\Projects\deimv2_x_v26_corpus${yr}\agg_${site}_shard$i.pkl"
            $lg  = "$repo\logs\corpus_v26_botsort_${yr}_${site}_shard_$i.log"
            $argline = "-u `"$repo\scripts\_track_op_from_store.py`" --store-dir `"$store`" " +
                       "--op-json `"$manifest`" --key-mode $KeyMode --footage-root `"$footage`" " +
                       "--decode-backend decord --reid-fp16 --frame-skip $FrameSkip " +
                       "--agg-out `"$agg`" --shard $i --num-shards $Shards"
            $procs += Start-Process -FilePath $py -ArgumentList $argline -RedirectStandardOutput $lg `
                                    -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
            Start-Sleep 4
        }
        "[$(Get-Date)] v26 botsort $yr $site launched ($Shards shards, pids $($procs.Id -join ','))" |
            Tee-Object "$repo\logs\corpus_v26_botsort.log" -Append
        $procs | Wait-Process
        "[$(Get-Date)] v26 botsort $yr $site DONE" | Tee-Object "$repo\logs\corpus_v26_botsort.log" -Append
    }
    & $py $ntfy "v26 BoT-SORT re-track year $yr done (both rivers)" --title "FishCV v26 botsort $yr" --priority 4 2>$null
}
"[$(Get-Date)] v26 BOTSORT ALL YEARS DONE" | Tee-Object "$repo\logs\corpus_v26_botsort.log" -Append
& $py $ntfy "v26 BoT-SORT re-track (all years) DONE" --title "FishCV v26 botsort DONE" --priority 5 2>$null

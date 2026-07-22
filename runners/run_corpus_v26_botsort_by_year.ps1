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

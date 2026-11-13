# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: none. Per-season form of the tracking step; like
# run_corpus_v27_detstore_by_year.ps1 it was first executed in September 2026 on 2026 footage.
# This is the committed version, without the later -Tag option for 2026 footage.
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps track --years 2022 2021 2020
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Multi-year BoT-SORT re-track -- Pass B of the manuscript corpus under DEIMv2-X v2.7 (leak-clean).
# Reads each year's v2.7 detect-store, re-decodes ONLY the stored detection frames (decord) for OSNet
# re-ID, and dumps per-track BoTSORT aggregates. Runs in the MAIN venv (boxmot needs numpy<2).
# Resumable per shard (agg skips tracked videos). Run AFTER the detect-store year lands.
#
# Shard count note: the v2.7 aggregates already on disk for 2022-2025 were written with 2 shards
# (agg_<site>_shard{0,1}.pkl). Keep -Shards 2 for the backfilled years so the agg glob
# "agg_*.pkl" stays consistent across years; a different shard count only changes how the work is
# split, not the result, but it does change the file names.
#
# Feed the aggs to count-r with:
#   python scripts/_corpus_countr_from_store.py --store-dir <store> --agg "<...>/agg_*.pkl" \
#       --videos-json <both-rivers manifest> --key-mode video_id --metric net --out-csv <year>_net_v27.csv
#
# Usage:
#   ./run_corpus_v27_botsort_by_year.ps1                            # the three missing years
#   ./run_corpus_v27_botsort_by_year.ps1 -Sites credit -Years 2022  # 2022 Credit only
param(
    [int[]]   $Years     = @(2022, 2021, 2020),
    [string[]]$Sites     = @("credit", "ganaraska"),
    [int]     $Shards    = 2,
    [int]     $FrameSkip = 3,
    [string]  $KeyMode   = "video_id"
)

$py     = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"   # MAIN venv (boxmot)
$repo   = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$footage= "F:\RiverWatcher"
$labels = "G:\Projects\label_batches"
$ntfy   = "$repo\scripts\_ntfy_send.py"
$log    = "$repo\logs\corpus_v27_botsort.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

foreach ($yr in $Years) {
    $store = "G:\Projects\deimv2_x_v27_corpus${yr}\store"
    $n = if (Test-Path $store) { (Get-ChildItem $store -File -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
    if ($n -eq 0) {
        "[$(Get-Date)] EMPTY store $store ($n files) -- skip (run detect-store first)" | Tee-Object $log -Append
        continue
    }
    foreach ($site in $Sites) {
        $manifest = "$labels\corpus_${yr}_${site}.json"
        if (-not (Test-Path $manifest)) { continue }
        $procs = @()
        for ($i = 0; $i -lt $Shards; $i++) {
            $agg = "G:\Projects\deimv2_x_v27_corpus${yr}\agg_${site}_shard$i.pkl"
            $lg  = "$repo\logs\corpus_v27_botsort_${yr}_${site}_shard_$i.log"
            $argline = "-u `"$repo\scripts\_track_op_from_store.py`" --store-dir `"$store`" " +
                       "--op-json `"$manifest`" --key-mode $KeyMode --footage-root `"$footage`" " +
                       "--decode-backend decord --reid-fp16 --frame-skip $FrameSkip " +
                       "--agg-out `"$agg`" --shard $i --num-shards $Shards"
            $procs += Start-Process -FilePath $py -ArgumentList $argline -RedirectStandardOutput $lg `
                                    -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
            Start-Sleep 4
        }
        "[$(Get-Date)] v27 botsort $yr $site launched ($Shards shards, pids $($procs.Id -join ','))" |
            Tee-Object $log -Append
        $procs | Wait-Process
        "[$(Get-Date)] v27 botsort $yr $site DONE" | Tee-Object $log -Append
    }
    & $py $ntfy "v27 BoT-SORT re-track year $yr done (sites: $($Sites -join ','))" --title "FishCV v27 botsort $yr" --priority 4 2>$null
}
"[$(Get-Date)] v27 BOTSORT REQUESTED YEARS DONE ($($Years -join ','))" | Tee-Object $log -Append
& $py $ntfy "v27 BoT-SORT re-track ($($Years -join ',')) DONE" --title "FishCV v27 botsort DONE" --priority 5 2>$null

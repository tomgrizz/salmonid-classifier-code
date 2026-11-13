# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: none. It was written during the run as a per-season form of
# the detection step, but the run logs show it was first executed in September 2026, on 2026
# footage that is outside the paper. It is kept because it states the detection settings in the
# simplest form. This is the committed version; a later uncommitted edit that adds a -Tag option
# for 2026 footage is not included.
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps detect --years 2022 2021 2020
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Multi-year detect-store -- SINGLE-STAGE DEIMv2-X v2.7 (checkpoint0017), full-distribution.
# Pass A of the manuscript corpus under the LEAK-CLEAN model. Identical to the v26 runner except for the
# output root and the checkpoint; the X architecture is unchanged (v25 config loads v27 weights).
#
# WHY THIS FILE EXISTS: the 2026-07-27 v2.7 corpus attempt completed 2023/2024/2025 and Ganaraska-2022,
# then failed silently on 2022-Credit, 2021 and 2020 with
#     _deimv2_detect_store.py: error: unrecognized arguments: G:/RiverWatcher
# i.e. TWO footage roots were passed to a single-value --footage-root. Every year is present on F:, so
# this runner passes ONE quoted root, exactly as run_corpus_v26_detstore_by_year.ps1 does. The failed
# years still wrote count CSVs from an EMPTY store ("loaded 0 tracked videos"), so those CSVs are junk
# and must be regenerated -- do not trust G:\Projects\deimv2_x_v27_corpus{2020,2021}\*_v27.csv.
#
# The store skips existing pkls, so this is resumable and safe to re-run over finished years.
#
# Usage:
#   ./run_corpus_v27_detstore_by_year.ps1                          # the three missing years
#   ./run_corpus_v27_detstore_by_year.ps1 -Years 2021 -Shards 2
#   ./run_corpus_v27_detstore_by_year.ps1 -Sites credit -Years 2022 # 2022 Credit only (Ganaraska is done)
param(
    [int[]]   $Years     = @(2022, 2021, 2020),
    [string[]]$Sites     = @("credit", "ganaraska"),
    [int]     $Shards    = 2,              # fp16: 2 shards = ~2610 vid/hr @ ~18GB (optimal; 1=1800, 3 risks OOM)
    [int]     $FrameSkip = 3,
    [int]     $BatchSize = 32,
    [switch]  $NoFp16                      # fp16 is ON by default (corpus counts); -NoFp16 for exact fp32
)
$fp16flag = if ($NoFp16) { "" } else { "--fp16" }

$py     = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv\.venv_deim\Scripts\python.exe"
$repo   = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$config = "$repo\single_stage\configs\deimv2\deimv2_fish_x_v25.yml"   # X-arch v25==v26==v27; loads v27 weights
$ckpt   = "C:\Users\lomu-\Documents\GitHub\DEIMv2\outputs\deimv2_fish_x_v27\checkpoint0017.pth"
$footage= "F:\RiverWatcher"            # SINGLE root -- all of 2020-2025 is mirrored here (see header)
$labels = "G:\Projects\label_batches"
$ntfy   = "$repo\scripts\_ntfy_send.py"
$log    = "$repo\logs\corpus_v27_detstore.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

if (-not (Test-Path $ckpt))   { throw "MISSING checkpoint $ckpt" }
if (-not (Test-Path $config)) { throw "MISSING config $config" }

foreach ($yr in $Years) {
    $store = "G:\Projects\deimv2_x_v27_corpus${yr}\store"
    New-Item -ItemType Directory -Force $store | Out-Null
    foreach ($site in $Sites) {
        $manifest = "$labels\corpus_${yr}_${site}.json"
        if (-not (Test-Path $manifest)) { "[$(Get-Date)] MISSING manifest $manifest -- skip" | Tee-Object $log -Append; continue }
        $procs = @()
        for ($i = 0; $i -lt $Shards; $i++) {
            $lg = "$repo\logs\corpus_v27_detstore_${yr}_${site}_shard_$i.log"
            $argline = "-u `"$repo\scripts\_deimv2_detect_store.py`" --config `"$config`" --ckpt `"$ckpt`" " +
                       "--op-json `"$manifest`" --out-dir `"$store`" --key-mode video_id " +
                       "--footage-root `"$footage`" --thresh 0.3 --mask-header-px 40 --full-dist $fp16flag " +
                       "--frame-batch-size $BatchSize --frame-skip $FrameSkip --shard $i --num-shards $Shards"
            $procs += Start-Process -FilePath $py -ArgumentList $argline -RedirectStandardOutput $lg `
                                    -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
            Start-Sleep 4
        }
        "[$(Get-Date)] v27 detstore $yr $site launched ($Shards shards, batch=$BatchSize, pids $($procs.Id -join ','))" |
            Tee-Object $log -Append
        $procs | Wait-Process
        $n = (Get-ChildItem $store -File -ErrorAction SilentlyContinue | Measure-Object).Count
        "[$(Get-Date)] v27 detstore $yr $site DONE (store now holds $n files)" | Tee-Object $log -Append
    }
    & $py $ntfy "v27 detect-store year $yr done (sites: $($Sites -join ','))" --title "FishCV v27 detstore $yr" --priority 4 2>$null
}
"[$(Get-Date)] v27 DETSTORE REQUESTED YEARS DONE ($($Years -join ','))" | Tee-Object $log -Append
& $py $ntfy "v27 detect-store ($($Years -join ',')) DONE" --title "FishCV v27 detstore DONE" --priority 5 2>$null

# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role: detection (Pass A) for the SUPERSEDED v2.6 model. No number in the paper comes from v2.6
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
# Multi-year detect-store -- SINGLE-STAGE DEIMv2-X v2.6 (checkpoint0007), full-distribution.
# Pass A of the manuscript corpus: write per-video detection stores for 2020-2024, both rivers.
# Reads footage from the F: NVMe (G: HDD is the I/O ceiling). Batched forward (numerically SET-identical
# to batch-1: detection set unchanged, retained values differ only by GPU batch-kernel rounding ~4e-4).
# video_id keying (multi-year safe; Ganaraska 2023 has 1,425 rel collisions). Resumable per shard (the
# store skips existing pkls). Newest-validatable year first (2024 -> 2020).
#
# Usage:
#   ./run_corpus_v26_detstore_by_year.ps1                              # all years, 3 shards, batch 32
#   ./run_corpus_v26_detstore_by_year.ps1 -Years 2024 -Shards 3        # one year
#   ./run_corpus_v26_detstore_by_year.ps1 -BatchSize 1                 # bit-identical (slower ~1.6x)
param(
    [int[]]$Years     = @(2024, 2023, 2022, 2021, 2020),
    [int]  $Shards    = 2,                 # fp16: 2 shards = ~2610 vid/hr @ ~18GB (optimal; 1=1800, 3 risks OOM)
    [int]  $FrameSkip = 3,
    [int]  $BatchSize = 32,
    [switch]$NoFp16                       # fp16 is ON by default (corpus counts); -NoFp16 for exact fp32
)
$fp16flag = if ($NoFp16) { "" } else { "--fp16" }

$py     = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv\.venv_deim\Scripts\python.exe"
$repo   = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$config = "$repo\single_stage\configs\deimv2\deimv2_fish_x_v25.yml"   # X-arch v25==v26; loads v26 weights
$ckpt   = "C:\Users\lomu-\Documents\GitHub\DEIMv2\outputs\deimv2_fish_x_v26\checkpoint0007.pth"
$footage= "F:\RiverWatcher"
$labels = "G:\Projects\label_batches"
$ntfy   = "$repo\scripts\_ntfy_send.py"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

foreach ($yr in $Years) {
    $store = "G:\Projects\deimv2_x_v26_corpus${yr}\store"
    New-Item -ItemType Directory -Force $store | Out-Null
    foreach ($site in @("credit", "ganaraska")) {
        $manifest = "$labels\corpus_${yr}_${site}.json"
        if (-not (Test-Path $manifest)) { "[$(Get-Date)] MISSING manifest $manifest -- skip" | Tee-Object "$repo\logs\corpus_v26_detstore.log" -Append; continue }
        $procs = @()
        for ($i = 0; $i -lt $Shards; $i++) {
            $lg = "$repo\logs\corpus_v26_detstore_${yr}_${site}_shard_$i.log"
            $argline = "-u `"$repo\scripts\_deimv2_detect_store.py`" --config `"$config`" --ckpt `"$ckpt`" " +
                       "--op-json `"$manifest`" --out-dir `"$store`" --key-mode video_id " +
                       "--footage-root `"$footage`" --thresh 0.3 --mask-header-px 40 --full-dist $fp16flag " +
                       "--frame-batch-size $BatchSize --frame-skip $FrameSkip --shard $i --num-shards $Shards"
            $procs += Start-Process -FilePath $py -ArgumentList $argline -RedirectStandardOutput $lg `
                                    -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
            Start-Sleep 4
        }
        "[$(Get-Date)] v26 detstore $yr $site launched ($Shards shards, batch=$BatchSize, pids $($procs.Id -join ','))" |
            Tee-Object "$repo\logs\corpus_v26_detstore.log" -Append
        $procs | Wait-Process
        "[$(Get-Date)] v26 detstore $yr $site DONE" | Tee-Object "$repo\logs\corpus_v26_detstore.log" -Append
    }
    & $py $ntfy "v26 detect-store year $yr done (both rivers)" --title "FishCV v26 detstore $yr" --priority 4 2>$null
}
"[$(Get-Date)] v26 DETSTORE ALL YEARS DONE" | Tee-Object "$repo\logs\corpus_v26_detstore.log" -Append
& $py $ntfy "v26 detect-store (all years 2020-2024) DONE" --title "FishCV v26 detstore DONE" --priority 5 2>$null

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

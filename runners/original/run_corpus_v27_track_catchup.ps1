# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: repeated the tracking pass (Pass B) for Credit 2022, Credit 2020
# and Ganaraska 2020 on 2 August 2026, after the first attempt ended at start-up because the
# appearance-model weights were not found.
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps track audit --years 2022 2020
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Catch-up: re-run ONLY the BoT-SORT tracking passes lost on 2026-08-01.
#
# What happened: `_track_op_from_store.py` held the OSNet re-ID weights as a bare RELATIVE path
# (`gradio_app/salmonid_app-main/botsort_weights/...`). The backfill driver was launched from `C:\`
# rather than the repo root, so boxmot did not find the file, fell back to downloading it from
# Google Drive, and died at tracker construction on a machine with no DNS. In the runner log that
# reads as "BoT-SORT DONE" after 15 seconds. Three finished detect-store years lost their tracking
# pass this way; the detect-stores themselves (the expensive GPU part, ~7.5 h) are intact.
#
# Fixed at source: `_track_op_from_store.py` now resolves the weights against the repo root and
# fails loudly if they are absent, so the working directory no longer matters. This script exists
# only to redo the three passes that were skipped.
#
# Cheap: ~14,100 videos at the observed ~11,900 videos/hour is a bit over an hour.
#
# Run AFTER run_corpus_v27_backfill.ps1 finishes, so the two do not contend for the GPU and for G:.
param(
    [int]   $Shards  = 2,
    [switch]$Force              # re-track even if a non-trivial aggregate already exists
)
$repo    = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$py      = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy    = "$repo\scripts\_ntfy_send.py"
$footage = "F:\RiverWatcher"
$labels  = "G:\Projects\label_batches"
$log     = "$repo\logs\corpus_v27_track_catchup.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

function Say($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object $log -Append }

# (tag, manifest, store, agg prefix) -- exactly the three pairs whose tracking was lost.
$jobs = @(
    @{ tag = "2022_credit";    man = "$labels\corpus_2022_credit.json"
       store = "G:\Projects\deimv2_x_v27_corpus2022\store"
       agg   = "G:\Projects\deimv2_x_v27_corpus2022\agg_credit" },
    @{ tag = "2020_credit";    man = "$labels\corpus_2020_credit.json"
       store = "G:\Projects\deimv2_x_v27_corpus2020\store"
       agg   = "G:\Projects\deimv2_x_v27_corpus2020\agg_credit" },
    @{ tag = "2020_ganaraska"; man = "$labels\corpus_2020_ganaraska.json"
       store = "G:\Projects\deimv2_x_v27_corpus2020\store"
       agg   = "G:\Projects\deimv2_x_v27_corpus2020\agg_ganaraska" }
)

Say "=== v2.7 tracking catch-up BEGIN ==="
foreach ($j in $jobs) {
    $n = if (Test-Path $j.store) { (Get-ChildItem $j.store -File -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
    if ($n -eq 0) { Say "$($j.tag): store empty -- skip"; continue }

    # The lost passes left 5-byte aggregates from an earlier failure. Those load as an empty dict,
    # so the tracker would "resume" from nothing, which is harmless but indistinguishable from a
    # real empty result downstream. Clear them so the rerun starts clean.
    $stale = @(Get-ChildItem "$($j.agg)_shard*.pkl" -ErrorAction SilentlyContinue |
               Where-Object { $_.Length -lt 1024 })
    foreach ($s in $stale) { Say "$($j.tag): removing stale $($s.Length)-byte $($s.Name)"; Remove-Item $s.FullName -Force }

    $good = @(Get-ChildItem "$($j.agg)_shard*.pkl" -ErrorAction SilentlyContinue |
              Where-Object { $_.Length -ge 1024 })
    if ($good -and -not $Force) { Say "$($j.tag): already has $($good.Count) real aggregate(s) -- skip (use -Force to redo)"; continue }

    Say "$($j.tag): BoT-SORT begin ($Shards shards)"
    $procs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $lg = "$repo\logs\corpus_v27_botsort_$($j.tag)_shard_$i.log"
        $a  = "-u `"$repo\scripts\_track_op_from_store.py`" --store-dir `"$($j.store)`" " +
              "--op-json `"$($j.man)`" --key-mode video_id --footage-root `"$footage`" " +
              "--decode-backend decord --reid-fp16 --frame-skip 3 " +
              "--agg-out `"$($j.agg)_shard$i.pkl`" --shard $i --num-shards $Shards"
        $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo `
                                -RedirectStandardOutput $lg `
                                -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
        Start-Sleep 4
    }
    $procs | Wait-Process

    $small = @(Get-ChildItem "$($j.agg)_shard*.pkl" -ErrorAction SilentlyContinue |
               Where-Object { $_.Length -lt 1024 })
    if ($small) { Say "$($j.tag): STILL FAILING -- $($small.Count) aggregate(s) under 1 KB; read the .log.err" }
    else        { Say "$($j.tag): BoT-SORT DONE" }
}
Say "=== v2.7 tracking catch-up COMPLETE ==="
& $py $ntfy "v2.7 tracking catch-up complete" --title "FishCV v27 catch-up" --priority 4 2>$null

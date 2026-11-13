# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: detection (Pass A) and tracking (Pass B) of the 2025, 2024 and
# 2023 seasons at both rivers and of the 2022 Ganaraska season, 27-30 July 2026. Its later steps
# for Credit 2022, 2021 and 2020 did nothing (see run_corpus_v27_backfill.ps1, which redid them),
# and the per-season count files it wrote (<year>_net_v27.csv, <year>_activity_v27.csv) were
# superseded by run_corpus_v27_countr_sweep.ps1 and are not used by the paper.
#
# Source: research repository, main branch (commits da8f3ba 2026-07-27, 6611e2b and 5d84978 2026-07-28).
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps detect audit-stores track audit --years 2025 2024 2023 2022
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# FULL CORPUS inference with the LEAK-CLEAN single-stage model, DEIMv2-X v2.7 (checkpoint0017).
#
# Produces a complete v2.7 counterpart to the v2.6 corpus so the paper can be moved onto a model whose
# test set is honest end-to-end, without re-deciding anything later. Per year, both rivers:
#   Pass A  detect-store  (.venv_deim, fp16 batched)
#   Pass B  BoT-SORT      (main venv, boxmot + OSNet)  -> gold-standard absolute counts
#   Pass C  aggregate     net + activity daily CSVs vs the technician record
# Each year completes end-to-end before the next starts, so partial results are usable throughout.
#
# CHECKPOINT: 0017 of the 30-epoch v2.7 run. Of the seven candidates it ties for best species macro-F1 on
# the fully-honest 366-track test_set_v6 (0.961, with 0019/best_stg2) AND posts the best bbox-gate scores
# any X-arch checkpoint has produced (Ganaraska 0.805 = PASS, Credit 0.737 vs v2.6's 0.692). Formal
# practice is to select on the tracked-op sweep and confirm once on test_set_v6; that sweep was not run
# for v2.7, and the seven candidates span 0.949-0.961 = 1-4 tracks, so this pick is defensible but is a
# test_set_v6 + gate pick, not an op-dev pick. Recorded here so the choice is traceable.
#
# 2025 uses the CORRECTED manifests (*_full.json, 47,997 videos). The originals were short by 13% --
# see docs/MANUSCRIPT_UPDATES_2026-07-27.md. Fresh store, so video_id keying throughout (multi-year safe;
# the v2.6 2025 store is rel-keyed for legacy reasons and is NOT interchangeable with this one).
#
#   powershell -File runners/run_corpus_v27_by_year.ps1
#   powershell -File runners/run_corpus_v27_by_year.ps1 -Years 2025 -Shards 2
param(
    [int[]]$Years     = @(2025, 2024, 2023, 2022, 2021, 2020),
    [int]  $Shards    = 2,                # fp16: 2 shards ~= 2,610 vid/hr @ ~18 GB (3 risks OOM)
    [int]  $FrameSkip = 3,
    [int]  $BatchSize = 32,
    # Sites to process. Exists for RESUME: detect-store skips videos already in the store, but Pass B
    # (BoT-SORT) has no such skip and would redo a completed site from scratch (~3 h for Ganaraska 2025).
    # After an interrupted year, resume the unfinished site alone, e.g. -Years 2025 -Sites credit.
    [string[]]$Sites  = @("ganaraska", "credit"),
    [switch]$NoFp16,
    [switch]$SkipTrack,
    [switch]$SkipAggregate,           # skip Pass C (use when finishing one site of a two-site year)
    [switch]$NoFinalNotify            # suppress the "FULL CORPUS DONE" ntfy when called as one step
)                                     # of a larger run (otherwise it claims all years are finished)
$fp16flag = if ($NoFp16) { "" } else { "--fp16" }

$repo   = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$deim   = "C:\Users\lomu-\Documents\GitHub\DEIMv2"
$vpy    = "$repo\.venv_deim\Scripts\python.exe"
$py     = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy   = "$repo\scripts\_ntfy_send.py"
$log    = "$repo\logs\corpus_v27.log"
$config = "$repo\single_stage\configs\deimv2\deimv2_fish_x_v27.yml"
$ckpt   = "$deim\outputs\deimv2_fish_x_v27\checkpoint0017.pth"
$lb     = "G:\Projects\label_batches"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null
Set-Location $repo
function Say($m) { "[$(Get-Date)] $m" | Tee-Object -FilePath $log -Append }

# Validate $Sites up front. A bad site name used to fall through to "MISSING manifest -- skip", so the
# year was declared COMPLETE having done nothing. That is exactly how a `powershell -File` array-flattening
# bug (-Years 2024,2023,... spilling onto the positional params) silently produced a "corpus DONE" ntfy and
# a model-total-zero CSV on 2026-07-28. Unknown site is now fatal.
$KNOWN_SITES = @("ganaraska", "credit")
foreach ($s in $Sites) {
    if ($KNOWN_SITES -notcontains $s.ToLower()) {
        Say "ABORT: unknown site '$s' (expected one of: $($KNOWN_SITES -join ', ')). Check how -Sites/-Years were passed."
        & $py $ntfy "v2.7 corpus ABORT: unknown site '$s'" --title "FishCV v2.7 corpus ABORT" --priority 5 2>$null
        exit 3
    }
}
foreach ($p in @($config, $ckpt)) {
    if (-not (Test-Path $p)) { Say "ABORT: missing $p"; & $py $ntfy "v2.7 corpus ABORT: missing $p" --title "FishCV v2.7 corpus ABORT" --priority 5 2>$null; exit 2 }
}
Say "=== v2.7 FULL CORPUS begin (ckpt0017, fp16=$(-not $NoFp16), $Shards shards, years $($Years -join ',')) ==="
& $py $ntfy "v2.7 full-corpus run begin ($($Years -join ',')); ~4-5 days" --title "FishCV v2.7 corpus" --priority 3 2>$null

foreach ($yr in $Years) {
    $out   = "G:\Projects\deimv2_x_v27_corpus$yr"
    $store = "$out\store"
    New-Item -ItemType Directory -Force $store | Out-Null

    foreach ($site in $Sites) {
        # 2025's original manifests omitted 13% of the footage -> always prefer *_full.json when present.
        $manifest = "$lb\corpus_${yr}_${site}_full.json"
        if (-not (Test-Path $manifest)) { $manifest = "$lb\corpus_${yr}_${site}.json" }
        if (-not (Test-Path $manifest)) { Say "MISSING manifest for $yr $site -- skip"; continue }

        Say "--- detect-store $yr $site  ($(Split-Path $manifest -Leaf)) ---"
        $procs = @()
        for ($i = 0; $i -lt $Shards; $i++) {
            $lg = "$repo\logs\corpus_v27_detstore_${yr}_${site}_shard_$i.log"
            $argline = "-u `"$repo\scripts\_deimv2_detect_store.py`" --config `"$config`" --ckpt `"$ckpt`" " +
                       "--op-json `"$manifest`" --out-dir `"$store`" --key-mode video_id " +
                       "--footage-root `"F:/RiverWatcher`" `"G:/RiverWatcher`" --thresh 0.3 " +
                       "--mask-header-px 40 --full-dist $fp16flag --frame-batch-size $BatchSize " +
                       "--frame-skip $FrameSkip --shard $i --num-shards $Shards"
            $procs += Start-Process -FilePath $vpy -ArgumentList $argline -RedirectStandardOutput $lg `
                                    -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
            Start-Sleep 4
        }
        $procs | Wait-Process
        Say "detect-store $yr $site DONE"

        if (-not $SkipTrack) {
            Say "--- BoT-SORT $yr $site ---"
            $procs = @()
            for ($i = 0; $i -lt $Shards; $i++) {
                $lg  = "$repo\logs\corpus_v27_botsort_${yr}_${site}_shard_$i.log"
                $agg = "$out\agg_${site}_shard$i.pkl"
                $argline = "-u `"$repo\scripts\_track_op_from_store.py`" --store-dir `"$store`" " +
                           "--op-json `"$manifest`" --key-mode video_id --footage-root `"F:/RiverWatcher`" " +
                           "--decode-backend decord --reid-fp16 --frame-skip $FrameSkip " +
                           "--agg-out `"$agg`" --shard $i --num-shards $Shards"
                $procs += Start-Process -FilePath $py -ArgumentList $argline -RedirectStandardOutput $lg `
                                        -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
                Start-Sleep 4
            }
            $procs | Wait-Process
            Say "BoT-SORT $yr $site DONE"
        }
    }

    # Pass C -- daily model-vs-tech CSVs for the year (cheap; re-runnable with other thresholds later)
    $bothman = "$lb\corpus_${yr}_both_full.json"
    if (-not (Test-Path $bothman)) { $bothman = "$lb\corpus_${yr}_both.json" }
    if ($SkipAggregate) { Say "Pass C skipped for $yr (-SkipAggregate)" }
    elseif (Test-Path $bothman) {
        foreach ($m in @("net", "activity")) {
            & $py -u "$repo\scripts\_corpus_countr_from_store.py" --store-dir $store --agg "$out\agg_*.pkl" `
                --videos-json $bothman --key-mode video_id --year $yr --metric $m `
                --out-csv "$out\${yr}_${m}_v27.csv" 2>&1 | Tee-Object -FilePath $log -Append
        }
    } else { Say "no both-manifest for $yr -- skipped aggregation (run _corpus_countr_from_store later)" }

    Say "=== YEAR $yr COMPLETE -> $out ==="
    & $py $ntfy "v2.7 corpus: $yr complete (both rivers, detect+track+counts)" --title "FishCV v2.7 $yr" --priority 4 2>$null
}

Say "=== v2.7 corpus: requested years DONE ($($Years -join ',')) ==="
if (-not $NoFinalNotify) {
    & $py $ntfy "v2.7 corpus DONE for years $($Years -join ',') -- compare vs v2.6 next" `
        --title "FishCV v2.7 corpus DONE" --priority 5 --tags fish 2>$null
}

# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: restarted run_corpus_v27_by_year.ps1 after the workstation went
# down on 28 July 2026 (finished Credit 2025, then continued with 2024 and earlier).
#
# Source: research repository, main branch (commits da8f3ba 2026-07-27, 6611e2b and 5d84978 2026-07-28).
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps detect audit-stores track audit
#     (every step resumes where it stopped)
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# RESUME the v2.7 full-corpus run after the 2026-07-28 02:38 machine crash.
#
# What the crash left behind (verified from the store + agg pickles, not from the log, which had not
# flushed past 22:41):
#   2025 Ganaraska  detect-store DONE (35,238) + BoT-SORT DONE (agg_ganaraska_shard{0,1}.pkl, 17.9/17.8 MB)
#   2025 Credit     detect-store ~11,680 / 12,759 (91.5%), BoT-SORT NOT started
#   2024-2020       not started
# System log: Event 6008, "previous system shutdown at 2:38:22 AM on 7/28/2026 was unexpected"; Event 41
# at 02:49. The run did not fail -- the machine went down under it.
#
# Resume strategy. detect-store skips videos already in the store, so re-running Ganaraska 2025 would be
# cheap; BoT-SORT has NO skip and would redo ~3 h of completed tracking and overwrite good agg pickles.
# So finish Credit 2025 ALONE first (-Sites credit), then run Pass C for the whole year once both sites
# are present, then continue 2024->2020 normally.
#
#   powershell -File runners/run_corpus_v27_resume.ps1
param(
    [int]$Shards = 2
)
$repo = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$py   = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy = "$repo\scripts\_ntfy_send.py"
$log  = "$repo\logs\corpus_v27.log"
$byyr = "$repo\runners\run_corpus_v27_by_year.ps1"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
Set-Location $repo
function Say($m) { "[$(Get-Date)] $m" | Tee-Object -FilePath $log -Append }

Say "=== v2.7 RESUME after 2026-07-28 02:38 crash ==="
& $py $ntfy "v2.7 corpus RESUMING after machine crash (2025 Credit, then 2024-2020)" `
    --title "FishCV v2.7 resume" --priority 3 2>$null

# NOTE on how the inner runner is invoked. `powershell -File $byyr -Years 2024, 2023, ...` does NOT work:
# -File flattens arguments onto a command line, so only "2024" binds to -Years and the remaining values
# spill onto the next POSITIONAL parameters (FrameSkip, BatchSize, Sites). On 2026-07-28 that silently gave
# Sites=@("2021","2020"), no manifest matched, and the year was declared COMPLETE having processed nothing --
# with a "corpus DONE" ntfy and a model-total-zero CSV to show for it. Call the script in-process with `&`
# and splatting instead, which passes real arrays. (-Sites is also validated in the runner now.)

# --- Step 1: finish 2025 Credit only. Pass C deferred: Ganaraska's half of the year is already done and
# aggregating on a half-finished Credit would write a misleading CSV.
Say "--- step 1/3: 2025 Credit (detect resumes from store, then BoT-SORT) ---"
$step1 = @{ Years = @(2025); Sites = @("credit"); Shards = $Shards
            SkipAggregate = $true; NoFinalNotify = $true }
& $byyr @step1

# --- Step 2: Pass C for 2025, now that BOTH sites are complete.
Say "--- step 2/3: 2025 Pass C (both sites present) ---"
$out = "G:\Projects\deimv2_x_v27_corpus2025"
$man = "G:\Projects\label_batches\corpus_2025_both_full.json"
if (Test-Path $man) {
    foreach ($m in @("net", "activity")) {
        & $py -u "$repo\scripts\_corpus_countr_from_store.py" --store-dir "$out\store" --agg "$out\agg_*.pkl" `
            --videos-json $man --key-mode video_id --year 2025 --metric $m `
            --out-csv "$out\2025_${m}_v27.csv" 2>&1 | Tee-Object -FilePath $log -Append
    }
} else { Say "WARN: $man missing -- 2025 Pass C skipped" }

# --- Step 3: the remaining years, untouched by the crash.
Say "--- step 3/3: 2024 -> 2020, both sites ---"
$step3 = @{ Years = @(2024, 2023, 2022, 2021, 2020); Shards = $Shards }
& $byyr @step3

Say "=== v2.7 RESUME COMPLETE ==="
& $py $ntfy "v2.7 corpus resume DONE (2025 + 2024-2020) -- compare vs v2.6 next" `
    --title "FishCV v2.7 corpus DONE" --priority 5 --tags fish 2>$null

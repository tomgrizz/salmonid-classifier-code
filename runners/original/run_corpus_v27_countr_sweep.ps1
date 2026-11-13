# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: produced every count file the paper uses. Eight direction rules
# x six seasons (data/net_rule_sweep/*.csv are copies of its output), the six activity files, and
# the two files for the recovered 2020 Ganaraska footage. Run for all seasons on 3 August 2026
# and again for 2025 on 5 August 2026, after 679 Credit 2025 videos lost to a damaged store file
# had been re-detected and re-tracked by hand (that repair has no script of its own).
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps count canonical compare
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# Pass C for v2.7: rebuild the whole count-r chain from the completed corpus aggregates.
#
# Emits, into G:\Projects\deimv2_x_v27_corpus2025\{net_rule_sweep,activity_sweep}\ (the same layout
# _build_canonical_results.py reads for v2.6, so only its SW constant has to change):
#   * <year>_<rule>.csv  for 8 directional-rule variants x 6 seasons  -> the net-rule sweep and the
#     leave-one-year-out validation of the 50/300 px rule
#   * <year>_activity.csv for 6 seasons
#   * 2020gan_recovered_{net,activity}_botsort.csv, dated from the RiverWatcher log via the datemap
#     rather than file mtime (the recovered footage carries no reliable mtime)
#
# CPU only -- reads the aggregates and the detect-store, loads no model. Safe to run while the GPU
# is busy, though it is heavy on G: so do not overlap it with a detect-store pass.
#
# The invocation form was validated by reproducing G:\...v27_corpus2024\2024_net_v27.csv byte for
# byte before this script was written.
#
# Usage:
#   ./run_corpus_v27_countr_sweep.ps1                 # everything
#   ./run_corpus_v27_countr_sweep.ps1 -Years 2024     # one season
#   ./run_corpus_v27_countr_sweep.ps1 -NetOnly        # skip the activity metric
param(
    [int[]] $Years   = @(2020, 2021, 2022, 2023, 2024, 2025),
    [switch]$NetOnly,
    [switch]$SkipRecovered
)
$repo = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$py   = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy = "$repo\scripts\_ntfy_send.py"
$labels = "G:\Projects\label_batches"
$OUT  = "G:\Projects\deimv2_x_v27_corpus2025"          # the sweep lives beside the newest season
$log  = "$repo\logs\corpus_v27_countr.log"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$OUT\net_rule_sweep", "$OUT\activity_sweep", "$repo\logs" | Out-Null

function Say($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object $log -Append }

# rule label -> (up threshold, down threshold). "up50" etc. keep the v2.6 file names so the
# canonical builder and the committed sweep CSVs in the public repo line up.
$RULES = [ordered]@{
    "up50"      = @(50,  300)      # the deployed rule
    "up150"     = @(150, 300)
    "up250"     = @(250, 300)
    "up300"     = @(300, 300)
    "u50_d150"  = @(50,  150)
    "u50_d50"   = @(50,  50)
    "u150_d150" = @(150, 150)
    "u1_d1"     = @(1,   1)
}

function Count-R($storeDir, $aggGlob, $manifest, $year, $metric, $up, $down, $outCsv, $dateMap) {
    $a = "-u `"$repo\scripts\_corpus_countr_from_store.py`" --store-dir `"$storeDir`" " +
         "--agg `"$aggGlob`" --videos-json `"$manifest`" --key-mode video_id " +
         "--year $year --metric $metric --up-thresh $up --down-thresh $down " +
         "--out-csv `"$outCsv`""
    if ($dateMap) { $a += " --date-map `"$dateMap`"" }
    $lg = "$repo\logs\countr_v27_tmp.log"
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo `
                       -RedirectStandardOutput $lg -RedirectStandardError "$lg.err" `
                       -NoNewWindow -PassThru
    $p | Wait-Process
    if (-not (Test-Path $outCsv)) { Say "  FAILED (no output): $outCsv -- see $lg.err"; return $false }
    # A count-r CSV with only a header means the aggregates were empty, which is the silent
    # failure this campaign kept hitting. Treat it as failure, not success.
    $rows = (Get-Content $outCsv | Measure-Object -Line).Lines - 1
    if ($rows -lt 1) { Say "  FAILED (0 data rows): $outCsv"; return $false }
    return $true
}

Say "=== v2.7 count-r + sweep BEGIN (years $($Years -join ',')) ==="

foreach ($yr in $Years) {
    $store    = "G:\Projects\deimv2_x_v27_corpus${yr}\store"
    $aggGlob  = "G:\Projects\deimv2_x_v27_corpus${yr}\agg_*.pkl"
    # 2025 was reprocessed on the backfilled manifests; every other season uses the plain one.
    $manifest = if ($yr -eq 2025) { "$labels\corpus_2025_both_full.json" } else { "$labels\corpus_${yr}_both.json" }
    if (-not (Test-Path $manifest)) { Say "$yr : MISSING manifest $manifest -- skip"; continue }
    if (-not (Test-Path $store))    { Say "$yr : MISSING store -- skip"; continue }

    foreach ($rule in $RULES.Keys) {
        $up, $down = $RULES[$rule]
        $dest = "$OUT\net_rule_sweep\${yr}_${rule}.csv"
        Say "$yr $rule (up=$up down=$down) -> net"
        [void](Count-R $store $aggGlob $manifest $yr "net" $up $down $dest $null)
    }
    if (-not $NetOnly) {
        Say "$yr activity"
        [void](Count-R $store $aggGlob $manifest $yr "activity" 50 300 "$OUT\activity_sweep\${yr}_activity.csv" $null)
    }
    Say "$yr DONE"
}

# The recovered 2020 Ganaraska season: its own store and aggregates, and dated from the
# RiverWatcher log via the v2.6 datemap (a property of the footage, not of the model, so it is
# reused rather than rebuilt).
if (-not $SkipRecovered -and $Years -contains 2020) {
    $recStore = "G:\Projects\deimv2_x_v27_corpus2020\store_gan_recovered"
    $recAgg   = "G:\Projects\deimv2_x_v27_corpus2020\agg_gan_recovered_*.pkl"
    $recMan   = "$labels\corpus_2020_gan_recovered.json"
    $dateMap  = "G:\Projects\deimv2_x_v26_corpus2020\gan_recovered_datemap.json"
    if ((Test-Path $recStore) -and (Test-Path $dateMap)) {
        foreach ($m in @("net", "activity")) {
            if ($NetOnly -and $m -eq "activity") { continue }
            $dest = "G:\Projects\deimv2_x_v27_corpus2020\2020gan_recovered_${m}_botsort.csv"
            Say "2020 gan_recovered -> $m"
            [void](Count-R $recStore $recAgg $recMan 2020 $m 50 300 $dest $dateMap)
        }
    } else {
        Say "2020 gan_recovered: store or datemap missing -- skip"
    }
}

Say "=== v2.7 count-r + sweep COMPLETE ==="
& $py $ntfy "v2.7 count-r + net-rule sweep complete" --title "FishCV v27 count-r" --priority 4 2>$null

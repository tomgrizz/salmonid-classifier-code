# ==================================================================================================
# HISTORICAL RECORD. NOT RUNNABLE FROM THIS REPOSITORY.
#
# Everything below this banner is the script exactly as it was on the authors' workstation.
# It is kept so that a reader can see which commands and settings produced the published
# results. It calls scripts by their names in the private research repository and uses that
# workstation's drive letters and folders, so it will not run anywhere else.
#
# Role in the published v2.7 run: detection and tracking of the seasons the first attempt left
# empty: Credit 2022, both rivers 2020 and 2021, and the recovered 2020 Ganaraska footage,
# 1-3 August 2026.
#
# Source: research repository commit d06535a (2026-08-05), "v2.7 corpus pipeline: runners, integrity guards, and the analysis scripts".
#
# To run the same step from this repository:
#     python -m salmonid.count.run_corpus --steps detect audit-stores track audit --years 2022 2021 2020
#
# Script names in the research repository and their equivalents in this repository:
#   scripts\_deimv2_detect_store.py       ->  python -m salmonid.detect.detect_store
#   scripts\_track_op_from_store.py       ->  python -m salmonid.detect.track_from_store
#   scripts\_corpus_countr_from_store.py  ->  python -m salmonid.count.corpus_countr_from_store
#   scripts\_build_canonical_results.py   ->  python -m salmonid.results.build_canonical_results
#   scripts\_audit_store_integrity.py     ->  python -m salmonid.detect.audit_store_integrity
#   scripts\_ntfy_send.py                 ->  salmonid.util.ntfy (off unless SALMONID_NTFY_TOPIC is set)
# ==================================================================================================
# One-shot driver that finishes the v2.7 corpus: the years the 2026-07-27 attempt lost to the
# double-footage-root argparse error (see run_corpus_v27_detstore_by_year.ps1 header), plus the
# recovered 2020 Ganaraska season that Table 5 / Figure 4 report.
#
# Already complete under v2.7 and NOT re-run here: 2023, 2024, 2025 (both rivers) and 2022 Ganaraska.
# NOT run here: 2019 Credit -- its corpus manifest no longer exists on disk, and it is a recovered
# season excluded from every agreement statistic. It enters only the pooled rank correlation
# (n = 39 species-years), which the year-inclusion cleanup drops to n = 36 (2020-2025, both rivers,
# three species). If 2019 Credit must stay, its manifest has to be rebuilt first.
#
# Ordered smallest-first so partial progress completes whole seasons:
#   1. 2022 Credit          6,500 videos  -> finishes 2022
#   2. 2020 both rivers     7,649         -> finishes plain 2020
#   3. 2021 both rivers    29,436         -> finishes 2021
#   4. 2020 gan_recovered  22,877         -> finishes the recovered Ganaraska season
# ~66,500 videos total; at the observed ~3,800 videos/GPU-hour that is ~18 h detect + ~6 h track.
# Every stage is resumable (the store skips existing pkls, the agg skips tracked videos), so this
# script is safe to re-run after an interrupt.
param(
    [switch]$SkipRecovered,          # stop after step 3
    [switch]$Only2020Recovered,      # run ONLY step 4 (used by run_corpus_v27_finish.ps1)
    [int]   $Shards = 2
)
$ErrorActionPreference = "Continue"
$repo = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$log  = "$repo\logs\corpus_v27_backfill.log"
$py   = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$pyd  = "$repo\.venv_deim\Scripts\python.exe"
$ntfy = "$repo\scripts\_ntfy_send.py"
$config  = "$repo\single_stage\configs\deimv2\deimv2_fish_x_v25.yml"
$ckpt    = "C:\Users\lomu-\Documents\GitHub\DEIMv2\outputs\deimv2_fish_x_v27\checkpoint0017.pth"
$footage = "F:\RiverWatcher"
$labels  = "G:\Projects\label_batches"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$repo\logs" | Out-Null

function Say($m) { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $m" | Tee-Object $log -Append }

# Run detect-store then BoT-SORT for one (manifest, store, agg-prefix) triple.
# $root overrides the footage root: the recovered 2020 Ganaraska season is NOT mirrored on F:.
# Its manifest video_ids are "Ganaraska/Ganaraska 2020/<reviewer>/..." and the footage sits at
# "D:\Ganaraska\Ganaraska 2020\...", so its root is bare "D:\" (matching config.RECOVERED_ROOT),
# not "D:\RiverWatcher". Passing F: silently resolved zero videos on 2026-08-02.
function Do-Pass($tag, $manifest, $store, $aggPrefix, $root = $null) {
    if (-not $root) { $root = $footage }
    if (-not (Test-Path $manifest)) { Say "MISSING manifest $manifest -- skip $tag"; return }
    New-Item -ItemType Directory -Force $store | Out-Null

    Say "$tag : detect-store begin ($Shards shards)"
    $before = if (Test-Path $store) { (Get-ChildItem $store -File -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
    $procs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $lg = "$repo\logs\corpus_v27_detstore_${tag}_shard_$i.log"
        $a  = "-u `"$repo\scripts\_deimv2_detect_store.py`" --config `"$config`" --ckpt `"$ckpt`" " +
              "--op-json `"$manifest`" --out-dir `"$store`" --key-mode video_id " +
              "--footage-root `"$root`" --thresh 0.3 --mask-header-px 40 --full-dist --fp16 " +
              "--frame-batch-size 32 --frame-skip 3 --shard $i --num-shards $Shards"
        $procs += Start-Process -FilePath $pyd -ArgumentList $a -WorkingDirectory $repo `
                                -RedirectStandardOutput $lg `
                                -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
        Start-Sleep 4
    }
    $procs | Wait-Process
    $n = (Get-ChildItem $store -File -ErrorAction SilentlyContinue | Measure-Object).Count
    # A detect-store pass that adds nothing has failed, whatever its exit code. This is the guard
    # that both 2026-08-01/02 silent failures needed: a wrong footage root and a mangled argument
    # list both look like instant success otherwise. Abort the stage rather than tracking nothing.
    if ($n -le $before) {
        Say "$tag : detect-store FAILED -- store went $before -> $n files. See ${tag}_shard*.log.err"
        & $py $ntfy "v27 $tag detect-store FAILED (stored nothing)" --title "FishCV v27 FAIL" --priority 5 2>$null
        return
    }
    Say "$tag : detect-store DONE (store holds $n files, +$($n - $before))"

    Say "$tag : BoT-SORT begin"
    $procs = @()
    for ($i = 0; $i -lt $Shards; $i++) {
        $lg  = "$repo\logs\corpus_v27_botsort_${tag}_shard_$i.log"
        $agg = "${aggPrefix}_shard$i.pkl"
        $a   = "-u `"$repo\scripts\_track_op_from_store.py`" --store-dir `"$store`" " +
               "--op-json `"$manifest`" --key-mode video_id --footage-root `"$root`" " +
               "--decode-backend decord --reid-fp16 --frame-skip 3 " +
               "--agg-out `"$agg`" --shard $i --num-shards $Shards"
        $procs += Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo `
                                -RedirectStandardOutput $lg `
                                -RedirectStandardError "$lg.err" -NoNewWindow -PassThru
        Start-Sleep 4
    }
    $procs | Wait-Process
    # Guard against the failure this script exists to prevent: a stage that "finishes" instantly
    # having produced nothing. A real BoT-SORT pass writes a multi-MB aggregate per shard.
    $small = @(Get-ChildItem "${aggPrefix}_shard*.pkl" -ErrorAction SilentlyContinue |
               Where-Object { $_.Length -lt 1024 })
    if ($small) {
        Say "$tag : BoT-SORT FAILED -- $($small.Count) aggregate(s) under 1 KB. See ${tag}_shard*.log.err"
        & $py $ntfy "v27 backfill $tag BoT-SORT FAILED (empty aggregate)" --title "FishCV v27 FAIL" --priority 5 2>$null
    } else {
        Say "$tag : BoT-SORT DONE"
    }
    & $py $ntfy "v27 backfill $tag done" --title "FishCV v27 backfill" --priority 4 2>$null
}

Say "=== v2.7 corpus backfill BEGIN (ckpt0017) ==="

if (-not $Only2020Recovered) {
# 1. 2022 Credit (Ganaraska 2022 already landed under v2.7)
Do-Pass "2022_credit" "$labels\corpus_2022_credit.json" `
        "G:\Projects\deimv2_x_v27_corpus2022\store" "G:\Projects\deimv2_x_v27_corpus2022\agg_credit"

# 2. 2020 both rivers (plain manifests)
Do-Pass "2020_credit" "$labels\corpus_2020_credit.json" `
        "G:\Projects\deimv2_x_v27_corpus2020\store" "G:\Projects\deimv2_x_v27_corpus2020\agg_credit"
Do-Pass "2020_ganaraska" "$labels\corpus_2020_ganaraska.json" `
        "G:\Projects\deimv2_x_v27_corpus2020\store" "G:\Projects\deimv2_x_v27_corpus2020\agg_ganaraska"

# 3. 2021 both rivers
Do-Pass "2021_credit" "$labels\corpus_2021_credit.json" `
        "G:\Projects\deimv2_x_v27_corpus2021\store" "G:\Projects\deimv2_x_v27_corpus2021\agg_credit"
Do-Pass "2021_ganaraska" "$labels\corpus_2021_ganaraska.json" `
        "G:\Projects\deimv2_x_v27_corpus2021\store" "G:\Projects\deimv2_x_v27_corpus2021\agg_ganaraska"
}

# 4. Recovered 2020 Ganaraska (its own store + agg prefix, matching the v2.6 layout)
if (-not $SkipRecovered) {
    Do-Pass "2020_gan_recovered" "$labels\corpus_2020_gan_recovered.json" `
            "G:\Projects\deimv2_x_v27_corpus2020\store_gan_recovered" `
            "G:\Projects\deimv2_x_v27_corpus2020\agg_gan_recovered" `
            "D:/"
    # NOTE the FORWARD slash. These runners build one command-line STRING and hand it to
    # Start-Process, so a value ending in a backslash ("D:\") puts \" into the string, which the
    # Windows argument parser reads as an escaped quote: the root swallows every argument after
    # it and --mask-header-px silently reverts to 0. "D:/" resolves identically via pathlib and
    # cannot do that. Do not "tidy" this back to a backslash.
}

Say "=== v2.7 corpus backfill COMPLETE ==="
& $py $ntfy "v2.7 corpus backfill COMPLETE -- all years now have non-empty aggs" --title "FishCV v27 backfill DONE" --priority 5 2>$null

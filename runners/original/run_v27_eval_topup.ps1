# ==================================================================================================
# HISTORICAL RECORD. Do not run this file; it is kept to be read.
#
# This is the PowerShell script that completed the held-out scoring of the PUBLISHED model
# (v2.7). The first scoring pass, at the end of run_deimv2_fish_x_v27.ps1, read footage from a
# fast mirror disk that was missing 8 of the 364 test clips, so two checkpoints were scored on
# 358 fish instead of 366. This script re-ran the detection step with the complete footage
# archive as a second source and re-scored every candidate on all 366. The published score
# (macro-F1 0.961, checkpoint 0017) comes from this second pass. Everything below this banner
# is the research repository's runners/run_v27_eval_topup.ps1, unchanged (commit d40b869).
#
# It cannot run from this deposit (workstation paths, research-repository script names).
# The portable equivalent is the --eval step of:  python -m salmonid.train.train_v27
# which reads the complete footage archive directly, so no second pass is needed.
# ==================================================================================================
# v2.7 eval TOP-UP + re-score on the COMPLETE 366-track test set.
#
# WHY: the main eval (run_deimv2_fish_x_v27.ps1) indexes footage from the F: mirror only. On 2026-07-25 the
# mirror was missing 24,565 clips that G: still had, including 8 test_set_v6 videos (5 Atlantic, 3 Chinook).
# The detect-store logged them as "unresolved" and they silently left the F1 denominator -- on a 60-track
# class, 5 missing Atlantic is a visible artifact, not a model property. _deimv2_detect_store.py now takes
# multiple --footage-root values (mirror first, archive as fallback); this script re-runs each finished
# store with both roots (existing pickles are skipped, so it only fetches the stragglers) and re-scores.
#
# Waits for the main eval to exit first, so it can be launched at any time.
#   powershell -File runners/run_v27_eval_topup.ps1 -WaitPid 26216
param(
    [int]$WaitPid = 0,
    [string[]]$Ckpts = @("0013", "0015", "0017", "0019", "0021", "0023", "best_stg2")
)

$deim  = "C:\Users\lomu-\Documents\GitHub\DEIMv2"
$repo  = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$vpy   = "$repo\.venv_deim\Scripts\python.exe"
$py    = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy  = "$repo\scripts\_ntfy_send.py"
$log   = "$repo\logs\deimv2_x_v27_topup.log"
$cfg   = "configs/deimv2/deimv2_fish_x_v27.yml"
$base  = "G:\Projects\deimv2_x_v27_eval"
$testvids = "G:/Projects/label_batches/test_set_v6_videos.json"
$res   = "$base\results_366.txt"

$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
Set-Location $repo
function Say($m) { "[$(Get-Date)] $m" | Tee-Object -FilePath $log -Append }

if ($WaitPid -gt 0) {
    Say "waiting for the main v2.7 eval (PID $WaitPid) to finish..."
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch { Say "PID $WaitPid already gone" }
}

"v2.7-X test_set_v6 on the COMPLETE 366 tracks (footage: F: mirror + G: archive)" | Out-File -FilePath $res -Encoding utf8
"baseline v2.6 ckpt0007: FULL 0.966 (contaminated: 62 of 366 test fish were trained on) | 304-track leak-clean 0.955" |
    Out-File -FilePath $res -Append -Encoding utf8

foreach ($tag in $Ckpts) {
    $cf = if ($tag -match '^\d+$') { "outputs/deimv2_fish_x_v27/checkpoint$tag.pth" }
          else { "outputs/deimv2_fish_x_v27/$tag.pth" }
    $store = "$base\store_$tag"
    if (-not (Test-Path (Join-Path $deim ($cf.Replace('/', '\'))))) { Say "SKIP $tag (no checkpoint)"; continue }
    if (-not (Test-Path $store)) { Say "SKIP $tag (no store -- main eval never reached it)"; continue }
    Say "--- top-up $tag ---"
    & $vpy -u "$repo\scripts\_deimv2_detect_store.py" --config $cfg --ckpt $cf --op-json $testvids `
        --key-mode rel --thresh 0.3 --mask-header-px 40 --full-dist --out-dir $store `
        --footage-root "F:/RiverWatcher" "G:/RiverWatcher" 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Say "$tag FAIL detect"; continue }
    & $py -u "$repo\scripts\_predict_testset_v6_from_store.py" --store-dir $store `
        --out "$base\preds_$tag.csv" --video-out "$base\videopreds_$tag.csv" 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Say "$tag FAIL predict"; continue }
    "===================== $tag =====================" | Out-File -FilePath $res -Append -Encoding utf8
    & $py -u "$repo\scripts\_score_testset_v6.py" --preds "$base\preds_$tag.csv" `
        --video-preds "$base\videopreds_$tag.csv" 2>&1 | Tee-Object -FilePath $log -Append |
        Out-File -FilePath $res -Append -Encoding utf8
}
Say "v2.7 366-track re-score done -> $res"
& $py $ntfy "v2.7 test_set_v6 re-scored on the full 366 (8 F:-missing clips recovered from G:)" --title "FishCV v2.7 366" --priority 5 --tags fish 2>$null

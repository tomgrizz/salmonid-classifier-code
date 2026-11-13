# ==================================================================================================
# HISTORICAL RECORD. Do not run this file; it is kept to be read.
#
# This is the PowerShell script that launched the training run of the PUBLISHED model (v2.7) on
# the lead author's workstation on 2026-07-24, and then scored the candidate checkpoints on the
# held-out test set. Everything below this banner is the research repository's
# runners/run_deimv2_fish_x_v27.ps1, unchanged (first committed there as d40b869, three days
# after the run; the launch log of the run shows the same steps and messages).
#
# It cannot run from this deposit: its paths are that workstation's, and it calls the helper
# scripts by their research-repository names (scripts/_audit_coco_holdout_leak.py and so on).
#
# The supported, portable equivalent, which performs the same steps in the same order, is:
#     python -m salmonid.train.train_v27 --dry-run      (print every step and guard, train nothing)
#     python -m salmonid.train.train_v27                (run it)
# What the two do differently is listed in the docstring of salmonid/train/train_v27.py.
# ==================================================================================================
# Fine-tune DEIMv2-X v2.7 = the LEAK-CLEAN publish candidate (2026-07-24).
#
# WHY: v2.5/v2.6 excluded test_set_v6 by (video_id, track_id) only, which does not identify a fish
# (track ids are per detection run; `track_id == -1` = "the clip's dominant track"). 62 of the 366
# held-out test fish were trained on WITH THEIR OWN LABEL -> v2.6's 0.966 is 0.955 leak-clean.
# v2.7 changes exactly two things vs v2.5:
#   (1) DATA  = masked_v26_clean (v2.6 corpus, holdout re-applied at VIDEO level, gated below)
#   (2) INIT  = ckpts/deimv2_x_coco.pth, NOT a v2.x checkpoint -- v2.6 warm-started from v2.5 ckpt0019
#       and the v2.3 base under both carried 149 leaked test tracks, so any warm start re-imports the
#       leak through the weights. Hence the full 30-epoch schedule, not v2.6's 10-epoch continuation.
#
# Reproducible-by-default: hard leak gate -> pinned seed -> run_manifest.json (both repo SHAs, DEIMv2
# worktree patch, config snapshot, train/val COCO sha256 + category counts, env, command) BEFORE training.
# Then (unless -NoEval) scores test_set_v6 for the candidate checkpoints. ASCII-only, unattended.
#
#   powershell -File runners/run_deimv2_fish_x_v27.ps1
#   powershell -File runners/run_deimv2_fish_x_v27.ps1 -NoEval          # train only
#   powershell -File runners/run_deimv2_fish_x_v27.ps1 -EvalOnly        # score existing checkpoints
param(
    [switch]$NoEval,
    [switch]$EvalOnly,
    [string[]]$EvalCkpts = @("0013", "0015", "0017", "0019", "0021", "0023", "best_stg2")
)

$deim  = "C:\Users\lomu-\Documents\GitHub\DEIMv2"
$repo  = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$vpy   = "$repo\.venv_deim\Scripts\python.exe"
$py    = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy  = "$repo\scripts\_ntfy_send.py"
$log   = "$repo\logs\deimv2_fish_x_v27.log"

$cfg     = "configs/deimv2/deimv2_fish_x_v27.yml"
$init    = "ckpts/deimv2_x_coco.pth"
$seed    = 0
$outdir  = "$deim\outputs\deimv2_fish_x_v27"
$trainj  = "C:/ssl_cache/deimv2_fish_6class_masked_v26_clean/train.json"
$valj    = "C:/ssl_cache/deimv2_fish_6class_masked_v26_clean/val_clean.json"
$evalbase = "G:\Projects\deimv2_x_v27_eval"
$testvids = "G:/Projects/label_batches/test_set_v6_videos.json"
$synth   = @("G:/Projects/sam2_synth_partial/synth_coco.json",
             "G:/Projects/sam2_synth_proximity/synth_coco.json",
             "G:/Projects/sam2_synth_v26_atlrbw/synth_coco.json",
             "G:/Projects/sam2_synth_v26_cohochin/synth_coco.json",
             "G:/Projects/sam2_synth_v26_cohorbw/synth_coco.json",
             "G:/Projects/sam2_synth_v26_partial/synth_coco.json")

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path "$repo\logs" | Out-Null
Set-Location $repo

function Say($msg) { "[$(Get-Date)] $msg" | Tee-Object -FilePath $log -Append }

if (-not $EvalOnly) {
    # --- Guard 1: the COCO exists ---
    if (-not (Test-Path $trainj)) {
        Say "ABORT: v2.7 train COCO missing: $trainj"
        & $py $ntfy "DEIMv2-X v2.7 ABORT: train COCO missing" --title "FishCV v2.7 ABORT" --priority 5 2>$null
        exit 2
    }

    # --- Guard 2: the COCO is leak-free. This is THE precondition for the whole run. ---
    Say "=== v2.7 leak gate (test_set_v6 + ls_test_v5, TRACK/VIDEO/SYNTH/same-fish-box) ==="
    & $py -u "$repo\scripts\_audit_coco_holdout_leak.py" --coco $trainj --deep --strict --synth-coco $synth `
        2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Say "ABORT: train COCO LEAKS held-out eval fish -- refusing to train"
        & $py $ntfy "DEIMv2-X v2.7 ABORT: COCO leak gate FAILED" --title "FishCV v2.7 ABORT" --priority 5 2>$null
        exit 3
    }
    & $py -u "$repo\scripts\_audit_coco_holdout_leak.py" --coco $valj --deep --strict 2>&1 |
        Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Say "ABORT: val COCO leaks"; exit 3 }

    Say "DEIMv2-X v2.7 begin (seed=$seed, leak-clean COCO, init from COCO weights, 30 epochs)"

    # --- Reproducibility manifest (pins code+config+data+seed+env BEFORE training) ---
    & $py "$repo\scripts\_record_run_manifest.py" `
        --output-dir  $outdir `
        --config      "$deim\$cfg" `
        --deimv2-dir  $deim `
        --repo-dir    $repo `
        --train-json  $trainj `
        --val-json    $valj `
        --train-python $vpy `
        --seed        $seed `
        --note        "DEIMv2-X v2.7 leak-clean publish candidate: v2.5 recipe + v2.6 data with the holdout re-applied at VIDEO level; init from deimv2_x_coco.pth (no v2.x warm start)" `
        --command     "train.py -c $cfg -t $init -d cuda --use-amp --seed $seed" `
        2>&1 | Tee-Object -FilePath $log -Append

    & $py $ntfy "DEIMv2-X v2.7 training begin (leak-clean; 30ep; ~26h; manifest pinned)" --title "FishCV DEIMv2-X v2.7" --priority 3 2>$null

    Set-Location $deim
    & $vpy -u train.py -c $cfg -t $init -d cuda --use-amp --seed $seed 2>&1 | Tee-Object -FilePath $log -Append
    $trainExit = $LASTEXITCODE
    Set-Location $repo
    if ($trainExit -ne 0) {
        Say "DEIMv2-X v2.7 training FAILED (exit $trainExit)"
        & $py $ntfy "DEIMv2-X v2.7 training FAILED (exit $trainExit)" --title "FishCV v2.7 FAIL" --priority 5 2>$null
        exit 1
    }
    Say "DEIMv2-X v2.7 training done -> $outdir"
    & $py $ntfy "DEIMv2-X v2.7 training DONE. Scoring test_set_v6..." --title "FishCV v2.7 trained" --priority 4 2>$null
}

if ($NoEval) { Say "eval skipped (-NoEval)"; exit 0 }

# --- test_set_v6 eval per candidate checkpoint -----------------------------------------------------
# NOTE --key-mode rel: _predict_testset_v6_from_store.py resolves stores by the legacy site__md5(session/file)
# key, so the store MUST be built in rel mode (the detect-store default is video_id).
New-Item -ItemType Directory -Force -Path $evalbase | Out-Null
$res = "$evalbase\results.txt"
"v2.7-X LEAK-CLEAN (v2.5 recipe, masked_v26_clean, COCO init) -- test_set_v6" | Out-File -FilePath $res -Encoding utf8
"baseline v2.6 ckpt0007: FULL 0.966 (Chin .971 Coho .950 Atl .966 Rbw .977 Brn .967) | LEAK-CLEAN 304-track 0.955" |
    Out-File -FilePath $res -Append -Encoding utf8

foreach ($tag in $EvalCkpts) {
    $cf = if ($tag -match '^\d+$') { "outputs/deimv2_fish_x_v27/checkpoint$tag.pth" }
          else { "outputs/deimv2_fish_x_v27/$tag.pth" }
    if (-not (Test-Path (Join-Path $deim ($cf.Replace('/', '\'))))) { Say "eval SKIP $tag (missing)"; continue }
    $store = "$evalbase\store_$tag"
    Say "--- eval $tag ---"
    if (-not (Test-Path "$store\.done")) {
        & $vpy -u "$repo\scripts\_deimv2_detect_store.py" --config $cfg --ckpt $cf --op-json $testvids `
            --key-mode rel --thresh 0.3 --mask-header-px 40 --full-dist --out-dir $store `
            2>&1 | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { Say "eval $tag FAIL detect"; continue }
        New-Item -ItemType File -Force -Path "$store\.done" | Out-Null
    }
    & $py -u "$repo\scripts\_predict_testset_v6_from_store.py" --store-dir $store `
        --out "$evalbase\preds_$tag.csv" --video-out "$evalbase\videopreds_$tag.csv" `
        2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { Say "eval $tag FAIL predict"; continue }
    "===================== $tag =====================" | Out-File -FilePath $res -Append -Encoding utf8
    & $py -u "$repo\scripts\_score_testset_v6.py" --preds "$evalbase\preds_$tag.csv" `
        --video-preds "$evalbase\videopreds_$tag.csv" 2>&1 |
        Tee-Object -FilePath $log -Append | Out-File -FilePath $res -Append -Encoding utf8
    # bbox gate (hard rule #2) -- record it for every candidate; v2.6 was never gated
    & $vpy -u "$repo\scripts\_deimv2_bbox_gate.py" --config $cfg --ckpt $cf --mask-header-px 40 `
        --overlay-dir "$repo\figures\deimv2_v27_gate_$tag" 2>&1 |
        Tee-Object -FilePath $log -Append | Select-String -Pattern "medIoU|Credit|Ganaraska|GATE:" |
        Out-File -FilePath $res -Append -Encoding utf8
}
Say "v2.7 eval done -> $res"
& $py $ntfy "DEIMv2-X v2.7 eval DONE -- see $res" --title "FishCV v2.7 DONE" --priority 5 --tags fish 2>$null

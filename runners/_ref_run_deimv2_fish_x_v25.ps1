# Fine-tune DEIMv2-X v2.5 = the DEPLOY / PUBLISH candidate (weekend run).
# v2.5 vs v2.4-X: (1) LR-FIX main lr 0.0005 -> 0.00025 (v2.4 ran batch 8 with the batch-32-tuned lr,
# ~2x v2.3-L per-sample; DEIMv2 has NO grad-accum so we halve). (2) DATA = v2.3 masked base +
# Tom-REVIEWED SAM2 synth (partial + coho/chinook proximity), replacing v2.4's noisy model-labelled
# mined co-occurrence (which regressed Coho). One clean data variable vs v2.4.
#
# This runner is REPRODUCIBLE-BY-DEFAULT: it pins an explicit --seed and writes a full run_manifest.json
# (both repo SHAs + DEIMv2 worktree patch + config snapshot + train/val COCO sha256 & category counts +
# env versions + command) into the output dir BEFORE training. See single_stage/REPRODUCE.md.
#
# PREREQS (built by the weekend build step, see single_stage/REPRODUCE.md):
#   C:/ssl_cache/deimv2_fish_6class_masked_v25/train.json   (synth-folded, holdout-filtered)
#   C:/ssl_cache/deimv2_fish_6class_masked_v23/val.json     (val reused from v23)
#   DEIMv2/ckpts/deimv2_x_coco.pth  +  dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth
# Isolated .venv_deim (cu128 torch for the 5090). ASCII-only, unattended/detached.

$deim  = "C:\Users\lomu-\Documents\GitHub\DEIMv2"
$repo  = "C:\Users\lomu-\Documents\GitHub\lake-ontario-fish-cv"
$vpy   = "$repo\.venv_deim\Scripts\python.exe"
$py    = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$ntfy  = "$repo\scripts\_ntfy_send.py"
$log   = "$repo\logs\deimv2_fish_x_v25.log"

$cfg     = "configs/deimv2/deimv2_fish_x_v25.yml"
$init    = "ckpts/deimv2_x_coco.pth"
$seed    = 0
$outdir  = "$deim\outputs\deimv2_fish_x_v25"
# train.json + val_clean.json are test_set_v6-EXCLUDED (holdout key dropped 369 base + 86 synth train imgs,
# 50 val imgs) so the held-out publishable test set is honest. Contaminated originals kept as *_v6contam.json.
$trainj  = "C:/ssl_cache/deimv2_fish_6class_masked_v25/train.json"
$valj    = "C:/ssl_cache/deimv2_fish_6class_masked_v25/val_clean.json"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# --- Guard: refuse to train on a missing/empty COCO (don't waste the weekend GPU) ---
if (-not (Test-Path $trainj)) {
    "[$(Get-Date)] ABORT: v2.5 train COCO missing: $trainj  (run the build step first)" | Tee-Object -FilePath $log
    & $py $ntfy "DEIMv2-X v2.5 ABORT: train COCO missing" --title "FishCV v2.5 ABORT" --priority 5 2>$null
    exit 2
}

"[$(Get-Date)] DEIMv2-X v2.5 begin (seed=$seed, LR-fix 0.00025, reviewed SAM2 synth)" | Tee-Object -FilePath $log

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
    --note        "DEIMv2-X v2.5 weekend deploy candidate: X arch + reviewed SAM2 synth + LR-fix" `
    --command     "train.py -c $cfg -t $init -d cuda --use-amp --seed $seed" `
    2>&1 | Tee-Object -FilePath $log -Append

& $py $ntfy "DEIMv2-X v2.5 training begin (deploy candidate; seed $seed; manifest pinned)" --title "FishCV DEIMv2-X v2.5" --priority 3 2>$null

Set-Location $deim
& $vpy -u train.py -c $cfg -t $init -d cuda --use-amp --seed $seed `
    2>&1 | Tee-Object -FilePath $log -Append

if ($LASTEXITCODE -ne 0) {
    & $py $ntfy "DEIMv2-X v2.5 training FAILED (exit $LASTEXITCODE)" --title "FishCV v2.5 FAIL" --priority 5 2>$null
    "[$(Get-Date)] DEIMv2-X v2.5 training FAILED" | Tee-Object -FilePath $log -Append
    exit 1
}
& $py $ntfy "DEIMv2-X v2.5 training DONE. Next: bbox gate + tracked-op ckpt sweep on LEAK-CLEAN op_dev (run_deimv2_x_v24_eval.ps1 pattern), then confirm op_test + count-r gate." --title "FishCV v2.5 DONE" --priority 5 --tags fish 2>$null
"[$(Get-Date)] DEIMv2-X v2.5 training done -> $outdir" | Tee-Object -FilePath $log -Append

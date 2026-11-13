# ==================================================================================================
# SUPERSEDED. Do not run this file; it is kept to be read.
#
# This runner trains v2.6, the model BEFORE the published one. v2.6 scored 0.966 on the held-out
# test set, but 62 of the 366 test fish were in its training pictures, so that score is not a
# clean test and no number in the manuscript comes from v2.6. The statement on the first line
# below ("the PUBLISHED single-stage model") was true when this file was written and is no
# longer true: the published model is v2.7 (python -m salmonid.train.train_v27).
#
# Unlike the two v2.7 files in this folder, this is NOT a copy of what was executed: the
# research repository never committed a v2.6 launch script. It was written for an earlier
# state of this deposit and is moved here unchanged below this banner. It also relies on
# SALMONID_COCO_CACHE only for its checks, while the training step reads the path written in
# the settings file, so on another machine it could check one file and train on another.
# ==================================================================================================
# Fine-tune DEIMv2-X v2.6 = the PUBLISHED single-stage model (test_set_v6 macro-F1 0.966).
#
# v2.6 is a WARM-START continuation of v2.5: it resumes from the v2.5 epoch-19 checkpoint and
# trains 10 more epochs at lr 1e-4 (set in configs/deimv2/deimv2_fish_x_v26.yml), on the
# masked_v26 COCO (v2.5 base + targeted Coho/Rainbow SAM2 synth). The published checkpoint is
# epoch 7 (checkpoint0007.pth), selected on tracked-op + test_set_v6 (NOT lab AP). See
# docs/REPRODUCE.md for the full recipe and docs/DEIMV2_PIN.md for the external-repo pin.
#
# REPRODUCIBLE-BY-DEFAULT: pins --seed and writes a full run_manifest.json (both repo SHAs +
# DEIMv2 worktree patch + config snapshot + train/val COCO sha256 & category counts + env +
# command) into the output dir BEFORE training.
#
# PREREQS (see docs/REPRODUCE.md + docs/DATA_AVAILABILITY.md):
#   $DEIM/outputs/deimv2_fish_x_v25/checkpoint0019.pth        (warm-start init)
#   $COCO/deimv2_fish_6class_masked_v26/{train.json,val_clean.json}
#   $DEIM reconstructed at the pinned SHA + worktree patch, with the fish configs copied in
#   Isolated .venv_deim (cu128 torch for the 5090). ASCII-only, unattended/detached.

# --- paths (override via SALMONID_* env vars; defaults = original workstation) ---
$repo = "C:\Users\lomu-\Documents\GitHub\salmonid-classifier-code"
$deim = if ($env:SALMONID_DEIM_DIR) { $env:SALMONID_DEIM_DIR } else { "C:\Users\lomu-\Documents\GitHub\DEIMv2" }
$coco = if ($env:SALMONID_COCO_CACHE) { $env:SALMONID_COCO_CACHE } else { "C:\ssl_cache" }
$vpy  = if ($env:SALMONID_DEIM_VENV_PY) { $env:SALMONID_DEIM_VENV_PY } else { "$deim\.venv_deim\Scripts\python.exe" }
$py   = "C:\Users\lomu-\AppData\Local\Programs\Python\Python311\python.exe"
$log  = "$repo\logs\deimv2_fish_x_v26.log"

$cfg    = "configs/deimv2/deimv2_fish_x_v26.yml"          # relative to $deim (train.py runs from there)
$init   = "outputs/deimv2_fish_x_v25/checkpoint0019.pth"  # WARM START (not the COCO init ckpt)
$seed   = 0
$outdir = "$deim\outputs\deimv2_fish_x_v26"
$trainj = "$coco/deimv2_fish_6class_masked_v26/train.json"
$valj   = "$coco/deimv2_fish_6class_masked_v26/val_clean.json"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

# --- Guard: refuse to train on a missing COCO or warm-start ckpt ---
if (-not (Test-Path $trainj)) {
    "[$(Get-Date)] ABORT: v2.6 train COCO missing: $trainj  (build it first, see docs/REPRODUCE.md)" | Tee-Object -FilePath $log
    exit 2
}
if (-not (Test-Path "$deim\$init")) {
    "[$(Get-Date)] ABORT: v2.6 warm-start ckpt missing: $deim\$init" | Tee-Object -FilePath $log -Append
    exit 2
}

"[$(Get-Date)] DEIMv2-X v2.6 begin (seed=$seed, warm-start from v2.5 ckpt0019, +targeted SAM2 synth)" | Tee-Object -FilePath $log

# --- Reproducibility manifest (pins code+config+data+seed+env BEFORE training) ---
& $py -m salmonid.provenance.record_run_manifest `
    --output-dir   $outdir `
    --config       "$deim\$cfg" `
    --deimv2-dir   $deim `
    --repo-dir     $repo `
    --train-json   $trainj `
    --val-json     $valj `
    --train-python $vpy `
    --seed         $seed `
    --note         "DEIMv2-X v2.6 PUBLISHED model: warm-start from v2.5 ckpt0019 + targeted Coho/Rainbow SAM2 synth" `
    --command      "train.py -c $cfg -t $init -d cuda --use-amp --seed $seed" `
    2>&1 | Tee-Object -FilePath $log -Append

Set-Location $deim
& $vpy -u train.py -c $cfg -t $init -d cuda --use-amp --seed $seed `
    2>&1 | Tee-Object -FilePath $log -Append

if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date)] DEIMv2-X v2.6 training FAILED (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append
    exit 1
}
"[$(Get-Date)] DEIMv2-X v2.6 training done -> $outdir" | Tee-Object -FilePath $log -Append
"[$(Get-Date)] NEXT: select the deploy checkpoint on tracked-op (leak-clean op_dev), confirm ONCE on op_test + test_set_v6, then the bbox gate + count-r gate. See docs/REPRODUCE.md step 3." | Tee-Object -FilePath $log -Append

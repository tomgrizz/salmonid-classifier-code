"""In plain terms: this script trains the published fish-identification model (v2.7) the same
way it was trained for the paper. It does not train straight away. First it checks that the
training pictures are the published ones, then it runs the check that none of the "exam" fish
are among them and refuses to continue if any are, then it writes a receipt recording exactly
what is about to be used. Only after that does it start the training program, with the same
settings file, the same starting weights and the same random seed as the published run. It can
then score the resulting model on the held-out test set. With ``--dry-run`` it performs the
checks and prints every command it would run, and trains nothing.

Inputs
------
All locations come from ``salmonid/config.py`` (set the ``SALMONID_*`` variables to relocate):

* the training and validation picture lists, ``config.COCO_V27 / {train,val_clean}.json``,
  and the image files they name;
* the settings file ``configs/deimv2/deimv2_fish_x_v27.yml`` (committed, unchanged from the run);
* the external DEIMv2 detector code at ``config.DEIM_DIR`` (pinned commit plus a small patch,
  see ``docs/DEIMV2_PIN.md``), its starting weights ``config.DEIM_INIT_COCO`` and the backbone
  weights named in the settings chain;
* the Python interpreter of the DEIMv2 environment (``requirements-deim.txt``): set
  ``SALMONID_DEIM_PYTHON`` to it (``config.DEIM_PYTHON``), or start this launcher from that
  environment, in which case the same interpreter is used for every step;
* the held-out sets (``salmonid/holdout.py``) and, for the leak gate's composite check, the six
  ``synth_coco.json`` records under ``config.PROJECTS_ROOT``;
* ``provenance/v2.7/run_manifest.json`` (committed): the receipt of the published run, used to
  confirm the pictures on disk are the published ones.

Outputs
-------
Under ``--work-dir`` (default ``output/train_v27/``, which git ignores; never ``data/``):

* ``run/``: the receipt (``run_manifest.json``, ``config_snapshot/``, ``deimv2_worktree.patch``)
  and the trained checkpoints (about 0.8 GB each, 15 files, 12 GB in all);
* ``eval/`` (with ``--eval``): stored detections, predictions and ``results.txt`` per checkpoint;
* ``launch.log``: everything printed by the steps below.

Feeds
-----
The published model itself, and through it every model-derived number in the manuscript. The
headline it must reproduce is the held-out species score of checkpoint 0017: macro-F1 0.961 on
366 fish (Results, held-out classification). Training is seeded but not bit-for-bit
repeatable (see "Will I get the same model?"), so expect a score close to, not identical to, that.

Run
---
::

    python -m salmonid.train.train_v27 --dry-run          # checks + the full plan; trains nothing
    python -m salmonid.train.train_v27                    # train (the published run took 27 h 46 min on one RTX 5090)
    python -m salmonid.train.train_v27 --eval             # train, then score the candidates
    python -m salmonid.train.train_v27 --eval-only        # score checkpoints already in run/

Tier
----
2. It needs a CUDA GPU, the DEIMv2 environment and several external downloads (training
pictures, starting weights; footage and stored detections for ``--eval``).
See ``docs/DATA_AVAILABILITY.md``.

The steps, in order (the same as the original launch script)
------------------------------------------------------------
The published run was started by a PowerShell script, kept unchanged for the record in
``runners/original/run_deimv2_fish_x_v27.ps1``. This module performs the same steps:

1. **The training list exists.** (exit 2 if not)
2. **Leak gate.** ``salmonid.data.audit_coco_holdout_leak --deep --strict`` on the training list
   (with the composite records) and on the validation list. Any held-out fish stops the run
   (exit 3). This is the precondition for the whole run: it is why the held-out score means
   what the paper says it means.
3. **Receipt.** ``salmonid.provenance.record_run_manifest`` pins code, settings, data
   fingerprints, seed and software versions BEFORE training starts.
4. **Train.** DEIMv2 ``train.py`` with seed 0, starting from the public COCO-pretrained
   detector weights. Not from an earlier fish model: the earlier fish models had seen held-out
   fish, and starting from their weights would carry that knowledge over however clean the
   pictures are. (exit 1 if training fails)
5. **Score (optional).** For each candidate checkpoint: detect on the held-out clips, match the
   detections to the test fish, compute the species score, and run the box-placement check.

What this launcher does differently from the original script
------------------------------------------------------------
None of these changes what is trained.

* **Checks added before step 2.** The fingerprint (sha256) of the two picture lists is compared
  with the published receipt; a sample of the image files is checked to exist; the DEIMv2
  code, its patch, the starting weights and the training interpreter are checked. The original
  relied on the workstation being set up.
* **The data files are passed to the trainer explicitly.** The settings file contains the
  original workstation's absolute paths (``C:/ssl_cache/...``) and is kept unchanged as the
  record of what ran. The trainer is given the two list paths from ``salmonid.config`` through
  DEIMv2's ``-u`` override option, so the file that is checked, fingerprinted and trained on
  is always the same file. On the original workstation the paths are identical to the ones in
  the settings file. To override them by hand, the equivalent trainer arguments are::

      -u train_dataloader.dataset.ann_file=<path>/train.json val_dataloader.dataset.ann_file=<path>/val_clean.json

* **Output goes to ``output/train_v27/run/``** (``--output-dir`` to the trainer) instead of the
  DEIMv2 folder's ``outputs/deimv2_fish_x_v27``, and the launcher refuses to start if that
  folder already holds checkpoints. A reproduction can therefore never overwrite the published
  checkpoints or the published receipt.
* **The settings file is read from this repository** (``configs/``) rather than from a copy
  inside the DEIMv2 folder. The two are byte-identical; DEIMv2 resolves the files a settings
  file includes relative to that file, and ``configs/`` has the layout it expects.
* **Scoring reads the fast footage mirror first and the complete archive second**
  (``config.FOOTAGE_ROOT_FAST``, then ``config.FOOTAGE_ROOT``) in one pass. The original first
  pass read only the mirror, which lacked 8 test clips, and needed a second pass with both
  (``runners/original/run_v27_eval_topup.ps1``).
* **No phone notifications** are sent.

Known limitation: the picture lists name images by absolute Windows path
-------------------------------------------------------------------------
Every picture in the published lists is recorded with the absolute path it had on the original
workstation (``C:\\ssl_cache\\...``, ``G:\\Projects\\...``), and the settings file sets the image
folder to ``/``. Relocating the LIST with ``SALMONID_COCO_CACHE`` therefore does not relocate
the IMAGES. To train elsewhere, the images must be available under the same paths, or the
``file_name`` entries must be rewritten, which changes the list's fingerprint (pass
``--allow-data-mismatch`` and say so when reporting). The pre-flight check below reports this
case in one sentence instead of failing hours into the run.

Will I get the same model?
--------------------------
Not bit-for-bit. The seed fixes the random choices, but the GPU library is not run in its
deterministic mode and mixed-precision arithmetic is used, as in the published run. Two runs
from the same inputs give models that differ slightly. The published weights are the file whose
fingerprint is in ``provenance/v2.7/CHECKSUMS.md``.

Which checkpoint is the published one, and why
----------------------------------------------
Checkpoint 0017. The reasons, with the scores of all seven candidates, are in
``provenance/v2.7/CHECKPOINT_SELECTION.md``. Note that the choice was made using the held-out
test set itself, which the manuscript discloses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from salmonid import config

# ---- The pinned recipe of the published run (do not change: these define "v2.7") ----------
SEED = 0                                   # train.py --seed, as in the published run
RUN_NAME = "deimv2_fish_x_v27"
# Checkpoints scored for the published run. DEIMv2 saves one every 2 epochs up to epoch 25
# and afterwards keeps only the best-by-validation one (best_stg2), so these seven are all the
# candidates that exist from epoch 13 onward.
EVAL_CKPTS = ["0013", "0015", "0017", "0019", "0021", "0023", "best_stg2"]
NOTE = ("DEIMv2-X v2.7 leak-clean publish candidate: v2.5 recipe + v2.6 data with the holdout "
        "re-applied at VIDEO level; init from deimv2_x_coco.pth (no v2.x warm start)")

# The composite batches folded into the training list. Their records say which real fish each
# composite was cut from; the merged list does not keep that, so the gate needs these files.
SYNTH_SETS = ["sam2_synth_partial", "sam2_synth_proximity", "sam2_synth_v26_atlrbw",
              "sam2_synth_v26_cohochin", "sam2_synth_v26_cohorbw", "sam2_synth_v26_partial"]

TRAIN_JSON = config.COCO_V27 / "train.json"
VAL_JSON = config.COCO_V27 / "val_clean.json"
PUBLISHED_MANIFEST = config.PROVENANCE_DIR / "v2.7" / "run_manifest.json"
TESTSET_VIDEOS = config.LABEL_BATCHES / "test_set_v6_videos.json"   # the 364 held-out clips
# The interpreter of the DEIMv2 environment (PyTorch + CUDA). config.DEIM_PYTHON is empty
# unless SALMONID_DEIM_PYTHON is set; empty means "the interpreter running this launcher".
DEIM_PY = Path(config.DEIM_PYTHON) if config.DEIM_PYTHON else Path(sys.executable)

# Exit codes, as in the original launch script.
EXIT_TRAIN_FAILED, EXIT_MISSING_INPUT, EXIT_LEAK = 1, 2, 3


class Launcher:
    """Holds the run's options and the two things every step needs: say() and run()."""

    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.dry = a.dry_run
        self.work = Path(a.work_dir)
        self.run_dir = self.work / "run"
        self.eval_dir = self.work / "eval"
        self.log_path = self.work / "launch.log"
        self.failures: list[str] = []        # dry-run only: guards that would have aborted
        # Child processes: find this package, print UTF-8, and use the GPU allocator setting of
        # the published run. PYTHONUTF8 matters on Windows: DEIMv2 opens the settings file
        # without naming an encoding, and the file contains non-ASCII characters in comments.
        self.env = dict(os.environ)
        self.env["PYTHONPATH"] = os.pathsep.join(
            [str(config.REPO_ROOT)] + ([self.env["PYTHONPATH"]] if self.env.get("PYTHONPATH") else []))
        self.env["PYTHONIOENCODING"] = "utf-8"
        self.env["PYTHONUTF8"] = "1"
        self.env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ---- output ----------------------------------------------------------------------
    def say(self, msg: str = "") -> None:
        """Print a line and, in a real run, append it to launch.log (a dry run writes nothing)."""
        print(msg, flush=True)
        if not self.dry:
            self.work.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def stamp(self, msg: str) -> None:
        self.say(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

    @staticmethod
    def show(cmd: list) -> str:
        """A command as one copy-pasteable line (arguments containing spaces are quoted)."""
        return " ".join(f'"{c}"' if (" " in str(c) or not str(c)) else str(c) for c in cmd)

    def run(self, cmd: list, cwd: Path | None = None, read_only: bool = False,
            capture_to: Path | None = None) -> int:
        """Run one step, echoing its output to the screen and the log. Returns its exit code.

        In a dry run the command is printed, and executed only if ``read_only`` (the leak
        gate reads files and writes nothing, so a dry run can show its real result).
        """
        cmd = [str(c) for c in cmd]
        self.say(f"  $ {self.show(cmd)}" + (f"      (in {cwd})" if cwd else ""))
        if self.dry and not read_only:
            self.say("    [dry run: not executed]")
            return 0
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        captured = []
        for line in proc.stdout:
            self.say("    " + line.rstrip("\n"))
            captured.append(line)
        code = proc.wait()
        if capture_to is not None and not self.dry:
            with open(capture_to, "a", encoding="utf-8") as f:
                f.writelines(captured)
        return code

    # ---- guards ----------------------------------------------------------------------
    def guard(self, name: str, ok: bool, detail: str, exit_code: int, fatal: bool = True) -> bool:
        """Report one check. A failed fatal check stops a real run; a dry run lists it and goes on."""
        status = "OK  " if ok else ("FAIL" if fatal else "WARN")
        self.say(f"  [{status}] {name}: {detail}")
        if not ok and fatal:
            if self.dry:
                self.failures.append(f"{name} (a real run would stop here, exit {exit_code})")
            else:
                self.stamp(f"ABORT: {name}")
                raise SystemExit(exit_code)
        return ok

    def train_cmd(self) -> list:
        """The exact trainer command. Kept in one place so the receipt records what is run."""
        return [DEIM_PY, "-u", "train.py",
                "-c", config.TRAIN_CONFIG_V27.as_posix(),
                "-t", config.DEIM_INIT_COCO.as_posix(),      # start from public COCO weights
                "-d", self.a.device, "--use-amp", "--seed", SEED,
                "--output-dir", self.run_dir.as_posix(),
                # -u must come last (it takes any number of values). These two overrides make
                # the trainer read the SAME list files that were checked and fingerprinted.
                "-u", f"train_dataloader.dataset.ann_file={TRAIN_JSON.as_posix()}",
                f"val_dataloader.dataset.ann_file={VAL_JSON.as_posix()}"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _probe_torch(python: Path) -> str:
    """'ok: torch <version>, <GPU name>' if that interpreter can train, else what is wrong."""
    code = ("import torch;"
            "print('ok: torch', torch.__version__ + ',', torch.cuda.get_device_name(0)) "
            "if torch.cuda.is_available() else print('PyTorch', torch.__version__, 'sees no CUDA GPU')")
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=180,
                             encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - turned into a reported, fatal guard by the caller
        return f"could not start it: {exc}"
    lines = (out.stdout.strip() or out.stderr.strip() or "no output").splitlines()
    return lines[-1] if out.returncode == 0 else f"PyTorch not usable: {lines[-1]}"


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - reported by the caller as a warning, never hidden
        return f"<could not read: {exc}>"


def step1_inputs(L: Launcher) -> None:
    """Step 1 (+ the added pre-flight checks): are the inputs present and the published ones?"""
    L.say("\n=== Step 1: inputs ===")
    L.guard("training list exists", TRAIN_JSON.is_file(), str(TRAIN_JSON), EXIT_MISSING_INPUT)
    L.guard("validation list exists", VAL_JSON.is_file(), str(VAL_JSON), EXIT_MISSING_INPUT)

    # Are these the PUBLISHED lists? Compare fingerprints with the committed receipt.
    published = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))["data"] \
        if PUBLISHED_MANIFEST.is_file() else None
    L.guard("published receipt present", published is not None, str(PUBLISHED_MANIFEST),
            EXIT_MISSING_INPUT)
    if published:
        for label, path, key in (("training", TRAIN_JSON, "train"), ("validation", VAL_JSON, "val")):
            if not path.is_file():
                continue
            got, want = _sha256(path), published[key]["sha256"]
            ok = got == want
            L.guard(f"{label} list is the published one", ok or L.a.allow_data_mismatch,
                    f"sha256 {got[:16]}… " + ("matches the published receipt" if ok else
                                              f"DIFFERS from the published {want[:16]}…"
                                              + (" (allowed by --allow-data-mismatch)"
                                                 if L.a.allow_data_mismatch else "")),
                    EXIT_MISSING_INPUT)

    # Do the image files the list names exist? (They are absolute paths; see module docstring.)
    if TRAIN_JSON.is_file():
        imgs = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))["images"]
        sample = imgs[::max(1, len(imgs) // 200)]
        # The settings file sets img_folder to "/", and the trainer joins that with file_name.
        missing = [im["file_name"] for im in sample if not os.path.isfile(os.path.join("/", im["file_name"]))]
        L.guard("image files exist (sample)", not missing,
                f"{len(sample) - len(missing)} of {len(sample)} sampled images found"
                + (f"; first missing: {missing[0]}  (the list names images by the original "
                   f"workstation's absolute paths; see this module's docstring)" if missing else ""),
                EXIT_MISSING_INPUT)

    # The external trainer, its patch, its weights and its interpreter.
    L.guard("DEIMv2 trainer present", (config.DEIM_DIR / "train.py").is_file(),
            str(config.DEIM_DIR / "train.py") + "  (SALMONID_DEIM_DIR; docs/DEIMV2_PIN.md)", EXIT_MISSING_INPUT)
    tf = config.DEIM_DIR / "engine" / "data" / "transforms" / "_transforms.py"
    patched = tf.is_file() and "ColorJitter = register()" in tf.read_text(encoding="utf-8", errors="replace")
    L.guard("DEIMv2 patch applied", patched,
            "ColorJitter is registered" if patched else
            "ColorJitter is NOT registered: apply docs/deimv2_worktree.patch (the settings file "
            "uses it, so the trainer would stop at start-up)", EXIT_MISSING_INPUT)
    if published is not None:
        want = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))["repos"]["deimv2"]["sha"]
        got = _git_head(config.DEIM_DIR)
        L.guard("DEIMv2 at the pinned commit", got == want,
                f"{got[:12]} " + ("= published run" if got == want else f"(published run used {want[:12]})"),
                0, fatal=False)
    L.guard("starting weights present", config.DEIM_INIT_COCO.is_file(), str(config.DEIM_INIT_COCO),
            EXIT_MISSING_INPUT)
    backbone = sorted(config.DEIM_CKPTS.glob("dinov3_vits16plus_pretrain_*.pth"))
    L.guard("backbone weights present", bool(backbone),
            str(backbone[0]) if backbone else f"no dinov3_vits16plus_pretrain_*.pth in {config.DEIM_CKPTS}",
            EXIT_MISSING_INPUT)
    L.guard("training interpreter present", DEIM_PY.is_file(),
            str(DEIM_PY) + ("  (SALMONID_DEIM_PYTHON)" if config.DEIM_PYTHON else
                            "  (SALMONID_DEIM_PYTHON is not set, so the interpreter running this "
                            "launcher is used)"), EXIT_MISSING_INPUT)
    if DEIM_PY.is_file():
        # Ask that interpreter whether it can train at all: PyTorch importable and a GPU visible.
        # Without this a wrong interpreter fails only after the receipt has been written.
        probe = _probe_torch(DEIM_PY)
        L.guard("training interpreter has PyTorch with CUDA", probe.startswith("ok"), probe,
                EXIT_MISSING_INPUT)
        if published is not None and probe.startswith("ok"):
            want = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))["env"].get("torch", "?")
            same = f"torch {want}," in probe
            L.guard("same PyTorch version as the published run", same,
                    f"torch {want}" if same else
                    f"published run used torch {want}; a different version still trains but is "
                    f"one more reason the result will not be identical", 0, fatal=False)
    L.guard("settings file present", config.TRAIN_CONFIG_V27.is_file(), str(config.TRAIN_CONFIG_V27),
            EXIT_MISSING_INPUT)

    # Never write over an existing run (in particular: never over the published one).
    existing = sorted(L.run_dir.glob("*.pth")) if L.run_dir.is_dir() else []
    L.guard("output folder holds no earlier run", not existing,
            str(L.run_dir) + (f" already contains {len(existing)} checkpoint file(s); choose another "
                              f"--work-dir or use --eval-only" if existing else " is free"),
            EXIT_MISSING_INPUT)


def step2_leak_gate(L: Launcher) -> None:
    """Step 2: no held-out fish in the training or validation pictures. THE precondition."""
    L.say("\n=== Step 2: leak gate (held-out test set; TRACK / VIDEO / SYNTH / same-fish-box) ===")
    synth = [config.PROJECTS_ROOT / s / "synth_coco.json" for s in SYNTH_SETS]
    absent = [p for p in synth if not p.is_file()]
    if absent and not L.a.no_synth_provenance:
        L.guard("composite records present", False,
                f"{len(absent)} of {len(synth)} synth_coco.json files not found (first: {absent[0]}). "
                "Without them the gate cannot check which real fish the composites were cut from. "
                "Pass --no-synth-provenance to run the gate without that level.", EXIT_MISSING_INPUT)
    use_synth = [] if (absent or L.a.no_synth_provenance) else synth
    if not use_synth:
        L.say("  WARNING: the SYNTH level of the gate is NOT being checked on this run.")

    if not (TRAIN_JSON.is_file() and VAL_JSON.is_file()):
        # Only reachable in a dry run (a real run stopped at step 1).
        L.say("  the picture lists are missing, so the gate cannot be run.")
        return
    audit = [sys.executable, "-u", "-m", "salmonid.data.audit_coco_holdout_leak"]
    cmd = audit + ["--coco", TRAIN_JSON, "--deep", "--strict"]
    if use_synth:
        cmd += ["--synth-coco", *use_synth]
    code = L.run(cmd, read_only=True)      # read-only, so a dry run shows the real verdict
    L.guard("training list is leak-free", code == 0,
            "gate passed" if code == 0 else f"gate exited {code}: refusing to train", EXIT_LEAK)
    code = L.run(audit + ["--coco", VAL_JSON, "--deep", "--strict"], read_only=True)
    L.guard("validation list is leak-free", code == 0,
            "gate passed" if code == 0 else f"gate exited {code}", EXIT_LEAK)


def step2b_verify_derivation(L: Launcher) -> None:
    """Optional: rebuild the published lists from the v2.6 lists and compare fingerprints.

    The published v2.7 lists are the v2.6 lists with every picture from a held-out clip
    removed by the audit's ``--out`` option. Repeating that and getting byte-identical files
    shows that the deposited audit is the cleaning step that produced the published data.
    Written to a temporary folder that is deleted afterwards.
    """
    L.say("\n=== Step 2b (optional): re-derive the published lists from the v2.6 lists ===")
    src_train, src_val = config.COCO_V26 / "train.json", config.COCO_V26 / "val_clean.json"
    synth = [config.PROJECTS_ROOT / s / "synth_coco.json" for s in SYNTH_SETS]
    need = [src_train, src_val, *synth]
    if any(not p.is_file() for p in need):
        L.guard("v2.6 lists and composite records present", False,
                "missing: " + ", ".join(str(p) for p in need if not p.is_file()), 0, fatal=False)
        return
    audit = [sys.executable, "-u", "-m", "salmonid.data.audit_coco_holdout_leak"]
    with tempfile.TemporaryDirectory(prefix="salmonid_derive_") as tmp:
        out_train, out_val = Path(tmp) / "train.json", Path(tmp) / "val_clean.json"
        L.run(audit + ["--coco", src_train, "--deep", "--synth-coco", *synth, "--out", out_train], read_only=True)
        L.run(audit + ["--coco", src_val, "--deep", "--out", out_val], read_only=True)
        for label, made, published in (("training", out_train, TRAIN_JSON), ("validation", out_val, VAL_JSON)):
            same = made.is_file() and published.is_file() and _sha256(made) == _sha256(published)
            L.guard(f"re-derived {label} list is byte-identical to the published one", same,
                    "sha256 equal" if same else "sha256 DIFFERS (or a file is missing)", 0, fatal=False)


def step3_manifest(L: Launcher) -> None:
    """Step 3: write the receipt BEFORE training (so it cannot be adjusted to fit the result)."""
    L.say("\n=== Step 3: run receipt (code + settings + data fingerprints + seed + versions) ===")
    cmd = [sys.executable, "-m", "salmonid.provenance.record_run_manifest",
           "--output-dir", L.run_dir,
           "--config", config.TRAIN_CONFIG_V27,
           "--deimv2-dir", config.DEIM_DIR,
           "--repo-dir", config.REPO_ROOT,
           "--train-json", TRAIN_JSON,
           "--val-json", VAL_JSON,
           "--train-python", DEIM_PY,
           "--seed", SEED,
           "--note", NOTE + " [reproduction via salmonid.train.train_v27]",
           "--command", Launcher.show([str(c) for c in L.train_cmd()[2:]])]
    code = L.run(cmd)
    # The original script did not test this step's exit status. A run with no receipt cannot
    # be traced afterwards, so here a failure stops the launch.
    if not L.dry:
        L.guard("receipt written", code == 0 and (L.run_dir / "run_manifest.json").is_file(),
                str(L.run_dir / "run_manifest.json"), EXIT_MISSING_INPUT)


def step4_train(L: Launcher) -> None:
    """Step 4: the training run itself (DEIMv2 train.py, seed pinned, public starting weights)."""
    L.say("\n=== Step 4: train ===")
    L.stamp(f"DEIMv2-X v2.7 begin (seed={SEED}, leak-clean lists, init from COCO weights, 30 epochs)")
    # train.py must be started from inside the DEIMv2 folder: the settings chain names the
    # backbone weights by a path relative to it (./ckpts/...).
    code = L.run(L.train_cmd(), cwd=config.DEIM_DIR)
    if code != 0:
        L.stamp(f"DEIMv2-X v2.7 training FAILED (exit {code})")
        raise SystemExit(EXIT_TRAIN_FAILED)
    L.stamp(f"DEIMv2-X v2.7 training done -> {L.run_dir}" + ("  [dry run: nothing was trained]" if L.dry else ""))


def step5_eval(L: Launcher) -> None:
    """Step 5 (optional): score each candidate checkpoint on the held-out test set."""
    L.say("\n=== Step 5: held-out scoring of the candidate checkpoints ===")
    L.guard("held-out clip list present", TESTSET_VIDEOS.is_file(), str(TESTSET_VIDEOS), EXIT_MISSING_INPUT)
    L.guard("footage archive present", config.FOOTAGE_ROOT.is_dir(),
            str(config.FOOTAGE_ROOT) + "  (SALMONID_FOOTAGE_ROOT)", EXIT_MISSING_INPUT)
    # Fast mirror first (if there is one), complete archive second. The archive is what makes
    # all 364 clips findable: the published run's first pass used the mirror alone and missed 8.
    roots = [r for r in dict.fromkeys([config.FOOTAGE_ROOT_FAST, config.FOOTAGE_ROOT]) if r.is_dir()] \
        or [config.FOOTAGE_ROOT]
    results = L.eval_dir / "results.txt"
    if not L.dry:
        L.eval_dir.mkdir(parents=True, exist_ok=True)
        results.write_text("v2.7 reproduction (salmonid.train.train_v27) -- held-out test set, 366 fish\n"
                           "published checkpoint 0017: macro-F1 0.961 "
                           "(see provenance/v2.7/CHECKPOINT_SELECTION.md)\n", encoding="utf-8")
    for tag in L.a.eval_ckpts:
        ckpt = L.run_dir / (f"checkpoint{tag}.pth" if tag.isdigit() else f"{tag}.pth")
        L.stamp(f"--- eval {tag} ---")
        if not ckpt.is_file() and not L.dry:
            L.say(f"  eval SKIP {tag} (no such checkpoint: {ckpt})")
            continue
        store = L.eval_dir / f"store_{tag}"
        preds, vpreds = L.eval_dir / f"preds_{tag}.csv", L.eval_dir / f"videopreds_{tag}.csv"
        # (a) Detect on every frame of the held-out clips. --mask-header-px 40 blacks out the
        #     date/river overlay exactly as in training. --key-mode rel is the naming scheme the
        #     next step expects. .done marks a finished store so a restart does not redo it.
        if L.dry or not (store / ".done").is_file():
            code = L.run([DEIM_PY, "-u", "-m", "salmonid.detect.detect_store",
                          "--config", config.TRAIN_CONFIG_V27, "--ckpt", ckpt,
                          "--op-json", TESTSET_VIDEOS, "--key-mode", "rel", "--thresh", "0.3",
                          "--mask-header-px", config.HEADER_MASK_PX, "--full-dist",
                          "--footage-root", *roots, "--out-dir", store])
            if code != 0:
                L.say(f"  eval {tag} FAIL detect (exit {code})")
                continue
            if not L.dry:
                (store / ".done").touch()
        # (b) Match detections to the 366 test fish and take each fish's predicted species.
        code = L.run([sys.executable, "-u", "-m", "salmonid.eval.predict_testset_v6_from_store",
                      "--store-dir", store, "--gt-file", config.TESTSET_V6_GT,
                      "--out", preds, "--video-out", vpreds])
        if code != 0:
            L.say(f"  eval {tag} FAIL predict (exit {code})")
            continue
        # (c) The species score (per-species precision / recall / F1 and their unweighted mean).
        if not L.dry:
            with open(results, "a", encoding="utf-8") as f:
                f.write(f"===================== {tag} =====================\n")
        L.run([sys.executable, "-u", "-m", "salmonid.eval.score_testset_v6",
               "--gt-file", config.TESTSET_V6_GT, "--preds", preds, "--video-preds", vpreds],
              capture_to=results)
        # (d) Box-placement check against hand-drawn boxes (recorded for every candidate).
        L.run([DEIM_PY, "-u", "-m", "salmonid.eval.deimv2_bbox_gate",
               "--config", config.TRAIN_CONFIG_V27, "--ckpt", ckpt,
               "--mask-header-px", config.HEADER_MASK_PX,
               "--overlay-dir", L.eval_dir / f"bbox_gate_{tag}"], capture_to=results)
    L.stamp(f"v2.7 eval done -> {results}" + ("  [dry run: nothing was scored]" if L.dry else ""))


def main() -> None:
    a = _parse_args()
    L = Launcher(a)
    L.stamp("salmonid.train.train_v27" + ("  [DRY RUN: checks are real, nothing is trained or written]"
                                          if L.dry else ""))
    L.say(f"  model            : {config.MODEL_VERSION} (seed {SEED})")
    L.say(f"  settings file    : {config.TRAIN_CONFIG_V27}")
    L.say(f"  training list    : {TRAIN_JSON}")
    L.say(f"  validation list  : {VAL_JSON}")
    L.say(f"  starting weights : {config.DEIM_INIT_COCO}")
    L.say(f"  DEIMv2 folder    : {config.DEIM_DIR}")
    L.say(f"  DEIMv2 python    : {DEIM_PY}")
    L.say(f"  work folder      : {L.work}")

    if not a.eval_only:
        step1_inputs(L)
        step2_leak_gate(L)
        if a.verify_derivation:
            step2b_verify_derivation(L)
        step3_manifest(L)
        step4_train(L)
    if a.eval or a.eval_only:
        step5_eval(L)
    else:
        L.say("\nheld-out scoring skipped (add --eval, or run again with --eval-only)")

    if L.dry:
        L.say("\n=== Dry run summary ===")
        if L.failures:
            L.say(f"  {len(L.failures)} check(s) would stop a real run:")
            for f in L.failures:
                L.say(f"    - {f}")
            raise SystemExit(1)
        else:
            L.say("  every check passed; a real run would execute the commands shown above.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the published v2.7 model: checks -> leak gate -> receipt -> train -> (score).")
    p.add_argument("--dry-run", action="store_true",
                   help="Run the read-only checks (including the leak gate), print every command "
                        "that would be executed, and train / write nothing.")
    p.add_argument("--eval", action="store_true", help="After training, score the candidate checkpoints.")
    p.add_argument("--eval-only", action="store_true", help="Skip training; score checkpoints already in run/.")
    p.add_argument("--eval-ckpts", nargs="+", default=EVAL_CKPTS, metavar="TAG",
                   help=f"Checkpoints to score (default: {' '.join(EVAL_CKPTS)}).")
    p.add_argument("--work-dir", default=str(config.OUTPUT_DIR / "train_v27"),
                   help="Where the run, its receipt, its log and its scores are written.")
    p.add_argument("--device", default="cuda", help="Passed to the trainer (default: cuda).")
    p.add_argument("--verify-derivation", action="store_true",
                   help="Also rebuild the published lists from the v2.6 lists and compare sha256.")
    p.add_argument("--no-synth-provenance", action="store_true",
                   help="Run the leak gate without the composite records (SYNTH level not checked).")
    p.add_argument("--allow-data-mismatch", action="store_true",
                   help="Continue although a picture list's sha256 differs from the published receipt.")
    return p.parse_args()


if __name__ == "__main__":
    main()

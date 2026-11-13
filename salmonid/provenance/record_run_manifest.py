"""In plain terms: this script writes a "receipt" for one training run of the fish-identifying
model. Before training starts it records which version of the code was used, which settings
file, which exact list of training pictures (by a fingerprint of the file, so a changed file
cannot pass for the original), the random seed, the software versions and the command. Anyone
can later compare their own copy of the training pictures against the receipt and know whether
they hold the same data the published model was trained on. It computes no number in the paper;
it is the record that lets the published model be traced to its ingredients.

Inputs
------
All given on the command line by the launcher (nothing is read from ``salmonid.config``):
the run output folder, the top settings file (DEIMv2 ``.yml``), the DEIMv2 folder and this
repository's folder, the training and validation picture lists, the seed, and the command.
``--train-python`` optionally names the interpreter used for training, so its PyTorch / CUDA /
GPU versions can be read back.

Outputs
-------
Written into ``--output-dir``:

* ``run_manifest.json``: the receipt (git commits of both repositories and whether they had
  uncommitted changes; sha256, size, picture count, box count and per-species box counts of the
  training and validation lists; seed; software versions; command).
* ``config_snapshot/*.yml``: a copy of the settings file and every file it includes.
* ``deimv2_worktree.patch``: the uncommitted changes in the DEIMv2 folder (empty if none).

Feeds
-----
The reproducibility statement for the published v2.7 model. The receipt of the actual v2.7
run, written by the research-repository version of this script on 2026-07-24, is committed
under ``provenance/v2.7/`` and explained in ``provenance/v2.7/README.md``.

Run
---
Normally called by ``python -m salmonid.train.train_v27``. On its own::

    python -m salmonid.provenance.record_run_manifest
        --output-dir <run output dir> --config <configs/deimv2/deimv2_fish_x_v27.yml>
        --deimv2-dir <DEIMv2 clone> --repo-dir <this repository>
        --train-json <COCO_V27>/train.json --val-json <COCO_V27>/val_clean.json
        --seed 0 --command "<the train.py command line>"

(one command; shown on several lines)

Tier
----
2 (it hashes the external training lists). It uses only the Python standard library and ``git``.

Why a receipt is needed
-----------------------
The training recipe is split across two repositories: this one (data builders, launcher) and
the external DEIMv2 detector code, which is used with a small local patch. A published model is
only traceable if both code states, the settings, the data and the seed are pinned together at
the moment training starts.

Two corrections relative to the script that wrote the v2.7 receipt
------------------------------------------------------------------
Neither changes what a receipt means; both fix how it is written. The committed v2.7 receipt
shows the old behaviour and is kept as it was written.

* **First character of the first changed-file name.** ``git status --porcelain`` prints two
  status characters and a space before each file name, and the first status character is often
  a space. The old code stripped leading whitespace from the whole output and then cut three
  characters from every line, so the first line lost one character of its file name. That is
  why the v2.7 receipt lists ``LAUDE.md`` and ``ngine/data/transforms/_transforms.py``; the real
  names are ``CLAUDE.md`` and ``engine/data/transforms/_transforms.py``.
* **The saved patch could not be applied.** It was written without its final newline and, on
  Windows, with CRLF line endings, and ``git apply`` rejects it ("corrupt patch"). It is now
  written with LF endings and its final newline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> str:
    """Run ``cmd`` and return its stripped stdout.

    Args:
        cmd: argv list to execute.
        cwd: working directory, or None for the current one.

    Returns:
        The command's stdout with TRAILING whitespace removed, or an ``<error: ...>``
        sentinel string on any failure. Provenance recording must never crash the
        training launch, so every exception (missing git, timeout, etc.) is turned
        into the return value rather than raised; the sentinel then appears in the
        manifest, so the failure is visible in the receipt.
    """
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace")
        # rstrip(), not strip(): leading whitespace is DATA in `git status --porcelain`
        # (" M file" = modified, not staged). strip() ate the first line's leading space and
        # `_git` then cut one character too many from that file name (see module docstring).
        return out.stdout.rstrip()
    except Exception as exc:  # noqa: BLE001 - provenance must never crash the run
        return f"<error: {exc}>"


def _git(repo: str) -> dict:
    """Pin the git state of a repo checkout.

    Args:
        repo: path to a git working tree.

    Returns:
        A dict with the full/short HEAD sha, branch, a ``dirty`` flag, and up to 200
        ``dirty_files`` (porcelain paths). If ``repo`` is not a directory, returns
        ``{"path": repo, "present": False}`` so a missing sibling repo is recorded
        rather than fatal.
    """
    if not Path(repo).is_dir():
        return {"path": repo, "present": False}
    sha = _run(["git", "-C", repo, "rev-parse", "HEAD"])
    short = _run(["git", "-C", repo, "rev-parse", "--short", "HEAD"])
    branch = _run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", repo, "status", "--porcelain"])
    return {
        "path": repo,
        "present": True,
        "sha": sha,
        "short_sha": short,
        "branch": branch,
        "dirty": bool(status.strip()),
        # Each porcelain line is "XY <path>": two status characters, one space, the path.
        "dirty_files": [ln[3:] for ln in status.splitlines()][:200],
    }


def _sha256(path: str) -> str:
    """Return the hex sha256 of a file, read in 1 MiB chunks.

    Streaming the file (rather than reading it whole) keeps the memory footprint
    flat even for the multi-hundred-MB COCO jsons.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1<<20 = 1 MiB blocks
            h.update(chunk)
    return h.hexdigest()


def _coco_stats(path: str) -> dict:
    """Summarise a COCO json for the manifest (content-addressed identity).

    Args:
        path: a COCO detection json (train or val split).

    Returns:
        A dict with the file's sha256 + byte size (its content fingerprint), image
        and annotation counts, the id->name category map, and per-category
        annotation counts (keyed by category name). Two runs whose ``sha256`` and
        ``per_category`` match consumed byte-identical training data. Returns
        ``{"path": path, "present": False}`` if the file is absent.
    """
    if not Path(path).is_file():
        return {"path": path, "present": False}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    per = Counter(a["category_id"] for a in d.get("annotations", []))
    return {
        "path": path,
        "present": True,
        "sha256": _sha256(path),
        "bytes": os.path.getsize(path),
        "images": len(d.get("images", [])),
        "annotations": len(d.get("annotations", [])),
        "categories": cats,
        "per_category": {cats.get(k, k): v for k, v in sorted(per.items())},
    }


def _include_chain(config: str) -> list[str]:
    """Resolve the DEIMv2 __include__ chain (relative to each file's dir)."""
    seen: set[str] = set()
    order: list[str] = []
    # pop(0) => breadth-first from the top config; `seen` de-dups shared includes so
    # a yml pulled in by two parents is snapshotted once, in first-encounter order.
    stack = [os.path.abspath(config)]
    while stack:
        f = stack.pop(0)
        if f in seen or not os.path.isfile(f):
            continue
        seen.add(f)
        order.append(f)
        txt = Path(f).read_text(encoding="utf-8")
        m = re.search(r"__include__:\s*\[([^\]]*)\]", txt)
        if m:
            for inc in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)):
                stack.append(os.path.abspath(os.path.join(os.path.dirname(f), inc)))
    return order


def _env(train_python: str | None) -> dict:
    """Capture interpreter/toolchain versions for the manifest.

    Always records the recorder's own Python version. If ``train_python`` points at
    a real interpreter (the ``.venv_deim`` used for training), it is launched in a
    subprocess to read back its Python / torch / torchvision / CUDA / GPU-name — the
    recorder's own env deliberately lacks torch, so the versions must be probed from
    the training env itself rather than imported here.

    Args:
        train_python: path to the training interpreter, or None to skip the probe.

    Returns:
        A dict of version strings; on probe failure the raw output is stored under
        ``train_python_probe`` instead of the parsed fields.
    """
    env = {"recorder_python": sys.version.split()[0]}
    if train_python and Path(train_python).is_file():
        probe = (
            "import torch,torchvision,sys,platform;"
            "g=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None;"
            "print(sys.version.split()[0]);print(torch.__version__);"
            "print(torchvision.__version__);print(torch.version.cuda);print(g)"
        )
        out = _run([train_python, "-c", probe])
        lines = out.splitlines()
        if len(lines) >= 5 and not out.startswith("<error"):
            env.update({
                "train_python": lines[0], "torch": lines[1], "torchvision": lines[2],
                "cuda": lines[3], "gpu": lines[4],
            })
        else:
            env["train_python_probe"] = out
    return env


def main() -> None:
    """CLI entry point: assemble and write the run manifest.

    Parses the run's paths/seed/command from argv, then writes three artifacts into
    ``--output-dir``: a verbatim ``config_snapshot/`` of the whole __include__ chain,
    the DEIMv2 working-tree ``deimv2_worktree.patch``, and ``run_manifest.json``
    tying together the repo SHAs, config, train/val COCO fingerprints, seed, and env.
    Prints a one-line summary to stdout. Side effects: filesystem writes under
    ``--output-dir`` and subprocess calls to ``git`` (and optionally the training
    interpreter).
    """
    p = argparse.ArgumentParser(description="Pin a reproducible run manifest.")
    p.add_argument("--output-dir", required=True, help="Run output dir (manifest written here).")
    p.add_argument("--config", required=True, help="Top DEIMv2 fish yml.")
    p.add_argument("--deimv2-dir", required=True)
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--train-json", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--train-python", default=None, help="Interpreter used for training (for versions/GPU).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--command", default="", help="The exact train.py command line.")
    p.add_argument("--note", default="", help="Free-text one-liner about this run.")
    a = p.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) snapshot the full config include-chain verbatim
    snap = out / "config_snapshot"
    snap.mkdir(exist_ok=True)
    chain = _include_chain(a.config)
    for f in chain:
        shutil.copy2(f, snap / Path(f).name)

    # 2) save the DEIMv2 uncommitted diff (the patch is part of the code state)
    diff = _run(["git", "-C", a.deimv2_dir, "diff", "HEAD"])
    # A patch must end with a newline and use LF line endings or `git apply` rejects it.
    # newline="\n" stops Windows from rewriting the line endings on the way to disk.
    (out / "deimv2_worktree.patch").write_text((diff + "\n") if diff else "",
                                               encoding="utf-8", newline="\n")

    manifest = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "note": a.note,
        "seed": a.seed,
        "command": a.command,
        "determinism": (
            "seed fixed via DEIMv2 --seed; cudnn NON-deterministic (setup_seed "
            "deterministic=False) + AMP, so runs are seed-pinned but not bitwise "
            "reproducible. Report as seeded, not deterministic."
        ),
        "repos": {"fish_cv": _git(a.repo_dir), "deimv2": _git(a.deimv2_dir)},
        "deimv2_worktree_patch": "deimv2_worktree.patch",
        "config": {
            "top": os.path.abspath(a.config),
            "include_chain": [Path(f).name for f in chain],
            "snapshot_dir": "config_snapshot",
        },
        "data": {"train": _coco_stats(a.train_json), "val": _coco_stats(a.val_json)},
        "env": _env(a.train_python),
    }

    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tr = manifest["data"]["train"]
    print(f"[manifest] wrote {out/'run_manifest.json'}")
    print(f"[manifest] fish_cv {manifest['repos']['fish_cv'].get('short_sha')} "
          f"(dirty={manifest['repos']['fish_cv'].get('dirty')}) | "
          f"deimv2 {manifest['repos']['deimv2'].get('short_sha')} "
          f"(dirty={manifest['repos']['deimv2'].get('dirty')})")
    if tr.get("present"):
        print(f"[manifest] train: {tr['images']} imgs / {tr['annotations']} anns  {tr['per_category']}")
        print(f"[manifest] train sha256 {tr['sha256'][:16]}…  seed={a.seed}")


if __name__ == "__main__":
    main()

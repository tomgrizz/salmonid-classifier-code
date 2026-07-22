"""
In plain terms: this file writes a detailed "receipt" for one training session of
the fish-identifying model — it records exactly which version of the code, which
settings, and which set of labelled fish photos were fed in. That receipt lets
anyone later re-run the training and trace the published model back to its exact
ingredients, which is what backs up the paper's claim that its results (the
0.966-accuracy single-stage model) can be reproduced. It does not itself compute any
number in the paper; it is the record-keeping step that makes the published number
verifiable.

Record a complete, reproducible provenance manifest for a single-stage DEIMv2
training run. Call this at launch time (from the runner) BEFORE train.py.

Pipeline position: this is a *provenance sidecar* to the DETECT / TRAIN stage of
the pipeline (data -> [detect/train] -> track -> eval/count -> results -> figures).
It touches no fish data itself; it snapshots the exact code + config + data state
that a training run consumes, so the resulting weights (and therefore every
downstream count, score, and figure) are traceable to a fixed input set. It is what
underwrites the manuscript's central reproducibility claim for the published
single-stage model (DEIMv2-X v2.6, held-out macro-F1 0.966).

Why: the single-stage experiment definition is split across two repos — this one
(scripts, runners, COCO builders) and the external DEIMv2 clone (train.py + the
fish configs), which is untracked and carries a working-tree patch. A published
number is only reproducible if we pin, per run: both repo SHAs, the DEIMv2
uncommitted diff, the resolved config set, the train/val COCO content hashes +
category counts, the seed, the env versions, and the exact command.

Environment: the recorder itself runs under the base Python (only stdlib +
``git``); it optionally shells out to the *training* interpreter (the isolated
``.venv_deim``) via ``--train-python`` purely to read back its torch/CUDA/GPU
versions. No path literals: every location is supplied as a CLI argument by the
runner, so this module needs nothing from ``salmonid.config``.

Inputs (all CLI args): the run output dir, the top DEIMv2 config yml, the DEIMv2
and fish-cv repo dirs, the train/val COCO json paths, the seed, and the command.
Outputs — writes into <output-dir>:
  run_manifest.json          — the machine-readable pin (everything below)
  config_snapshot/*.yml      — the full __include__ chain, copied verbatim
  deimv2_worktree.patch      — `git diff` of the DEIMv2 clone (empty if clean)

Usage (from the runner, base Python is fine):
  python scripts/_record_run_manifest.py \
      --output-dir  C:/Users/.../DEIMv2/outputs/deimv2_fish_x_v25 \
      --config      C:/Users/.../DEIMv2/configs/deimv2/deimv2_fish_x_v25.yml \
      --deimv2-dir  C:/Users/.../DEIMv2 \
      --repo-dir    C:/Users/.../lake-ontario-fish-cv \
      --train-json  C:/ssl_cache/deimv2_fish_6class_masked_v25/train.json \
      --val-json    C:/ssl_cache/deimv2_fish_6class_masked_v23/val.json \
      --train-python C:/Users/.../.venv_deim/Scripts/python.exe \
      --seed 0 \
      --command "train.py -c configs/deimv2/deimv2_fish_x_v25.yml -t ckpts/deimv2_x_coco.pth -d cuda --use-amp --seed 0"
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
        The command's stdout (stripped), or an ``<error: ...>`` sentinel string on
        any failure. Provenance recording must never crash the training launch, so
        every exception (missing git, timeout, etc.) is swallowed into the return
        value rather than raised.
    """
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
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
    d = json.loads(Path(path).read_text())
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
    (out / "deimv2_worktree.patch").write_text(diff or "", encoding="utf-8")

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

    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
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

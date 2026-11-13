"""In plain terms: this script runs the whole counting chain for the seasons you name, in the
right order, and stops with a clear message as soon as a step fails. The chain is: (1) the
model looks at every video and saves what it sees; (2) the saved files are checked for
damage; (3) the detections are joined into one track per fish; (4) the tracks are checked for
missing videos; (5) the tracks are counted per day under each counting rule and set beside
the technician counts; (6) the per-season tables are stacked into the two master tables the
paper reads from; (7) the result is compared with the published tables. With ``--dry-run`` it
only prints the exact commands, which is the quickest way to see what was done.

One-command driver for the published (v2.7) counts: detect -> audit -> track -> audit ->
count -> canonical tables -> comparison with the committed data.

Pipeline stage: this module runs the stages; it contains no counting logic of its own.

Inputs : everything the individual steps need, located through ``salmonid.config``:
         video lists (``config.LABEL_BATCHES``), detect-stores and track aggregates
         (``config.corpus_dir(year)``), the technician files, and for steps 1 and 3 the raw
         footage, the model weights and a GPU.
Outputs: under ``--out-dir`` (default ``output/corpus_v27/``):
           net_rule_sweep/<year>_<rule>.csv     8 rules x each season (net counts)
           activity_sweep/<year>_activity.csv   each season (activity counts)
           2020gan_recovered_{net,activity}_botsort.csv   the recovered 2020 Ganaraska footage
           date_maps/<year>.json                the passage date used for every video
           logs/                                one log per launched job
         and ``output/canonical_daily.csv`` + ``output/canonical_summary.csv``.
         Steps 1 and 3 write detect-stores and track aggregates into ``config.corpus_dir(year)``
         on the project data drive, because they are far too large for the repository.
         Nothing is ever written into ``data/``.
Feeds  : Tables 2-5 and Figures 4-7; the rule comparison of Section 3.4.
Run    : python -m salmonid.count.run_corpus                  (steps 4-7, from stored tracks)
         python -m salmonid.count.run_corpus --dry-run        (print the commands only)
         python -m salmonid.count.run_corpus --steps all --years 2024   (from raw video; GPU, hours)
Tier   : 2. Steps 4-7 need the track aggregates (about 370 MB for six seasons), the video
         lists, the technician files, and the passage date of each video. The dates come
         either from the raw footage (file timestamps; about five minutes) or from date maps
         saved by an earlier run (``--date-map-dir``; under two minutes, no footage needed).
         Both routes were checked to reproduce the published tables exactly.
         Steps 1-3 need the raw footage and a GPU: 246,070 videos, several days.

The steps (``--steps``; default: audit count canonical compare)
  detect        Pass A, ``salmonid.detect.detect_store``. Every third frame, half precision,
                top 40 rows masked, detections above 0.30 kept. Runs in the DEIMv2
                environment (``config.DEIM_PYTHON``), two copies side by side.
  audit-stores  ``salmonid.detect.audit_store_integrity --stores-only``: no damaged store
                files. Opens every file (about 40 minutes for six seasons on a hard disk).
  track         Pass B, ``salmonid.detect.track_from_store``: BoT-SORT, one track per fish.
  audit         ``salmonid.detect.audit_store_integrity --aggregates-only``: every listed
                video is present in the track aggregates. Counting does not start otherwise.
  count         ``salmonid.count.corpus_countr_from_store``: per season, the net count under
                each of eight direction rules, and the activity count.
  canonical     ``salmonid.results.build_canonical_results``: the two master tables.
  compare       compares the regenerated files with ``data/`` and reports any difference.

The eight direction rules (upstream threshold / downstream threshold, in pixels). The paper
reports ``up50`` (50 / 300); the other seven exist only for the comparison in Section 3.4 and
the leave-one-year-out check (``salmonid.count.net_rule_loyo``).

Where this comes from: it replaces the PowerShell scripts that drove the original run, which
are kept unchanged in ``runners/original/`` as the record of what was executed. The commands
built here carry the same settings. Two differences from the original run are deliberate:
the detection step is always given both footage folders (fast copy first, archive second),
and all per-season count files are written to ``output/`` instead of the data drive.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from salmonid import config
from salmonid.detect.audit_store_integrity import manifest_for, store_key_vid

ALL_STEPS = ["detect", "audit-stores", "track", "audit", "count", "canonical", "compare"]
DEFAULT_STEPS = ["audit", "count", "canonical", "compare"]

# Rule name -> (pixels needed to count as upstream, pixels needed to count as downstream).
# The names are the file names used in data/net_rule_sweep/.
RULES = {
    "up50":      (50, 300),     # the published rule
    "up150":     (150, 300),
    "up250":     (250, 300),
    "up300":     (300, 300),
    "u50_d150":  (50, 150),
    "u50_d50":   (50, 50),
    "u150_d150": (150, 150),
    "u1_d1":     (1, 1),
}

DRY = False
LOG_DIR = None


def _show(cmd):
    """One command as a single copy-and-paste line."""
    cmd = [str(c) for c in cmd]
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def _env():
    """Environment for the launched steps: make ``salmonid`` importable and output readable."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _fail(msg):
    """Stop the whole run with one message (and a notification if those are switched on)."""
    _notify(f"run_corpus FAILED: {msg}", priority=5)
    raise SystemExit(f"\nSTOPPED: {msg}")


def _notify(message, priority=4):
    """Send a "job finished" notification, only if the user has set ``SALMONID_NTFY_TOPIC``."""
    if DRY or not config.NTFY_TOPIC:
        return
    try:
        from salmonid.util import ntfy
        ntfy.send(message, title="salmonid run_corpus", priority=priority)
    except Exception as e:                                      # noqa: BLE001
        print(f"(notification not sent: {type(e).__name__})")


def run(cmd, log_name=None):
    """Print one command, and unless this is a dry run, execute it and stop on failure."""
    print("  " + _show(cmd), flush=True)
    if DRY:
        return
    if log_name:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / log_name, "w", encoding="utf-8") as lg:
            rc = subprocess.call([str(c) for c in cmd], cwd=config.REPO_ROOT, env=_env(),
                                 stdout=lg, stderr=subprocess.STDOUT)
        tail = (LOG_DIR / log_name).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        for line in tail:
            print("      " + line)
    else:
        rc = subprocess.call([str(c) for c in cmd], cwd=config.REPO_ROOT, env=_env())
    if rc != 0:
        _fail(f"command returned exit code {rc}: {_show(cmd)}"
              + (f"\n  full log: {LOG_DIR / log_name}" if log_name else ""))


def run_parallel(cmds, log_names):
    """Run several commands side by side (the "shards" of one job) and wait for all of them."""
    for c in cmds:
        print("  " + _show(c), flush=True)
    if DRY:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    for c, name in zip(cmds, log_names):
        lg = open(LOG_DIR / name, "w", encoding="utf-8")
        procs.append((subprocess.Popen([str(x) for x in c], cwd=config.REPO_ROOT, env=_env(),
                                       stdout=lg, stderr=subprocess.STDOUT), lg, name))
        time.sleep(4)        # start the copies a few seconds apart so they do not load the model at once
    bad = []
    for p, lg, name in procs:
        rc = p.wait(); lg.close()
        if rc != 0:
            bad.append(f"{name} (exit code {rc})")
    if bad:
        _fail("a parallel job failed: " + ", ".join(bad) + f"\n  logs: {LOG_DIR}")


def _jobs(years, sites, tag, skip_recovered):
    """The (label, video list, store folder, aggregate prefix, footage roots) of every detect/track job."""
    jobs = []
    for year in years:
        base = config.corpus_dir(year, tag)
        for site in sites:
            jobs.append({"label": f"{year}_{site}", "manifest": manifest_for(year, site),
                         "store": base / "store", "agg": base / f"agg_{site}",
                         "detect_roots": [config.FOOTAGE_ROOT_FAST, config.FOOTAGE_ROOT],
                         "track_root": config.FOOTAGE_ROOT_FAST})
        if year == 2020 and not skip_recovered:
            # The recovered 2020 Ganaraska footage lives on its own drive and has its own video
            # list, store and aggregates, apart from the regular 2020 footage.
            jobs.append({"label": "2020_gan_recovered",
                         "manifest": config.LABEL_BATCHES / "corpus_2020_gan_recovered.json",
                         "store": base / "store_gan_recovered", "agg": base / "agg_gan_recovered",
                         "detect_roots": [config.RECOVERED_ROOT], "track_root": config.RECOVERED_ROOT})
    return jobs


def _n_in_store(job):
    """How many of a job's listed videos have a store file."""
    import json
    vids = json.load(open(job["manifest"], encoding="utf-8"))["videos"]
    return sum((job["store"] / store_key_vid(v["video_id"])).is_file() for v in vids), len(vids)


def step_detect(a, jobs):
    """Pass A for every job: N copies side by side, each taking every Nth video."""
    py = a.deim_python or config.DEIM_PYTHON or sys.executable
    for j in jobs:
        print(f"\n[detect] {j['label']}")
        if not DRY:
            config.require(j["manifest"], f"the video list for {j['label']}", "SALMONID_PROJECTS_ROOT")
        cmds, logs = [], []
        for i in range(a.shards):
            cmds.append([py, "-u", "-m", "salmonid.detect.detect_store",
                         "--config", config.INFER_CONFIG, "--ckpt", config.CHECKPOINT_V27,
                         "--op-json", j["manifest"], "--out-dir", j["store"], "--key-mode", "video_id",
                         "--footage-root", *j["detect_roots"],
                         "--thresh", "0.3", "--mask-header-px", "40", "--full-dist", "--fp16",
                         "--frame-batch-size", "32", "--frame-skip", "3",
                         "--shard", i, "--num-shards", a.shards])
            logs.append(f"detect_{j['label']}_shard{i}.log")
        run_parallel(cmds, logs)
        if not DRY:
            # A pass that leaves nothing in the store has failed, whatever its exit code. In the
            # original run a wrong footage folder looked like instant success for exactly this reason.
            have, total = _n_in_store(j)
            print(f"  store holds {have:,} of {total:,} listed videos")
            if have == 0:
                _fail(f"detect step stored nothing for {j['label']} (wrong footage folder?). See {LOG_DIR}")
            if have < total:
                print(f"  WARNING: {total - have:,} listed videos have no store file (footage not found). "
                      "The audit step will report them as missing.")


def step_track(a, jobs):
    """Pass B for every job: one aggregate file per copy."""
    for j in jobs:
        print(f"\n[track] {j['label']}")
        cmds, logs = [], []
        for i in range(a.shards):
            cmds.append([sys.executable, "-u", "-m", "salmonid.detect.track_from_store",
                         "--store-dir", j["store"], "--op-json", j["manifest"], "--key-mode", "video_id",
                         "--footage-root", j["track_root"], "--decode-backend", "decord", "--reid-fp16",
                         "--frame-skip", "3", "--agg-out", f"{j['agg']}_shard{i}.pkl",
                         "--shard", i, "--num-shards", a.shards])
            logs.append(f"track_{j['label']}_shard{i}.log")
        run_parallel(cmds, logs)
        if not DRY:
            # A real tracking pass writes megabytes. A file under 1 KB is an empty result from a
            # job that died at start-up (in the original run: missing appearance-model weights).
            small = [p for p in Path(j["agg"]).parent.glob(f"{Path(j['agg']).name}_shard*.pkl")
                     if p.stat().st_size < 1024]
            if small:
                _fail(f"track step produced empty aggregate(s) for {j['label']}: "
                      + ", ".join(p.name for p in small) + f". See {LOG_DIR}")


def step_audit(a, stores_only):
    """Run the read-only audit; a non-zero result stops the run before anything is counted.

    ``audit-stores`` opens every store file (tens of minutes on a hard disk) and belongs
    between detection and tracking. ``audit`` checks that every listed video has tracks; it
    takes seconds and is the check that protects the counts.
    """
    print(f"\n[{'audit-stores' if stores_only else 'audit'}]")
    cmd = [sys.executable, "-m", "salmonid.detect.audit_store_integrity", "--tag", a.tag,
           "--years", *a.years, "--stores-only" if stores_only else "--aggregates-only"]
    run(cmd)


def _count(a, store, agg_glob, manifest, year, metric, up, down, out_csv, date_map=None, dump=None):
    """One count run, followed by the check that it produced data."""
    cmd = [sys.executable, "-u", "-m", "salmonid.count.corpus_countr_from_store",
           "--store-dir", store, "--agg", agg_glob, "--videos-json", manifest, "--key-mode", "video_id",
           "--year", year, "--metric", metric, "--up-thresh", up, "--down-thresh", down,
           "--tech-dir", config.TECH_2025_DIR, "--out-csv", out_csv]
    if date_map:
        cmd += ["--date-map", date_map]
    if dump:
        cmd += ["--dump-date-map", dump]
    run(cmd, log_name=f"count_{Path(out_csv).stem}.log")
    if not DRY:
        # A table with a header and no rows means the track files were empty or not found.
        with open(out_csv, newline="", encoding="utf-8") as f:
            n = sum(1 for _ in csv.reader(f)) - 1
        if n < 1:
            _fail(f"count step wrote no data rows: {out_csv}")


def step_count(a, out):
    """Per season: the net count under each of the eight rules, then the activity count."""
    for year in a.years:
        print(f"\n[count] {year}")
        base = config.corpus_dir(year, a.tag)
        manifest = manifest_for(year, "both")       # 2025: the corrected "_full" list
        if not DRY:
            config.require(base, f"the {year} season folder (detect-store and track aggregates)",
                           "SALMONID_PROJECTS_ROOT")
            config.require(manifest, f"the {year} video list for both rivers", "SALMONID_PROJECTS_ROOT")
        # A date map (video -> passage date) saved by an earlier run lets the count be redone
        # without the raw footage. Without one, each video's date is read from its file.
        dm = None
        if a.date_map_dir:
            cand = Path(a.date_map_dir) / f"{year}.json"
            dm = cand if (DRY or cand.is_file()) else None
        dump = out / "date_maps" / f"{year}.json"
        first = True
        for rule, (up, down) in RULES.items():
            _count(a, base / "store", base / "agg_*.pkl", manifest, year, "net", up, down,
                   out / "net_rule_sweep" / f"{year}_{rule}.csv", date_map=dm,
                   dump=dump if first else None)
            first = False
        _count(a, base / "store", base / "agg_*.pkl", manifest, year, "activity", 50, 300,
               out / "activity_sweep" / f"{year}_activity.csv", date_map=dm)
        if year == 2020 and not a.skip_recovered:
            # The recovered 2020 Ganaraska footage: its own tracks, dated from the counter log
            # through the date map (its files do not carry a usable modification time).
            print("\n[count] 2020 Ganaraska, recovered footage")
            for metric in ("net", "activity"):
                _count(a, base / "store_gan_recovered", base / "agg_gan_recovered_*.pkl",
                       config.LABEL_BATCHES / "corpus_2020_gan_recovered.json", 2020, metric, 50, 300,
                       out / f"2020gan_recovered_{metric}_botsort.csv",
                       date_map=config.RECOVERED_2020_DATEMAP)


def step_canonical(a, out):
    """Stack the per-season files into canonical_daily.csv and canonical_summary.csv."""
    print("\n[canonical]")
    run([sys.executable, "-m", "salmonid.results.build_canonical_results", "--tag", a.tag,
         "--sweep-root", out, "--recovered-dir", out, "--years", *a.years,
         "--out-dir", a.canonical_out_dir])


def _same(new, old):
    """Compare two CSV files by content (line endings ignored). Returns (identical, description)."""
    if not Path(new).is_file():
        return False, "not regenerated"
    a = Path(new).read_text(encoding="utf-8").replace("\r\n", "\n")
    b = Path(old).read_text(encoding="utf-8").replace("\r\n", "\n")
    if a == b:
        return True, "identical"
    ra, rb = a.strip().split("\n"), b.strip().split("\n")
    only_new = len(set(ra) - set(rb)); only_old = len(set(rb) - set(ra))
    return False, (f"DIFFERENT: {len(ra) - 1} rows regenerated vs {len(rb) - 1} published; "
                   f"{only_new} rows only in regenerated, {only_old} only in published")


def step_compare(a, out):
    """Compare the regenerated files with the published copies in data/ (content, not line endings)."""
    print("\n[compare] regenerated files against the published copies in data/")
    if DRY:
        print("  (dry run: nothing to compare)")
        return True
    results = []
    for year in a.years:
        for rule in RULES:
            old = config.NET_RULE_SWEEP / f"{year}_{rule}.csv"
            if old.is_file():
                results.append((old.name, *_same(out / "net_rule_sweep" / old.name, old)))
    full = sorted(a.years) == sorted(config.CORPUS_YEARS) and not a.skip_recovered
    if full:
        for old in (config.CANONICAL_DAILY, config.CANONICAL_SUMMARY):
            results.append((old.name, *_same(Path(a.canonical_out_dir) / old.name, old)))
    else:
        print("  (canonical tables not compared: the published ones cover all of 2020-2025 "
              "including the recovered 2020 footage)")
    n_ok = sum(1 for _, ok, _ in results if ok)
    for name, ok, why in results:
        if not ok:
            print(f"  {name}: {why}")
    print(f"  {n_ok} of {len(results)} files identical to the published copies")
    return n_ok == len(results)


def main():
    """Parse the options and run the chosen steps in pipeline order."""
    global DRY, LOG_DIR
    ap = argparse.ArgumentParser(
        description="Run the published counting chain (detect, audit, track, audit, count, canonical, compare).")
    ap.add_argument("--years", nargs="+", type=int, default=list(config.CORPUS_YEARS),
                    help="Seasons to process (default: 2020-2025).")
    ap.add_argument("--steps", nargs="+", default=DEFAULT_STEPS, choices=ALL_STEPS + ["all"],
                    help="Steps to run, always in pipeline order. Default: audit count canonical compare "
                         "(everything that works from stored tracks). 'all' adds the GPU steps "
                         "detect, audit-stores and track.")
    ap.add_argument("--sites", nargs="+", default=list(config.CORPUS_SITES), choices=config.CORPUS_SITES,
                    help="Rivers for the detect and track steps (default: both).")
    ap.add_argument("--shards", type=int, default=2,
                    help="Copies run side by side in the detect and track steps (published run: 2; "
                         "3 or more can exhaust a 32 GB GPU).")
    ap.add_argument("--tag", default=config.CORPUS_TAG, help="Model tag in the folder names (default: v27).")
    ap.add_argument("--skip-recovered", action="store_true",
                    help="Leave out the recovered 2020 Ganaraska footage (needs its own drive).")
    ap.add_argument("--out-dir", default=None,
                    help="Folder for the per-season count files (default: output/corpus_<tag>/).")
    ap.add_argument("--canonical-out-dir", default=str(config.OUTPUT_DIR),
                    help="Folder for canonical_daily.csv and canonical_summary.csv (default: output/).")
    ap.add_argument("--date-map-dir", default=None,
                    help="Folder of <year>.json date maps written by an earlier run (see "
                         "<out-dir>/date_maps/). With it the count step does not need the raw footage.")
    ap.add_argument("--deim-python", default="",
                    help="Python of the DEIMv2 environment for the detect step "
                         "(default: config.DEIM_PYTHON, else this Python).")
    ap.add_argument("--dry-run", action="store_true", help="Print every command; run nothing.")
    a = ap.parse_args()

    DRY = a.dry_run
    steps = ALL_STEPS if "all" in a.steps else [s for s in ALL_STEPS if s in a.steps]
    out = Path(a.out_dir) if a.out_dir else config.OUTPUT_DIR / f"corpus_{a.tag}"
    LOG_DIR = out / "logs"
    for d in (out, Path(a.canonical_out_dir)):
        if d.resolve() == config.REPO_DATA.resolve() or config.REPO_DATA.resolve() in d.resolve().parents:
            raise SystemExit("Refusing to write into data/: it holds the published copies. Use output/.")
    print(f"run_corpus: seasons {a.years}; steps {steps}; model tag {a.tag}"
          + ("; DRY RUN (commands are printed, nothing is executed)" if DRY else ""))
    print(f"  per-season count files -> {out}")

    jobs = _jobs(a.years, a.sites, a.tag, a.skip_recovered)
    ok = True
    if "detect" in steps:
        step_detect(a, jobs)
    if "audit-stores" in steps:
        step_audit(a, stores_only=True)
    if "track" in steps:
        step_track(a, jobs)
    if "audit" in steps:
        step_audit(a, stores_only=False)
    if "count" in steps:
        step_count(a, out)
    if "canonical" in steps:
        step_canonical(a, out)
    if "compare" in steps:
        ok = step_compare(a, out)
    print("\nrun_corpus: finished" + ("" if ok else " (with differences from the published files; see above)"))
    _notify(f"run_corpus finished: seasons {a.years}, steps {steps}"
            + ("" if ok else " -- DIFFERENCES from published files"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

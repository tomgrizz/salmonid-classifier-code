"""In plain terms: this script is a stock-take. Before any fish are counted it checks two
things for every season and river: that none of the saved detection files is damaged, and
that every video on the season's list actually made it through the tracking step. It changes
nothing; it only reads and reports. A season is safe to count when it shows 0 damaged files
and 0 missing videos. If videos are missing, the counts for that season would be too low
without any other warning, which is exactly what this check exists to catch.

Read-only audit of the detect-stores (Pass A output) and track aggregates (Pass B output).

Pipeline stage: data -> detect -> **audit (this module)** -> track -> **audit again** -> count.

Why it exists: during the original v2.7 run, one damaged detection file out of 47,997 stopped
a tracking job part-way and silently removed 679 videos from the 2025 Credit River season.
Nothing downstream noticed, because a shorter list of tracks looks the same as fewer fish.
The loss was found by a separate coverage check, repaired, and the season re-counted on
5 August 2026, before the manuscript numbers were produced. This audit was written then so
that the same thing cannot go unnoticed again.

Two checks per season and river:
  INTEGRITY     every store file is non-empty and does not start with a zero byte (the mark of
                a write that was cut short). ``--deep`` opens and fully reads every file
                instead, which is exact but slow (hundreds of thousands of files).
  COMPLETENESS  every video in the season's list (manifest) is present in the track aggregate.

Inputs : ``config.corpus_dir(year)/store`` (and ``store_gan_recovered`` for 2020),
         ``config.corpus_dir(year)/agg_<river>_shard*.pkl``, and the manifests
         ``config.LABEL_BATCHES/corpus_<year>_<river>.json`` (``..._full.json`` for 2025).
Outputs: a table printed to the screen. Nothing is written. The exit code is 1 if anything
         is damaged or missing, so a calling script can stop before counting.
Feeds  : no manuscript number directly. It is the guard on the inputs to Tables 2-5 and
         Figures 4-7.
Run    : python -m salmonid.detect.audit_store_integrity
         python -m salmonid.detect.audit_store_integrity --years 2025 --deep
         python -m salmonid.detect.audit_store_integrity --stores-only      (between Pass A and Pass B)
         python -m salmonid.detect.audit_store_integrity --aggregates-only  (before counting; seconds)
Tier   : 2 (needs the detect-stores and aggregates; base Python environment, no GPU).
         The integrity check opens every store file (246,070 for the six seasons) and took
         about 40 minutes on a hard disk. The completeness check alone takes seconds.

How to read the table: one line per river. "files" and "corrupt" describe the store folder
(shared by both rivers of a season); "manifest" is the number of videos on the list, "in agg"
how many of them have tracks, and "missing" the difference. For the 2025 season the list is
the corrected ``_full`` manifest; the first 2025 lists omitted about 13% of the videos.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

from salmonid import config

YEARS = list(config.CORPUS_YEARS)
SITES = ["credit", "ganaraska"]


def store_key_vid(video_id):
    """Store file name for a video: a 16-character hash of its full path (same as Pass A / Pass B)."""
    return f"{hashlib.md5(video_id.encode()).hexdigest()[:16]}.pkl"


def manifest_for(year, site):
    """The video list used for one season and river.

    ``site`` is "ganaraska", "credit" or "both". If a corrected ``..._full.json`` list exists
    it is used (this is the case for 2025 only); otherwise the plain list.
    """
    full = config.LABEL_BATCHES / f"corpus_{year}_{site}_full.json"
    return full if full.is_file() else config.LABEL_BATCHES / f"corpus_{year}_{site}.json"


def scan_store(store, deep):
    """Count the files in a store folder and list the damaged ones. Returns ``(n_files, bad_names)``.

    The quick check (file not empty, first byte not zero) catches a write that was cut short,
    which is the only kind of damage seen in practice. ``deep`` reads every file completely.
    """
    bad = []
    n = 0
    if not store.is_dir():
        return 0, bad
    for e in os.scandir(store):
        if not e.name.endswith(".pkl"):
            continue
        n += 1
        try:
            if e.stat().st_size == 0:
                bad.append(e.name); continue
            if deep:
                with open(e.path, "rb") as fh:
                    pickle.load(fh)
            else:
                with open(e.path, "rb") as fh:
                    if fh.read(1)[:1] == b"\x00":
                        bad.append(e.name)
        except Exception:                                   # noqa: BLE001
            bad.append(e.name)
    return n, bad


def main():
    """Print the audit table; exit with code 1 if any file is damaged or any video is missing."""
    ap = argparse.ArgumentParser(description="Audit detect-stores for damaged files and aggregates for missing videos.")
    ap.add_argument("--deep", action="store_true", help="fully read every store file (slow but exact)")
    ap.add_argument("--years", nargs="+", type=int, default=YEARS,
                    help="seasons to audit (default: 2020-2025)")
    ap.add_argument("--tag", default=config.CORPUS_TAG,
                    help="model tag in the folder names (default: the published model, v27)")
    ap.add_argument("--version", dest="tag", help=argparse.SUPPRESS)   # the original script's name for --tag
    ap.add_argument("--stores-only", action="store_true",
                    help="check the store files only and do not count videos missing from the aggregates "
                         "(for use after Pass A, before the aggregates exist)")
    ap.add_argument("--aggregates-only", action="store_true",
                    help="skip the store files and only check that every listed video is in the track "
                         "aggregates (seconds; this is the check that matters before counting)")
    a = ap.parse_args()
    if a.stores_only and a.aggregates_only:
        ap.error("--stores-only and --aggregates-only cannot be combined")

    config.require(config.LABEL_BATCHES, "the folder of video lists (manifests), 'label_batches'",
                   "SALMONID_PROJECTS_ROOT")
    print(f"{'year':6}{'store':26}{'files':>9}{'corrupt':>9}"
          f"{'   site':12}{'manifest':>10}{'in agg':>9}{'missing':>9}")
    print("-" * 92)
    total_bad = total_missing = n_stores = bad_aggs = 0
    for year in a.years:
        base = config.corpus_dir(year, a.tag)
        # "store_gan_recovered" exists only for 2020: the recovered Ganaraska footage was
        # processed into its own store and aggregates, apart from the regular 2020 footage.
        for store_name in ["store", "store_gan_recovered"]:
            store = base / store_name
            if a.aggregates_only:
                # The store folder is not opened (and need not be present). The recovered
                # 2020 store is represented by its aggregates.
                if store_name == "store_gan_recovered" and not glob.glob(str(base / "agg_gan_recovered_*.pkl")):
                    continue
                if store_name == "store" and not glob.glob(str(base / "agg_*.pkl")):
                    continue
                n_stores += 1
                n, bad = 0, []
            else:
                if not store.is_dir():
                    continue
                n_stores += 1
                n, bad = scan_store(store, a.deep)
            total_bad += len(bad)
            flag = f"  <<< {len(bad)} CORRUPT" if bad else ""
            printed = False
            for site in SITES:
                if store_name == "store_gan_recovered":
                    man = config.LABEL_BATCHES / "corpus_2020_gan_recovered.json"
                    pat = str(base / "agg_gan_recovered_*.pkl")
                    label = "gan_recovered"
                    if site != "credit":                     # one line only for this store
                        continue
                else:
                    man = manifest_for(year, site)
                    pat = str(base / f"agg_{site}_shard*.pkl")
                    label = site
                if not man.is_file():
                    continue
                keys = {store_key_vid(v["video_id"])
                        for v in json.load(open(man, encoding="utf-8"))["videos"]}
                agg = set()
                for fn in sorted(glob.glob(pat)):
                    try:
                        with open(fn, "rb") as fh:
                            agg |= set(pickle.load(fh))
                    except Exception as e:                   # noqa: BLE001
                        print(f"  !! unreadable aggregate {Path(fn).name}: {type(e).__name__}")
                        bad_aggs += 1
                missing = len(keys - agg)
                if not a.stores_only:
                    total_missing += missing
                counts = f"{'-':>9}{'-':>9}" if a.aggregates_only else f"{n:>9,}{len(bad):>9}"
                head = (f"{year:<6}{store_name:26}{counts}" if not printed
                        else " " * 50)
                printed = True
                print(f"{head}   {label:12}{len(keys):>10,}{len(keys) - missing:>9,}{missing:>9,}"
                      + ("  <<< GAP" if missing else "") + (flag if flag and head.strip() else ""))
            for b in bad:
                print(f"        corrupt: {store}/{b}")

    print("-" * 92)
    if n_stores == 0:
        config.require(config.corpus_dir(a.years[0], a.tag) / ("agg_*.pkl" if a.aggregates_only else "store"),
                       f"the detect-stores / track aggregates for the requested seasons (tag {a.tag})",
                       "SALMONID_PROJECTS_ROOT")
    if a.stores_only:
        print("(--stores-only: the 'missing' column is shown for information and does not affect the result)")
    if a.aggregates_only:
        print("(--aggregates-only: store files were not opened)")
    print(f"TOTAL corrupt store files: {total_bad}   TOTAL videos missing from aggregates: {total_missing:,}"
          + (f"   UNREADABLE aggregate files: {bad_aggs}" if bad_aggs else ""))
    if total_bad:
        print("Re-run salmonid.detect.detect_store for that season: it re-detects any store file that")
        print("fails the validity check. Then re-run salmonid.detect.track_from_store, which resumes")
        print("and adds the videos that were missing.")
    if total_bad or total_missing or bad_aggs:
        sys.exit(1)


if __name__ == "__main__":
    main()

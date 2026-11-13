"""In plain terms: the 2025 technician record came out of two review tools over five
periods, and only one of the tools lets a reviewer say a fish was "holding" (it entered the
counter and milled without passing). This script counts, for each river and period, how many
identifications the record holds and how many were logged as holding. It shows how much of
the reference the model is compared with could not record the holding outcome that the model
itself uses.

Inputs
------
Tier 2 only: the 2025 technician review exports under ``config.TECH_2025_DIR`` (third-party
OMNR records, not stored in git; set ``SALMONID_TECH_2025_DIR``):

* ``fish_counts_export_*_vaki_matched.csv``: reviewed in the manufacturer's software and
  linked to video afterwards (upstream / downstream only);
* other ``fish_counts_export_*.csv``: reviewed in the web application (adds "Stay" = holding);
* ``CounterExport_04_04_06_24.csv``: the spring Ganaraska River record, with no video link.

Outputs
-------
Printed, and ``table_s3_review_record.csv`` in ``--out-dir`` (default ``output/paper/``).

Feeds
-----
* Table S3 (records and holding totals by river and period).
* Section 4.5: technicians logged 24% of Credit River events and 40% of Ganaraska River
  events as holding in the web-application weeks.

Run
---
::

    python -m salmonid.paper.technician_record_2025

Tier
----
2 (the exports are not redistributed with this repository).

Provenance
----------
WRITTEN FOR THE DEPOSIT. Table S3 was assembled from row counts of the export files; no
earlier script existed. A "record" is one row of an export; "holding" is a row whose
movement is "Stay".
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import pandas as pd

from salmonid import config
from salmonid.paper._common import add_out_dir, out_dir

FOLDERS = {
    "Credit": config.TECH_2025_DIR / "Credit_Counter_Analysis_2025" / "data" / "raw_data",
    "Ganaraska": config.TECH_2025_DIR / "Ganaraska_Counter_Analysis_2025" / "data" / "raw",
}


def summarize(river, folder):
    config.require(folder, "2025 technician review exports (third-party OMNR records)", "SALMONID_TECH_2025_DIR")
    groups = {}
    for f in sorted(glob.glob(str(folder / "fish_counts_export_*.csv"))):
        tool = "Manufacturer's software" if "vaki_matched" in Path(f).name else "Web application"
        rows = list(csv.DictReader(open(f, encoding="latin-1")))
        g = groups.setdefault(tool, dict(records=0, holding=0, first="9999", last="0000", files=0))
        g["records"] += len(rows)
        g["holding"] += sum(1 for r in rows if (r.get("movement") or "").strip().lower() == "stay")
        ts = sorted(r["ts"][:10] for r in rows if r.get("ts"))
        g["first"], g["last"], g["files"] = min(g["first"], ts[0]), max(g["last"], ts[-1]), g["files"] + 1
    out = [dict(river=river, review_tool=t, linked_to_video="Yes", first_day=g["first"], last_day=g["last"],
                files=g["files"], records=g["records"], logged_as_holding=g["holding"],
                holding_pct=round(100 * g["holding"] / g["records"], 1)) for t, g in groups.items()]
    spring = folder / "CounterExport_04_04_06_24.csv"      # Ganaraska River only
    if spring.is_file():
        n = sum(1 for _ in csv.DictReader(open(spring, encoding="latin-1")))
        out.append(dict(river=river, review_tool="Manufacturer's software", linked_to_video="No",
                        first_day="2025-04-04", last_day="2025-06-24", files=1, records=n,
                        logged_as_holding=0, holding_pct=0.0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_out_dir(ap)
    args = ap.parse_args()
    out = out_dir(args)
    rows = [r for river, folder in FOLDERS.items() for r in summarize(river, folder)]
    t = pd.DataFrame(rows).sort_values(["river", "first_day"]).reset_index(drop=True)
    print(t.to_string(index=False))
    t.to_csv(out / "table_s3_review_record.csv", index=False)
    print(f"\nwrote: {out / 'table_s3_review_record.csv'}")


if __name__ == "__main__":
    main()

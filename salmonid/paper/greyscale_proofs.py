"""In plain terms: the journal asks that every figure can still be read when printed in black
and white, so no two series may differ by colour alone. This script makes a greyscale copy
of each figure for that check. It changes no figure and computes no result.

Inputs
------
The figure PNGs in ``config.FIGURES_DIR`` (``figures/``), produced by the figure scripts
(``python run_tier1.py`` regenerates the Tier-1 ones). Figures that are not there are
reported as missing and skipped.

Outputs
-------
``<name>_grey.png`` for each figure, in ``--out-dir``/``greyscale`` (default
``output/paper/greyscale/``).

Feeds
-----
No manuscript number. It supports the journal's requirement that figures remain
interpretable in monochrome (all figures, main and supplementary).

Run
---
::

    python -m salmonid.paper.greyscale_proofs

Tier
----
1 for the figures drawn from committed tables; Figures 1, 3 and S1 need their Tier-2 inputs
to exist first.

Provenance
----------
Ported from ``scripts/make_greyscale_proofs.py`` (manuscript repository). The conversion is
unchanged (ITU-R 601-2 luma, the weighting a monochrome print process approximates).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from salmonid import config

# The figures the manuscript places, in the order they appear.
FIGURES = [
    "fig1_study_map",                 # Figure 1
    "fig_confusion_heldout",          # Figure 2
    "fig_species_morphology",         # Figure 3
    "fig_hero_performance",           # Figure 4
    "fig_species_phenology",          # Figure 5
    "fig_concordance",                # Figure 6
    "fig_ganaraska_sanctuary",        # Figure 7
    "bbox_sample_grid",               # Figure S1
    "fig_daily_net_2025_credit",      # Figure S2a
    "fig_daily_net_2025_ganaraska",   # Figure S2b
    "fig_bland_altman",               # Figure S3
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fig-dir", default=str(config.FIGURES_DIR), help="Folder holding the colour figures.")
    ap.add_argument("--out-dir", default=str(config.PAPER_OUTPUT),
                    help="Proofs are written to <out-dir>/greyscale (default: output/paper).")
    args = ap.parse_args()
    SRC = Path(args.fig_dir)
    DST = Path(args.out_dir) / "greyscale"

    Image.MAX_IMAGE_PIXELS = None          # the production-spec figures exceed the default guard
    DST.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        src = SRC / f"{name}.png"
        if not src.is_file():
            print(f"{name:<32} MISSING, skipped")
            continue
        with Image.open(src) as im:
            dpi = im.info.get("dpi")
            size = im.size
            grey = im.convert("L")         # "L" = 8-bit luma: 0.299 R + 0.587 G + 0.114 B
        out = DST / f"{name}_grey.png"
        grey.save(out, dpi=dpi) if dpi else grey.save(out)
        dpi_note = f"{dpi[0]:.0f} dpi" if dpi else "no dpi tag"
        print(f"{name:<32} {size[0]:>5} x {size[1]:<5} {dpi_note:>10}  ->  {out}")


if __name__ == "__main__":
    main()

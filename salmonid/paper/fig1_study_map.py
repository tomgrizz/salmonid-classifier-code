"""In plain terms: this script draws the study-area map (Figure 1): the north shore of Lake
Ontario, the Ganaraska and Credit rivers, the two fishway counting stations, three cities
for orientation, a scale bar, a north arrow and a small locator map of the Great Lakes
region. It computes no result; it only draws.

Inputs
------
Map layers, read from ``config.SPATIAL_DIR`` (default ``data/paper/spatial/``; set
``SALMONID_SPATIAL_DIR`` to use another folder):

* ``ganaraska_full.json``: Ganaraska River and North Ganaraska River centre lines.
  COMMITTED. OpenStreetMap data (Overpass export of 2026-06-10), (c) OpenStreetMap
  contributors, Open Database License (ODbL) 1.0.
* ``credit_river.geojson``: Credit River centre line. NOT COMMITTED: its source and licence
  were not recorded, so it is not redistributed here.
* ``lake_ontario.gpkg``: Lake Ontario shoreline polygon. NOT COMMITTED, for the same reason.
* Natural Earth 1:50m countries and lakes for the locator inset (public domain). Downloaded
  on first run into ``<SPATIAL_DIR>/ne_cache/``; needs an internet connection once.

``data/paper/README.md`` says where equivalent open layers can be obtained.

Outputs
-------
``fig1_study_map.png`` (600 dpi) and ``fig1_study_map.pdf`` (vector) in ``--out-dir``
(default ``output/paper/``).

Feeds
-----
Figure 1. The station coordinates drawn are those of Section 2.1.

Run
---
::

    python -m salmonid.paper.fig1_study_map

Needs ``geopandas`` and ``shapely`` in addition to ``requirements.txt``.

Tier
----
2 (two of the three local layers are not redistributed, and the inset needs a download).

Provenance
----------
Ported from ``scripts/make_fig1_map.py`` (manuscript repository). The drawing code is
unchanged; only the input and output locations were changed.
"""
from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from shapely.geometry import box, Point, LineString

from salmonid import config

DATA = config.SPATIAL_DIR
NE_CACHE = DATA / "ne_cache"
OUT = config.PAPER_OUTPUT / "fig1_study_map.png"

LAKE_GPKG = DATA / "lake_ontario.gpkg"
CREDIT_GJ = DATA / "credit_river.geojson"
GANA_OSM = DATA / "ganaraska_full.json"   # main stem + North Ganaraska River

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master"
EXTS = (".shp", ".shx", ".dbf", ".prj")

# professional serif, consistent across every label (matches the manuscript)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Georgia", "DejaVu Serif"]
plt.rcParams["pdf.fonttype"] = 42          # embed TrueType; the old PDF carried a Type 3 font

COL2_IN = 18.2 / 2.54                      # CJFAS 2-column width
K = COL2_IN / 9.5                          # layout designed at 9.5 in wide

# Station positions (latitude, longitude) and which side the label sits on.
SITES = [
    ("Port Hope fishway\n(Ganaraska River)", 43.96890415, -78.29518116, "right"),
    ("Streetsville fishway\n(Credit River)", 43.57763889, -79.70322222, "left"),
]
CITIES = [("Toronto", 43.6532, -79.3832, "left"),
          ("Hamilton", 43.2557, -79.8711, "left"),
          ("Cobourg", 43.9593, -78.1677, "left")]

# Map frame and inset frame in degrees; UTM zone 17N so the scale bar is true in metres.
LON0, LON1, LAT0, LAT1 = -80.20, -77.85, 43.12, 44.18
ILON0, ILON1, ILAT0, ILAT1 = -92.5, -74.5, 40.2, 49.3
UTM17, WEBM = "EPSG:32617", "EPSG:3857"

LAND = "#f6f5f1"
WATER = "#cfe3ef"
SHORE = "#5a8fa8"
RIVER = "#2f6f95"
SITE_C = "#c0392b"
CITY_C = "#3a3a3a"


# S and LW rescale font sizes and line widths from the 9.5 in design canvas to the journal's
# 18.2 cm page width without letting text fall below 7 pt or lines below 0.5 pt.
def S(size, floor=7.0):
    """Font size scaled to the print width, never below the 7 pt floor."""
    return max(floor, size * K)


def LW(width, floor=0.5):
    """Line width scaled to the print width, never below 0.5 pt (spec minimum 0.35 pt)."""
    return max(floor, width * K)


def fetch_ne(group, name):
    NE_CACHE.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    for ext in EXTS:
        dst = NE_CACHE / f"{name}{ext}"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        req = urllib.request.Request(f"{NE_BASE}/{group}/{name}{ext}",
                                     headers={"User-Agent": "study-map/1.0"})
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r, open(dst, "wb") as f:
            f.write(r.read())
    return gpd.read_file(NE_CACHE / f"{name}.shp")


def load_osm_ways(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = [LineString([(p["lon"], p["lat"]) for p in e["geometry"]])
             for e in d.get("elements", [])
             if e.get("type") == "way" and len(e.get("geometry", [])) > 1]
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def scalebar(ax, x, y, length_m):
    """Two-segment zebra scale bar; (x, y) is the LEFT end of the bar."""
    seg = length_m / 2
    ax.add_patch(Rectangle((x, y), seg, length_m * 0.05, facecolor="black",
                           edgecolor="black", zorder=20))
    ax.add_patch(Rectangle((x + seg, y), seg, length_m * 0.05, facecolor="white",
                           edgecolor="black", zorder=20))
    for xx, lab in [(x, "0"), (x + seg, f"{length_m/2000:.0f}"),
                    (x + length_m, f"{length_m/1000:.0f} km")]:
        ax.text(xx, y + length_m * 0.09, lab, ha="center", va="bottom",
                fontsize=S(8), zorder=21)


def north_arrow(ax, x, y, size):
    """Clean two-tone compass-style north pointer with an 'N' label."""
    tip = (x, y + size)
    notch = (x, y + size * 0.18)
    base_l = (x - size * 0.34, y - size * 0.40)
    base_r = (x + size * 0.34, y - size * 0.40)
    ax.add_patch(Polygon([tip, base_l, notch], closed=True, facecolor="black",
                         edgecolor="black", linewidth=0.5, zorder=21))
    ax.add_patch(Polygon([tip, base_r, notch], closed=True, facecolor="white",
                         edgecolor="black", linewidth=0.5, zorder=21))
    ax.text(x, y + size * 1.28, "N", ha="center", va="bottom", fontsize=S(13),
            fontweight="bold", zorder=22,
            path_effects=[pe.withStroke(linewidth=2 * K, foreground="white")])


def main():
    print("Loading layers...")
    clip = box(LON0, LAT0, LON1, LAT1)
    # clip the lake past the frame so the water runs cleanly to the map border
    # (no straight shoreline-coloured edge along the visible cut)
    lake_clip = box(LON0, LAT0 - 0.3, LON1 + 1.5, LAT1 + 0.3)
    lake = gpd.read_file(LAKE_GPKG).to_crs("EPSG:4326")
    credit = gpd.read_file(CREDIT_GJ).to_crs("EPSG:4326")
    gana = load_osm_ways(GANA_OSM)

    lake_m = gpd.clip(lake, lake_clip).to_crs(UTM17)
    credit_m = gpd.clip(credit, clip).to_crs(UTM17)
    gana_m = gpd.clip(gana, clip).to_crs(UTM17)
    lake_m["geometry"] = lake_m.simplify(40)
    credit_m["geometry"] = credit_m.simplify(25)
    gana_m["geometry"] = gana_m.simplify(25)

    sites = gpd.GeoDataFrame(
        {"label": [s[0] for s in SITES], "ha": [s[3] for s in SITES]},
        geometry=[Point(s[2], s[1]) for s in SITES], crs="EPSG:4326").to_crs(UTM17)
    cities = gpd.GeoDataFrame(
        {"name": [c[0] for c in CITIES], "ha": [c[3] for c in CITIES]},
        geometry=[Point(c[2], c[1]) for c in CITIES], crs="EPSG:4326").to_crs(UTM17)
    bnds = gpd.GeoSeries([clip], crs="EPSG:4326").to_crs(UTM17).total_bounds

    fig = plt.figure(figsize=(COL2_IN, COL2_IN * 6.6 / 9.5), dpi=300)
    ax = fig.add_axes([0.06, 0.07, 0.90, 0.88])
    ax.set_facecolor(LAND)

    lake_m.plot(ax=ax, facecolor=WATER, edgecolor=SHORE, linewidth=LW(0.8), zorder=2)
    gana_m.plot(ax=ax, color=RIVER, linewidth=LW(1.6), zorder=3)
    credit_m.plot(ax=ax, color=RIVER, linewidth=LW(1.6), zorder=3)

    gana_union = gana_m.union_all()
    gpt = gana_union.interpolate(0.60, normalized=True)
    ax.text(gpt.x, gpt.y + 6500, "Ganaraska R.", fontsize=S(12), style="italic",
            color=RIVER, rotation=0, ha="center", va="bottom", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5 * K, foreground="white")])
    cg = credit_m.union_all()
    ax.text(cg.centroid.x - 6000, cg.centroid.y - 3000, "Credit R.", fontsize=S(12),
            style="italic", color=RIVER, rotation=0, ha="center", va="center",
            zorder=6, path_effects=[pe.withStroke(linewidth=2.5 * K, foreground="white")])

    # Cobourg label pushed down into the water so it clears the north arrow
    city_special = {"Cobourg": (0, -13 * K, "center", "top")}
    cities.plot(ax=ax, marker="o", color=CITY_C, markersize=24 * K * K, zorder=7)
    for name, ha, g in zip(cities["name"], cities["ha"], cities.geometry):
        if name in city_special:
            dxp, dyp, cha, cva = city_special[name]
        else:
            dxp, dyp, cha, cva = (10 * K if ha == "left" else -10 * K), 4 * K, ha, "baseline"
        ax.annotate(name, (g.x, g.y), xytext=(dxp, dyp), textcoords="offset points",
                    fontsize=S(11), color=CITY_C, ha=cha, va=cva, zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.2 * K, foreground="white")])

    sites.plot(ax=ax, marker="*", color=SITE_C, edgecolor="black", linewidth=LW(0.6),
               markersize=380 * K * K, zorder=10)
    for label, ha, g in zip(sites["label"], sites["ha"], sites.geometry):
        dx = 10000 if ha == "left" else -10000
        ax.annotate(label, (g.x, g.y), xytext=(g.x + dx, g.y - 6500), ha=ha,
                    fontsize=S(10.5), fontweight="bold", color="#7b241c", zorder=11,
                    path_effects=[pe.withStroke(linewidth=2.5 * K, foreground="white")])

    ax.text(0.60, 0.42, "Lake Ontario", transform=ax.transAxes, ha="center",
            fontsize=S(22), style="italic", color="#2c6b8f", zorder=5,
            path_effects=[pe.withStroke(linewidth=2.5 * K, foreground="white")])

    ax.set_xlim(bnds[0], bnds[2]); ax.set_ylim(bnds[1], bnds[3])

    xt_lon = np.arange(-80.0, -77.9, 0.5); yt_lat = np.arange(43.25, 44.2, 0.25)
    xt = gpd.GeoSeries([Point(lo, (LAT0 + LAT1) / 2) for lo in xt_lon],
                       crs="EPSG:4326").to_crs(UTM17)
    yt = gpd.GeoSeries([Point((LON0 + LON1) / 2, la) for la in yt_lat],
                       crs="EPSG:4326").to_crs(UTM17)
    ax.set_xticks([p.x for p in xt]); ax.set_xticklabels([f"{abs(l):.1f}°W" for l in xt_lon], fontsize=S(11))
    ax.set_yticks([p.y for p in yt]); ax.set_yticklabels([f"{l:.2f}°N" for l in yt_lat], fontsize=S(11))
    ax.tick_params(direction="in", length=3, width=LW(0.8))
    for s in ax.spines.values():
        s.set_edgecolor("#444444"); s.set_linewidth(LW(0.8))

    # scale bar: bottom-right corner; north arrow: top-right land portion
    scalebar(ax, bnds[2] - 54000, bnds[1] + 11000, 40000)
    north_arrow(ax, bnds[2] - 14000, bnds[3] - 15000, 6000)

    # ---- regional locator inset ----
    try:
        iax = fig.add_axes([0.075, 0.62, 0.30, 0.30])
        iclip = box(ILON0, ILAT0, ILON1, ILAT1)
        iland = fetch_ne("50m_cultural", "ne_50m_admin_0_countries")
        ilake = fetch_ne("50m_physical", "ne_50m_lakes")
        iax.set_facecolor(LAND)
        gpd.clip(iland, iclip).to_crs(WEBM).boundary.plot(ax=iax, color="#9a9a9a", linewidth=0.5, zorder=1)
        gpd.clip(ilake, iclip).to_crs(WEBM).plot(ax=iax, facecolor=WATER, edgecolor=SHORE, linewidth=0.5, zorder=2)
        mbox = gpd.GeoSeries([box(LON0, LAT0, LON1, LAT1)], crs="EPSG:4326").to_crs(WEBM)
        mbox.boundary.plot(ax=iax, color=SITE_C, linewidth=LW(1.4), zorder=5)
        ib = gpd.GeoSeries([iclip], crs="EPSG:4326").to_crs(WEBM).total_bounds
        iax.set_xlim(ib[0], ib[2]); iax.set_ylim(ib[1], ib[3])
        iax.set_xticks([]); iax.set_yticks([])
        for s in iax.spines.values():
            s.set_edgecolor("#444444"); s.set_linewidth(LW(0.8))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] inset skipped: {exc}")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
                metadata={"CreationDate": None})
    print(f"Saved {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(config.PAPER_OUTPUT),
                    help="Folder for the figure (default: output/paper, git-ignored).")
    args = ap.parse_args()
    OUT = Path(args.out_dir) / "fig1_study_map.png"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for layer, what in ((LAKE_GPKG, "Lake Ontario shoreline polygon"),
                        (CREDIT_GJ, "Credit River centre line"),
                        (GANA_OSM, "Ganaraska River centre lines")):
        config.require(layer, f"{what} for Figure 1 (see data/paper/README.md)", "SALMONID_SPATIAL_DIR")
    main()

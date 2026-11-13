"""In plain terms: this script makes extra practice pictures for the fish detector. It takes
real fish that a person has already identified, cut out of a video frame with the background
removed, and pastes them onto real empty frames from the same camera tunnel. It builds two hard
situations on purpose: a fish only partly in view at the edge of the frame, and two fish
overlapping. Because every pasted fish was identified by a person, the species label of each
composite is known, not guessed. Before making anything it throws away every cut-out that came
from a held-out test clip, so no exam fish can re-enter training inside a composite.

Inputs
------
* The cut-out library: ``<--cutouts-dir>/manifest.csv`` plus one transparent PNG per fish
  (default ``config.SAM2_CUTOUTS``), with ``review.csv`` holding the lead author's
  reject / relabel decisions for individual cut-outs.
* Empty tunnel frames: ``<--backgrounds-dir>/index.csv`` (or ``Credit/`` and ``Ganaraska/`` folders).
* The held-out sets: the canonical ones from ``salmonid.holdout`` plus two older evaluation
  lists (``--op-testset``, ``--ls-test-queue``).

Outputs
-------
Under ``--output-dir`` (default ``<SALMONID_PROJECTS_ROOT>/sam2_synth_<mode>``):

* ``images/*.jpg`` and ``synth_coco.json``: the composites and their boxes (five species, one box
  per visible fish). Each record lists the real fish it was cut from (``srcs``); that list is
  what ``salmonid.data.audit_coco_holdout_leak --synth-coco`` checks.
* ``crops/*.jpg`` and ``crops.csv``: small square crops for a separate classifier (not used by
  the published single-stage model; skip with ``--no-crops``).

Feeds
-----
No manuscript number directly. Its composites, after ``salmonid.data.merge_synth_into_coco``,
are 7,391 of the 45,249 pictures (11,644 of the 52,444 fish boxes) the published v2.7 model was
trained on (Methods, training-set composition).

Run
---
::

    # single fish partly out of frame, with a preview gallery
    python -m salmonid.data.sam2_composite partial   --n 2000 --sweep --mask-header-px 40 --gallery
    # one fish partly hidden behind another
    python -m salmonid.data.sam2_composite proximity --n 1200 --pair coho_chinook --mask-header-px 40 --gallery

Tier
----
2. The cut-out library, the empty backgrounds and ``review.csv`` are external and were not
produced by code in this deposit (see "What cannot be regenerated").

The two modes
-------------
::

  partial   — reveal only a swept fraction of a known fish past a frame edge -> teaches "a Chinook tail
              is still a Chinook" AND, because the species is known, gives an accuracy-vs-exposure
              DIAGNOSTIC axis (100/75/50/25% visible).
  proximity — paste two known cutouts adjacent/overlapping (default Coho trailing a Chinook, the observed
              two-fish confusion) with correct boxes for BOTH -> teaches fish-to-fish separation for
              crowded passages and the two-fish Coho->Chinook mis-ID.

Both paste onto REAL empty frames from the SAME site, so the tunnel geometry, water and lighting
match real footage. Header strip: composites keep the camera's date/river overlay unless
``--mask-header-px 40`` is passed, which blacks it out the same way the real training frames are.

How this differs from what built the published data
---------------------------------------------------
The guard here is the corrected, video-level one (research repository commit ``d40b869``). The
composite batches in the published training list were generated earlier, with the pair-keyed
guard: 23 of them used a cut-out from a held-out clip and were removed afterwards by
``salmonid.data.audit_coco_holdout_leak --out``. None remain in the v2.7 list.

Known limitations (left as they were)
-------------------------------------
* ``--no-holdout-filter`` switches the whole guard off, including the canonical test set. It
  exists for making demonstration images; output made with it must not be used for training.
  The script now prints a warning when it is used.
* Background frames carry no record of which clip they came from, so they are not tested
  against the holdout. They are empty frames (no fish), chosen by a separate harvest step.
* In ``_apply_review``, a cut-out relabelled to something other than the five species (for
  example "Small Salmonid") is kept under its ORIGINAL species instead of being dropped.

What cannot be regenerated from this deposit
--------------------------------------------
The random choices are seeded (``--seed``, default 42), so the same library and backgrounds give
the same composites. But the cut-out extraction step, the background harvest step and the
review tool are not part of this deposit, and the exact commands (counts, species pairs) used
for the four targeted v2.6 batches were not recorded in any repository. The composites
themselves are therefore a frozen external artifact.

Environment: base Python 3.11 (``cv2`` + ``numpy``). It does not need the DEIMv2 environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from salmonid import config

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow Trout", 4: "Brown Trout"}  # mirrors salmonid.config.CLASS_TO_SP
LABEL_TO_CLASS = {v: k for k, v in SPECIES.items()}  # mirrors salmonid.config.SP_TO_CLASS
EDGES = ("left", "right", "top", "bottom")
SWEEP = (1.0, 0.75, 0.5, 0.25)


# ---------------------------------------------------------------------------
# Library / background loading
# ---------------------------------------------------------------------------

def _load_manifest(cut_dir: Path) -> list[dict]:
    """Load the cutout library manifest.

    Args:
        cut_dir: cutout directory holding manifest.csv + the RGBA PNGs it references.
    Returns:
        List of manifest rows (dicts) with an added ``_png`` Path, filtered to rows whose PNG exists.
    """
    rows = list(csv.DictReader(open(cut_dir / "manifest.csv", newline="", encoding="utf-8")))
    for r in rows:
        r["_png"] = cut_dir / r["png"]
    return [r for r in rows if r["_png"].is_file()]


_NAME2ID = {"Chinook": "0", "Coho": "1", "Atlantic": "2", "Rainbow Trout": "3", "Brown Trout": "4"}


def _apply_review(cuts: list[dict], cut_dir: Path, review_csv: str | None) -> list[dict]:
    """Gate the library by the lead author's cut-out review. DROP cutout_ids marked 'reject';
    for 'relabel' keep the cutout under the corrected species (the reviewer is the species authority).
    Last write per cutout_id wins, so a later 'keep' undoes an earlier reject.

    Known limitation, unchanged: a 'relabel' to a name outside the five species falls through
    the `sp in _NAME2ID` test below and the cut-out is kept under its ORIGINAL species."""
    p = Path(review_csv) if review_csv else (cut_dir / "review.csv")
    if not p.is_file():
        return cuts
    latest: dict[str, tuple[str, str]] = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        latest[r["cutout_id"]] = (r["action"], r.get("species", ""))
    out, n_rej, n_rel = [], 0, 0
    for c in cuts:
        act, sp = latest.get(c["cutout_id"], ("keep", ""))
        if act == "reject":
            n_rej += 1
            continue
        if act == "relabel" and sp in _NAME2ID:
            c = dict(c); c["species"] = sp; c["class_id"] = _NAME2ID[sp]; n_rel += 1
        out.append(c)
    print(f"review filter ({p.name}): dropped {n_rej} rejected, relabelled {n_rel} -> {len(out)} cutouts")
    return out


def _short_rel(video_id: str) -> str:
    """Last two path components (timestamp_folder/file.mp4) — the op_testset video_rel convention."""
    return "/".join(video_id.split("/")[-2:])


def _load_holdout(op_path: str, ls_path: str):
    """Return (op_keys {(site, short_rel)}, ls_keys {(video_id, track_id)}) for the EVAL holdouts.

    Cutouts come from labelled tracks, some of which ARE the op_testset / ls_test_set_v5 eval fish
    (measured: 145 ls + 11 op of 751). Compositing those into training data re-introduces exactly the
    leak the eval-hygiene work fought. This gate drops them BY DEFAULT.
    """
    op_keys, ls_keys = set(), set()
    for label, path in (("--op-testset", op_path), ("--ls-test-queue", ls_path)):
        # The original skipped a missing file without a word. These two lists are secondary to
        # the canonical guard (fatal when its own files are missing), so a missing one is still
        # not an error, but it is now said out loud.
        if path and not Path(path).is_file():
            print(f"  WARNING: {label} file not found, so it excludes nothing: {path}")
    if op_path and Path(op_path).is_file():
        for v in json.loads(Path(op_path).read_text(encoding="utf-8")).get("videos", []):
            op_keys.add((v["site"], v["video_rel"].replace("\\", "/")))
    if ls_path and Path(ls_path).is_file():
        for it in json.loads(Path(ls_path).read_text(encoding="utf-8")).get("items", []):
            ls_keys.add((it["video_id"], int(it["track_id"])))
    return op_keys, ls_keys


def _is_holdout(row: dict, op_keys, ls_keys) -> bool:
    """True if a cutout row belongs to an eval-holdout track (by op site/rel key or ls video/track key)."""
    vid = row["video_id"]
    site = vid.split("/", 1)[0]
    if (site, _short_rel(vid)) in op_keys:
        return True
    if (vid, int(row["track_id"])) in ls_keys:
        return True
    return False


def _load_backgrounds(bg_dir: Path) -> dict[str, list[Path]]:
    """Index empty-tunnel background frames by site.

    Prefers ``index.csv`` (site,bg columns); falls back to scanning Credit/ and Ganaraska/ subdirs.
    Returns {site: [background_paths]} keeping only files that exist on disk.
    """
    idx = bg_dir / "index.csv"
    by_site: dict[str, list[Path]] = defaultdict(list)
    if idx.is_file():
        for r in csv.DictReader(open(idx, newline="", encoding="utf-8")):
            p = bg_dir / r["bg"]
            if p.is_file():
                by_site[r["site"]].append(p)
    else:  # fall back to scanning site subdirs
        for site in ("Credit", "Ganaraska"):
            by_site[site] = sorted((bg_dir / site).glob("*.jpg"))
    return by_site


def _src(row: dict) -> dict:
    """Per-fish provenance for a synth image (re-auditable leak trail)."""
    return {"cutout_id": row["cutout_id"], "video_id": row["video_id"],
            "track_id": int(row["track_id"]), "species": row["species"]}


def _flt(v, default=0.0):
    """Best-effort float() coercion, returning ``default`` for None / non-numeric manifest fields."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _read_rgba(path: Path):
    """Read a stored BGRA cutout PNG.

    Returns (rgb HxWx3, alpha HxW) with the colour channels converted to RGB, or (None, None) if the
    file is missing or is not 4-channel.
    """
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[2] != 4:
        return None, None
    rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)  # stored BGRA
    return rgb, im[:, :, 3]


# ---------------------------------------------------------------------------
# Core paste
# ---------------------------------------------------------------------------

def _place(bg_bgr, cut_rgb, cut_alpha, x, y, harmonize=0.25, feather=1.0):
    """Alpha-blend a cutout onto bg at top-left (x,y) (may be off-edge -> clipped).

    Returns (composite_bgr, mask_full HxW bool, visible_box xyxy) or None if fully off-frame.
    Boxes are computed from the hard (>=0.5) alpha within the frame, so partial-past-edge fish get
    a tight box on the VISIBLE fragment only.
    """
    H, W = bg_bgr.shape[:2]
    h, w = cut_alpha.shape
    bx1, by1 = max(0, x), max(0, y)
    bx2, by2 = min(W, x + w), min(H, y + h)
    if bx2 <= bx1 or by2 <= by1:
        return None
    cx1, cy1 = bx1 - x, by1 - y
    cx2, cy2 = cx1 + (bx2 - bx1), cy1 + (by2 - by1)

    a = cut_alpha[cy1:cy2, cx1:cx2].astype(np.float32) / 255.0
    rgb = cut_rgb[cy1:cy2, cx1:cx2].astype(np.float32)
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)
    dst = bg_bgr[by1:by2, bx1:bx2].astype(np.float32)

    hard = a >= 0.5
    if not hard.any():
        return None
    if harmonize > 0 and hard.sum() > 50:
        cut_lum = bgr[hard].mean()
        bg_lum = dst.mean()
        if cut_lum > 1:
            gain = (bg_lum / cut_lum) ** harmonize     # partial pull toward local bg brightness
            bgr = np.clip(bgr * gain, 0, 255)
    if feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), feather)
    a3 = a[:, :, None]
    blended = a3 * bgr + (1 - a3) * dst

    comp = bg_bgr.copy()
    comp[by1:by2, bx1:bx2] = blended.astype(np.uint8)
    mask_full = np.zeros((H, W), bool)
    mask_full[by1:by2, bx1:bx2] = hard
    ys, xs = np.where(mask_full)
    vbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return comp, mask_full, vbox


def _scaled(cut_rgb, cut_alpha, scale):
    """Resize a cutout (rgb bilinear, alpha nearest to keep the mask hard); identity if scale~=1."""
    if abs(scale - 1.0) < 1e-3:
        return cut_rgb, cut_alpha
    h, w = cut_alpha.shape
    nw, nh = max(4, int(w * scale)), max(4, int(h * scale))
    return (cv2.resize(cut_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR),
            cv2.resize(cut_alpha, (nw, nh), interpolation=cv2.INTER_NEAREST))


def _maybe_flip(cut_rgb, cut_alpha, do):
    """Horizontally mirror the cutout+alpha when ``do`` is True (fish can swim either direction)."""
    return (cut_rgb[:, ::-1], cut_alpha[:, ::-1]) if do else (cut_rgb, cut_alpha)


def _letterbox_crop(frame_bgr, box, pad_frac=0.20, size=224):
    """Cut a padded, aspect-preserving 224 letterbox crop around ``box`` (xyxy).

    Matches the crop geometry of the earlier two-stage classifier's training crops, so synth
    classifier crops are pixel-consistent with real ones:
    pad the box by ``pad_frac`` per side, resize the longest side to ``size``, and centre it on a
    black square canvas. Returns the HxWx3 crop, or None for a degenerate box / empty crop.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    fh, fw = frame_bgr.shape[:2]
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(fw, x2 + px), min(fh, y2 + py)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    s = size / max(ch, cw)
    rz = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    yo, xo = (size - rz.shape[0]) // 2, (size - rz.shape[1]) // 2
    canvas[yo:yo + rz.shape[0], xo:xo + rz.shape[1]] = rz
    return canvas


# ---------------------------------------------------------------------------
# Placement policies
# ---------------------------------------------------------------------------

def _partial_xy(edge, expo, W, H, w, h, rng):
    """Top-left (x,y) so that fraction `expo` of the cutout stays inside the frame past `edge`."""
    # Free coordinate along the non-crossing axis, biased to the fish swim band (0.35-0.80 H).
    fy = int(rng.uniform(0.35, 0.72) * H) - h // 2
    fx = rng.randint(0, max(1, W - w))
    if edge == "right":
        return W - int(expo * w), fy
    if edge == "left":
        return -int((1 - expo) * w), fy
    if edge == "bottom":
        return fx, H - int(expo * h)
    if edge == "top":                      # rarely a real case, but completes the sweep
        return fx, -int((1 - expo) * h)
    return fx, fy


# ---------------------------------------------------------------------------
# COCO / crops writer
# ---------------------------------------------------------------------------

class Emitter:
    """Accumulates COCO images/annotations + classifier crops, writes at the end."""

    def __init__(self, out_dir: Path, source_tag: str, emit_crops: bool):
        """Prepare the output tree.

        Args:
            out_dir: target directory (images/ always created; crops/ only when emit_crops).
            source_tag: value stamped on every COCO image's ``synth_source`` and each crop row's
                ``source`` (e.g. "synth_partial"), so folded synth stays attributable.
            emit_crops: also write 224 classifier crops + crops.csv alongside the COCO.
        """
        self.out = out_dir
        self.img_dir = out_dir / "images"; self.img_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir = out_dir / "crops"
        self.source = source_tag
        self.emit_crops = emit_crops
        if emit_crops:
            self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.images, self.annotations, self.crop_rows = [], [], []
        self._img_id = 0; self._ann_id = 0

    def add(self, comp_bgr, fishes, stem, extra=None, srcs=None):
        """fishes = [(class_id, xyxy_box)]. extra = per-image metadata. srcs = per-fish provenance
        [{cutout_id, video_id, track_id, species}] so the merge is re-auditable for leaks."""
        H, W = comp_bgr.shape[:2]
        img_path = self.img_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_path), comp_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        self._img_id += 1
        img_rec = {"id": self._img_id, "file_name": str(img_path.resolve()),
                   "width": W, "height": H, "synth_source": self.source}
        if srcs:
            img_rec["srcs"] = srcs
        if extra:
            img_rec.update(extra)
        self.images.append(img_rec)
        for cid, (x1, y1, x2, y2) in fishes:
            w, h = x2 - x1, y2 - y1
            self._ann_id += 1
            self.annotations.append({
                "id": self._ann_id, "image_id": self._img_id, "category_id": cid,
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0})
            if self.emit_crops:
                crop = _letterbox_crop(comp_bgr, (x1, y1, x2, y2))
                if crop is not None:
                    cname = f"{stem}_c{self._ann_id}.jpg"
                    cv2.imwrite(str(self.crop_dir / cname), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    self.crop_rows.append({"path": cname, "class_id": cid,
                                           "species": SPECIES[cid], "source": self.source})

    def write(self):
        """Flush accumulated records to disk (synth_coco.json, and crops.csv if crops were emitted).

        Returns (n_images, n_annotations). Side effect: writes files under ``self.out``.
        """
        coco = {
            "categories": [{"id": k, "name": v, "supercategory": "fish"} for k, v in SPECIES.items()],
            "images": self.images, "annotations": self.annotations,
        }
        (self.out / "synth_coco.json").write_text(json.dumps(coco, indent=2))
        if self.emit_crops and self.crop_rows:
            with open(self.out / "crops.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["path", "class_id", "species", "source"])
                w.writeheader(); w.writerows(self.crop_rows)
        return len(self.images), len(self.annotations)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _pick_bg(bgs, site, rng):
    """Random background path for ``site``; falls back to the other site, then any, else None."""
    pool = bgs.get(site) or bgs.get("Credit") or bgs.get("Ganaraska") or []
    if not pool:
        allp = [p for v in bgs.values() for p in v]
        return rng.choice(allp) if allp else None
    return rng.choice(pool)


def run_partial(a, cuts, bgs, rng):
    """Generate the PARTIAL set: single known fish revealed a swept fraction past a frame edge.

    Args:
        a: parsed CLI args (n, edges, sweep/min_expo, scale/harmonize/feather, mask_header_px, gallery...).
        cuts: filtered cutout library rows.
        bgs: {site: [background_paths]}.
        rng: seeded random.Random for reproducibility.
    Side effects: writes composites + COCO (+ optional crops/gallery) via an Emitter; prints a summary.
    Returns the Emitter.
    """
    em = Emitter(Path(a.output_dir), "synth_partial", not a.no_crops)
    gal = Path(a.output_dir) / "gallery"
    if a.gallery:
        gal.mkdir(parents=True, exist_ok=True)
    by_sp = Counter()
    made = 0
    attempts = 0
    # Optional species filter (e.g. focus partials on the classes that default to Atlantic).
    pool = [c for c in cuts if (not a.species or c["species"] in a.species)]
    while made < a.n and attempts < a.n * 20:
        attempts += 1
        c = rng.choice(pool)
        rgb, alpha = _read_rgba(c["_png"])
        if rgb is None:
            continue
        rgb, alpha = _maybe_flip(rgb, alpha, rng.random() < 0.5)
        rgb, alpha = _scaled(rgb, alpha, rng.uniform(a.scale_min, a.scale_max))
        bgp = _pick_bg(bgs, c["site"], rng)
        if bgp is None:
            continue
        bg = cv2.imread(str(bgp))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        edge = rng.choice(a.edges)
        expo = rng.choice(SWEEP) if a.sweep else rng.uniform(a.min_expo, 1.0)
        h, w = alpha.shape
        x, y = _partial_xy(edge, expo, W, H, w, h, rng)
        placed = _place(bg, rgb, alpha, x, y, a.harmonize, a.feather)
        if placed is None:
            continue
        comp, mask, vbox = placed
        # Guard: a valid visible fragment (not a sliver, not the whole thing when expo<1).
        vis_area = mask.sum()
        vw, vh = vbox[2] - vbox[0], vbox[3] - vbox[1]
        if vw < 12 or vh < 12 or vis_area < 400:
            continue
        if a.mask_header_px > 0:
            comp[:a.mask_header_px] = 0
            if vbox[1] < a.mask_header_px:  # box no longer valid if it was in the header
                continue
        cid = LABEL_TO_CLASS[c["species"]]
        stem = f"partial_{made:05d}_{c['species'].replace(' ', '')}_{edge}_{int(expo*100)}"
        em.add(comp, [(cid, vbox)], stem,
               extra={"exposure_frac": round(expo, 3), "edge": edge, "species": c["species"],
                      "site": c["site"]},
               srcs=[_src(c)])
        by_sp[c["species"]] += 1; made += 1
        if a.gallery and made <= a.gallery_n:
            _preview(comp, [(c["species"], vbox)], gal / f"{stem}.jpg", f"expo={expo:.2f} {edge}")
    ni, na = em.write()
    print(f"[partial] {ni} images / {na} boxes; by species " + dict(by_sp).__repr__())
    if a.sweep:
        print("  exposure sweep counts: " + repr(Counter(
            im["exposure_frac"] for im in em.images)))
    return em


def run_proximity(a, cuts, bgs, rng):
    """Generate the PROXIMITY set: a trailer fish partly occluded behind a leader fish, boxes for BOTH.

    Args:
        a: parsed CLI args (n, pair, overlap/size-ratio/scale, harmonize/feather, mask_header_px...).
        cuts: filtered cutout library rows.
        bgs: {site: [background_paths]}.
        rng: seeded random.Random for reproducibility.
    Side effects: writes composites + COCO (+ optional crops/gallery) via an Emitter; prints a summary.
    Returns the Emitter.
    """
    em = Emitter(Path(a.output_dir), "synth_proximity", not a.no_crops)
    gal = Path(a.output_dir) / "gallery"
    if a.gallery:
        gal.mkdir(parents=True, exist_ok=True)
    by_sp = defaultdict(int)
    lead_sp, trail_sp = _pair_species(a.pair)
    leaders = [c for c in cuts if c["species"] in lead_sp]
    trailers = [c for c in cuts if c["species"] in trail_sp]
    if not leaders or not trailers:
        print(f"[proximity] need leaders({lead_sp}) and trailers({trail_sp}); "
              f"have {len(leaders)}/{len(trailers)} — build more library first.")
        em.write(); return em
    made = attempts = 0
    while made < a.n and attempts < a.n * 30:
        attempts += 1
        lc, tc = rng.choice(leaders), rng.choice(trailers)
        if lc["cutout_id"] == tc["cutout_id"]:
            continue
        site = lc["site"]
        bgp = _pick_bg(bgs, site, rng)
        if bgp is None:
            continue
        bg = cv2.imread(str(bgp))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        lrgb, lal = _read_rgba(lc["_png"]); trgb, tal = _read_rgba(tc["_png"])
        if lrgb is None or trgb is None:
            continue
        # Size-MATCH the two fish. Cutouts carry their real per-frame pixel size, which differs wildly
        # (near vs far fish); pasting both at one scale factor keeps that mismatch and looks OOD. Two
        # co-migrating fish are similar-sized -> scale the trailer so its length ~= leader length x ratio.
        s = rng.uniform(a.scale_min, a.scale_max)
        lrgb, lal = _scaled(lrgb, lal, s)
        ll = _flt(lc.get("length_px")); tl = _flt(tc.get("length_px"))
        if ll > 5 and tl > 5:
            ratio = rng.uniform(a.size_ratio_min, a.size_ratio_max)
            ts = max(0.3, min(2.5, (ll * s * ratio) / tl))
        else:
            ts = s
        trgb, tal = _scaled(trgb, tal, ts)
        flip = rng.random() < 0.5
        lrgb, lal = _maybe_flip(lrgb, lal, flip); trgb, tal = _maybe_flip(trgb, tal, flip)
        lh, lw = lal.shape; th, tw = tal.shape
        # Leader placed in the swim band; trailer offset behind/beside with controlled overlap.
        lx = rng.randint(int(0.10 * W), max(int(0.10 * W) + 1, int(0.55 * W)))
        ly = int(rng.uniform(0.38, 0.66) * H) - lh // 2
        ov = rng.uniform(a.overlap_min, a.overlap_max)         # fraction of trailer width overlapped
        dxsign = 1 if flip else -1                              # trailer sits behind the leader's tail
        tx = lx + dxsign * int((1 - ov) * lw)
        ty = ly + rng.randint(-int(0.25 * lh), int(0.25 * lh))
        # Paste trailer FIRST, then leader on top -> leader occludes the trailing fish (realistic).
        p_t = _place(bg, trgb, tal, tx, ty, a.harmonize, a.feather)
        if p_t is None:
            continue
        comp_t, tmask, _ = p_t
        p_l = _place(comp_t, lrgb, lal, lx, ly, a.harmonize, a.feather)
        if p_l is None:
            continue
        comp, lmask, lbox = p_l
        # Trailer's VISIBLE box = its mask minus where the leader now covers it.
        tvis = tmask & ~lmask
        if tvis.sum() < 500:
            continue
        ys, xs = np.where(tvis)
        tbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if (tbox[2] - tbox[0]) < 16 or (tbox[3] - tbox[1]) < 16:
            continue
        if a.mask_header_px > 0:
            comp[:a.mask_header_px] = 0
            if lbox[1] < a.mask_header_px or tbox[1] < a.mask_header_px:
                continue
        fishes = [(LABEL_TO_CLASS[lc["species"]], lbox), (LABEL_TO_CLASS[tc["species"]], tbox)]
        stem = f"prox_{made:05d}_{tc['species'].replace(' ', '')}_behind_{lc['species'].replace(' ', '')}"
        em.add(comp, fishes, stem,
               extra={"leader": lc["species"], "trailer": tc["species"], "overlap": round(ov, 3),
                      "site": site},
               srcs=[_src(lc), _src(tc)])
        by_sp[f"{tc['species']}<-{lc['species']}"] += 1; made += 1
        if a.gallery and made <= a.gallery_n:
            _preview(comp, [(lc["species"], lbox), (tc["species"], tbox)],
                     gal / f"{stem}.jpg", f"ov={ov:.2f}")
    ni, na = em.write()
    print(f"[proximity] {ni} images / {na} boxes; pairs " + dict(by_sp).__repr__())
    return em


def _pair_species(pair):
    """Map a --pair token to (leader_species_set, trailer_species_set).

    'coho_chinook' => Chinook leads, Coho trails (the observed two-fish confusion); 'any' => any/any;
    'lead_trail' (any underscore form, e.g. 'chinook_coho') => explicit leader/trailer species names.
    """
    if pair == "coho_chinook":
        return {"Chinook"}, {"Coho"}
    if pair == "any":
        return set(SPECIES.values()), set(SPECIES.values())
    if "_" in pair:  # e.g. "chinook_coho" -> leader chinook, trailer coho
        a, b = pair.split("_", 1)
        n = {s.lower().replace(" ", ""): s for s in SPECIES.values()}
        return {n.get(a, a.title())}, {n.get(b, b.title())}
    return {"Chinook"}, {"Coho"}


def _preview(comp_bgr, fishes, out_path: Path, tag):
    """Write a 640x480 annotated QA thumbnail (species-coloured boxes + a corner tag) for the gallery."""
    im = comp_bgr.copy()
    cols = {"Chinook": (255, 60, 60), "Coho": (0, 160, 255), "Atlantic": (60, 200, 60),
            "Rainbow Trout": (0, 0, 255), "Brown Trout": (200, 60, 200)}
    for sp, (x1, y1, x2, y2) in fishes:
        c = cols.get(sp, (255, 255, 255))
        cv2.rectangle(im, (x1, y1), (x2, y2), c, 2)
        cv2.putText(im, sp, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    cv2.putText(im, tag, (6, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.resize(im, (640, 480)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: load + filter the library, then run the requested compositor mode.

    Order matters: review filter (drop rejects / apply relabels) BEFORE the holdout filter, and both
    BEFORE any composite is made, so no rejected or eval-holdout fish can enter training synth.
    Side effects: writes the synth COCO / crops / gallery under a.output_dir; prints progress.
    """
    a = _parse_args()
    rng = random.Random(a.seed)
    cut_dir = Path(a.cutouts_dir)
    cuts = _load_manifest(cut_dir)
    # Apply the cut-out review (reject/relabel) FIRST so bad/wrong-species cutouts never reach synth.
    if not a.no_review_filter:
        cuts = _apply_review(cuts, cut_dir, a.review_csv)
    # Drop EVAL-holdout cutouts before any training synth is made (on by default — see _load_holdout).
    if a.no_holdout_filter:
        print("WARNING: --no-holdout-filter given: held-out test fish are NOT excluded. "
              "Composites from this run must not be used for training.")
    if not a.no_holdout_filter:
        op_keys, ls_keys = _load_holdout(a.op_testset, a.ls_test_queue)
        # Permanent leak guard: always add the canonical frozen eval sets (test_set_v6 + ls_test_v5).
        # Assign (don't |=) so the VIDEO-aware HoldoutKeys survives: a cutout of the test fish mined
        # under another run's track id is still the test fish. `ls_keys |= load_holdout_pairs()`
        # (the earlier form) copied the pairs into a plain set and lost the video test.
        from salmonid.holdout import load_holdout_pairs
        ls_keys = load_holdout_pairs(extra_pairs=ls_keys, verbose=False)
        before = len(cuts)
        dropped = Counter(c["species"] for c in cuts if _is_holdout(c, op_keys, ls_keys))
        cuts = [c for c in cuts if not _is_holdout(c, op_keys, ls_keys)]
        print(f"holdout filter: dropped {before - len(cuts)} eval cutouts ({dict(dropped)})")
    bgs = _load_backgrounds(Path(a.backgrounds_dir))
    print(f"library: {len(cuts)} cutouts ({Counter(c['species'] for c in cuts)}); "
          f"backgrounds: {{{', '.join(f'{k} {len(v)}' for k, v in bgs.items())}}}")
    if not cuts:
        print("No cutouts — the cut-out library is empty or missing (see docs/DATA_AVAILABILITY.md)."); return
    if not any(bgs.values()):
        print("No backgrounds — the empty-frame library is empty or missing (see docs/DATA_AVAILABILITY.md)."); return
    (run_partial if a.mode == "partial" else run_proximity)(a, cuts, bgs, rng)
    print(f"-> {a.output_dir}  (synth_coco.json + images/"
          + ("" if a.no_crops else " + crops/ + crops.csv") + ")")


def _parse_args():
    """Define and parse the CLI. Path defaults resolve via salmonid.config (same absolute locations
    as the original workstation, overridable by SALMONID_* env vars). Returns the argparse Namespace,
    with output_dir defaulted to config.PROJECTS_ROOT/sam2_synth_<mode> when not supplied."""
    p = argparse.ArgumentParser(description="SAM2 partial/proximity compositor")
    p.add_argument("mode", choices=["partial", "proximity"])
    p.add_argument("--cutouts-dir", default=str(config.SAM2_CUTOUTS))
    p.add_argument("--backgrounds-dir", default=str(config.PROJECTS_ROOT / "sam2_backgrounds"))
    p.add_argument("--output-dir", default=None,
                   help="Default <config.PROJECTS_ROOT>/sam2_synth_<mode>.")
    p.add_argument("--n", type=int, default=300, help="Composites to generate.")
    p.add_argument("--species", action="append", default=None,
                   help="(partial) restrict source cutouts to these species.")
    p.add_argument("--pair", default="coho_chinook",
                   help="(proximity) 'coho_chinook' (Coho behind Chinook), 'any', or 'lead_trail'.")
    p.add_argument("--sweep", action="store_true",
                   help="(partial) draw exposure from the fixed 100/75/50/25% diagnostic set.")
    p.add_argument("--min-expo", type=float, default=0.25, help="(partial, non-sweep) min visible frac.")
    p.add_argument("--edges", nargs="+", default=["left", "right"], choices=list(EDGES),
                   help="(partial) which frame edges fish enter from (default the horizontal pair).")
    p.add_argument("--overlap-min", type=float, default=0.10)
    p.add_argument("--overlap-max", type=float, default=0.45)
    p.add_argument("--size-ratio-min", type=float, default=0.60,
                   help="(proximity) trailer length as a fraction of leader length — low end.")
    p.add_argument("--size-ratio-max", type=float, default=1.05,
                   help="(proximity) trailer/leader length ratio — high end (keeps the pair co-sized).")
    p.add_argument("--scale-min", type=float, default=0.85)
    p.add_argument("--scale-max", type=float, default=1.15)
    p.add_argument("--harmonize", type=float, default=0.25,
                   help="Luminance pull toward local bg (0 off, 1 full match). Keep low to preserve fish.")
    p.add_argument("--feather", type=float, default=1.0, help="Alpha edge blur sigma (px).")
    p.add_argument("--mask-header-px", type=int, default=0,
                   help="Pre-zero the top N px. Default 0 = keep header, mask downstream with real COCO.")
    p.add_argument("--no-holdout-filter", action="store_true",
                   help="Do NOT drop held-out cutouts (switches off the canonical guard too). "
                        "Only for non-training use, e.g. demonstration images.")
    p.add_argument("--review-csv", default=None,
                   help="Cutout review decisions (default <cutouts-dir>/review.csv); "
                        "rejects dropped, relabels applied.")
    p.add_argument("--no-review-filter", action="store_true",
                   help="Ignore review.csv (use the raw library). Default: apply the reject/relabel decisions.")
    p.add_argument("--op-testset", default=str(config.LABEL_BATCHES / "op_testset_v1.json"))
    p.add_argument("--ls-test-queue", default=str(config.LABEL_BATCHES / "ls_test_set_v5.json"))
    p.add_argument("--no-crops", action="store_true", help="Skip the classifier crops emitter.")
    p.add_argument("--gallery", action="store_true")
    p.add_argument("--gallery-n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output_dir is None:
        a.output_dir = str(config.PROJECTS_ROOT / f"sam2_synth_{a.mode}")
    return a


if __name__ == "__main__":
    main()

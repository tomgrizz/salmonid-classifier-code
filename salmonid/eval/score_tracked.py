"""In plain terms: this script checks how well a trained model spots which fish species are
present in each video. It reuses results that were already computed (the detections and the
fish-following step that links one fish across video frames), so it can quickly re-test different
settings without re-running anything on the GPU. It reports a per-species presence score — the
F1 score, a single number combining how often the model is right and how complete it is — which
the team uses as the go/no-go gate for deploying a model; the presence-F1 baselines quoted in the
paper (two-stage 0.894, single-stage DEIMv2 0.837) are produced here.

Tracked-op presence-F1 sweep — fast OFFLINE scorer over cached DEIMv2 detect-store aggregates.

Role in the pipeline
--------------------
data -> detect -> track -> **eval/count** -> results -> figures. This is an *evaluation*
knob-sweeper, not an inference step: it re-scores an already-computed per-track aggregate under
different post-processing settings (``min_frames`` / edge-filter / vote-mode / ``min_displacement``)
so config sweeps cost nothing — no GPU and no re-tracking. It is how the published single-stage
checkpoint's operating point is *selected* on the leak-clean tracked-op split.

Inputs
------
* ``--agg`` : a pickle written by ``_track_op_from_store.py`` — a dict ``{video_key: video_dict}``
  where each ``video_dict`` carries the frame size (``fw`` / ``fh``), the ``site`` (Ganaraska or
  Credit), the technician species ground-truth set (``tech_set``), and per-track ``boxes``
  (N x 4 xyxy) + ``pvs`` (N x 5 per-frame softmax species vectors). It lives under
  ``config.PROJECTS_ROOT`` (default ``.../deimv2_op_detstore/_agg_best_stg1.pkl``).
* Optional ``--keep-keys`` / ``--drop-keys`` : one-video-key-per-line files that RESTRICT or
  EXCLUDE videos before scoring. These implement the leak-clean op-dev/op-test split and drop the
  ~150 op_testset videos that leak into training, so the reported op-F1 is honest.

Outputs
-------
Prints, per species, a presence confusion (TP / FP / FN), precision / recall / F1, the MACRO-F1,
and the net-upstream migration tally per species. Nothing is written to disk — purely diagnostic.

Manuscript claim
----------------
Produces the tracked-op **presence-F1** used as the go/no-go operating-point gate for checkpoint
selection (baselines the sweep prints for context: two-stage 0.894, single-stage DEIMv2
mean/edge0 0.837). See docs/counting_and_movement.md and the deployment-readiness scorecard.

Environment
-----------
Runs in the BASE env (numpy only — no torch / boxmot). The heavy detection + tracking that
produced the ``--agg`` pickle runs in the DEIMv2 ``.venv_deim``; this scorer is downstream of it.

Usage
-----
    python -m salmonid.eval.score_tracked --agg <detect-store>/_agg_best_stg1.pkl \\
        --min-frames 5 --edge-margin-px 8 --vote-mode best-frame
    # or sweep the edge x vote-mode grid:
    python -m salmonid.eval.score_tracked --agg <detect-store>/_agg_best_stg1.pkl --sweep
"""
import argparse, pickle
from pathlib import Path
import numpy as np

SPECIES = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]  # mirrors salmonid.config.SPECIES
# Per-site upstream screen direction: which +/-x heading is UPSTREAM. Mirrors the sign
# convention in salmonid.config.UP_SIGN (Ganaraska upstream = +x/Right, Credit = -x/Left).
UP_DIR = {"Ganaraska": "Right", "Credit": "Left"}


def score(agg, min_frames, edge_px, vote_mode, min_disp):
    """Score one aggregate under a fixed post-processing config.

    Args:
        agg: {video_key: video_dict} as loaded from the --agg pickle. Each video_dict has
            fw/fh (frame size), site, tech_set (technician species present), and tracks
            {track_id: {"boxes": Nx4 xyxy, "pvs": Nx5 per-frame species softmax}}.
        min_frames: drop tracks with fewer than this many detections (short == spurious).
        edge_px: per-frame edge margin; frames whose box touches within edge_px of any border
            are excluded from the species vote (entering/leaving fish are partial and bias the
            classifier toward Atlantic). 0 disables the gate.
        vote_mode: "best-frame" votes with the single largest-area (nearest/clearest) frame;
            "mean" sums the per-frame species vectors across the kept frames.
        min_disp: minimum |delta_x| (px) for a track to count as up/down movement; below it the
            track is treated as milling and contributes to neither the up nor the down tally.

    Returns:
        (macro_f1, rows, net) where rows is a per-species
        (species, TP, FP, FN, precision, recall, F1) list and net maps species -> net upstream
        count (up minus down).

    Side effects: none (pure function; does not read or write files).
    """
    import collections
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    m_up = collections.Counter(); m_dn = collections.Counter()
    for vk, vd in agg.items():
        fw, fh, site = vd["fw"], vd["fh"], vd["site"]
        tech_set = set(vd["tech_set"]); model_set = set()
        for tid, t in vd["tracks"].items():
            boxes, pvs = t["boxes"], t["pvs"]
            n = len(boxes)
            if n < min_frames:
                continue
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            # edge gate per frame (does not vote, still part of track)
            if edge_px > 0:
                keep = ~((boxes[:, 0] <= edge_px) | (boxes[:, 1] <= edge_px)
                         | (boxes[:, 2] >= fw - edge_px) | (boxes[:, 3] >= fh - edge_px))
            else:
                keep = np.ones(n, bool)
            vpvs = pvs[keep]
            if vote_mode == "best-frame":
                vb = boxes[keep]
                if len(vb):
                    # largest box == fish nearest the camera / least occluded -> most reliable ID
                    areas = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
                    vote = vpvs[int(np.argmax(areas))]
                else:
                    vote = np.zeros(5, np.float32)
            else:
                vote = vpvs.sum(0) if len(vpvs) else np.zeros(5, np.float32)
            if vote.sum() == 0:                      # all frames edge-gated -> fall back
                vote = pvs.sum(0)
            if vote.sum() == 0:
                continue
            sp = int(np.argmax(vote)); model_set.add(SPECIES[sp])
            w = min(3, n)
            # net direction from centroid drift: mean-x of the last w frames minus the first w
            dx = float(np.mean(cx[-w:]) - np.mean(cx[:w]))
            if abs(dx) >= min_disp:                   # below min_disp == milling: counts neither way
                d = "Right" if dx > 0 else "Left"
                (m_up if d == UP_DIR.get(site) else m_dn)[sp] += 1
        for sp in SPECIES:
            if sp in model_set and sp in tech_set: tp[sp] += 1
            elif sp in model_set: fp[sp] += 1
            elif sp in tech_set: fn[sp] += 1
    f1s = []
    rows = []
    for sp in SPECIES:
        p = tp[sp] / max(tp[sp] + fp[sp], 1); r = tp[sp] / max(tp[sp] + fn[sp], 1)
        f1 = 2 * p * r / max(p + r, 1e-9); f1s.append(f1)
        rows.append((sp, tp[sp], fp[sp], fn[sp], p, r, f1))
    return float(np.mean(f1s)), rows, {SPECIES[s]: m_up[s] - m_dn[s] for s in range(5)}


def report(agg, min_frames, edge_px, vote_mode, min_disp):
    """Run score() for one config and print the per-species table + net tally to stdout.

    Returns the macro-F1 (so the --sweep loop can pick the best config).
    """
    macro, rows, net = score(agg, min_frames, edge_px, vote_mode, min_disp)
    print(f"\n== min_frames={min_frames} edge={edge_px} vote={vote_mode} min_disp={min_disp} ==")
    print(f"{'species':14s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    for sp, t, f, n, p, r, f1 in rows:
        print(f"{sp:14s} {t:>4d} {f:>4d} {n:>4d} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
    print(f"{'MACRO F1':14s} {macro:>46.3f}")
    print("net upstream:", net)
    return macro


def main():
    """CLI entry point: load the --agg pickle, apply the optional keep/drop key filters, then
    either sweep the edge x vote-mode grid (--sweep) or score the single supplied config.
    Prints results to stdout; returns nothing.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--edge-margin-px", type=int, default=0)
    ap.add_argument("--vote-mode", choices=["mean", "best-frame"], default="mean")
    ap.add_argument("--min-disp-px", type=float, default=300.0)
    ap.add_argument("--sweep", action="store_true", help="grid over edge x vote-mode")
    ap.add_argument("--drop-keys", default=None,
                    help="file of agg store_keys to EXCLUDE before scoring (one per line); used to drop "
                         "op_testset videos that leak into training (see _audit_op_testset_leak.py -> "
                         "op_testset_cropleak_keys.txt, 150 keys) so the op-F1 number is honest.")
    ap.add_argument("--keep-keys", default=None,
                    help="file of agg store_keys to RESTRICT scoring to (one per line); used to score on a "
                         "single half of the op_dev/op_test split (_build_op_devtest_split.py). Tune config "
                         "on op_dev_keys.txt, report the locked config once on op_test_keys.txt.")
    args = ap.parse_args()
    agg = pickle.load(open(args.agg, "rb"))
    print(f"loaded {len(agg)} videos from {Path(args.agg).name}")
    if args.keep_keys and Path(args.keep_keys).is_file():
        keep = {ln.strip() for ln in open(args.keep_keys) if ln.strip()}
        before = len(agg)
        agg = {k: v for k, v in agg.items() if k in keep}
        print(f"restricted to {len(agg)} videos in {Path(args.keep_keys).name} (from {before})")
    if args.drop_keys and Path(args.drop_keys).is_file():
        drop = {ln.strip() for ln in open(args.drop_keys) if ln.strip()}
        before = len(agg)
        agg = {k: v for k, v in agg.items() if k not in drop}
        print(f"dropped {before - len(agg)} leaked videos (op_testset train-leak) -> {len(agg)} remain")
    if args.sweep:
        best = (None, -1)
        for edge in (0, 8, 16):
            for vm in ("mean", "best-frame"):
                m = report(agg, args.min_frames, edge, vm, args.min_disp_px)
                if m > best[1]:
                    best = ((edge, vm), m)
        print(f"\nBEST: edge={best[0][0]} vote={best[0][1]} -> macro {best[1]:.3f}  "
              f"(baselines: two-stage 0.894 | Nano 0.826 | DEIMv2 mean/edge0 = 0.837)")
    else:
        report(agg, args.min_frames, args.edge_margin_px, args.vote_mode, args.min_disp_px)


if __name__ == "__main__":
    main()

"""In plain terms: this script checks the automated counter against a person. An independent
reviewer re-watched a sample of video clips without seeing the model's answers and recorded, for
each fish, its species and which way it swam; this script lines those human answers up against the
model's and reports how often they agree. The species-agreement figure it prints is the paper's
~94.7% blind-human agreement number, and it also breaks down the counting errors (fish the model
double-counted, missed, or sent the wrong direction).

Blind-review scorer: score the reference-standard (blind human recount) against the model.

Pipeline position
-----------------
``data -> detect -> track -> eval/count -> RESULTS``. This is a *results/eval* script: it consumes
the model's corpus tracks (already detected + tracked + adjudicated by a reviewer) and an
independent, blind human recount of a stratified clip sample, and emits the reference-standard
agreement numbers reported in the manuscript. It runs in the **base environment** (stdlib only:
csv/json/collections/random) — no torch / boxmot / DEIMv2 required.

What it computes (model vs. the reviewer's blind truth, on the reviewed clips)
-----------------------------------------------------------------------------
  * per-track **species accuracy** and **direction accuracy** on kept, fully-adjudicated tracks —
    the species figure here is the manuscript's **~94.7% blind-human** agreement number;
  * an **error decomposition**: milling over-call (model "up" / you stay-or-down), crowd
    under-count (real fish you saw the model missed), over-segmentation / non-fish (tracks you
    removed), and species-swap;
  * **net magnitude** (model vs. truth), reported both on the reviewed sample and
    **Horvitz-Thompson reweighted to the corpus** (design weight = stratum population / reviewed
    clips in that stratum), with **clip-bootstrap 95% CIs**;
  * **activity vs. net** (which is the better proxy), a **Credit 3-way** model/tech/truth table
    (per-clip tech from the Credit counter export), and a per-species rare-class breakdown.

Inputs
------
  * ``--recount``  reference_recount.csv — the reviewer's per-track adjudication (your_species,
    your_dir, removed, missed flags), keyed by video_id.
  * ``--sample``   reference_sample_2025.json — the stratified sampling frame (video_id, stratum);
    used to derive the HT design weights.
  * Credit per-clip tech CSVs (globbed, repo-relative) for the optional 3-way comparison.

Output
------
A formatted agreement report to stdout (no files are written).

  python -m salmonid.results.score_reference_sample
"""
import argparse, csv, json, collections, random
from pathlib import Path

from salmonid import config

# Corpus detect-store root for the 2025 blind-review sample (v2.6 single-stage tracks).
CORP = config.PROJECTS_ROOT / "deimv2_x_v26_corpus2025"
DIRV = {"up": 1, "down": -1, "stay": 0, "": 0}


def net(rows):
    """Sum a sequence of direction strings to a signed net movement (up=+1, down=-1, else 0)."""
    return sum(DIRV.get(r, 0) for r in rows)


def boot_ci(clip_vals, stat, n=2000, seed=13):
    """clip_vals: list of per-clip objects; stat(list)->float. Returns (lo, hi) 95% by resampling clips."""
    rng = random.Random(seed); N = len(clip_vals); out = []
    for _ in range(n):
        samp = [clip_vals[rng.randrange(N)] for _ in range(N)]
        try:
            out.append(stat(samp))
        except Exception:
            pass
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))]) if out else (float("nan"), float("nan"))


def main():
    """Load the recount + sampling frame, compute every model-vs-truth agreement statistic, and
    print the reference-standard report to stdout. Side effects: reads the two input files (and the
    globbed Credit tech CSVs) and writes only to stdout."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", default=str(CORP / "reference_recount.csv"))
    ap.add_argument("--sample", default=str(CORP / "reference_sample_2025.json"))
    a = ap.parse_args()

    sample = {c["video_id"]: c for c in json.load(open(a.sample))}
    rows = list(csv.DictReader(open(a.recount, encoding="utf-8")))
    by_clip = collections.defaultdict(list)
    for r in rows:
        by_clip[r["video_id"]].append(r)
    reviewed = [v for v in by_clip if v in sample]

    # HT weight per clip = stratum population / reviewed clips in that stratum
    strat_pop = {}
    for c in sample.values():
        strat_pop[c["stratum"]] = strat_pop.get(c["stratum"], 0) + 1
    rev_per_strat = collections.Counter(sample[v]["stratum"] for v in reviewed)
    w = {v: strat_pop[sample[v]["stratum"]] / rev_per_strat[sample[v]["stratum"]] for v in reviewed}

    # per-clip records
    clips = []
    kept_tracks = []                          # (model_sp, your_sp, model_dir, your_dir)
    n_removed = n_missed = 0
    for v in reviewed:
        rs = by_clip[v]
        mt = [r for r in rs if r.get("missed") != "1"]
        kept = [r for r in mt if r.get("removed") != "1"]
        removed = [r for r in mt if r.get("removed") == "1"]
        missed = [r for r in rs if r.get("missed") == "1"]
        n_removed += len(removed); n_missed += len(missed)
        for r in kept:
            if r.get("your_species") and r.get("your_dir"):
                kept_tracks.append((r["model_species"], r["your_species"], r["model_dir"], r["your_dir"]))
        model_net = net([r["model_dir"] for r in mt])                      # model counts ALL its tracks
        truth_net = net([r["your_dir"] for r in kept] + [r["your_dir"] for r in missed])  # your real fish
        clips.append({"v": v, "w": w[v], "model_net": model_net, "truth_net": truth_net,
                      "n_model": len(mt), "n_kept": len(kept), "n_missed": len(missed), "n_removed": len(removed),
                      "kept": [(r["model_species"], r["your_species"], r["model_dir"], r["your_dir"])
                               for r in kept if r.get("your_species") and r.get("your_dir")]})

    print(f"# REFERENCE-STANDARD SCORING — {len(reviewed)} clips reviewed (of {len(sample)} sampled)")
    print(f"model tracks: {sum(c['n_model'] for c in clips)}  | you removed (dup/non-fish): {n_removed}  "
          f"| you added (model MISSED): {n_missed}")

    # ---- species + direction accuracy (on kept, fully-adjudicated tracks) ----
    sp_ok = sum(1 for m, y, _, _ in kept_tracks if m == y)
    unsure = sum(1 for _, y, _, _ in kept_tracks if y == "unsure")
    dir_ok = sum(1 for _, _, m, y in kept_tracks if m == y)
    K = len(kept_tracks)
    print(f"\n=== per-track agreement (n={K} adjudicated kept tracks; {unsure} 'unsure') ===")
    # species model==you over kept tracks = the manuscript's ~94.7% blind-human species agreement.
    print(f"  species: model==you  {sp_ok}/{K} = {100*sp_ok/K:.1f}%")
    print(f"  direction: model==you {dir_ok}/{K} = {100*dir_ok/K:.1f}%")
    # species confusion (model -> you), only mismatches
    mism = collections.Counter((m, y) for m, y, _, _ in kept_tracks if m != y)
    if mism:
        print("  species mismatches (model -> you):")
        for (m, y), c in mism.most_common():
            print(f"    {m:14s} -> {y:14s}  x{c}")
    dmism = collections.Counter((m, y) for _, _, m, y in kept_tracks if m != y)
    if dmism:
        print("  direction mismatches (model -> you):", dict(dmism))

    # ---- error decomposition (model up-calls) ----
    up_tracks = [(m, y, md, yd) for m, y, md, yd in kept_tracks if md == "up"]
    milling = sum(1 for _, _, md, yd in up_tracks if yd != "up")
    print(f"\n=== error decomposition ===")
    print(f"  MILLING over-call (model up, you stay/down): {milling}/{len(up_tracks)} up-calls "
          f"= {100*milling/max(1,len(up_tracks)):.0f}%")
    print(f"  CROWD under-count (fish you saw, model missed): {n_missed} over {len(reviewed)} clips "
          f"= {n_missed/len(reviewed):.2f}/clip")
    print(f"  OVER-SEG / non-fish (you removed): {n_removed} ({100*n_removed/sum(c['n_model'] for c in clips):.1f}% of model tracks)")
    print(f"  SPECIES-swap: {K-sp_ok}/{K} = {100*(K-sp_ok)/K:.1f}%")

    # ---- net magnitude: model vs truth, reviewed + reweighted ----
    m_raw = sum(c["model_net"] for c in clips); t_raw = sum(c["truth_net"] for c in clips)
    m_rw = sum(c["w"] * c["model_net"] for c in clips); t_rw = sum(c["w"] * c["truth_net"] for c in clips)
    print(f"\n=== NET magnitude: model vs YOUR truth ===")
    print(f"  reviewed sample:  model {m_raw}  vs  you {t_raw}   -> {100*(m_raw-t_raw)/t_raw:+.1f}%" if t_raw else "  (truth net 0)")
    if t_rw:
        # pe = HT-reweighted percent error of net (model vs truth); boot_ci resamples whole clips.
        pe = lambda cs: 100 * (sum(c['w']*c['model_net'] for c in cs) - sum(c['w']*c['truth_net'] for c in cs)) / max(1e-9, sum(c['w']*c['truth_net'] for c in cs))
        lo, hi = boot_ci(clips, pe)
        print(f"  reweighted (corpus HT): model {m_rw:.0f}  vs  you {t_rw:.0f}   -> {pe(clips):+.1f}%  (95% CI {lo:+.0f}..{hi:+.0f}%)")
    # per-clip exact-net agreement
    exact = sum(1 for c in clips if c["model_net"] == c["truth_net"])
    print(f"  per-clip net agreement (model net == your net): {exact}/{len(clips)} = {100*exact/len(clips):.0f}% of clips")

    # ---- ACTIVITY vs NET: is activity the better proxy? (both rivers, model vs truth) ----
    m_act = sum(c["n_model"] for c in clips)                 # model reports every track
    t_act = sum(c["n_kept"] + c["n_missed"] for c in clips)  # real fish = kept + missed (removed = non-fish)
    print(f"\n=== ACTIVITY vs NET — model vs YOUR truth (both rivers, {len(clips)} clips) ===")
    print(f"  ACTIVITY (total fish):  model {m_act}  vs  you {t_act}   -> {100*(m_act-t_act)/t_act:+.1f}%"
          f"   [{'clean detection-recall gap' if m_act < t_act else 'over-detect'}]")
    print(f"  NET (up-down):          model {m_raw}  vs  you {t_raw}   -> {100*(m_raw-t_raw)/t_raw:+.1f}%"
          f"   [inherits milling/direction ambiguity]")
    act_agree = sum(1 for c in clips if c["n_model"] == c["n_kept"] + c["n_missed"])
    print(f"  per-clip ACTIVITY agreement: {act_agree}/{len(clips)} = {100*act_agree/len(clips):.0f}% of clips "
          f"(vs net {100*exact/len(clips):.0f}%)")

    # ---- Credit per-clip 3-way: model vs truth vs TECH (activity + net) ----
    ctech = collections.defaultdict(lambda: {"act": 0, "net": 0})
    SIGN = {"up": 1, "down": -1, "stay": 0, "upstream": 1, "downstream": -1}
    for f in __import__("glob").glob("docs/Riverwatcher_2025/Credit_Counter_Analysis_2025/data/raw_data/fish_counts_export_*.csv"):
        for r in csv.DictReader(open(f, encoding="latin-1")):
            if str(r.get("false_trigger", "")).strip().lower() in ("1", "true", "yes"):
                continue
            if r.get("species", "").strip().lower() in ("non fish", "non-fish", "nonfish", ""):
                continue
            try:
                cnt = int(float(r.get("count") or 1))
            except ValueError:
                cnt = 1
            k = "/".join(r["video_rel"].replace("\\", "/").split("/")[-2:]).strip().lower()
            ctech[k]["act"] += cnt; ctech[k]["net"] += cnt * SIGN.get(r.get("movement", "").strip().lower(), 0)

    def kf(v):
        """Normalise a video path/id to a 'site_or_dir/file' key for joining to the tech export."""
        return "/".join(v.replace("\\", "/").split("/")[-2:]).strip().lower()
    three = [c for c in clips if c["v"].split("/", 1)[0] == "Credit" and kf(c["v"]) in ctech]
    if three:
        mA = sum(c["n_model"] for c in three); tA = sum(c["n_kept"] + c["n_missed"] for c in three)
        techA = sum(ctech[kf(c["v"])]["act"] for c in three)
        mN = sum(c["model_net"] for c in three); tN = sum(c["truth_net"] for c in three)
        techN = sum(ctech[kf(c["v"])]["net"] for c in three)
        print(f"\n=== CREDIT per-clip 3-way vs TRUTH (n={len(three)} clips w/ per-clip tech) ===")
        print(f"  {'':10s}{'model':>8s}{'tech':>8s}{'YOU(truth)':>12s}")
        print(f"  {'ACTIVITY':10s}{mA:8d}{techA:8d}{tA:12d}    model {100*(mA-tA)/max(1,tA):+.0f}% | tech {100*(techA-tA)/max(1,tA):+.0f}% vs truth")
        print(f"  {'NET':10s}{mN:8d}{techN:8d}{tN:12d}    model {100*(mN-tN)/max(1,tN):+.0f}% | tech {100*(techN-tN)/max(1,tN):+.0f}% vs truth")
        print(f"  (Ganaraska per-clip tech unavailable — 87% video_rel=NA; daily-only)")

    # ---- per-species (rare-class check) ----
    print(f"\n=== per-species (your truth species; kept tracks) ===")
    bysp = collections.defaultdict(lambda: [0, 0])   # your_sp -> [n, model-agrees]
    for m, y, _, _ in kept_tracks:
        bysp[y][0] += 1; bysp[y][1] += (m == y)
    for sp in ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout", "unsure"]:
        if bysp[sp][0]:
            n, ok = bysp[sp]
            print(f"  {sp:14s} n={n:3d}  model-correct {ok}/{n} ({100*ok/n:.0f}%)")


if __name__ == "__main__":
    main()

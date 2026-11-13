"""In plain terms: to check the model against a person rather than against the technicians'
field record, a reviewer watched a sample of 2025 video clips without being shown the model's
species, and wrote down every fish: its species and which way it went. This script lines the
reviewer's record up against the model tracks that were shown in the review tool and prints
how often the two agree on species and on direction, how many fish the model missed or
counted twice, and how the net count compares. It scores the reviewer against the tracks of
the EARLIER model (v2.6), because those were the tracks in the review tool. The blind-review
figures in the manuscript are for the published model (v2.7) and were recomputed from the
same reviewer record by the blind-review recomputation in ``salmonid/paper/``; this script
does not produce them.

What this script gives and what the manuscript reports
------------------------------------------------------
* **This script** (earlier model's tracks, the reviewer's verdict on each track): species
  agreement 180 of 190 tracks = 94.7% (run on the review files on 2026-11-10).
* **Manuscript Section 3.2** (published model's tracks, paired with the reviewer's fish
  within each clip by species): 196 of 208 paired fish = 94.2%, kappa 0.90, from the
  blind-review recomputation in ``salmonid/paper/``. The reviewer's record is the same;
  the model tracks and the pairing method differ, so the two figures are not expected to
  be equal.

Only the species-agreement idea carries over to the manuscript. The other sections printed
here (direction agreement, the error breakdown, the net and activity comparisons, the
three-way table with the Credit River technician record) were working diagnostics for the
earlier model and are not quoted in the manuscript.

How the reviewer's record is read
---------------------------------
The review tool listed the model's tracks for a clip. For each one the reviewer either kept
it, giving a species and a direction, or removed it (a duplicate of another track, or not a
fish), and could add fish the model had missed. In ``reference_recount.csv`` each row is one
such fish, with the columns ``model_species``, ``model_dir`` (the model's call),
``your_species``, ``your_dir`` (the reviewer's call), ``removed`` and ``missed``.

* Species and direction agreement are computed over kept tracks for which the reviewer gave
  both a species and a direction. A fish the reviewer marked "unsure" stays in and counts
  as a disagreement.
* The model's net count of a clip is upstream minus downstream over all of its tracks; the
  reviewer's is the same over kept plus missed fish.

Limitations a reader should know
--------------------------------
* **The "reweighted" net line is not a whole-season estimate.** 206 clips were sampled and
  155 were reviewed. The weight used here is, for each stratum, clips sampled divided by
  clips reviewed. It corrects only for the sampled clips that were not reviewed. The weight
  that would scale the sample up to the season (written by ``build_reference_sample`` into
  each clip's ``weight`` field) is not used, and would not be a valid design weight anyway
  because of the rare-species top-up (see that module). The line is labelled accordingly.
* **The agreement percentages are unweighted** rates over the reviewed tracks, from a sample
  that deliberately over-represents clips with rare species.
* **The reviewed clips are not an even subset of the sampled ones.** Of the 51 sampled clips
  that were not reviewed, 48 are late-phase Ganaraska River clips: 62 were drawn there and
  14 were reviewed. The other ten groups were reviewed completely or nearly so.
* **Rare-species clips were chosen by the model's own call.** The top-up adds clips in which
  the (earlier) model reported the species. All six fish the reviewer identified as
  Atlantic Salmon are in such clips. Agreement on those fish therefore shows that the
  reviewer confirmed Atlantic Salmon the model had reported; it cannot show how many
  Atlantic Salmon the model called something else, because clips were not selected for that.
* **The intervals** resample whole clips 2,000 times (seed 13).
* The three-way table needs the Credit River technician export; if it is not found the
  table is skipped and a line says so.

Inputs
------
All Tier 2 (not stored in git), in ``config.BLIND_REVIEW_DIR``:

* ``--recount``: ``reference_recount.csv``, the reviewer's fish-by-fish record.
* ``--sample``: ``reference_sample_2025.json``, the sampled clips with their strata.
* Optional: the Credit River technician review export under ``config.TECH_2025_DIR``.

Outputs
-------
A report printed to the screen. Nothing is written to disk.

Behaviour changes relative to the original research script
-----------------------------------------------------------
Every number printed is computed as before (checked on the review files: identical output).

* The interval routine discarded any resample on which the statistic raised an error,
  without saying so. The error is now allowed to stop the run. With the statistic used here
  no resample fails, so the interval is unchanged.
* The technician export was looked for relative to the folder the command was run from, and
  the three-way table vanished silently if it was not there. It is now looked for under
  ``config.TECH_2025_DIR`` and a line is printed when it is not found. Rows whose count is
  not a number were silently counted as one fish; they still are, and their number is printed.
* Missing review files stop the run with one sentence naming the file. A run with no
  reviewed clip, or no fully reviewed track, stops instead of dividing by zero.
* Printed wording only: "you" is now "reviewer", and the "reweighted (corpus HT)" line is
  labelled for what it is.

Feeds
-----
Development check on the earlier model's tracks; no manuscript number. (The manuscript's
blind-review figures, Section 3.2 and 4.4, come from ``salmonid/paper/``.)

Run
---
::

    python -m salmonid.results.score_reference_sample

Base Python, standard library only.

Tier
----
2 (needs the review files; they hold no footage but are not stored in git).

Status
------
Supporting check (the original scorer of the blind review; superseded for the published
figures by the recomputation in ``salmonid/paper/``).
"""
import argparse, csv, glob, json, collections, random
from pathlib import Path

from salmonid import config

# Folder holding the two blind-review files. The clips were sampled from the earlier (v2.6)
# 2025 run, so by default this is that run's folder (see config.BLIND_REVIEW_DIR).
CORP = config.BLIND_REVIEW_DIR
DIRV = {"up": 1, "down": -1, "stay": 0, "": 0}


def net(rows):
    """Net movement of a list of directions: +1 per "up", -1 per "down", 0 otherwise."""
    return sum(DIRV.get(r, 0) for r in rows)


def boot_ci(clip_vals, stat, n=2000, seed=13):
    """95% interval of ``stat`` by redrawing whole clips with replacement, ``n`` times.

    Whole clips are redrawn (not single fish) because the fish in one clip are not
    independent observations: they share the same water, light and crowding.
    """
    rng = random.Random(seed); N = len(clip_vals); out = []
    for _ in range(n):
        samp = [clip_vals[rng.randrange(N)] for _ in range(N)]
        out.append(stat(samp))
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))]) if out else (float("nan"), float("nan"))


def _pct(a, b):
    """Format the percentage difference of ``a`` from ``b``; "n/a" when ``b`` is zero."""
    return f"{100*(a-b)/b:+.1f}%" if b else "n/a (reference is 0)"


def main():
    """Read the reviewer's record and the sample, and print the agreement report."""
    ap = argparse.ArgumentParser(
        description="Score the blind reviewer's record against the model tracks shown in the "
                    "review tool (earlier model, v2.6). The manuscript's blind-review figures "
                    "for the published model come from salmonid/paper/.")
    ap.add_argument("--recount", default=str(CORP / "reference_recount.csv"),
                    help="the reviewer's fish-by-fish record")
    ap.add_argument("--sample", default=str(CORP / "reference_sample_2025.json"),
                    help="the sampled clips with their strata")
    a = ap.parse_args()

    config.require(a.sample, "the blind-review sample (reference_sample_2025.json)",
                   "SALMONID_BLIND_REVIEW_DIR")
    config.require(a.recount, "the blind reviewer's record (reference_recount.csv)",
                   "SALMONID_BLIND_REVIEW_DIR")
    with open(a.sample, encoding="utf-8") as f:
        sample = {c["video_id"]: c for c in json.load(f)}
    with open(a.recount, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_clip = collections.defaultdict(list)
    for r in rows:
        by_clip[r["video_id"]].append(r)
    reviewed = [v for v in by_clip if v in sample]
    not_in_sample = len(by_clip) - len(reviewed)
    if not_in_sample:
        print(f"NOTE: {not_in_sample} clips in the reviewer's record are not in the sample "
              f"file and are left out.")
    if not reviewed:
        raise SystemExit("Nothing to score: none of the clips in the reviewer's record is in "
                         "the sample file.")

    # Weight per clip = clips SAMPLED in its stratum / clips REVIEWED in its stratum. This
    # makes up for sampled clips that were not reviewed. It does not scale to the season:
    # strat_pop counts the sample, not all the season's clips.
    strat_pop = {}
    for c in sample.values():
        strat_pop[c["stratum"]] = strat_pop.get(c["stratum"], 0) + 1
    rev_per_strat = collections.Counter(sample[v]["stratum"] for v in reviewed)
    w = {v: strat_pop[sample[v]["stratum"]] / rev_per_strat[sample[v]["stratum"]] for v in reviewed}

    # One record per reviewed clip.
    clips = []
    kept_tracks = []                          # (model species, reviewer species, model dir, reviewer dir)
    n_removed = n_missed = 0
    for v in reviewed:
        rs = by_clip[v]
        mt = [r for r in rs if r.get("missed") != "1"]            # the model's tracks
        kept = [r for r in mt if r.get("removed") != "1"]         # ...that the reviewer accepted as a fish
        removed = [r for r in mt if r.get("removed") == "1"]      # ...that the reviewer struck out
        missed = [r for r in rs if r.get("missed") == "1"]        # fish the reviewer added
        n_removed += len(removed); n_missed += len(missed)
        for r in kept:
            if r.get("your_species") and r.get("your_dir"):
                kept_tracks.append((r["model_species"], r["your_species"], r["model_dir"], r["your_dir"]))
        model_net = net([r["model_dir"] for r in mt])                      # the model counts all of its tracks
        truth_net = net([r["your_dir"] for r in kept] + [r["your_dir"] for r in missed])  # the reviewer's fish
        clips.append({"v": v, "w": w[v], "model_net": model_net, "truth_net": truth_net,
                      "n_model": len(mt), "n_kept": len(kept), "n_missed": len(missed), "n_removed": len(removed),
                      "kept": [(r["model_species"], r["your_species"], r["model_dir"], r["your_dir"])
                               for r in kept if r.get("your_species") and r.get("your_dir")]})

    print(f"# BLIND-REVIEW SCORING (earlier model's tracks) — {len(reviewed)} clips reviewed "
          f"(of {len(sample)} sampled)")
    n_model_tracks = sum(c['n_model'] for c in clips)
    print(f"model tracks: {n_model_tracks}  | removed by the reviewer (duplicate/not a fish): {n_removed}  "
          f"| added by the reviewer (model missed): {n_missed}")

    # ---- species and direction agreement (kept tracks with both a species and a direction) ----
    sp_ok = sum(1 for m, y, _, _ in kept_tracks if m == y)
    unsure = sum(1 for _, y, _, _ in kept_tracks if y == "unsure")
    dir_ok = sum(1 for _, _, m, y in kept_tracks if m == y)
    K = len(kept_tracks)
    if K == 0:
        raise SystemExit("Nothing to score: no kept track has both a reviewer species and a "
                         "reviewer direction.")
    print(f"\n=== per-track agreement (n={K} kept tracks with species and direction; {unsure} 'unsure') ===")
    # "unsure" is not equal to any model species, so it counts as a disagreement.
    print(f"  species: model == reviewer  {sp_ok}/{K} = {100*sp_ok/K:.1f}%")
    print(f"  direction: model == reviewer {dir_ok}/{K} = {100*dir_ok/K:.1f}%")
    mism = collections.Counter((m, y) for m, y, _, _ in kept_tracks if m != y)
    if mism:
        print("  species mismatches (model -> reviewer):")
        for (m, y), c in mism.most_common():
            print(f"    {m:14s} -> {y:14s}  x{c}")
    dmism = collections.Counter((m, y) for _, _, m, y in kept_tracks if m != y)
    if dmism:
        print("  direction mismatches (model -> reviewer):", dict(dmism))

    # ---- where the model's errors come from ----
    # "Milling" = a fish that moves about in front of the camera without passing. If the
    # model calls it an upstream passage and the reviewer does not, the net count is inflated.
    up_tracks = [(m, y, md, yd) for m, y, md, yd in kept_tracks if md == "up"]
    milling = sum(1 for _, _, md, yd in up_tracks if yd != "up")
    print(f"\n=== error breakdown ===")
    print(f"  MILLING over-call (model up, reviewer stay/down): {milling}/{len(up_tracks)} up-calls "
          f"= {100*milling/max(1,len(up_tracks)):.0f}%")
    print(f"  MISSED fish (seen by the reviewer, no model track): {n_missed} over {len(reviewed)} clips "
          f"= {n_missed/len(reviewed):.2f}/clip")
    print(f"  DUPLICATE or not-a-fish tracks (removed by the reviewer): {n_removed} "
          f"({100*n_removed/max(1, n_model_tracks):.1f}% of model tracks)")
    print(f"  SPECIES disagreement: {K-sp_ok}/{K} = {100*(K-sp_ok)/K:.1f}%")

    # ---- net count: model vs reviewer, on the reviewed clips and reweighted ----
    m_raw = sum(c["model_net"] for c in clips); t_raw = sum(c["truth_net"] for c in clips)
    m_rw = sum(c["w"] * c["model_net"] for c in clips); t_rw = sum(c["w"] * c["truth_net"] for c in clips)
    print(f"\n=== NET count (up minus down): model vs reviewer ===")
    print(f"  reviewed clips:  model {m_raw}  vs  reviewer {t_raw}   -> {100*(m_raw-t_raw)/t_raw:+.1f}%" if t_raw else "  (reviewer net is 0)")
    if t_rw:
        # Percentage difference of the weighted net counts; the interval redraws whole clips.
        pe = lambda cs: 100 * (sum(c['w']*c['model_net'] for c in cs) - sum(c['w']*c['truth_net'] for c in cs)) / max(1e-9, sum(c['w']*c['truth_net'] for c in cs))
        lo, hi = boot_ci(clips, pe)
        print(f"  reweighted to the {len(sample)} sampled clips (NOT to the season): model {m_rw:.0f}  vs  "
              f"reviewer {t_rw:.0f}   -> {pe(clips):+.1f}%  (95% CI {lo:+.0f}..{hi:+.0f}%)")
    exact = sum(1 for c in clips if c["model_net"] == c["truth_net"])
    print(f"  clips where the two net counts are equal: {exact}/{len(clips)} = {100*exact/len(clips):.0f}% of clips")

    # ---- activity (every fish counted once) beside net ----
    m_act = sum(c["n_model"] for c in clips)                 # the model reports every track
    t_act = sum(c["n_kept"] + c["n_missed"] for c in clips)  # the reviewer's fish: kept + missed
    print(f"\n=== ACTIVITY vs NET — model vs reviewer (both rivers, {len(clips)} clips) ===")
    print(f"  ACTIVITY (all fish):    model {m_act}  vs  reviewer {t_act}   -> {_pct(m_act, t_act)}"
          f"   [{'model finds fewer fish' if m_act < t_act else 'model finds as many or more'}]")
    print(f"  NET (up minus down):    model {m_raw}  vs  reviewer {t_raw}   -> {_pct(m_raw, t_raw)}"
          f"   [also depends on the direction calls]")
    act_agree = sum(1 for c in clips if c["n_model"] == c["n_kept"] + c["n_missed"])
    print(f"  clips where the two fish counts are equal: {act_agree}/{len(clips)} = {100*act_agree/len(clips):.0f}% of clips "
          f"(net: {100*exact/len(clips):.0f}%)")

    # ---- Credit River: model, technician and reviewer side by side, per clip ----
    ctech = collections.defaultdict(lambda: {"act": 0, "net": 0})
    SIGN = {"up": 1, "down": -1, "stay": 0, "upstream": 1, "downstream": -1}
    tech_glob = str(config.TECH_2025_DIR / "Credit_Counter_Analysis_2025" / "data" / "raw_data"
                    / "fish_counts_export_*.csv")
    tech_files = sorted(glob.glob(tech_glob))
    n_bad_count = 0
    for f in tech_files:
        with open(f, newline="", encoding="latin-1") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("false_trigger", "")).strip().lower() in ("1", "true", "yes"):
                    continue
                if r.get("species", "").strip().lower() in ("non fish", "non-fish", "nonfish", ""):
                    continue
                try:
                    cnt = int(float(r.get("count") or 1))
                except ValueError:
                    cnt = 1                  # count is not a number: treated as one fish
                    n_bad_count += 1
                # A clip is identified by its recording folder and file name.
                k = "/".join(r["video_rel"].replace("\\", "/").split("/")[-2:]).strip().lower()
                ctech[k]["act"] += cnt; ctech[k]["net"] += cnt * SIGN.get(r.get("movement", "").strip().lower(), 0)

    def kf(v):
        """Reduce a video path to 'recording folder/file name', the key used for the technician export."""
        return "/".join(v.replace("\\", "/").split("/")[-2:]).strip().lower()
    three = [c for c in clips if c["v"].split("/", 1)[0] == "Credit" and kf(c["v"]) in ctech]
    if not tech_files:
        print(f"\n=== CREDIT three-way table SKIPPED: no technician export found at {tech_glob} "
              f"(set SALMONID_TECH_2025_DIR) ===")
    elif not three:
        print("\n=== CREDIT three-way table SKIPPED: no reviewed Credit River clip appears in the "
              "technician export ===")
    if n_bad_count:
        print(f"  NOTE: {n_bad_count} technician rows had a non-numeric count and were taken as 1 fish.")
    if three:
        mA = sum(c["n_model"] for c in three); tA = sum(c["n_kept"] + c["n_missed"] for c in three)
        techA = sum(ctech[kf(c["v"])]["act"] for c in three)
        mN = sum(c["model_net"] for c in three); tN = sum(c["truth_net"] for c in three)
        techN = sum(ctech[kf(c["v"])]["net"] for c in three)
        print(f"\n=== CREDIT per-clip three-way comparison (n={len(three)} clips found in the technician export) ===")
        print(f"  {'':10s}{'model':>8s}{'tech':>8s}{'reviewer':>12s}")
        print(f"  {'ACTIVITY':10s}{mA:8d}{techA:8d}{tA:12d}    model {100*(mA-tA)/max(1,tA):+.0f}% | tech {100*(techA-tA)/max(1,tA):+.0f}% vs reviewer")
        print(f"  {'NET':10s}{mN:8d}{techN:8d}{tN:12d}    model {100*(mN-tN)/max(1,tN):+.0f}% | tech {100*(techN-tN)/max(1,tN):+.0f}% vs reviewer")
        print(f"  (No such table for the Ganaraska River: most rows of its technician export are "
              f"not linked to a video.)")

    # ---- per species, by the reviewer's species ----
    print(f"\n=== per species (reviewer's species; kept tracks) ===")
    bysp = collections.defaultdict(lambda: [0, 0])   # reviewer species -> [n, n where the model agrees]
    for m, y, _, _ in kept_tracks:
        bysp[y][0] += 1; bysp[y][1] += (m == y)
    for sp in ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout", "unsure"]:
        if bysp[sp][0]:
            n, ok = bysp[sp]
            print(f"  {sp:14s} n={n:3d}  model agrees {ok}/{n} ({100*ok/n:.0f}%)")


if __name__ == "__main__":
    main()

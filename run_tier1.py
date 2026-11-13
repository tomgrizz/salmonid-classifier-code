"""In plain terms: this is the one command to run first. It re-computes every result in the
paper that can be re-computed from the small data files stored in this repository (no video,
no trained model, no graphics card), in the order the paper presents them, and then checks
each re-computed result against the published one and prints a PASS / FAIL table.

Run it from the repository root, after installing the four packages in requirements.txt:

    python run_tier1.py              # run every step, then verify           (about 1 minute)
    python run_tier1.py --lock-check # also report whether your package versions match the lock file
    python run_tier1.py --list       # show the steps and what each one feeds, run nothing
    python run_tier1.py --only 1 3   # run just steps 1 and 3
    python run_tier1.py --no-figures # statistics only (skips the figure steps)

What "Tier 1" means
-------------------
Tier 1 = everything that runs from the files committed under data/. That is the held-out
species score, every agreement statistic, and every figure that is drawn from count tables.
Tier 2 = the steps that need large external files (the trained model, the raw video, the
stored detections); those are documented in docs/REPRODUCE.md and are not run here.

Where things go
---------------
* data/         the published tables, exactly as the manuscript quotes them. NEVER written to.
* output/tier1/ and output/paper/  what this script regenerates (git-ignored). Safe to delete.
* figures/      regenerated figures (git-ignored).

How the check works
-------------------
Every random step in this repository is seeded, so a re-run is expected to reproduce the
published tables exactly, not approximately. For each table the script compares the
regenerated file with the committed one, value by value. It also re-derives the headline
numbers quoted in the manuscript (for example macro-F1 0.961, or the Bland-Altman limits)
and compares them with the printed values listed in EXPECTED below.

The script only starts other modules in order; all of the analysis lives in the salmonid/
package. Each step prints the exact command it runs, so any step can be repeated by hand.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
OUT = REPO / "output" / "tier1"
PAPER_OUT = REPO / "output" / "paper"     # where the salmonid.paper scripts write by default
FIGS = REPO / "figures"

# ---------------------------------------------------------------------------------------
# The steps, in the order the manuscript presents the results.
#   module : what is run (python -m <module> <args>)
#   feeds  : the manuscript item(s) it reproduces
#   tables : files it writes that must equal the committed copy under data/
#   figures: files it writes to figures/
#   outdir : where its tables land (output/tier1/ unless stated)
# Steps 1 to 9 are the classification score, the count tables, the agreement statistics and
# the figures. Steps 10 to 18 are the remaining manuscript tables and in-text statistics
# (salmonid/paper/).
# ---------------------------------------------------------------------------------------
STEPS = [
    dict(n=1, title="Held-out species score",
         module="salmonid.eval.score_testset_v6", args=[],
         feeds="Table 1 (point estimates), Figure 2 counts, Section 3.1 error breakdown",
         tables=[], figures=[]),
    dict(n=2, title="Confidence intervals for the species score",
         module="salmonid.results.classification_ci", args=[],
         feeds="Table 1 (95% intervals; macro-F1 0.961 [0.939, 0.980])",
         tables=["classification_ci.csv"], figures=[]),
    dict(n=3, title="Rebuild the canonical daily and seasonal count tables",
         module="salmonid.results.build_canonical_results",
         args=["--sweep-root", "data", "--recovered-dir", "data/recovered_2020",
               "--out-dir", "output/tier1"],
         feeds="data/canonical_daily.csv and canonical_summary.csv, which every count "
               "statistic and count figure below is computed from",
         tables=["canonical_daily.csv", "canonical_summary.csv"], figures=[]),
    dict(n=4, title="Concordance of daily counts (Lin's CCC) + Bland-Altman",
         module="salmonid.results.compute_concordance", args=[],
         feeds="Table 5, Figure 6, Figure S3",
         tables=["concordance_ccc.csv"],
         figures=["fig_concordance.png", "fig_concordance.pdf",
                  "fig_bland_altman.png", "fig_bland_altman.pdf"]),
    dict(n=5, title="Agreement statistics that respect the structure of daily counts",
         module="salmonid.results.agreement_stats", args=[],
         feeds="Sections 3.3 and 3.5 (per-river Bland-Altman, count bands, moving-block "
               "intervals, inter-annual agreement, Spearman rho in Table 5)",
         tables=["agreement_bland_altman.csv", "agreement_ba_by_count_band.csv",
                 "agreement_ccc_block.csv", "agreement_interannual.csv"], figures=[]),
    dict(n=6, title="Leave-one-year-out check of the upstream/downstream rule",
         module="salmonid.count.net_rule_loyo", args=[],
         feeds="Section 3.4 (50/300 px rule chosen in 4 of 4 folds; 3.6% error)",
         tables=["net_rule_loyo.csv"], figures=[]),
    dict(n=7, title="Figure: sensitivity of the count to the rule thresholds",
         module="salmonid.figures.render_sensitivity_figure", args=[],
         feeds="Section 3.4 (alternatives 10% to 26%); supporting figure",
         tables=["net_rule_sensitivity.json"], figures=["fig_net_rule_sensitivity.png"],
         is_figure=True),
    dict(n=8, title="Figures: confusion matrix and 2025 cumulative counts",
         module="salmonid.figures.render_manuscript_figures", args=[],
         feeds="Figure 2, Figure S2",
         tables=[],
         figures=["fig_confusion_heldout.png", "fig_confusion_heldout.pdf",
                  "fig_daily_net_2025_credit.png", "fig_daily_net_2025_credit.pdf",
                  "fig_daily_net_2025_ganaraska.png", "fig_daily_net_2025_ganaraska.pdf"],
         is_figure=True),
    dict(n=9, title="Figures: annual counts, run timing, sanctuary window",
         module="salmonid.figures.render_hero_regulation", args=[],
         feeds="Figure 4, Figure 5, Figure 7",
         tables=[],
         figures=["fig_hero_performance.png", "fig_hero_performance.pdf",
                  "fig_species_phenology.png", "fig_species_phenology.pdf",
                  "fig_ganaraska_sanctuary.png", "fig_ganaraska_sanctuary.pdf"],
         is_figure=True),
    # ---- the remaining manuscript tables and in-text statistics (salmonid/paper/) ----------
    dict(n=10, title="2025 per-video species agreement and Cohen's kappa",
         module="salmonid.paper.pervideo_agreement", args=[], outdir=PAPER_OUT,
         feeds="Section 3.2 (98.6% and 90.6%; kappa 0.93 and 0.71), Table 2 per-video columns",
         tables=["pervideo_summary_v27.csv", "pervideo_per_species_v27.csv"], figures=[]),
    dict(n=11, title="2025 detection outcomes on technician-reviewed videos",
         module="salmonid.paper.detection_outcomes", args=[], outdir=PAPER_OUT,
         feeds="Section 3.2 (share of fish videos with a counted track; false triggers)",
         tables=["detection_outcomes_2025.csv"], figures=[]),
    dict(n=12, title="Blind review of 150 clips",
         module="salmonid.paper.blind_review", args=[], outdir=PAPER_OUT,
         feeds="Section 3.2 (196 of 208 fish, 94.2%; kappa 0.90)",
         tables=["blind_v27_pairs.csv", "blind_v27_summary.csv"], figures=[]),
    dict(n=13, title="Cohen's kappa on the held-out test set",
         module="salmonid.paper.heldout_kappa", args=[], outdir=PAPER_OUT,
         feeds="Section 3.1 (kappa 0.95)",
         tables=["heldout_kappa.csv", "heldout_kappa_per_species.csv"], figures=[]),
    dict(n=14, title="Held-out score without tracks that have a neighbouring training clip",
         module="salmonid.paper.heldout_adjacent_clips", args=[], outdir=PAPER_OUT,
         feeds="Sections 3.1, 4.1 and 4.5 (macro-F1 0.950 on the 290 remaining tracks)",
         tables=["heldout_adjacent_clip_sensitivity.csv"], figures=[]),
    dict(n=15, title="Backbone comparison of three classifiers",
         module="salmonid.paper.backbone_comparison", args=[], outdir=PAPER_OUT,
         feeds="Table S4, Sections 2.3 and 3.1",
         tables=["table_s4.csv", "backbone_mcnemar.csv"], figures=[]),
    dict(n=16, title="Count tables: 2025 by species, multi-year Chinook, three species",
         module="salmonid.paper.count_tables", args=[], outdir=PAPER_OUT,
         feeds="Tables 2 (count columns), 3 and 4; Figure 4 annotations; Section 3.5 "
               "by-year differences; Figure 7 window shares",
         tables=["table2_count_columns.csv", "table3_chinook_multiyear.csv",
                 "table4_three_species.csv", "fig4_panel_annotations.csv",
                 "chinook_daily_difference_by_year.csv", "sanctuary_window_share.csv",
                 "quiet_day_sums.csv"], figures=[]),
    dict(n=17, title="Daily totals and the busiest days",
         module="salmonid.paper.daily_totals_and_peak_days", args=[], outdir=PAPER_OUT,
         feeds="Abstract and Conclusions (median daily difference of three fish; busiest days)",
         tables=["daily_totals_and_peak_days.csv"], figures=[]),
    dict(n=18, title="Checks against the published annual technician totals",
         module="salmonid.paper.annual_report_checks", args=[], outdir=PAPER_OUT,
         feeds="Section 2.1 ranges; Section 2.2 share of unidentified salmonids; Tables S1, S2",
         tables=["annual_report_checks.csv"], figures=[]),
]


def reference_for(name: str) -> Path:
    """The committed copy a regenerated table is compared with.

    Published tables sit in data/ or data/paper/. For scripts whose output the manuscript
    quotes only as numbers in the text, a reference copy is kept in data/paper/expected/.
    """
    for folder in (DATA, DATA / "net_rule_sweep", DATA / "paper", DATA / "paper" / "expected"):
        if (folder / name).is_file():
            return folder / name
    return DATA / name

# ---------------------------------------------------------------------------------------
# Headline numbers as PRINTED in the manuscript. Each is re-derived from the regenerated
# tables (or directly from data/) and must round to the printed value.
# ---------------------------------------------------------------------------------------
EXPECTED = [
    # (label, where it is printed, expected value, number of decimals it is printed to)
    ("held-out macro-F1", "Abstract, Table 1", 0.961, 3),
    ("held-out tracks scored", "Table 1", 366, 0),
    ("held-out tracks correct", "Figure 2 caption", 352, 0),
    ("held-out accuracy", "Section 3.1", 0.962, 3),
    ("macro-F1 95% CI, lower", "Table 1", 0.939, 3),
    ("macro-F1 95% CI, upper", "Table 1", 0.980, 3),
    ("Chinook daily CCC, Ganaraska", "Table 5", 0.964, 3),
    ("Chinook daily CCC, Credit", "Table 5", 0.958, 3),
    ("Bland-Altman mean difference (fish/day)", "Figure S3", -1.5, 1),
    ("Bland-Altman lower limit", "Figure S3", -122, 0),
    ("Bland-Altman upper limit", "Figure S3", 119, 0),
    ("Bland-Altman river-days", "Figure S3", 857, 0),
    ("pooled inter-annual Spearman rho", "Section 3.3", 0.91, 2),
    ("species-years in the pooled rho", "Section 3.3", 33, 0),
    ("leave-one-year-out error, %", "Section 3.4", 3.6, 1),
    ("folds choosing the 50/300 px rule", "Section 3.4", 4, 0),
    ("per-video agreement, Credit, %", "Section 3.2", 98.6, 1),
    ("per-video agreement, Ganaraska, %", "Section 3.2", 90.6, 1),
    ("per-video kappa, Credit", "Section 3.2", 0.93, 2),
    ("per-video kappa, Ganaraska", "Section 3.2", 0.71, 2),
    ("blind review: fish agreeing", "Section 3.2", 196, 0),
    ("blind review: fish paired", "Section 3.2", 208, 0),
    ("blind review kappa", "Section 3.2", 0.90, 2),
    ("held-out kappa", "Section 3.1", 0.95, 2),
    ("macro-F1 without neighbouring-clip tracks", "Section 3.1", 0.950, 3),
    ("median daily difference, fish", "Abstract", 3, 0),
]


# ---------------------------------------------------------------------------------------
# Running a step
# ---------------------------------------------------------------------------------------
def child_env() -> dict:
    """Environment for the step subprocesses.

    PYTHONPATH makes ``import salmonid`` work straight from a clone (no ``pip install -e .``
    needed); Agg lets matplotlib draw without a display; utf-8 keeps Windows consoles from
    failing on a non-ASCII character in a printed table.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(step: dict, log_dir: Path) -> dict:
    """Run one step as ``python -m <module>``; keep its full printout in output/tier1/logs/."""
    cmd = [sys.executable, "-m", step["module"], *step["args"]]
    shown = "python -m " + step["module"] + (" " + " ".join(step["args"]) if step["args"] else "")
    print(f"\n[{step['n']}/{len(STEPS)}] {step['title']}")
    print(f"      feeds : {step['feeds']}")
    print(f"      runs  : {shown}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, env=child_env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    secs = time.time() - t0
    log = log_dir / f"step{step['n']:02d}_{step['module'].rsplit('.', 1)[-1]}.log"
    log.write_text(f"$ {shown}\n\n{proc.stdout}\n{proc.stderr}", encoding="utf-8")
    ok = proc.returncode == 0
    print(f"      {'ok' if ok else 'FAILED'} in {secs:.1f} s  (full printout: "
          f"{log.relative_to(REPO).as_posix()})")
    if not ok:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
        print("      " + "\n      ".join(tail))
    return dict(step=step, ok=ok, secs=secs)


# ---------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------
def _cells_equal(a: str, b: str) -> bool:
    """Two table cells are equal if identical as text or equal as numbers (nan == nan)."""
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(fa) and math.isnan(fb):
        return True
    return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)


def compare_csv(new: Path, ref: Path) -> tuple[bool, str]:
    """Value-by-value comparison of a regenerated table with the committed one."""
    with open(new, newline="", encoding="utf-8") as fh:
        a = list(csv.reader(fh))
    with open(ref, newline="", encoding="utf-8") as fh:
        b = list(csv.reader(fh))
    if len(a) != len(b):
        return False, f"{len(a)} rows regenerated vs {len(b)} committed"
    bad = 0
    first = ""
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb) or not all(_cells_equal(x, y) for x, y in zip(ra, rb)):
            bad += 1
            if not first:
                first = f"first difference at row {i + 1}: {ra} vs {rb}"
    if bad:
        return False, f"{bad} of {len(a)} rows differ; {first}"
    n_cells = sum(len(r) for r in a)
    return True, f"{len(a) - 1} rows, {n_cells} cells identical"


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}{k}/")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}{i}/")
    else:
        yield prefix, obj


def compare_json(new: Path, ref: Path) -> tuple[bool, str]:
    a = dict(_flatten(json.loads(new.read_text(encoding="utf-8"))))
    b = dict(_flatten(json.loads(ref.read_text(encoding="utf-8"))))
    if a.keys() != b.keys():
        return False, "different keys"
    bad = [k for k in a if not _cells_equal(str(a[k]), str(b[k]))]
    if bad:
        return False, f"{len(bad)} of {len(a)} values differ (e.g. {bad[0]}: {a[bad[0]]} vs {b[bad[0]]})"
    return True, f"{len(a)} values identical"


def verify_tables(steps_run: list[dict]) -> list[tuple]:
    rows = []
    for res in steps_run:
        for name in res["step"]["tables"]:
            new = res["step"].get("outdir", OUT) / name
            ref = reference_for(name)
            where = ref.relative_to(REPO).as_posix()
            if not res["ok"] or not new.is_file():
                rows.append(("FAIL", name, where, "not regenerated (the step failed)"))
                continue
            ok, detail = (compare_json if name.endswith(".json") else compare_csv)(new, ref)
            rows.append(("PASS" if ok else "FAIL", name, where, detail))
    return rows


def headline_values() -> dict:
    """Re-derive the manuscript's headline numbers from the regenerated tables and data/."""
    sys.path.insert(0, str(REPO))
    from salmonid import config
    from salmonid.eval.score_testset_v6 import load_gt_file, load_track_preds, species_f1
    from salmonid.results import agreement_stats

    got = {}
    track_gt, _tiers, _presence = load_gt_file(str(config.TESTSET_V6_GT))
    macro, acc, _rows, cm, n = species_f1(track_gt, load_track_preds(str(config.PUBLISHED_PREDS)))
    got["held-out macro-F1"] = macro
    got["held-out tracks scored"] = n
    got["held-out tracks correct"] = sum(cm[s][s] for s in cm)
    got["held-out accuracy"] = acc

    def table(name):
        path = OUT / name
        if not path.is_file():
            return []
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    for r in table("classification_ci.csv"):
        if r["species"] == "MACRO":
            got["macro-F1 95% CI, lower"] = float(r["f1_lo"])
            got["macro-F1 95% CI, upper"] = float(r["f1_hi"])
    for r in table("concordance_ccc.csv"):
        if r["level"] == "daily" and r["species"] == "Chinook":
            got[f"Chinook daily CCC, {r['river']}"] = float(r["ccc"])
    for r in table("agreement_bland_altman.csv"):
        if r["scope"] == "pooled":
            got["Bland-Altman mean difference (fish/day)"] = float(r["bias"])
            got["Bland-Altman lower limit"] = float(r["loa_lo"])
            got["Bland-Altman upper limit"] = float(r["loa_hi"])
            got["Bland-Altman river-days"] = int(r["river_days"])
    _daily, summ = agreement_stats.load()
    pr = agreement_stats.pooled_rank_correlation(summ)
    got["pooled inter-annual Spearman rho"] = pr["rho"]
    got["species-years in the pooled rho"] = pr["n_species_years"]
    loyo = [r for r in table("net_rule_loyo.csv") if r["view"] == "pooled_all5"]
    for r in loyo:
        if r["held_out_year"] == "MEAN_OUT_OF_SAMPLE":
            got["leave-one-year-out error, %"] = float(r["heldout_err_pct"])
    got["folds choosing the 50/300 px rule"] = sum(r["chosen_rule"] == "up50/dn300" for r in loyo
                                                    if r["held_out_year"].isdigit())

    # ---- the manuscript-analysis tables written by salmonid/paper/ ----
    def ptable(name):
        path = PAPER_OUT / name
        if not path.is_file():
            return []
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    for r in ptable("pervideo_summary_v27.csv"):
        # "<river>_fish" rows use the denominator the manuscript describes: videos in which the
        # technician logged one record with a count of one.
        if r["name"].endswith("_fish"):
            river = r["name"].split("_")[0]
            got[f"per-video agreement, {river}, %"] = 100 * float(r["acc"])
            got[f"per-video kappa, {river}"] = float(r["kappa"])
    for r in ptable("blind_v27_summary.csv"):
        if r["group"] == "all paired fish":
            got["blind review: fish agreeing"] = int(r["agree"])
            got["blind review: fish paired"] = int(r["n"])
            got["blind review kappa"] = float(r["kappa"])
    for r in ptable("heldout_kappa.csv"):
        got["held-out kappa"] = float(r["kappa"])
    for r in ptable("heldout_adjacent_clip_sensitivity.csv"):
        if r["subset"] == "excluding flagged tracks":
            got["macro-F1 without neighbouring-clip tracks"] = float(r["macro_f1"])
    for r in ptable("daily_totals_and_peak_days.csv"):
        if r["statistic"].startswith("median absolute difference") and r["scope"] == "all species rows":
            got["median daily difference, fish"] = float(r["median_abs_diff"])
    return got


def verify_headlines() -> list[tuple]:
    try:
        got = headline_values()
    except Exception as exc:  # report the reason rather than hiding the verification table
        return [("FAIL", "headline numbers", "", f"could not be derived: {exc!r}")]
    rows = []
    for label, where, want, decimals in EXPECTED:
        if label not in got:
            rows.append(("SKIP", label, where, "step not run"))
            continue
        have = round(float(got[label]), decimals)
        ok = math.isclose(have, want, abs_tol=0.5 * 10 ** -(decimals + 2))
        fmt = f"{{:.{decimals}f}}"
        rows.append(("PASS" if ok else "FAIL", label, where,
                     f"regenerated {fmt.format(have)}, manuscript prints {fmt.format(want)}"))
    return rows


def verify_figures(steps_run: list[dict]) -> list[tuple]:
    """Figures: confirm each file was written, and say whether it is pixel-identical.

    A figure's bytes depend on the plotting-library version and its fonts, so on another
    machine a correct figure can differ in the last pixel. A mismatch is therefore reported
    as NOTE, not FAIL: the numbers behind every figure are verified by the table checks.
    """
    ref_file = DATA / "figure_checksums.json"
    ref = json.loads(ref_file.read_text(encoding="utf-8"))["sha256"] if ref_file.is_file() else {}
    rows = []
    for res in steps_run:
        for name in res["step"]["figures"]:
            path = FIGS / name
            if not res["ok"] or not path.is_file():
                rows.append(("FAIL", name, "", "not written"))
                continue
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if name not in ref:
                rows.append(("PASS", name, "", "written (no reference checksum recorded)"))
            elif ref[name] == sha:
                rows.append(("PASS", name, "", "written; byte-identical to the published figure"))
            else:
                rows.append(("NOTE", name, "", "written; bytes differ from the reference build "
                             "(expected if your matplotlib or fonts differ from requirements.lock)"))
    return rows


def lock_check() -> None:
    """Report installed packages whose version differs from the exact published environment."""
    from importlib import metadata
    lock = REPO / "environment" / "tier1.lock.txt"
    diffs = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, want = line.split("==")
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not installed"
        if have != want:
            diffs.append(f"  {name}: installed {have}, lock file {want}")
    if diffs:
        print("Package versions that differ from environment/tier1.lock.txt:")
        print("\n".join(diffs))
        print("(Numbers are expected to match regardless; figure files may differ in the last pixel.)")
    else:
        print("Installed packages match environment/tier1.lock.txt exactly.")


def print_block(title: str, rows: list[tuple]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    for status, name, where, detail in rows:
        loc = f" [{where}]" if where else ""
        print(f"  {status:4s}  {name}{loc}\n        {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run and verify every Tier-1 reproduction step.")
    ap.add_argument("--list", action="store_true", help="List the steps and exit.")
    ap.add_argument("--only", nargs="+", type=int, metavar="N", help="Run only these step numbers.")
    ap.add_argument("--no-figures", action="store_true", help="Skip the figure steps.")
    ap.add_argument("--lock-check", action="store_true",
                    help="Also list installed package versions that differ from environment/tier1.lock.txt.")
    a = ap.parse_args()

    if a.list:
        print("Tier-1 steps (run in this order):\n")
        for s in STEPS:
            print(f"  {s['n']}. {s['title']}\n       python -m {s['module']}\n       feeds: {s['feeds']}")
        return 0

    todo = [s for s in STEPS
            if (not a.only or s["n"] in a.only) and not (a.no_figures and s.get("is_figure"))]
    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "logs"
    log_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("Tier-1 reproduction: recompute the paper's results from the committed data")
    print(f"python {sys.version.split()[0]}  |  writing to output/tier1/, output/paper/ and figures/")
    print("=" * 78)
    if a.lock_check:
        lock_check()
    results = [run_step(s, log_dir) for s in todo]

    print("\n" + "=" * 78)
    print("VERIFICATION")
    print("=" * 78)
    step_rows = [("PASS" if r["ok"] else "FAIL", f"step {r['step']['n']}: {r['step']['title']}",
                  "", f"{r['secs']:.1f} s") for r in results]
    table_rows = verify_tables(results)
    headline_rows = verify_headlines()
    figure_rows = verify_figures(results)
    print_block("Steps completed", step_rows)
    print_block("Regenerated tables vs the committed tables in data/", table_rows)
    print_block("Headline numbers vs the values printed in the manuscript", headline_rows)
    print_block("Figures", figure_rows)

    everything = step_rows + table_rows + headline_rows + figure_rows
    n_fail = sum(r[0] == "FAIL" for r in everything)
    n_pass = sum(r[0] == "PASS" for r in everything)
    n_note = sum(r[0] == "NOTE" for r in everything)
    n_skip = sum(r[0] == "SKIP" for r in everything)
    print("\n" + "=" * 78)
    print(f"RESULT: {n_pass} passed, {n_fail} failed, {n_note} notes, {n_skip} skipped")
    if n_fail == 0:
        print("Every regenerated result matches the published one.")
    else:
        print("At least one result did not reproduce. See the FAIL lines above and the step logs.")
    print("=" * 78)

    report = OUT / "verification_report.txt"
    with open(report, "w", encoding="utf-8", newline="\n") as fh:
        for title, rows in [("steps", step_rows), ("tables", table_rows),
                            ("headline numbers", headline_rows), ("figures", figure_rows)]:
            fh.write(f"[{title}]\n")
            for status, name, where, detail in rows:
                fh.write(f"{status}\t{name}\t{where}\t{detail}\n")
        fh.write(f"\nRESULT\t{n_pass} passed\t{n_fail} failed\t{n_note} notes\t{n_skip} skipped\n")
    print(f"Report saved to {report.relative_to(REPO).as_posix()}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

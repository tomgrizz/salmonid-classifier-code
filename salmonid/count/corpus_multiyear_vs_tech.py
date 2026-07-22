"""
In plain terms: this script checks how well the automated fish counter agrees with the human
technicians who reviewed the same tunnel-camera video and counted the fish by hand. It lines the
computer's daily counts up against the technicians' counts for each species, river (Ganaraska and
Credit), and year from 2020 to 2025, then reports the season totals, the percent difference, and how
closely the day-to-day patterns match. These comparisons are the paper's multi-year monitoring
result: the year-by-year deployment scorecard table and its accompanying heatmap figure showing how
well the counter tracks the technician counts over time.

Multi-year deployment scorecard engine (2020-2025, both rivers): the automated single-stage
counter's full-corpus counts vs TECHNICIAN counts - ACTIVITY + NET, exact per-day from the Vaki
AbsDb / 2025 event exports - plus the full technician-source registry that stitches every year's raw
truth. Exposes ``cmd_year`` (one year -> daily CSV) and ``cmd_scorecard`` (pool years -> scorecard
CSV + heatmap figure).

Pipeline position:  data -> detect -> track -> [THIS: eval/count] -> results -> figures.
Environment:  the BASE Python env (pandas / pyarrow / numpy / matplotlib). This module does NOT run
detection or training, so the DEIMv2 ``.venv_deim`` is not needed - model counts are RE-DERIVED from
the per-frame detection parquets produced upstream.

Inputs:
  * Model detection parquets under ``config.MODEL_OUTPUTS / run_corpus_v6_<year>_shard_*`` (one
    parquet per video), dated by footage-file mtime under the footage roots.
  * Technician truth: the committed ``data/tech_raw_data`` bundle + sibling OMNRF repos
    (``config.MODEL_PRO_DIR`` / ``config.LOMU_INTERVIEW_DIR``) for pre-2024 official NET.
Outputs:
  * ``config.FIGURES_DIR / deployment / <year>_daily.csv``  (per --year)
  * ``config.FIGURES_DIR / deployment_scorecard_2022_2025.csv`` (+ .png)  (per --scorecard)
Manuscript claim: the multi-year monitoring result - species-level abundance/activity tracked
against the technician counts across 2020-2025 on both the Ganaraska and the Credit.

Same validated model method as `_corpus_2024_vs_tech.py` / `_corpus_historical_vs_tech.py`: model
counts RE-DERIVED from per-frame PARQUETS at --min-frames (default 3), AUTO_FLOOR species_conf triage
(Atlantic 1.01 = never auto-counted), file-mtime dates, ASYMMETRIC net (up>=--up-thresh,
down>=--down-thresh), UP_SIGN per site.

TECH BASELINE = the VIDEO-OBSERVABLE counts (Vaki AbsDb / 2025 reviewed event exports): one reviewed
row = one fish the tech saw ON CAMERA (DateTime, Direction Up/Down, sortname). This is the fair
baseline for a camera model - it sees the same passage the tech reviewed from video. HANDLIFTS (fish
intercepted by hand at the fishway, never on camera) are NOT in the AbsDb but ARE in the official
published totals (Ganny_Total / FINAL_CREDIT), so official >= video; that gap (~handlifts) is carried
only as the `net_official` context metric, NOT used to score the model. (Confirmed: AbsDb net matches
official for Chinook/Coho/Atlantic/Brown; official is higher for Ganaraska Rainbow = handlifted steelhead.)

Tech sources (all in data/tech_raw_data unless noted):
  2020 Credit     event  Riverwatcher Data/Credit/2020/FINAL_YUILLE(<10-01)+FINAL_JANG(>=10-01)  (MAYNARD dropped: all not-fish)
  2020 Ganaraska  NET only (daily SPC/CNT) -> Ganny_Total  (official, +handlift; activity n/a)
  2021 Ganaraska  event  Riverwatcher Data/Ganaraska/2021/*.csv  (4 tiles sliced 05-21/08-25/09-14)
  2021 Credit     event  Riverwatcher Data/Credit/2021/*.csv  (2 tiles sliced 08-27)
  2022 Ganaraska  AbsDb  'All Ganaraska Videos Completed.xlsx'  (single, full season)
  2022 Credit     event  Riverwatcher Data/Credit/2022/2022Credit_finalclean.csv  (full season; was net-only, now activity)
  2023 Ganaraska  AbsDb  'Ganaraska 2023 Final.xlsx'(<09-25) + 'Ganraska File Sept 25 to Nov 20.xlsx'(>=09-25)
  2023 Credit     AbsDb  Credit_20230403-20230915(<09-15) + Credit_20230915-20231124(>=09-15)
  2024 Ganaraska  AbsDb  (from 2024ganny.zip) 3 canonical tiles, dups dropped
  2024 Credit     AbsDb  model_pro/.../tech_reviewed/2024/Credit_2024_*.xlsx
  2025 Ganaraska  event  all_counts_gan.csv  (sep=';', species_main, skip false_trigger)
  2025 Credit     event  fish_counts_export_*.csv
Official net context: Ganny_Total (Ganaraska <=2024), FINAL_CREDIT_<year> (Credit <=2023).

Usage:
  python -m salmonid.count.corpus_multiyear_vs_tech --check-tech       # validate tech stitch (gates), fast
  python -m salmonid.count.corpus_multiyear_vs_tech --year 2024        # tech+model, writes figures/deployment/2024_daily.csv
  python -m salmonid.count.corpus_multiyear_vs_tech --scorecard        # combine per-year CSVs -> scorecard + figure
"""
from __future__ import annotations
import argparse, glob, os, io, zipfile, collections, datetime, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from salmonid import config  # centralised paths/constants (defaults resolve to the original locations)
try:
    from scipy.stats import pearsonr
except Exception:  # noqa: BLE001
    def pearsonr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 2 or a.std() == 0 or b.std() == 0: return (float("nan"), 1.0)
        return (float(np.corrcoef(a, b)[0, 1]), 0.0)

SPECIES = {0: "Chinook", 1: "Coho", 2: "Atlantic", 3: "Rainbow", 4: "Brown"}  # mirrors salmonid.config.SPECIES
AUTO_FLOOR = {0: 0.50, 1: 0.60, 2: 1.01, 3: 0.60, 4: 0.60}  # mirrors salmonid.config.AUTO_FLOOR (Atlantic 1.01 => never auto-counted)
_PROB = ["prob_chinook", "prob_coho", "prob_atlantic", "prob_rainbow", "prob_brown"]
UP_SIGN = {"Ganaraska": 1.0, "Credit": -1.0}           # mirrors salmonid.config.UP_SIGN (per-site upstream sign of delta_x)
SPC = {73: 1, 75: 0, 76: 3, 77: 2, 78: 4}              # OMNRF SPC -> class idx (70 = Salmonid, unclassified); mirrors salmonid.config.SPC_TO_CLASS

TRAW = config.REPO_DATA / "tech_raw_data"              # committed technician raw-data bundle (data/tech_raw_data)
RWDATA = TRAW / "Riverwatcher Data"                    # raw per-event Vaki CSV exports (2017-2023, both rivers)
ZIP24 = TRAW / "2024ganny.zip"
# Sibling OMNRF data repos (external; env-overridable via salmonid.config) holding the pre-2024 technician truth.
CREDIT24 = config.MODEL_PRO_DIR / "tech_data" / "data" / "tech_reviewed" / "2024"
GANNY_TOTAL = config.LOMU_INTERVIEW_DIR / "data" / "Ganny_Total.csv"
CREDIT_HIST = config.MODEL_PRO_DIR / "tech_data" / "data" / "tech_reviewed" / "2023-2018"
GANA2025 = config.DOCS_DIR / "Riverwatcher_2025/Ganaraska_Counter_Analysis_2025/all_counts_gan.csv"
CREDIT2025_GLOB = str(config.DOCS_DIR / "Riverwatcher_2025/Credit_Counter_Analysis_2025/**/fish_counts_export_*.csv")
OUTDIR = config.FIGURES_DIR / "deployment"             # per-year daily CSVs + scorecard artifacts land here
PUBLISHED_2024_GANA_NET = {0: 19514, 3: 9032, 1: 3201}  # Chinook / Rainbow / Coho (script-cited published)

# AbsDb sources: (year, site) -> [(kind, ref, dmin_inclusive, dmax_exclusive), ...]
ABSDB = {
    (2022, "Ganaraska"): [("file", TRAW / "All Ganaraska Videos Completed.xlsx", None, None)],
    (2023, "Ganaraska"): [("file", TRAW / "Ganaraska 2023 Final.xlsx", None, "2023-09-25"),
                          ("file", TRAW / "Ganraska File Sept 25 to Nov 20.xlsx", "2023-09-25", None)],
    (2023, "Credit"):    [("file", TRAW / "Credit_20230403-20230915.xlsx", None, "2023-09-15"),
                          ("file", TRAW / "Credit_20230915-20231124.xlsx", "2023-09-15", None)],
    (2024, "Ganaraska"): [("zip", "Ganaraska_2024_03_05-2024_09_16.xlsx", None, "2024-09-16"),
                          ("zip", "Ganaraska_2024_09_16-2024_09_24.xlsx", "2024-09-16", "2024-09-24"),
                          ("zip", "Ganaraska_2024_09_24-2024_11_27.xlsx", "2024-09-24", None)],
    (2024, "Credit"):    [("file", CREDIT24 / "Credit_2024_05_16_to_2024_09_03.xlsx", None, "2024-09-03"),
                          ("file", CREDIT24 / "Credit_2024_09_03_to_2024_11_27.xlsx", "2024-09-03", None)],
    # --- raw per-event Vaki CSV exports (Riverwatcher Data): one row = one fish, same DateTime/Direction/sortname ---
    (2021, "Ganaraska"): [("csv", RWDATA / "Ganaraska" / "2021" / "Ganaraska_20210324_20210521.csv", None, "2021-05-21"),
                          ("csv", RWDATA / "Ganaraska" / "2021" / "Ganaraska_20210521_20210825.csv", "2021-05-21", "2021-08-25"),
                          ("csv", RWDATA / "Ganaraska" / "2021" / "Ganaraska_20210825_20210914.csv", "2021-08-25", "2021-09-14"),
                          ("csv", RWDATA / "Ganaraska" / "2021" / "Ganaraska_20210914_20211121.csv", "2021-09-14", None)],
    (2021, "Credit"):    [("csv", RWDATA / "Credit" / "2021" / "Credit_20210329_20210826.csv", None, "2021-08-27"),
                          ("csv", RWDATA / "Credit" / "2021" / "Credit_20210827_20211104.csv", "2021-08-27", None)],
    (2022, "Credit"):    [("csv", RWDATA / "Credit" / "2022" / "2022Credit_finalclean.csv", None, None)],
    # 2020 Credit reviewer files overlap -> partition YUILLE(<10-01)+JANG(>=10-01); drop MAYNARD (all not-fish, inside YUILLE)
    (2020, "Credit"):    [("csv", RWDATA / "Credit" / "2020" / "FINAL_YUILLE_2020CREDIT.csv", None, "2020-10-01"),
                          ("csv", RWDATA / "Credit" / "2020" / "FINAL_JANG_2020CREDIT.csv", "2020-10-01", None)],
}


def _vd(d):
    """True iff ``d`` is a well-formed 'YYYY-MM-DD' string (cheap date-string sanity filter)."""
    return isinstance(d, str) and len(d) == 10 and d[4] == "-" and d[7] == "-" and d[:4].isdigit()


def _sp(n):
    """Map a free-text technician species/sortname to our class index 0-4, else None.

    Tolerates known reviewer typos ('chinnok', 'atltantic'); folds 'steelhead' into Rainbow (3);
    returns None for non-salmonid and brook trout (brook != our Brown) so they are not counted.
    """
    n = (n if isinstance(n, str) else "").strip().lower()
    if "non-salmonid" in n or "non salmonid" in n: return None     # not a salmonid
    if "brook" in n: return None                                   # brook trout != our brown
    if "chinook" in n or "chinnok" in n: return 0          # 'chinnok' typo (Credit 2019/raw exports)
    if "coho" in n: return 1
    if "atlantic" in n or "atltantic" in n: return 2               # 'Unclip Atltantic' typo (2024)
    if "rainbow" in n or "steelhead" in n: return 3
    if "brown" in n: return 4
    return None


def _is_salmonid_uncl(n):
    """True for a generic 'Salmonid' label carrying no species (tallied as unclassified activity only)."""
    n = (n if isinstance(n, str) else "").strip().lower()
    return "salmonid" in n and "non" not in n


def _mv(direction):
    """Direction string -> movement sign: +1 upstream, -1 downstream, 0 unknown/other."""
    d = (direction if isinstance(direction, str) else "").strip().lower()
    return 1 if d == "up" else (-1 if d == "down" else 0)


def _absdb_rows(year, site):
    """yield (date, species_str, direction_str, count) for an AbsDb (year, site)."""
    import pandas as pd
    for kind, ref, dmin, dmax in ABSDB[(year, site)]:
        if kind == "zip":
            with zipfile.ZipFile(ZIP24) as zf:
                df = pd.read_excel(io.BytesIO(zf.read(ref)), sheet_name="AbsDb")
        elif kind == "csv":
            df = pd.read_csv(ref, dtype=str, keep_default_na=False)   # raw per-event export (1 row/fish)
        else:
            df = pd.read_excel(ref, sheet_name="AbsDb")
        df["d"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.strftime("%Y-%m-%d")
        for _, r in df.iterrows():
            date = r["d"]
            if not _vd(date): continue
            if dmin is not None and date < dmin: continue
            if dmax is not None and date >= dmax: continue
            yield date, r.get("sortname"), r.get("Direction"), 1


def _event2025_rows(site):
    """yield (date, species_str, direction_str, count) for 2025 reviewed event exports."""
    import pandas as pd
    if site == "Ganaraska":
        df = pd.read_csv(GANA2025, sep=";")
        spcol = "species_main" if "species_main" in df.columns else "species"
        files = [(df, spcol)]
    else:
        files = [(pd.read_csv(f), "species") for f in sorted(glob.glob(CREDIT2025_GLOB, recursive=True))]
    for df, spcol in files:
        df["d"] = pd.to_datetime(df["ts"], errors="coerce").dt.strftime("%Y-%m-%d")
        for _, r in df.iterrows():
            if int(r.get("false_trigger") or 0) == 1: continue
            date = r["d"]
            if not _vd(date): continue
            c = r.get("count")
            cnt = 1 if (c is None or (isinstance(c, float) and np.isnan(c))) else int(c)
            yield date, r.get(spcol), r.get("movement"), cnt


def _credit_official_net(year):
    """FINAL_CREDIT_<year>.csv signed daily NET (official, incl. handlifts; 2018-2023)."""
    import pandas as pd
    out = collections.defaultdict(lambda: collections.defaultdict(int)); uncl = 0
    cf = CREDIT_HIST / str(year) / f"FINAL_CREDIT_{year}.csv"
    c = pd.read_csv(cf); c["d"] = pd.to_datetime(c["DATE"], errors="coerce").dt.strftime("%Y-%m-%d")
    for _, r in c.iterrows():
        code = int(r["SPC"]); date = r["d"]
        if not _vd(date): continue
        if code not in SPC:
            if code == 70: uncl += int(r["CNT"])
            continue
        out[date][SPC[code]] += int(r["CNT"])
    return out, uncl


def _ganny_total_net(year):
    """Ganny_Total.csv signed daily NET (official, incl. handlifts) for a year."""
    import pandas as pd
    out = collections.defaultdict(lambda: collections.defaultdict(int))
    g = pd.read_csv(GANNY_TOTAL); g = g[g.YEAR == year]
    for _, r in g.iterrows():
        code = int(r["SPC"]); date = str(r["DATE"])[:10]
        if not _vd(date) or code not in SPC: continue
        out[date][SPC[code]] += int(r["CNT"])
    return out


def build_tech(year):
    """ACTIVITY + NET from the per-event AbsDb/2025 exports = VIDEO-OBSERVABLE tech (the fair model
    baseline). OFFICIAL net (Ganny_Total <=2024 / FINAL_CREDIT <=2023, handlift-inclusive) is carried
    as context only (t_net_off_*). Credit 2022 has no video AbsDb, so its net falls back to official."""
    t_act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_off_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_net_off_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_up_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_up_t = collections.defaultdict(lambda: collections.defaultdict(int))
    t_down_d = collections.defaultdict(lambda: collections.defaultdict(int))
    t_down_t = collections.defaultdict(lambda: collections.defaultdict(int))
    uncl = collections.defaultdict(int); has_act = set(); net_src = {}

    for site in ("Ganaraska", "Credit"):
        if (year, site) in ABSDB:
            rows = _absdb_rows(year, site); has_act.add(site); net_src[site] = "absdb(video)"
        elif year == 2025:
            rows = _event2025_rows(site); has_act.add(site); net_src[site] = "event2025(video)"
        else:
            continue
        for date, sps, direction, cnt in rows:
            sp = _sp(sps)
            if sp is None:
                if _is_salmonid_uncl(sps): uncl[site] += cnt
                continue
            mv = _mv(direction)
            t_act_d[(site, date)][sp] += cnt; t_act_t[site][sp] += cnt
            t_net_d[(site, date)][sp] += mv * cnt; t_net_t[site][sp] += mv * cnt
            if mv == 1: t_up_d[(site, date)][sp] += cnt; t_up_t[site][sp] += cnt
            elif mv == -1: t_down_d[(site, date)][sp] += cnt; t_down_t[site][sp] += cnt

    def _fill_off(site, daily):
        for date, spd in daily.items():
            for sp, c in spd.items():
                t_net_off_d[(site, date)][sp] += c; t_net_off_t[site][sp] += c
    if year <= 2024: _fill_off("Ganaraska", _ganny_total_net(year))
    if year <= 2023:
        cn, u = _credit_official_net(year); uncl["Credit"] += u; _fill_off("Credit", cn)
    # net-only years (no video AbsDb at all, e.g. 2021/2020): official net is the primary baseline
    for s in ("Ganaraska", "Credit"):
        if not t_net_t[s] and t_net_off_t[s]:
            for (ss, date) in list(t_net_off_d):
                if ss == s:
                    for sp, c in t_net_off_d[(ss, date)].items():
                        t_net_d[(s, date)][sp] += c; t_net_t[s][sp] += c
            net_src[s] = "official(+handlift)"
    return dict(t_act_d=t_act_d, t_net_d=t_net_d, t_act_t=t_act_t, t_net_t=t_net_t,
                t_net_off_d=t_net_off_d, t_net_off_t=t_net_off_t,
                t_up_d=t_up_d, t_up_t=t_up_t, t_down_d=t_down_d, t_down_t=t_down_t,
                uncl=uncl, has_act=has_act, net_src=net_src)


def derive_model(year, shard_dirs, froots, MF, uT, dT, atl_floor=0.60):
    """Re-derive model ACTIVITY + NET counts from the per-frame detection parquets.

    For every parquet found under ``shard_dirs`` (glob patterns), rows are grouped into tracks;
    per-track class probabilities are averaged and the top class kept iff its mean confidence
    clears the species floor (AUTO_FLOOR, or ``atl_floor`` for Atlantic) AND the track spans at
    least ``MF`` frames (min_frames re-derivation). Videos are dated by footage-file mtime (first
    ``froots`` entry that contains ``video_id``). NET applies the asymmetric rule with UP_SIGN per
    site: count UP when |delta_x| >= ``uT`` and DOWN only when |delta_x| >= ``dT`` (strict down
    threshold suppresses milling), where delta_x = mean of the last-3 minus first-3 box centroids.

    Args:
        year: corpus year (context only; dating is by file mtime, not year).
        shard_dirs: glob patterns for shard dirs, each holding ``detections/*.parquet``.
        froots: ordered footage roots; the first containing ``video_id`` dates that video.
        MF: min_frames - minimum detections for a track to count.
        uT, dT: upstream / downstream |delta_x| pixel thresholds (asymmetric net rule).
        atl_floor: analysis-only Atlantic confidence floor (deploy keeps Atlantic review-only).
    Returns:
        dict of daily/total ACTIVITY, NET, UP, DOWN defaultdicts + ``n_tr`` / ``nfiles`` counters.
    """
    m_act_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_act_t = collections.defaultdict(lambda: collections.defaultdict(int))
    m_net_t = collections.defaultdict(lambda: collections.defaultdict(int))
    m_up_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_up_t = collections.defaultdict(lambda: collections.defaultdict(int))
    m_down_d = collections.defaultdict(lambda: collections.defaultdict(int))
    m_down_t = collections.defaultdict(lambda: collections.defaultdict(int))
    cache: dict[str, str] = {}
    cols = ["video_id", "track_id", "frame_number", "bbox_x1", "bbox_x2", *_PROB]
    n_tr = nfiles = 0
    for d in sorted({x for pat in shard_dirs for x in glob.glob(pat)}):
        for pf in glob.glob(os.path.join(d, "detections", "*.parquet")):
            t = pq.read_table(pf, columns=cols)
            if t.num_rows == 0: continue
            vid = t.column("video_id")[0].as_py(); site = vid.split("/")[0]
            if site not in UP_SIGN: continue
            date = cache.get(vid)
            if date is None:
                date = ""
                for fr in froots:
                    try: date = datetime.date.fromtimestamp(os.path.getmtime(fr / vid)).isoformat(); break
                    except OSError: continue
                cache[vid] = date
            if not _vd(date): continue
            nfiles += 1
            tid = t.column("track_id").to_numpy(); frame = t.column("frame_number").to_numpy()
            cx = (t.column("bbox_x1").to_numpy().astype(np.float64) + t.column("bbox_x2").to_numpy().astype(np.float64)) / 2.0
            probs = np.column_stack([t.column(c).to_numpy() for c in _PROB])
            # sort rows into contiguous per-track blocks (by track, then frame) so a single
            # add.reduceat over the block boundaries gives the per-track mean class-probability vector
            order = np.lexsort((frame, tid)); tid_s, cx_s, probs_s = tid[order], cx[order], probs[order]
            uniq, idx, counts = np.unique(tid_s, return_index=True, return_counts=True)
            means = np.add.reduceat(probs_s, idx, axis=0) / counts[:, None]
            spk = means.argmax(axis=1); confk = means[np.arange(len(uniq)), spk]
            for k in range(len(uniq)):
                s = int(spk[k])
                # Atlantic counted RAW here (atl_floor) for analysis visibility - at DEPLOY it stays
                # review-only (never auto-counted) via AUTO_FLOOR[2]=1.01; this CSV is analysis, not the counter.
                floor = atl_floor if s == 2 else AUTO_FLOOR[s]
                if confk[k] < floor or counts[k] < MF: continue
                # track displacement = mean-x of the last 3 detections minus the first 3
                seg = cx_s[idx[k]: idx[k] + counts[k]]; m3 = min(3, len(seg))
                delta_x = seg[-m3:].mean() - seg[:m3].mean(); n_tr += 1
                m_act_d[(site, date)][s] += 1; m_act_t[site][s] += 1
                sgn = delta_x * UP_SIGN[site]  # positive sgn = travelling upstream at this site
                # asymmetric net rule: UP if |dx| >= uT (liberal), DOWN only if |dx| >= dT (strict; kills milling)
                nc = 1 if (sgn > 0 and abs(delta_x) >= uT) else (-1 if (sgn < 0 and abs(delta_x) >= dT) else 0)
                m_net_d[(site, date)][s] += nc; m_net_t[site][s] += nc
                if nc == 1: m_up_d[(site, date)][s] += 1; m_up_t[site][s] += 1
                elif nc == -1: m_down_d[(site, date)][s] += 1; m_down_t[site][s] += 1
    return dict(m_act_d=m_act_d, m_net_d=m_net_d, m_act_t=m_act_t, m_net_t=m_net_t,
                m_up_d=m_up_d, m_up_t=m_up_t, m_down_d=m_down_d, m_down_t=m_down_t,
                n_tr=n_tr, nfiles=nfiles)


def _r(md, td):
    """Pearson r between the model and tech daily series, aligned on the union of their dates
    (missing dates count as 0); NaN if either series is empty/flat."""
    dates = sorted(set(md) | set(td))
    m = [md.get(x, 0) for x in dates]; t = [td.get(x, 0) for x in dates]
    return pearsonr(m, t)[0] if (any(m) and any(t)) else float("nan")


def _site_daily(dct, site, sp):
    """Flatten a {(site, date): {sp: value}} dict to {date: value} for one site + species."""
    return {dt: dct[(s, dt)][sp] for (s, dt) in dct if s == site}


def cmd_check_tech(years):
    """--check-tech: print the technician stitch (VIDEO baseline vs OFFICIAL handlift-inclusive net)
    per year and, for 2024, gate the Ganaraska official net against the published totals (flag >5%)."""
    print("=== TECH stitch validation (VIDEO baseline vs OFFICIAL handlift-inclusive) ===")
    for yr in years:
        T = build_tech(yr)
        print(f"\n--- {yr} ---  activity sites: {sorted(T['has_act'])}  net source: {T['net_src']}  "
              f"unclassified-salmonid(activity): {dict(T['uncl'])}")
        for site in ("Ganaraska", "Credit"):
            act = T["t_act_t"][site]; net = T["t_net_t"][site]; off = T["t_net_off_t"][site]
            if not act and not net and not off: continue
            print(f"  {site}")
            print("     ACT(video)  " + "  ".join(f"{SPECIES[s]}={act[s]:,}" for s in range(5)))
            print("     NET(video)  " + "  ".join(f"{SPECIES[s]}={net[s]:,}" for s in range(5)))
            if off:
                gap = "  ".join(f"{SPECIES[s]}={off[s]-net[s]:+,}" for s in range(5))
                print("     NET(offic)  " + "  ".join(f"{SPECIES[s]}={off[s]:,}" for s in range(5)))
                print(f"     handlift gap (official-video): {gap}")
        if yr == 2024:
            print("  GATE 2024 Gana official net vs published:")
            for sp, pub in PUBLISHED_2024_GANA_NET.items():
                got = T["t_net_off_t"]["Ganaraska"][sp]; err = 100 * (got - pub) / pub
                flag = "OK" if abs(err) <= 5 else "** CHECK **"
                print(f"     {SPECIES[sp]:9} official {got:,} vs published {pub:,}  ({err:+.0f}%) {flag}")


def cmd_year(yr, args):
    """--year: build the technician stitch + re-derive model counts for one year, write
    ``figures/deployment/<year>_daily.csv``, and print per-site/species model-vs-tech totals,
    % error, and daily correlation for activity / up / down / net(video) / net(official)."""
    froots = [Path(p) for p in args.footage_roots]
    shard_dirs = args.shard_dirs or [str(config.MODEL_OUTPUTS / f"run_corpus_v6_{yr}_shard_*")]
    MF, uT, dT = args.min_frames, args.up_thresh, args.down_thresh
    T = build_tech(yr)
    M = derive_model(yr, shard_dirs, froots, MF, uT, dT, args.atl_floor)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outp = OUTDIR / f"{yr}_daily.csv"
    print(f"\n=== {yr} model v6 vs tech  [min_frames={MF}, net up>={int(uT)}/down>={int(dT)}px]  "
          f"tracks={M['n_tr']:,} from {M['nfiles']:,} videos ===")
    rows = []
    for site in ("Ganaraska", "Credit"):
        metrics = [("activity", M["m_act_d"], T["t_act_d"], site in T["has_act"]),
                   ("up", M["m_up_d"], T["t_up_d"], site in T["has_act"]),
                   ("down", M["m_down_d"], T["t_down_d"], site in T["has_act"]),
                   ("net", M["m_net_d"], T["t_net_d"], bool(T["t_net_t"][site])),
                   ("net_official", M["m_net_d"], T["t_net_off_d"], bool(T["t_net_off_t"][site]))]
        for metric, md_all, td_all, has in metrics:
            for sp in range(5):
                md = _site_daily(md_all, site, sp); td = _site_daily(td_all, site, sp) if has else {}
                for dt in sorted(d for d in (set(md) | set(td)) if _vd(d)):
                    rows.append([yr, site, dt, SPECIES[sp], metric, md.get(dt, 0),
                                 (td.get(dt, 0) if has else "")])
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["year", "site", "date", "species", "metric", "model", "tech"])
        w.writerows(rows)
    print(f"wrote {outp}  ({len(rows)} rows)  net_src={T['net_src']}")
    for site in ("Ganaraska", "Credit"):
        print(f"\n  {site} {yr}:")
        for sp in range(5):
            for metric, md_all, td_all, tt, has in [
                ("activity    ", M["m_act_d"], T["t_act_d"], T["t_act_t"][site], site in T["has_act"]),
                ("up          ", M["m_up_d"], T["t_up_d"], T["t_up_t"][site], site in T["has_act"]),
                ("down        ", M["m_down_d"], T["t_down_d"], T["t_down_t"][site], site in T["has_act"]),
                ("net(video)  ", M["m_net_d"], T["t_net_d"], T["t_net_t"][site], bool(T["t_net_t"][site])),
                ("net(official)", M["m_net_d"], T["t_net_off_d"], T["t_net_off_t"][site], bool(T["t_net_off_t"][site])),
            ]:
                md = _site_daily(md_all, site, sp); mt = sum(md.values())
                if not has:
                    print(f"     {SPECIES[sp]:9} {metric}: model {mt:>7,}  tech   n/a"); continue
                td = _site_daily(td_all, site, sp); r = _r(md, td); tt_v = tt[sp]
                pct = f"{100*(mt-tt_v)/tt_v:+.0f}%" if tt_v else "n/a"
                print(f"     {SPECIES[sp]:9} {metric}: model {mt:>7,} / tech {tt_v:>7,} ({pct:>5})  daily-r={r:6.3f}")


def _grade(species, metric, r, model, tech):
    """Letter-grade an ACTIVITY row: A (r>=.95 & |err|<=10%), B (r>=.90 & |err|<=25%), else C.
    Atlantic is always C (review-only: a precision/false-positive problem, not a recall one).
    Non-activity metrics and empty/zero-tech rows are ungraded ('')."""
    if metric != "activity" or tech in ("", 0) or tech is None: return ""
    if species == "Atlantic": return "C"                      # precision (false Atlantics), review-only
    try: err = abs(model - tech) / tech
    except Exception: return ""
    if not np.isfinite(r): return ""
    if r >= 0.95 and err <= 0.10: return "A"
    if r >= 0.90 and err <= 0.25: return "B"
    return "C"


def _bump(fig, k=1.45):
    """Scale every text element in ``fig`` by ``k`` (uniformly bumps font sizes for print legibility)."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        try: t.set_fontsize(t.get_fontsize() * k)
        except Exception: pass


def cmd_scorecard():
    """--scorecard: pool the per-year ``<year>_daily.csv`` files into the deployment scorecard
    (CSV + an ACTIVITY daily-r heatmap PNG). 2020 is dropped from grading (inference incomplete -
    Ganaraska barely ran) and kept only as context."""
    import pandas as pd
    files = sorted(glob.glob(str(OUTDIR / "*_daily.csv")))
    if not files: raise SystemExit(f"no per-year CSVs in {OUTDIR}; run --year first")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tech"] = pd.to_numeric(df["tech"], errors="coerce")
    df = df[df.year != 2020]   # 2020 inference incomplete (Ganaraska barely ran) -> context only, never graded
    out = []
    for (yr, site, sp, metric), g in df.groupby(["year", "site", "species", "metric"]):
        model_tot = int(g["model"].sum()); has_tech = g["tech"].notna().any()
        tech_tot = int(g["tech"].sum()) if has_tech else ""
        r = (_r({d: m for d, m in zip(g["date"], g["model"])},
                {d: t for d, t in zip(g["date"], g["tech"].fillna(0))}) if has_tech else float("nan"))
        pct = (round(100 * (model_tot - tech_tot) / tech_tot, 1) if (has_tech and tech_tot) else "")
        out.append({"year": yr, "site": site, "species": sp, "metric": metric,
                    "model": model_tot, "tech": tech_tot, "pct_err": pct,
                    "daily_r": (round(r, 3) if np.isfinite(r) else ""),
                    "grade": _grade(sp, metric, r, model_tot, tech_tot)})
    sc = pd.DataFrame(out).sort_values(["species", "site", "metric", "year"])
    scp = config.FIGURES_DIR / "deployment_scorecard_2022_2025.csv"
    sc.to_csv(scp, index=False)
    print(f"wrote {scp}  ({len(sc)} rows)")
    with pd.option_context("display.max_rows", 300, "display.width", 200):
        print(sc.to_string(index=False))
    # figure: ACTIVITY daily-r heatmap per river (rows=species, cols=year), annotated with %err
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        act = sc[sc.metric == "activity"]; years = sorted(act.year.unique())
        sps = ["Chinook", "Coho", "Atlantic", "Rainbow", "Brown"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, site in zip(axes, ["Ganaraska", "Credit"]):
            grid = np.full((len(sps), len(years)), np.nan); txt = [["" for _ in years] for _ in sps]
            for i, s in enumerate(sps):
                for j, y in enumerate(years):
                    row = act[(act.site == site) & (act.species == s) & (act.year == y)]
                    if not len(row): txt[i][j] = "n/a"; continue
                    rr = pd.to_numeric(row.iloc[0]["daily_r"], errors="coerce")
                    pe = pd.to_numeric(row.iloc[0]["pct_err"], errors="coerce")
                    if not pd.isna(rr): grid[i, j] = rr
                    parts = ([f"r={rr:.2f}"] if not pd.isna(rr) else []) + ([f"{pe:+.0f}%"] if not pd.isna(pe) else [])
                    txt[i][j] = "\n".join(parts) if parts else "n/a"
            ax.imshow(grid, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(years))); ax.set_xticklabels(years)
            ax.set_yticks(range(len(sps))); ax.set_yticklabels(sps)
            ax.set_title(f"{site} - ACTIVITY daily-r + % total error")
            for i in range(len(sps)):
                for j in range(len(years)): ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8)
        fig.suptitle(f"Deployment scorecard {min(years)}-{max(years)} - model v6 vs VIDEO technician counts "
                     "(green=high daily correlation; Atlantic shown raw, review-only at deploy)", y=1.03, fontsize=11)
        _bump(fig); fig.tight_layout()
        figp = config.FIGURES_DIR / "deployment_scorecard_2022_2025.png"
        fig.savefig(figp, dpi=140, bbox_inches="tight"); print(f"wrote {figp}")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


def main():
    """CLI entry point: parse args and dispatch --check-tech / --year / --scorecard."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-tech", action="store_true")
    ap.add_argument("--year", type=int)
    ap.add_argument("--scorecard", action="store_true")
    ap.add_argument("--shard-dirs", nargs="+", default=None)
    ap.add_argument("--footage-roots", nargs="+", default=[str(config.FOOTAGE_ROOT_FAST), str(config.FOOTAGE_ROOT)])
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--up-thresh", type=float, default=50.0)
    ap.add_argument("--down-thresh", type=float, default=300.0)
    ap.add_argument("--atl-floor", type=float, default=0.60,
                    help="species_conf floor for Atlantic in ANALYSIS counts; deploy keeps Atlantic review-only (AUTO_FLOOR 1.01)")
    args = ap.parse_args()
    if args.check_tech: cmd_check_tech([2020, 2021, 2022, 2023, 2024, 2025])
    elif args.scorecard: cmd_scorecard()
    elif args.year: cmd_year(args.year, args)
    else: ap.error("one of --check-tech / --year / --scorecard required")


if __name__ == "__main__":
    main()

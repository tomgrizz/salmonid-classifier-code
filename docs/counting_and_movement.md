# Counting & Movement — Methodology and Approach

How the system turns raw inference into the two numbers OMNRF cares about — **activity** (how many
fish) and **net upstream movement** (up − down) — and how each is validated against the technician
counts. Written 2026-06-13 from the full-year 2025 corpus analysis (clean_v2 detector + DINOv3 v6
classifier + BotSORT). Companion tools live in `scripts/_corpus_*`/`_net_*`; deep history in
`docs/experiment_log.md` and `docs/net_upstream_milling_2026-06-11.md`.

## 1. Two metrics, two confidence levels

| Metric | Definition | Use | Status |
|---|---|---|---|
| **Activity** | total fish per species per day (direction-agnostic) | run timing, relative abundance, year-over-year | **deploy-grade** (daily-r 0.98–0.99; Chinook ±2% at min_frames=3) |
| **Net upstream** | up − down per species per day (what OMNRF reports) | escapement / run magnitude | **good with the asymmetric rule + calibration** (net-r 0.95–0.98; magnitude within a few % for Chinook) |

Activity is the robust primary. Net is intrinsically harder (downstream fish are weakly observable
and milling is ambiguous) and needs the direction handling in §5.

## 2. Pipeline & per-track record

Each video → RT-DETR (clean_v2) detection → BotSORT track → DINOv3 v6 species on the best crop. Per
track we persist (`inference/schema.py`): `predicted_class`, per-class `mean_prob_*`, `delta_x`
(end_x − start_x), `direction`/`upstream`/`entrance_side`/`exit_side`, `site`. Counting/movement are
computed **offline from these fields**, so any rule below is re-derivable for every year **without
re-running inference**.

## 3. Dating every clip — use file mtime, not the folder

Per-day counts need a reliable date for **every** video, including the ~85% the technicians never
reviewed. Two sources, one trap:
- ❌ **Timestamp folder** (`.../25  10  14  10  40/...`) is a **batch-start reference**, ~1 day off
  the true clip time — it matched tech dates only **9.5%** of the time.
- ✅ **File mtime** (preserved by robocopy `/COPY:DAT` onto the F: NVMe) is the true recording time —
  it matched the tech event date **100%** (5,016/5,016, zero off-by-one). **Use mtime.**

## 4. Technician ground truth (full year, the right files)

| River | File(s) | Notes |
|---|---|---|
| Ganaraska | `Ganaraska_Counter_Analysis_2025/all_counts_gan.csv` | complete event list; `species_main` (**incl. Brown**), `movement` Up/Down/Stay, `count`, `ts` |
| Credit | `Credit_Counter_Analysis_2025/data/raw_data/Credit_river_Counts_*.csv` + reviewed `fish_counts_export_*.csv` | raw counter (Apr–Oct, Vaki-auto species) + reviewed fall exports (human species) |

**Pitfalls (both learned the hard way):**
- **Do NOT use `daily_Ganaraska.csv`** — that derived aggregate silently **drops Brown Trout** and
  under-states the others (Chinook 22,896 vs the true 25,418). Use `all_counts_gan.csv`.
- **Skip `*_vaki_matched.csv`** for species — Vaki auto-species is unreliable (over-calls Coho ~28×);
  use reviewed exports / `species_main`.
- Tech has a **"Salmonid"/unclassified bucket** (~2,300 Ganaraska + ~2,100 Credit) the counter
  couldn't put to species but the model does — it absorbs part of the trout/Coho magnitude gap.

## 5. Activity counting

Per track, count 1 toward `(species, mtime-date, site)` if it clears the **deploy review-triage
species_conf floor** (`inference/config.py::auto_count_species_conf`: Chinook 0.50, Coho/Rainbow/
Brown 0.60; **Atlantic always routes to human review**, never auto-counted). Aggregate to daily and
seasonal totals.

### Count from the parquets at min_frames=3 (DEFAULT), not tracks.sqlite

`tracks.sqlite` only stores tracks ≥ `min_frames_for_track` (5), but the **per-frame parquets keep
every emitted detection** (the 5-frame filter is applied only when building the track summaries). So
re-deriving counts from the parquets at a **lower min_frames recovers short tracks the counts dropped**
— in 2025 that's **+8,214 tracks (+15%)** the detector+tracker found but never counted.
`_corpus_fullyear_vs_tech.py` (and `_track_reaggregate.py`) re-derive species + `delta_x` per
`track_id` straight from the parquet and count at `--min-frames` (default **3**). No re-inference.

**Why 3:** Chinook *over*-counts below 3 (its 1–2-frame tracks are mostly fragments of fish already
counted), while the minority species' short tracks are real **co-occurring** fish (a Coho/Rainbow
briefly tracked in a busy Chinook clip). So 3 is the sweet spot — Chinook honest, trout/Coho
recovered, **daily-r unchanged** (recovered short tracks add count, not timing noise).

**Full-year 2025 vs tech, min_frames 5 → 3** (`_corpus_fullyear_vs_tech.py --min-frames 3`):

| species | Ganaraska %err (5→3) | Credit %err | Ganaraska daily-r |
|---|---|---|---|
| Chinook | −5% → **+2%** | −7% → **−0%** | **0.991** |
| Rainbow | −21% → **−15%** | −33% → −23% | 0.980 → **0.987** |
| Coho | −30% → −25% | small-n | 0.980 |
| Brown | −20% → −17% | small-n | 0.983 |

Activity **daily-r is 0.98–0.99 for every species** and **Chinook is now essentially exact**. The
residual on the trout/Coho is the *never-emitted* fish of §6c (not in the parquets — needs a tracker
re-run) plus tech's unclassified "Salmonid" bucket. **Activity is deploy-grade.**

## 6. Movement — net upstream

**Direction** comes from the sign of `delta_x` plus the per-site upstream convention
(`inference/config.py::upstream_dirs`): **Ganaraska up = Right (Δx>0), Credit up = Left (Δx<0)**. A
displacement threshold decides whether a track is directional at all (below it = "Stay"/Unknown,
excluded — mirroring the technician "Stay" category).

### 6a. Why a single (symmetric) threshold can't win — the up/down decomposition

Decomposing model vs tech into up and down separately (`_net_direction_decomp.py`) is the key insight:

| Chinook | model UP / tech UP | model DOWN / tech DOWN |
|---|---|---|
| Ganaraska @ 0px | 94% ✓ | **193%** ✗ |
| Credit @ 0px | 87% ✓ | **152%** ✗ |

**Upstream is captured well (~90%); downstream is over-counted ~2×** at a low threshold — milling
fish drift slightly back and get logged as a downstream *passage* instead of a "Stay". A symmetric
threshold faces a dilemma: low keeps the spurious downs (net too low); high (300px) kills the spurious
downs **but also discards real upstream** (Chinook up drops to 66%). Either way net lands ~−20–28%.

### 6b. The fix — an ASYMMETRIC direction rule (liberal up, strict down)

Count **upstream** at a low bar (`|Δx| ≥ ~50px`) but **downstream** only when unambiguous
(`|Δx| ≥ ~300px`). Biologically sound: these are hard upstream runs, so a marginal-displacement
"downstream" track is almost always milling, not a true down passage.

**Net %err, symmetric 300px → asymmetric (up 50 / down 300):**

| species | Ganaraska | Credit | Ganaraska net daily-r |
|---|---|---|---|
| Chinook | −28% → **−4%** | −33% → **−13%** | 0.970 → **0.983** |
| Coho | −43% → −27% | −25% → −9% | 0.966 → 0.975 |
| Rainbow | −37% → −12% | −44% → −29% | 0.911 → 0.954 |
| Brown* | −17% → +25% | −26% → −8% | 0.946 → 0.971 |

\*Brown is small-n (399 Ganaraska) and **over-corrects** under aggressive asymmetry — keep it
conservative or treat as noise. **Chinook net snaps to within a few percent in both rivers**, and
every species' net daily-r improves. No threshold (symmetric or asymmetric, even 0px) ever produces a
net *over*-count for the main species — the milling-over-count feared in the earlier clip-level
analysis does not appear in this per-track full-year computation.

### 6c. Residual = multi-fish under-tracking (part post-recoverable, part not)

The residual under-count (after asymmetry + min_frames=3: Coho-Gana ≈ −23%, Rainbow-Credit ≈ −27%) is
the **multi-fish effect**: in busy clips BotSORT forms fewer tracks than the technician logs fish,
mostly losing the minority species co-occurring with Chinook (worse at the denser Credit tunnel — the
tell that it's tracking, not direction). The per-species diagnostic (`_coho_rainbow_diag.py`,
single-species videos) splits *where* the fish are lost: **Rainbow miss = 86% tracking / 14%
classifier; Coho = 57% tracking / 43% classifier** (Coho→Chinook).

Two parts, two fixes:
- **Short tracks (formed but <5 frames) are recoverable in post** — they sit in the parquets; the
  `min_frames=3` re-derivation (§5) claws them back (Rainbow −21%→−15%; Coho moves little, confirming
  its loss is mostly the next bullet).
- **Never-emitted fish are not** — the tracker never spawned a track (the co-occurring fish fell below
  BotSORT `new_track_thresh` 0.9). Recovering those needs a **targeted tracker-only re-run** (lower
  `new_track_thresh` on the dense fall clips; the detector already found them, so **no retrain**). The
  Coho→Chinook half additionally wants the classifier hard-pairs.
- **Reported magnitude** is meanwhile closed by a **per-river-per-species calibration factor**
  (`tech/model`, fit on 2025; near ×1 after asymmetry + min_frames=3). A scalar doesn't change
  correlation, so daily-r stays 0.95–0.99.

## 7. Known limitations & caveats

- **Net < activity in reliability.** Downstream is weakly observable; milling is ambiguous. Trust
  activity first.
- **Multi-fish under-tracking** caps net (and activity) recall in dense clips — the dominant residual.
  The real fix is better multi-fish tracking / cross-clip re-ID; the calibration factor is a stopgap.
- **Calibration factors are fit on 2025** and assume similar per-river fish density in other years —
  a first-order correction, not exact, for the tech-free years (2020–2024).
- **Brown / Atlantic are low-n** — Brown over-corrects under asymmetry; Atlantic is review-only
  (auto-count = 0 by policy).
- **Credit early-season species are Vaki-auto** (less reliable than the reviewed fall).

## 8. Tooling

| Script | Purpose |
|---|---|
| `_corpus_year_gate.py` | GO/NO-GO daily-r gate vs reviewed tech (per year, before trusting it) |
| `_corpus_year_analysis.py` | daily model-vs-tech on reviewed videos (tech-date) |
| `_corpus_fullyear_vs_tech.py` | **full-run** activity + net **re-derived from the parquets at `--min-frames` (default 3)**, mtime-dated, totals + daily-r + figures; asymmetric `--up-thresh/--down-thresh` |
| `_track_reaggregate.py` | re-derive counts from parquets across a min_frames sweep vs tech (short-track recovery) |
| `_coho_rainbow_diag.py` | tracking-vs-classifier split for a species (single-species videos) |
| `_net_threshold_sweep.py` | per-river net %err + daily-r across thresholds; calibration factors |
| `_net_direction_decomp.py` | up vs down decomposition; symmetric vs asymmetric comparison |

All read the saved corpus **parquets / `tracks.sqlite`** + F: footage metadata — CPU only, safe to run
while inference is live, and reusable for every year as it completes.

## 9. Deploy implications (`talking-to-counters`)

1. **Activity** — ship as-is (triage floors already in `config.py`); **count at min_frames=3** by
   re-deriving from the per-frame detections, not the ≥5-frame track summaries (§5).
2. **Net direction** — replace the single `min_displacement` with the **asymmetric rule**
   (up ≈ 50px, down ≈ 300px) in the direction logic; in `config.py` lowering the symmetric knob alone
   also requires dropping `min_displacement_frac` (0.234×1280 = 300 otherwise caps it).
3. **Net magnitude** — apply the per-river-per-species calibration factors at counting time.
4. **Atlantic** — always to the review queue.

## 10. Cross-year validation — 2024 Credit (held-out)

The method was *developed* on 2025. **2024 is the first fully held-out test** — the model never saw a
2024 frame or tech count. **Both rivers** have 2024 tech, from two different sources:
- **Credit** → Vaki **"AbsDb" reviewed export** (`model_pro/tech_data/data/tech_reviewed/2024/`, two
  non-overlapping season xlsx; `sortname` = human species, `Direction` Up/Down, one reviewed row = one
  fish) → gives **activity AND net**.
- **Ganaraska** → `lomu_interview/data/Ganny_Total.csv` — official OMNRF **daily NET** (SPC code,
  signed `CNT`, 2017–2024). NET only (no separate up/down), so Ganaraska gets a **net** comparison; its
  totals match the published `salmonid-classifier-paper/tables/ganaraska_annual_counts.csv` (2024
  Chinook 19,514, Rainbow 9,032, Coho 3,201), confirming it's the right ground truth.

Tool: `scripts/_corpus_2024_vs_tech.py` (both rivers, one parquet pass; identical min_frames=3 +
asymmetric net + file-mtime method as §5–6).

**Ganaraska NET — the big river, the metric OMNRF reports (held-out):**

| species | net model / tech (%err) | net daily-r |
|---|---|---|
| **Chinook** | 18,121 / 19,513 (**−7%**) | **0.988** |
| **Rainbow** | 8,390 / 8,634 (**−3%**) | **0.989** |
| **Coho** | 2,370 / 3,200 (−26%) | **0.989** |
| Brown | 383 / 336 (+14%) | 0.874 (small-n) |
| Atlantic | 0 / 4 (review-only) | — |

**Credit (held-out):**

| species | activity model/tech (%err) | activity daily-r | net model/tech (%err) | net daily-r |
|---|---|---|---|---|
| **Chinook** | 11,849 / 13,710 (−14%) | **0.994** | 6,187 / 6,500 (−5%) | 0.976 |
| Rainbow | 441 / 794 (−44%) | 0.840 | 218 / 268 (−19%) | 0.944 |
| Brown | 43 / 50 (−14%) | 0.947 | 24 / 28 (−14%) | 0.796 |
| Coho | 36 / 116 (−69%) | 0.282 (small-n) | 17 / 60 | — |
| Atlantic | 0 / 55 (review-only) | — | 0 / 23 | — |

**Read:** on a fully held-out year the **daily curve reproduces the technician at r ≈ 0.99 for the
dominant species in BOTH rivers** (Ganaraska Chinook/Rainbow/Coho net-r 0.988–0.989; Credit Chinook
activity-r 0.994) — timing/shape generalizes, the model is "not getting lost." Ganaraska net magnitude
is within **−3% (Rainbow) to −7% (Chinook)** — deploy-grade out of the box. The larger magnitude gaps
(Coho −26%, Credit-Rainbow −44%) are the §6c multi-fish under-tracking residual: note the daily-r stays
~0.99, i.e. a near-constant scale factor closed by the per-river-per-species calibration, not a timing
failure.

**Caveats specific to 2024:** (a) Credit tech is the **raw AbsDb**, which over-logs downstream the same
way the model mills (so raw-AbsDb net is a loose secondary); (b) the old
`model_pro/tech_data/data/metrics_*.csv` (e.g. "Tech Net 8,143") is a **superseded analysis of a worse
model** (−54%..−160% net) — do **not** anchor to it. Historical tech for the same validation as
inference completes: **Ganaraska** daily net 2017–2024 in `Ganny_Total.csv`; **Credit** 2018–2023 in
`tech_reviewed/2023-2018/FINAL_CREDIT_<year>.csv`.

### 10a. 2023 — the record year (held-out, NET both rivers)

2023 is the largest run in the 2017–2025 series (Ganaraska **28,294** Chinook). Both rivers have only
official daily-**NET** tech for the pre-2024 years (Ganaraska `Ganny_Total.csv`; Credit
`FINAL_CREDIT_2023.csv` — both reconcile to the published annual counts). Tool:
`scripts/_corpus_historical_vs_tech.py --year 2023` (reusable for 2018–2023; NET-only, same
min_frames=3 + asymmetric method).

| species | Ganaraska net model/tech (%err) | Gana net-r | Credit net model/tech (%err) | Credit net-r |
|---|---|---|---|---|
| **Chinook** | 26,138 / 28,294 (**−8%**) | **0.969** | 7,586 / 7,938 (**−4%**) | **0.996** |
| **Coho** | 2,351 / 2,491 (**−6%**) | **0.979** | 41 / 65 (small-n) | 0.800 |
| Rainbow | 3,790 / 5,520 (−31%) | 0.843 | 430 / 584 (−26%) | 0.972 |
| Brown | 287 / 160 (+79%, small-n) | 0.782 | 41 / 30 (small-n) | 0.732 |
| Atlantic | 0 / 15 (review-only) | — | 0 / 40 (review-only) | — |

**Record density did NOT break the model** — Chinook net −8%/−4% (Credit net-r 0.996), Coho −6%
(r 0.979). Brown's over-count is small-n + the known asymmetric-net over-correction.

**Rainbow net (−31% in 2023 vs −3% in 2024) — ROOT-CAUSED 2026-06-15, it is steelhead MILLING, not
density (`_rainbow_deep_diag.py`, `_rainbow_spring_net.py`).** Earlier drafts blamed "record Chinook
crowding out fall Rainbow" — **that is wrong**: ~90% of Rainbow are **spring** steelhead (Mar–Jun, a
season that is ~93% Rainbow with NO Chinook present), fall Rainbow are only ~600/yr. The model does NOT
miss Rainbow — in spring it classifies 96–97% of tracks as Rainbow (tech 99%) and produces **131–159%
as many tracks as tech fish**. The entire net gap is **movement attribution on milling steelhead**:
spring steelhead hold/mill in the tunnel, and that behaviour is **flow-year-dependent** — 2023 (milly):
30% of Rainbow tracks are "Stay" (|Δx|<50px), only 36% make a strong (≥300px) upstream move, 31% of
movers drift down; 2024 (clean): 17% Stay, 50% strong-up, 24% down. The asymmetric rule then discards
the 2023 milling majority → −44% spring net, but matches 2024. **No per-clip lever fixes it** —
up-threshold (even up≥20px → 2023 still −36%), conf floor, and min_frames are all flat; a fixed
calibration fails too (model/tech ratio 1.31 in 2023 vs 1.59 in 2024). This is the **cross-clip
steelhead-identity problem**: a steelhead holding across many clips is logged by the technician as one
net-up passage, but the per-video pipeline sees milling (Stay/down) in each clip and cannot reconstruct
the multi-clip trajectory.

**Post-hoc fix attempts — ALL tested 2026-06-15, none generalizes (so the displacement rule stays):**
- **Within-clip de-fragmentation** (`_rainbow_step1_defrag.py`): a NO-OP — only ~1.06× merge, i.e. the
  model is NOT over-fragmenting one fish into many *within* a clip. Rules out that hypothesis.
- **Season-aware net** (spring Rainbow = upstream migrant): best daily-r (0.95–0.99) but overshoots the
  total by +25–49%, inconsistently (ignores kelts + cross-clip duplicates). Confirms *timing* is fully
  recoverable; magnitude is not.
- **Crossing / exit-side rule** (`_crossing_net_test.py`, count up = enter downstream-half → exit
  upstream-half, matching how the technician counts): **strictly WORSE** — under-counts EVERY species
  ~35–50% (Chinook 2025 −0%→−32%), because **tracks do not span the midline** (the detector catches each
  fish over too short a slice of its transit, so it's only ever seen on one side of x=640). The
  displacement rule survives this because a +50px move *anywhere* counts.
- **Fixed calibration** fails: tech/model ratio is 1.46 (2023) / 1.03 (2024) / 1.11 (2025).

**Conclusion:** the **asymmetric displacement rule is the best definition for the tracking we have**
(Chinook/Coho within a few %, all species' daily-r 0.84–0.99). The crossing rule is correct *in spirit*
(tech-matched) but needs **complete entry→exit tracks we don't have** — so the genuine remaining lever
is **more complete tracking (detector/tracker tuned to hold a fish across the whole tunnel) + a re-run**,
NOT a post-hoc rule. Until/unless that re-run happens: Rainbow **timing is reliable** (daily-r 0.84–0.99);
Rainbow **net magnitude in milling years (e.g. 2023 −31%) is a documented caveat**.

**VISUAL CONFIRMATION (Tom, 2026-06-15, `_rainbow_clip_inspect.py` rendered clips).** Annotated clips of
representative spring-2023 Ganaraska Rainbow tracks were reviewed by eye. Findings, domain-expert
confirmed: (1) the **committed-mover** case (`ONE_SIDED_UP`, e.g. a clear steelhead +97px upstream but
only ever seen on the upstream side of the midline) — the displacement rule correctly counts it, the
crossing rule wrongly drops it (proves *why* crossing under-counts ~40%); (2) the **milling** cases are
**genuinely ambiguous even to the expert** — a 30-s clip does not contain a milling fish's eventual
up/down fate, so the per-clip model honestly abstaining (Stay) is *correct behaviour*, not a bug; (3)
milling clips contain **many fish at once** ("a ton of milling fish"), so the regime is hard two ways:
per-fish direction is uncallable AND the crowd exceeds clean tracking. **This is an information limit of
per-clip counting, not a tracking-quality defect** — the technician's number comes from *continuous*
observation. **Settled disposition:** report Rainbow as timing (reliable) + magnitude as a calibrated
estimate with an honest ~±15–20% band; do NOT chase per-clip direction further. Only continuous /
cross-clip tracking (mimicking the live watch) could close it, and it is capped by the crowding — future
work, not a deploy blocker. Chinook/Coho/Brown stay deploy-grade.

### 10b. Calibration + the commit-fraction signal (2026-06-15, `_trajectory_signal.py`)

Full per-track displacement analysis (Ganaraska, all species, 2023–25; cache
`G:/Projects/traj_signal_cache.parquet`) **overturned the "Rainbow mills more" framing**: pooled, all
species have **near-identical dx distributions** (same zero-spike + committed tail) — the direction rule
is NOT biased against Rainbow.

**Per-species calibration (tech_net / model_asym_net):** Chinook **1.05** (CV 3%, stable) · Rainbow
**1.20** (CV 16%, year-variable) · Coho **1.24** (classification, not direction) · Brown **0.75** (model
*over*-counts, small-n).

**The signal:** the net under-count is driven NOT by milling/stay-fraction (r −0.40, weak) but by
**commit-fraction** (% tracks with |dx|≥300px): **r = +0.84** for the direction-limited species
(Chinook+Rainbow), fit **under% ≈ −71 + 1.41·commit%** (Rainbow-2023 commit 33% → pred −24%, actual −31%;
Rainbow-2024 commit 45% → pred −6%, actual −3%). A *less-committed* run year under-counts predictably
more. **commit% is model-side (no tech needed) → a self-calibrating Rainbow/Chinook magnitude
correction usable in tech-free years**, and strictly better than a fixed factor (auto-adjusts to each
year's commitment). **Coho is the exception (corr −0.76, opposite sign) — its gap is classification
(Coho→Chinook), fixed by classifier hard-pairs, not movement.** Caveat: the fit is 6 points — validate/
refine on 2020–2022 (Ganaraska tech back to 2017 in `Ganny_Total.csv`) before trusting it as deploy.

### 10c. Peak-day under-count is the FOOTAGE, not the tracker (2026-06-15, `_peak_tracker_diag.py`, `_render_crowd_clip.py`)

The full-year Rainbow −31% is dominated by the high-volume peak days (Apr 7–10 2023: tech net 2,889).
Investigated end-to-end and **eliminated every model-side cause:**
- **NOT crowding:** the most crowded clip across 4 peak days has only **3 simultaneous detections** (mostly
  1 fish in frame — rendered clip `figures/rainbow_clips/crowd_peak_2023.mp4`). Fish arrive in rapid
  succession (clips ~9s apart), not in a pile.
- **NOT detection recall:** the model produces ~as many Rainbow tracks as the tech counted fish (2,665 vs
  2,889). It finds them.
- **NOT the direction rule per se:** of 2,665 tracks only 875 (33%) clear the threshold → −70% net.
- **NOT tracking quality / tracker params:** the dropped "Stay" tracks are **well-observed (median 19
  detections ≈2s), not fragments**; and a loose-tracking proxy (aggressively merge within-clip tracks =
  lower IOU / higher track_buffer) makes net **WORSE** (−70%→−74%), never better. So tuning BotSORT
  `match_thresh`/`track_buffer` (the old Gradio app's knobs — which our config already mirrors) would
  **hurt**, not help. **Do not re-run for tracker tuning.**

**Root cause (CORRECTED + final, after Tom pushed):** clips are NOT short — they are **~16s at 10fps**,
and each fish is on screen **~7s** (`_chinook_vs_rainbow_deep.py`). The real Rainbow-specific factor:
at matched density, **spring steelhead traverse ~HALF the tunnel of fall Chinook per clip (167–184px vs
340px; %commit 14% vs 26%)** — they hold/work against high spring flow and genuinely net-move little in
any 7s window, while Chinook charge through. Chinook nets close (−8%) because it clears the threshold;
40% of Rainbow net <50px → "Stay".

**Every model-side lever tested + ELIMINATED with data (do not re-try):**
- better direction math — `slope` (full-track linear trend) and `netfull` (true first/last) leave Rainbow
  at −29 to −31% (`_slope_direction_test.py`); the milling fish have NO hidden upstream trend.
- tracker IOU/`match_thresh` & frame-gap/`track_buffer` — loose-tracking proxy makes net WORSE (§10c).
- cross-clip stitching — worse in the flood (−70%→−88%).
- **detection recall — re-detected 120 spring Rainbow clips through the real pipeline at box_score 0.30
  vs 0.10 (`_rainbow_recall_test.py`): IDENTICAL (217 tracks, net 68, 25 frames, 184px x-range, 72px |dx|
  — flat to the pixel). The detector already catches the fish on every visible frame; 184px is the fish's
  REAL in-clip movement, not a detection clip.**

**So the residual is genuine slow-steelhead behaviour** — a fraction net-move <50px per 7s window because
their real passage spans minutes / many clips; the model captures each window faithfully, the technician
wins only by watching the whole sequence. **No downstream change (rule, tracker, detector, even re-run)
recovers it** → the commit-fraction calibration (§10b) is the honest, mechanism-grounded aggregate fix.

### 10d. Flow vs holding (2026-06-15, Tom's gauge data; `_flow_vs_holding.py`, `_flow_daily_test.py`, `_flow_daily_multiyear.py`)

Tested whether the holding is driven by river flow (Ganaraska gauge 02HD012; daily discharge in
`data/daily_ganny_flow_2022_2026.csv` from a fresh WSC QRD export, + HYDAT `data/daily_ganny_waterflow.txt`).
**2024 spring discharge is a genuine WSC data gap** (0 April days — gauge outage), so only 2023 + 2025
are testable.
- **Flow IS a real DAILY driver of holding** — within-season corr(flow, %Stay)=+0.36 / corr(flow, |dx|)=
  −0.51 pooled; **survives day-of-season DETREND (+0.24)** so it's not just the seasonal confound; 2025
  is strong (+0.60 / −0.79). High-flow days → fish hold → fewer passages registered. Whole-population
  median split (2023): 2×flow → +10pt %Stay, −30% speed.
- **But flow does NOT explain the YEAR-to-year difference** — 2023 and 2025 had near-identical mean
  spring flow (4.2 vs 4.3 m³/s) yet 47% vs 31% holding. 2023 was median-flow (rank #22/48), a
  high-*holding* year for a reason flow doesn't capture (temperature / run timing / fish condition —
  beyond this data). The year baseline drives the under-count variance (−44% vs −10%).
- **Implication:** keep the model-side **commit-fraction calibration**, NOT a flow-based one — commit-
  fraction measures holding directly (captures both the daily flow effect AND the unexplained year
  baseline); a flow correction would fail on 2023 (same flow as 2025, 2× the gap). Flow is a confirmed
  *mechanism*, not a usable *predictor* of the annual gap.

### 10e. CANDIDATE reporting basis: per-behaviour metric (activity vs net) (2026-06-16, `_activity_vs_net_report.py`)

The up/down decomposition (`_rainbow_run_graph.py`) showed the technician resolves ~98.5% of Rainbow to a
direction (only 208 "Stay" of 13,659 in 2025) while the model leaves 17–30% as "Stay" (the holding fish)
→ the **net** under-count IS those holding fish, and it is year-variable. But **activity** (track count)
is direction-agnostic, so it sidesteps the holding problem entirely.

**Principle (general, behaviour-based — not fit to the data):** report a species on **NET** if its fish
make committed directional passages (model measures direction well); report on **ACTIVITY × factor** if
its fish **HOLD/MILL** (per-clip direction corrupted but every fish is still seen). The per-species
outcome falls out of biology: Chinook/Coho/Brown commit → net; Rainbow holds → activity.

**Empirical support (Ganaraska, 2023–25, leave-one-YEAR-out = honest out-of-sample):**

| species | basis | LOO mean\|err\| | deploy factor (model→tech net) |
|---|---|---|---|
| Chinook | **NET** | 5% | net ×1.05 |
| Coho | **NET** | 15% | net ×1.22 |
| **Rainbow** | **ACTIVITY** | **8%** (net basis = 21%) | **activity ×0.61** |
| Brown | NET (small-n) | 28% | net ×0.72 |

Rainbow activity×0.61 lands 2023/24/25 at −4% / +8% / −6% — and the **activity advantage survives LOO**
(8% vs net's 21%), so it is not just in-sample fitting. Behaviour-matched overall LOO 14% < always-net
17% < always-activity 20% (the whole gain is the Rainbow switch).

**⚠️ Status = CANDIDATE, provisional.** Caveats (Tom flagged the data-fitting risk, correctly): only 3
years; factors are empirical (the metric *choice* is principled, the *number* is fit); LOO mitigates but
does not eliminate the thinness; activity still depends on the model SEEING the fish (the 2023 fall-Rainbow
miss is baked into the factor, not fixed). **VALIDATE the 0.61 factor on 2020–2022 as they complete before
trusting it as deploy.** If it holds, activity-based Rainbow reporting may make the session-level re-ID
build (§ the open Rainbow plan) unnecessary for the net total — pursue re-ID only if the up/down split
itself is needed.

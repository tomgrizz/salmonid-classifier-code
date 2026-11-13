# Software, models and methods used by this code, and how to cite them

This page lists every outside piece of software, every pretrained model, every dataset and
every named statistical method that the code in this repository relies on, with the version
that was used, what it does for this project, its licence, and the reference its authors ask
for. It exists so that the people whose work this analysis is built on are credited, and so
that a reader can rebuild the same software environment.

How the list was made:

1. **Inventory from the code.** Every `import` line in `salmonid/` (including the analysis
   scripts in `salmonid/paper/`), in the manuscript repository's `scripts/` and `docs/`, the
   PowerShell runners and the YAML configuration files was read. Nothing was added from memory.
2. **Versions from the machines that ran the work.** Versions are those installed on
   2026-11-10 in the two Python environments: the *base* environment (tracking, counting,
   statistics, figures) and the *DEIMv2* environment (training and running the detector).
3. **Citations checked at the source.** Each reference was confirmed on 2026-11-10 against
   the publisher's DOI record (Crossref), arXiv, Zenodo/DataCite, PubMed, the conference
   proceedings site, or the project's own citation file. Anything that could not be confirmed
   is marked **UNVERIFIED**.

The matching reference entries are in [`software.bib`](software.bib) (BibTeX, same keys as
the tables) and [`_citation_cff_references.yml`](_citation_cff_references.yml) (ready to
paste into `CITATION.cff`).

> **Two cautions on versions.**
>
> 1. *The Tier 1 environment is settled.* Everything that recomputes the paper's statistics
>    and figures from the committed tables ran under Python 3.11.9 with NumPy 2.4.4, pandas
>    2.3.3, SciPy 1.15.3 and Matplotlib 3.10.8, which is what `requirements.txt` pins.
> 2. *The tracker version is not yet settled.* Two copies of BoxMOT are installed: **10.0.52
>    in the base environment** and **21.0.0 in the DEIMv2 environment** (`.venv_deim`). The
>    code's own documentation names 10.0.52 in the base environment. **Which of the two
>    produced the published counts is being confirmed separately**, and this page reports both
>    until it is. Two facts bear on it and are recorded here without drawing the conclusion:
>    BoxMOT 10.0.52 declares that it requires NumPy 1.24.4, while the base environment now
>    holds NumPy 2.4.4; BoxMOT 21.0.0 declares NumPy 2.2 or newer. Both versions carry the
>    same licence (AGPL-3.0), so the licence note below holds either way; only the version
>    number and the citation key change (`brostrom2024boxmot` for 10.0.52,
>    `brostrom2026boxmot` for 21.0.0).
>
> For any other library, the versions below are what is installed now; the per-run
> `run_manifest.json` files are the authority for a single run.

---

## 1. Python libraries

Version columns: **base** = base Python environment, **DEIMv2** = detector environment. A
dash means the library is not used in that environment.

| Library | Version (base / DEIMv2) | What it does here | Where it is used | Licence | Citation key |
|---|---|---|---|---|---|
| NumPy | 2.4.4 / 2.4.4 | The arithmetic underneath every count, statistic and image array. | Almost every module | BSD-3-Clause | `harris2020numpy` |
| pandas | 2.3.3 / - | Reads the daily count tables and groups them by river, species and year. | `count/`, `results/`, `paper/`, `figures/`, `data/prepare_dataset` | BSD-3-Clause | `mckinney2010pandas`, `pandas2025software` |
| SciPy | 1.15.3 / 1.17.1 | Pearson and Spearman correlations, the normal quantile for the Wilson interval, and (inside DEIMv2) the matching of predicted to true boxes during training. | `results/agreement_stats`, `results/classification_ci`, `paper/correlation_stats`, `paper/count_tables`, `count/corpus_*_vs_tech`, `eval/corpus_year_gate`; DEIMv2 engine | BSD-3-Clause | `virtanen2020scipy` |
| Matplotlib | 3.10.8 / 3.10.9 | Draws every figure. | `figures/`, `results/compute_concordance`, `count/corpus_multiyear_vs_tech` | Matplotlib licence (PSF-style, permissive) | `hunter2007matplotlib`, `matplotlib2025software` |
| Apache Arrow (`pyarrow`) | 22.0.0 / - | Reads and writes the Parquet files that hold per-frame detections and per-track summaries. | `count/`, `data/`, `eval/`, `util/schema` | Apache-2.0 | `apachearrow2025` |
| OpenCV (`opencv-python`) | 4.13.0 / 4.13.0 | Reads video frames for the detection pass, reads and writes images, and builds the synthetic training composites. | `detect/detect_store`, `detect/track_from_store`, `data/`, `figures/fig_bbox_grid`, `figures/fig_species_morphology` | Apache-2.0 | `bradski2000opencv` |
| Pillow | 12.1.1 / 12.2.0 | Opens frames for the detector and saves figure files. | `detect/deimv2_infer`, `eval/*bbox_gate`, `figures/manuscript_style`; DEIMv2 engine | MIT-CMU | `murray2026pillow` |
| Decord | 0.6.0 / 0.6.0 | Jumps straight to a chosen frame of a video, so the tracking pass reads only the frames that hold a detection. | `detect/track_from_store` (used for all published seasons), `detect/detect_store` (optional), `data/prepare_dataset` | Apache-2.0 | `decord2021` |
| PyTorch (`torch`) | 2.11.0+cu128 / 2.7.0+cu128 | The deep-learning framework the detector and the re-identification network run in. | `detect/deimv2_infer`; DEIMv2 engine; BoxMOT | BSD-3-Clause | `paszke2019pytorch` (optional: `ansel2024pytorch2`) |
| TorchVision | 0.26.0+cu128 / 0.22.0+cu128 | Image resizing and the training-time image augmentations. | `detect/deimv2_infer`; DEIMv2 engine | BSD-3-Clause | `torchvision2016` |
| BoxMOT (`boxmot`) | **10.0.52** (base) / **21.0.0** (DEIMv2); which one produced the published counts is being confirmed | The tracker: the BoT-SORT implementation that links detections of one fish across frames, and the ByteTrack implementation used for the tracker comparison. | `detect/track_from_store` | **AGPL-3.0**, both versions (see Licence notes) | `brostrom2024boxmot` if 10.0.52, `brostrom2026boxmot` if 21.0.0 |
| Transformers | 5.5.4 / 5.10.2 | Loads the earlier two-stage detectors so their boxes can be checked against hand-drawn boxes. | `eval/bbox_regression_gate` | Apache-2.0 | `wolf2020transformers` |
| Supervision | 0.27.0.post2 / - | Supplies a second ByteTrack, used only by an optional quick tracker that works from stored boxes. The tracker comparison in the manuscript uses BoxMOT's ByteTrack, so cite Supervision only if a result from this quick tracker is reported. | `count/corpus_countr_from_store` (`--tracker bytetrack`) | MIT | `roboflow2026supervision` |
| Faster-COCO-Eval | - / 1.7.2 | Scores box accuracy during detector training. | DEIMv2 engine (not imported by `salmonid/`) | Apache-2.0 | `mixaill2026fastercocoeval` |
| PyYAML | 6.0.3 / 6.0.3 | Reads the training configuration files in `configs/`. | DEIMv2 engine | MIT | `simonov2025pyyaml` |
| TensorBoard | 2.20.0 / 2.20.0 | Logs training curves. | DEIMv2 engine | Apache-2.0 | `tensorboard2025` |
| calflops | - / 0.3.2 | Reports the detector's size and computing cost when it is built. | DEIMv2 engine | MIT | `ye2023calflops` |
| GeoPandas | 1.1.3 / - | Reads the map layers and reprojects them for the study-area map (Fig. 1). | Manuscript repository, `scripts/make_fig1_map.py` | BSD-3-Clause | `fleischmann2026geopandas` |
| Shapely | 2.1.2 / - | Builds the points, lines and clipping boxes on the Fig. 1 map. | Manuscript repository, `scripts/make_fig1_map.py` | BSD-3-Clause | `gillies2025shapely` |
| tqdm | 4.67.3 / 4.68.1 | Progress bars. **Not imported by any module here**; it was only listed in an earlier `requirements.txt`. No citation needed. | none | MPL-2.0 and MIT | `dacostaluis2019tqdm` |

Libraries that are installed only because the ones above need them (for example
`huggingface-hub`, `safetensors`, `lapx`, `filterpy`, `gdown`, `pyproj`, `pyogrio`) are not
listed: the code never calls them directly, and it is not customary to cite them.

**Version-specific DOIs.** Several projects mint a DOI for each release. The papers in the
table are the citations the projects ask for; if the journal wants the exact version, add:

| Library | Version | Version DOI | All-versions DOI |
|---|---|---|---|
| pandas | 2.3.3 | 10.5281/zenodo.17229934 | 10.5281/zenodo.3509134 |
| SciPy | 1.15.3 (base) | 10.5281/zenodo.15366870 | 10.5281/zenodo.595738 |
| SciPy | 1.17.1 (DEIMv2) | 10.5281/zenodo.18736568 | 10.5281/zenodo.595738 |
| Matplotlib | 3.10.8 (base) | 10.5281/zenodo.17595503 | 10.5281/zenodo.592536 |
| Matplotlib | 3.10.9 (DEIMv2) | 10.5281/zenodo.19716234 | 10.5281/zenodo.592536 |
| Pillow | 12.1.1 (base) | 10.5281/zenodo.18604291 | 10.5281/zenodo.596518 |
| Pillow | 12.2.0 (DEIMv2) | 10.5281/zenodo.19371132 | 10.5281/zenodo.596518 |
| GeoPandas | 1.1.3 | 10.5281/zenodo.18934477 | 10.5281/zenodo.2585848 |
| Shapely | 2.1.2 | 10.5281/zenodo.17193310 | 10.5281/zenodo.5597138 |
| BoxMOT | 10.0.52 or 21.0.0 | none exists for either (Zenodo holds versions 8, 9 and 10.0.15 only) | 10.5281/zenodo.7452873 |

NumPy, PyTorch, TorchVision, Transformers, OpenCV, Decord and PyArrow do not mint per-version
DOIs; state the version number in the text instead.

---

## 2. Models, pretrained weights, datasets and algorithms

| Item | Exact version or file | What it does here | Where it is used | Licence | Citation key |
|---|---|---|---|---|---|
| **DEIMv2** (detector code and design) | Repository `Intellindust-AI-Lab/DEIMv2` at commit `0fff8d4` (24 March 2026), plus `docs/deimv2_worktree.patch` | The detector that finds each fish and names its species in one pass. | Training (`runners/original/`, `configs/`), `detect/deimv2_infer` | Apache-2.0 **at the pinned commit**; upstream changed to a non-commercial licence on 24 August 2026 (see Licence notes) | `huang2025deimv2` |
| DEIM | (design inherited by DEIMv2) | The training scheme that DEIMv2 extends. | Inside DEIMv2 | Repository licence not examined (GitHub lists it as "Other"); no DEIM code is used directly | `huang2025deim` |
| D-FINE | (design inherited by DEIMv2) | The way the detector refines box edges. | Inside DEIMv2 (`loss_fgl`, `loss_ddf` in `configs/base/deimv2.yml`) | Apache-2.0 (repository) | `peng2025dfine` |
| RT-DETR | (design inherited by DEIMv2) | The real-time detection-transformer design the family is built on. | Inside DEIMv2 | Apache-2.0 (repository) | `lv2023rtdetr` |
| DETR | (design) | The original set-based detection transformer. | Inside DEIMv2 | not applicable (method) | `carion2020detr` |
| Vision Transformer | (design) | The network type of the backbone. | Inside DINOv3 | not applicable (method) | `dosovitskiy2021vit` |
| **DINOv3 backbone weights** | ViT-S+/16 pretrained on LVD-1689M: `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` (115 MB). The file's SHA-256 begins `4057cbaa`, matching Meta's file name. It was downloaded from a third-party Hugging Face copy (`xysun917/dinov3-vits16plus`); Meta's own copy is `facebook/dinov3-vits16plus-pretrain-lvd1689m` | The part of the detector that turns pixels into visual features, already trained on unlabelled images before it saw any fish. | `configs/deimv2/deimv2_dinov3_x_coco.yml` | **DINOv3 License** (custom; see Licence notes) | `simeoni2025dinov3` |
| **COCO-pretrained DEIMv2-X checkpoint** | `deimv2_x_coco.pth` (205 MB), converted from Hugging Face `Intellindust/DEIMv2_DINOv3_X_COCO` | The starting point for fine-tuning on fishway images. | `config.DEIM_INIT_COCO`; training | No licence is stated on the model page; the repository licence applies (see Licence notes). Contains DINOv3-derived weights | `huang2025deimv2`, `lin2014coco` |
| MS COCO | 2017 detection set (through the checkpoint above) | The everyday-objects image set the starting checkpoint was trained on. | Indirect | Annotations CC BY 4.0; images under the Flickr terms of use | `lin2014coco` |
| BoT-SORT | Algorithm, as implemented in BoxMOT (version as above) | Links detections into one track per fish using predicted motion and appearance. | `detect/track_from_store` | Implementation AGPL-3.0 (BoxMOT) | `aharon2022botsort` |
| ByteTrack | Algorithm, as implemented in BoxMOT (`BYTETracker`) | The motion-only tracker used as the comparison in Section 3.4. | `detect/track_from_store --tracker bytetrack` | Implementation AGPL-3.0 (BoxMOT) | `zhang2022bytetrack` |
| OSNet re-identification weights | `osnet_x0_25_msmt17.pt`, downloaded automatically by BoxMOT from the Torchreid model zoo | Compares the appearance of two detections so the tracker can tell whether they are the same fish. Trained on images of people and used unchanged. | `config.REID_WEIGHTS`, `detect/track_from_store` | Torchreid code is MIT; no separate licence is stated for the weights | `zhou2019osnet` (optional: `zhou2022osnet`) |
| Torchreid | Model zoo | The library that trained and distributes the OSNet weights. | Indirect | MIT | `zhou2019torchreid` |
| MSMT17 | Pedestrian dataset | The images the OSNet weights were trained on. | Indirect | Usage terms **UNVERIFIED** (the dataset page could not be reached) | `wei2018msmt17` |
| **SAM 2** | SAM 2.1 Hiera Base+ (`facebook/sam2.1-hiera-base-plus`), Python package `sam-2` 1.0 | Cuts verified fish out of real frames; the cutouts are pasted into crowded scenes for training and shown in the body-shape figure. | Research repository `scripts/_sam2_*`; output consumed by `data/sam2_composite`, `figures/fig_species_morphology` | Apache-2.0 (code and checkpoints) | `ravi2024sam2` |
| Copy-paste augmentation | Method | Pasting cut-out fish into new scenes to give the detector more examples of rare species. | `data/sam2_composite`, `data/merge_synth_into_coco` | not applicable (method) | `ghiasi2021copypaste` |
| ResNet-50 | TorchVision weights `IMAGENET1K_V1` | One of three classifiers in the backbone comparison (Table S4). | `paper/backbone_comparison` (also manuscript repository, `docs/backbone_comparison_2026-11-09/`) | BSD-3-Clause (TorchVision) | `he2016resnet` |
| ImageNet | ILSVRC-2012 (through the ResNet-50 weights) | The labelled images ResNet-50 was pretrained on. | Indirect | Terms not checked (not part of the deposit) | `deng2009imagenet` |
| DINOv2 | ViT-B/14, `facebook/dinov2-base` | Second classifier in the backbone comparison. | Backbone comparison | Apache-2.0 | `oquab2024dinov2` |
| DINOv3 (comparison classifier) | ViT-B/16, `facebook/dinov3-vitb16-pretrain-lvd1689m` | Third classifier in the backbone comparison. | Backbone comparison | DINOv3 License | `simeoni2025dinov3` |
| RT-DETRv2 | `PekingU/rtdetr_v2_r50vd` | Base of the earlier two-stage detector whose boxes one gate script still scores. Cite only if that earlier detector is described. | `eval/bbox_regression_gate` | Apache-2.0 | `lv2024rtdetrv2` |

Other method references that are already in the manuscript and were not re-checked here,
because they are training settings and not software: the AdamW optimizer
(`loshchilov2019adamw`) and the focal loss (`lin2017focal`).

---

## 3. Statistical methods implemented in the code

Only methods that the code actually computes are listed.

| Method | What it answers here | Where it is implemented | Citation key |
|---|---|---|---|
| Lin's concordance correlation coefficient (CCC) | Do model and technician daily counts agree in level as well as in pattern? | `results/compute_concordance.lin_ccc` | `lin1989concordance` |
| McBride's strength-of-agreement bands | Turns a CCC value into a word (moderate, substantial, and so on). | `results/compute_concordance.mcbride` | `mcbride2005proposal` |
| Bland-Altman limits of agreement | The range within which most daily differences fall. | `results/compute_concordance`, `results/agreement_stats.ba_constant` | `bland1986statistical` |
| Bland-Altman for differences that grow with the count (ratio limits and proportional limits) | The same question when the disagreement is larger on busy days. | `results/agreement_stats.ba_ratio`, `ba_proportional` | `bland1999measuring` |
| Wilson score interval | An honest uncertainty range for precision and recall on small samples. | `results/classification_ci.wilson`, `paper/_common.wilson` (used by `paper/blind_review`) | `wilson1927probable` (the code's docstring also cites `brown2001interval` for why Wilson is preferred) |
| Percentile bootstrap | Confidence intervals for CCC, F1, macro-F1 and kappa by resampling. | `results/compute_concordance.boot_ci`, `results/classification_ci`, `results/agreement_stats` | `efron1979bootstrap` |
| Moving-block bootstrap | The same, but resampling runs of consecutive days so that day-to-day dependence is kept. | `results/agreement_stats._season_blocks`, `block_bootstrap_ccc`; `paper/_common.block_boot_kappa` | `kunsch1989jackknife` |
| Cohen's kappa | Agreement on species beyond what chance would give. | `paper/_common.kappa`, `paper/heldout_kappa`, `paper/blind_review`, `paper/pervideo_agreement` | `cohen1960coefficient` |
| Large-sample standard error of kappa | A formula-based interval for kappa. Computed, but the manuscript reports the bootstrap interval, so cite only if this one is reported. | `paper/heldout_kappa.kappa_se_fce` | `fleiss1969large` |
| Landis and Koch bands | Words for kappa values. Computed, but not quoted in the manuscript, so cite only if used. | `paper/heldout_kappa.landis_koch` | `landis1977measurement` |
| Fisher z-transformation | The 95% interval on the daily Pearson correlation. | `paper/correlation_stats`, `paper/count_tables.fisher_ci` | `fisher1915frequency` |
| Spearman's rank correlation | Do model and technician rank the days, and the years, in the same order? | `results/agreement_stats`, `paper/correlation_stats` (through SciPy) | `spearman1904proof` |
| Exact McNemar test | Is one backbone wrong on significantly more of the same fish than another (Table S4)? | `paper/backbone_comparison.mcnemar_exact` | `mcnemar1947note` |
| Okabe-Ito colour palette | Species colours that stay distinguishable for colour-blind readers and in greyscale. | `figures/manuscript_style`, `results/compute_concordance` | `okabe2008color` (and usually `wong2011points`) |
| Pearson correlation | Daily co-variation of the two counts. | Through SciPy (`scipy.stats.pearsonr`) | none needed beyond `virtanen2020scipy` |

`paper/heldout_kappa` also computes the prevalence-adjusted kappa (PABAK), Krippendorff's
alpha, Scott's pi and Gwet's AC1. None of them appears in the manuscript, so none is cited; if
any is reported later it will need its own reference.

---

## 4. Tools and base data

| Tool or data | Version | What it does here | Licence | Citation key |
|---|---|---|---|---|
| Python | 3.11.9 (both environments) | The programming language. | PSF-2.0 | `python3119` |
| NVIDIA CUDA | 12.8 (both environments) | Lets PyTorch compute on the graphics card. Not usually cited. | NVIDIA proprietary | optional `nickolls2008cuda` |
| NVIDIA cuDNN | 9.19 (base) / 9.7 (DEIMv2) | Fast neural-network routines on the graphics card. Not usually cited. | NVIDIA proprietary | optional `chetlur2014cudnn` |
| Pandoc | **UNVERIFIED** (not installed on this machine's command path) | Turns `manuscript.md` and `references.bib` into the Word file. Manuscript build only. | GPL-2.0-or-later | `macfarlane2026pandoc` |
| Quarto | **UNVERIFIED** (as above) | Supplies Pandoc and Typst when they are not installed separately. Manuscript build only. | MIT (current repository; the 2022 Zenodo record says GPL-2.0-or-later) | `allaire2022quarto` |
| Typst | **UNVERIFIED** (as above) | Makes the PDF. Manuscript build only. | Apache-2.0 | `madje2026typst` |
| pypandoc | **UNVERIFIED** (not installed in the base environment) | A thin Python wrapper that calls Pandoc from `scripts/build_docx.py`. | MIT | none needed |
| Natural Earth | 1:50m countries and lakes, fetched from `nvkelso/natural-earth-vector` | The small inset map in Fig. 1. | Public domain | `naturalearth` |
| OpenStreetMap | Overpass extract with base timestamp 10 June 2026 (`scripts/spatial/ganaraska_full.json`) | The Ganaraska River lines in Fig. 1. | ODbL 1.0 (**attribution required**) | `openstreetmap2026` |
| `lake_ontario.gpkg` | unknown | The Lake Ontario shoreline in Fig. 1. | **UNVERIFIED: source not recorded anywhere in either repository** | none yet |
| `credit_river.geojson` | unknown | The Credit River lines in Fig. 1. | **UNVERIFIED: source not recorded** (its `hydr_order` field suggests a government hydrography layer) | none yet |

**AI assistance.** No new statement is proposed here. The manuscript's Acknowledgements
already disclose that Claude Code (Anthropic; Claude Opus models, versions 4.8 to 5.5, and
Claude Fable 5 models) was used to build the video-review applications, to develop and check
the training, testing and monitoring code, and to draft, edit and check the text, that
Grammarly and Microsoft Editor were used for wording, and that the authors reviewed all
output and take responsibility for the content.

---

## Licence notes

These are the facts found in the licence texts and what they mean in practice for this
deposit. **They are not legal advice. The choice of licence for the code and for the model
weights rests with the authors and the Ontario Ministry of Natural Resources.** The order is
from most to least consequential.

### 1. The fine-tuned model weights contain DINOv3 weights, which carry a custom licence

*Facts.* The 819 MB fine-tuned checkpoint is a DEIMv2-X detector whose backbone is the DINOv3
ViT-S+/16, further trained on fishway images. Meta releases DINOv3 code and weights under the
**DINOv3 License** (last updated 19 August 2025; text read from
`facebookresearch/dinov3/LICENSE.md`). It is not one of the standard open-source licences.
The parts that matter here:

* Section 1.a grants the right to "use, reproduce, distribute, copy, create derivative works
  of, and make modifications to the DINO Materials".
* Section 1.b.i: "If you distribute or make the DINO Materials, or any derivative works
  thereof, available to a third party, you may only do so under the terms of this Agreement
  and you shall provide a copy of this Agreement with any such DINO Materials."
* Section 1.b.ii: "If you submit for publication the results of research you perform on,
  using, or otherwise in connection with DINO Materials, you must acknowledge the use of DINO
  Materials in your publication."
* Section 1.b.v forbids use for activities subject to the International Traffic in Arms
  Regulations and for "military or warfare purposes, nuclear industries or applications,
  espionage, or the development or use of guns or illegal weapons".
* Section 5.a: you own the derivative works you make, subject to Meta's ownership of the
  original.
* Meta's own download page is gated: a user must accept the licence before downloading.

*Practical consequence.*

* The licence allows the fine-tuned checkpoint to be shared, but **only under the DINOv3
  License, with a copy of that licence beside the file**. A fine-tuned checkpoint is
  reasonably read as a "derivative work" of the weights. It therefore cannot simply be
  labelled Apache-2.0, MIT or a Creative Commons licence.
* **Dryad will not work for the weights.** Dryad publishes everything under a CC0 waiver,
  which cannot be combined with a licence that must travel with the file. Zenodo accepts
  custom licences. The manuscript's Data Availability text names both; for the weights it
  should name one that permits a custom licence.
* The acknowledgement condition is met by citing DINOv3 (`simeoni2025dinov3`), which the
  manuscript already does.
* `docs/DATA_AVAILABILITY.md` currently sends readers to the third-party copy
  `xysun917/dinov3-vits16plus`, which shows no licence at all. The file is genuine (its hash
  matches Meta's file name), but readers should be pointed to Meta's own gated source so they
  see and accept the licence.
* The placeholder `LICENSE` file in this repository says only "see the upstream model card for
  terms". It should name the DINOv3 License.

### 2. The tracker library, BoxMOT, is under the AGPL-3.0

*Facts.* BoxMOT is licensed under the GNU Affero General Public License version 3. This is
so for both installed versions: 10.0.52 (package metadata, repository `LICENSE` at tag
`v10.0.52`, and its `CITATION.cff`) and 21.0.0 (package metadata and `CITATION.cff` at tag
`v21.0.0`). One module here imports it: `salmonid/detect/track_from_store.py`
(`from boxmot import BoTSORT, BYTETracker`).
This repository does not contain any BoxMOT code; BoxMOT is installed separately with `pip`.
The licence text says:

* Section 5(c): a work "based on the Program" must be licensed "as a whole, under this
  License to anyone who comes into possession of a copy".
* The same section exempts an "aggregate": separate and independent works that "are not
  combined with it such as to form a larger program".
* Section 13 adds that a *modified* version offered to users over a network must offer them
  its source.

Whether a script that imports an AGPL library is a work "based on" it, or a separate work, is
a question of interpretation that the licence text does not settle in so many words. The Free
Software Foundation, which wrote the licence, states in its GPL FAQ that linking a covered work
"statically or dynamically with other modules is making a combined work" to which the licence
then applies as a whole.

*Practical consequence.* If the deposit is released under Apache-2.0 while one module imports
an AGPL-3.0 library, there are four ways to leave no doubt, and the choice is the authors':

1. Release the whole repository under AGPL-3.0. (The Apache Software Foundation's guidance is
   that Apache-2.0 software can be included in GPLv3 projects but GPLv3 software cannot be
   included in Apache-licensed projects; the AGPL-3.0 is the network variant of the GPLv3.)
2. Keep Apache-2.0 for the repository and mark the one tracking module, in its header and in
   `LICENSE`, as depending on an AGPL-3.0 library and as subject to that licence when combined
   with it.
3. Move the tracking step into a small separate program under AGPL-3.0, with the rest of the
   repository reading its output files.
4. Replace the tracker with a permissively licensed implementation (the BoT-SORT authors' own
   repository is MIT, as is Supervision). This would change the software that produced the
   published counts, so it is not a neutral swap and the counts would have to be re-run.

Whatever is chosen, the manuscript should cite BoxMOT itself and its version
(`brostrom2024boxmot` or `brostrom2026boxmot`, once the version is confirmed), because the published counts depend on that implementation and not
only on the BoT-SORT paper.

### 3. DEIMv2 changed its licence after the commit this work is pinned to

*Facts.*

* The commit this work pins (`0fff8d4`, 24 March 2026) contains a `LICENSE` file granting the
  **Apache License 2.0**. Apache-2.0 describes its grant as "perpetual, worldwide,
  non-exclusive, no-charge, royalty-free, irrevocable".
* On **24 August 2026** the upstream repository replaced that file with `LICENSE.md`, the
  **"DEIMv2 License"** (commit `bb64e5e`, "docs: update license information"). The new licence
  permits use, modification and redistribution "solely for Non-Commercial Purposes", requires
  a separate commercial licence otherwise, and states that "Model Weights distributed by
  [Intellindust] are subject to this License unless a separate license is explicitly
  provided". Its section 10 says that "Rights granted under a particular version remain
  governed by the License applicable to that release unless otherwise agreed."
* The starting checkpoint `deimv2_x_coco.pth` was downloaded on 1 July 2026, before the
  change, from a Hugging Face model page that states no licence of its own.
* This repository redistributes DEIMv2 material: the files under `configs/base/`,
  `configs/dataset/`, `configs/runtime.yml` and `configs/deimv2/deimv2_dinov3_x_coco.yml` are
  copies (the last is identical to upstream at the pinned commit), and
  `docs/deimv2_worktree.patch` is a modification of a DEIMv2 source file.

*Practical consequence.*

* Statements in `README.md`, `LICENSE` and `docs/DEIMV2_PIN.md` that DEIMv2 is "Apache-2.0"
  are true of the pinned commit and no longer true of the current upstream repository. They
  should say "Apache-2.0 at the pinned commit". A reader who clones today's upstream gets the
  non-commercial licence.
* Apache-2.0 section 4 sets conditions on redistributing the copied configuration files and
  the patch: give recipients a copy of the Apache licence, keep the copyright notice, and mark
  changed files as changed. The repository does not yet include that licence text or notice.
* Whether the later non-commercial terms reach a checkpoint obtained while the repository was
  Apache-2.0 is a question for the authors and the Ministry, and if needed for Intellindust.
  It bears on the fine-tuned weights, which are derived from that checkpoint.

### 4. OpenStreetMap data in Fig. 1 needs a credit line

*Facts.* The Ganaraska River lines in Fig. 1 come from OpenStreetMap, which is licensed under
the Open Database License 1.0 and requires that OpenStreetMap be credited. Natural Earth (the
inset) is public domain and asks for no credit. The Fig. 1 caption currently credits neither,
and the sources of the Lake Ontario shoreline and Credit River layers are not recorded.

*Practical consequence.* Add "© OpenStreetMap contributors" to the Fig. 1 caption or the
Acknowledgements, and identify the source and licence of `lake_ontario.gpkg` and
`credit_river.geojson` before submission. If they turn out to be government open-data layers,
check the attribution wording their licence asks for.

### 5. The re-identification weights should not be bundled

*Facts.* BoxMOT downloads `osnet_x0_25_msmt17.pt` by itself the first time the tracker runs.
The weights come from the Torchreid model zoo. Torchreid's code is MIT; no licence is stated
for the weights, and they were trained on MSMT17, a dataset of images of people whose usage
terms could not be confirmed here.

*Practical consequence.* Leave it as it is: do not copy the weight file into the deposit.
Cite OSNet (and, if space allows, Torchreid and MSMT17).

### 6. Items with no constraint on this deposit

* **SAM 2** code and checkpoints are Apache-2.0. The fish cutouts are the project's own
  images; SAM 2's licence places no condition on them.
* **MS COCO** annotations are CC BY 4.0, which asks for attribution; citing `lin2014coco`
  provides it. No COCO images or annotations are redistributed here.
* **ImageNet**, **DINOv2** and the comparison classifiers are not part of the deposit.
* All remaining Python libraries are under permissive licences (BSD, MIT, Apache-2.0, the
  Matplotlib and Pillow licences, and MPL-2.0 for tqdm). None of them is redistributed here,
  and none restricts how this repository is licensed.
* **Pandoc** is GPL, but it is a separate program that is only run to build the manuscript.
* **CUDA** and **cuDNN** are proprietary NVIDIA software obtained through the PyTorch
  installer; they are not redistributed here.

---

## Suggested sentence for the manuscript Methods

The manuscript currently reads: "Analyses used Python 3.11.9 with SciPy, pandas, and NumPy
[@virtanen2020scipy], and figures were made with Matplotlib." Only SciPy is cited. A
replacement that credits each tool, in the manuscript's own style:

> Analyses used Python 3.11.9 [@python3119] with NumPy 2.4.4 [@harris2020numpy], pandas 2.3.3
> [@mckinney2010pandas; @pandas2025software], and SciPy 1.15.3 [@virtanen2020scipy], and
> figures were made with Matplotlib 3.10.8 [@hunter2007matplotlib] in a colour-blind-safe
> palette [@okabe2008color]. The detector was trained and run with the DEIMv2 code
> [@huang2025deimv2] in PyTorch [@paszke2019pytorch] and TorchVision [@torchvision2016], video
> frames were read with OpenCV [@bradski2000opencv] and Decord [@decord2021], and both the
> BoT-SORT tracker and the ByteTrack comparison used the implementations in BoxMOT 10.0.52
> [@brostrom2024boxmot].

The BoxMOT version in that sentence is the one the code documents. If the separate check
finds that the published counts came from the copy in the DEIMv2 environment, write "BoxMOT
21.0.0 [@brostrom2026boxmot]" instead.

If the study-area map is credited in the text and not in the caption:

> The study-area map was drawn with GeoPandas [@fleischmann2026geopandas] from OpenStreetMap
> [@openstreetmap2026] and Natural Earth [@naturalearth] data.

References to attach where a method is already named in the text without one:

| Where the manuscript names it | Add |
|---|---|
| "95% confidence interval (Fisher z-transformation)" (Section 2.6) | `[@fisher1915frequency]` |
| "bootstrap 95% confidence intervals (2,000 resamples)" (Sections 2.4 and 2.6) | `[@efron1979bootstrap]` |
| "moving blocks of 10 consecutive days" (Section 2.4) and "a moving-block resample that preserves day-to-day dependence" (Section 2.6) | `[@kunsch1989jackknife]` |
| "Spearman's ρ" (Section 2.6) | `[@spearman1904proof]` |
| "exact McNemar test" (Section 2.3) | `[@mcnemar1947note]` |
| "Wilson intervals" (Table 1 caption) | `[@wilson1927probable]` |
| The ratio or proportional Bland-Altman limits, if reported | `[@bland1999measuring]` |

---

## Already cited in the manuscript

Matched against the manuscript repository's `references.bib` and the citations in
`manuscript.md`. Each of these has an entry and is cited in the text. All were re-checked.

| Item | Existing key | Check result |
|---|---|---|
| DEIMv2 | `huang2025deimv2` | Correct. Still an arXiv preprint (version 5, 27 October 2026). |
| DETR | `carion2020detr` | Correct. |
| RT-DETR | `lv2023rtdetr` | Correct. DOI missing: 10.1109/CVPR52733.2024.01605. |
| DINOv3 | `simeoni2025dinov3` | Correct. |
| Vision Transformer | `dosovitskiy2021vit` | Correct. |
| DINOv2 | `oquab2024dinov2` | Correct. URL missing: https://openreview.net/forum?id=a68SUt6zFt. |
| MS COCO | `lin2014coco` | Correct. |
| ResNet | `he2016resnet` | Correct. |
| BoT-SORT | `aharon2022botsort` | Correct; the "VERIFY" comment above the entry can be removed. |
| OSNet | `zhou2019osnet` | **Page range differs from the DOI record** (see below). |
| ByteTrack | `zhang2022bytetrack` | Correct. |
| SAM 2 | `ravi2024sam2` | Correct as an arXiv citation. The paper has since appeared at ICLR 2025. |
| Copy-paste augmentation | `ghiasi2021copypaste` | Correct. DOI now confirmed: 10.1109/CVPR46437.2021.00294. |
| PyTorch | `paszke2019pytorch` | Correct. |
| SciPy | `virtanen2020scipy` | **Wrong DOI** (see below). |
| Lin's CCC | `lin1989concordance` | Correct. |
| Bland-Altman | `bland1986statistical` | Correct. |
| McBride bands | `mcbride2005proposal` | Correct (report number HAM2005-062 confirmed from citing sources; the report has no DOI). |
| Cohen's kappa | `cohen1960coefficient` | Correct. |

### Existing entries whose metadata should be corrected

1. **`virtanen2020scipy`: wrong DOI.** The entry gives `10.1038/s41592-020-0772-5`. Crossref
   resolves that DOI to the *Author Correction* notice (Nature Methods 17, page 352). The
   article itself, volume 17, pages 261 to 272, is **`10.1038/s41592-019-0686-2`**. As it
   stands, the reference list would link readers to the correction notice.
2. **`zhou2019osnet`: pages.** The entry gives 3702 to 3712 with the IEEE DOI
   `10.1109/ICCV.2019.00380`; the IEEE record for that DOI gives **3701 to 3711**. The
   one-page offset is the difference between the open-access (CVF) and IEEE Xplore page
   numbering. Since the entry cites the IEEE DOI, the IEEE range is the consistent one.
3. **`ghiasi2021copypaste`: same offset, noted for consistency.** The entry's 2918 to 2928
   is the CVF numbering (confirmed on the CVF page); the IEEE record for the DOI gives 2917 to
   2927. Either is defensible; choose one convention for both entries.

One thing seen while checking that is not a reference error: the code labels a CCC below
0.90 as "weak" (`compute_concordance.mcbride`), while the manuscript calls the same band
"poor". The McBride report itself could not be retrieved to confirm which word it uses, so
this is flagged and not resolved.

## Missing from the manuscript's reference list

Used by the code, and not in `references.bib`. Entries for all of them are in
[`software.bib`](software.bib).

**Named in the manuscript text but not cited**

* NumPy: `harris2020numpy`
* pandas: `mckinney2010pandas`, `pandas2025software`
* Matplotlib: `hunter2007matplotlib`
* Python 3.11.9: `python3119`
* Fisher z-transformation: `fisher1915frequency`
* Bootstrap confidence intervals: `efron1979bootstrap`
* Moving-block bootstrap: `kunsch1989jackknife`
* Spearman's ρ: `spearman1904proof`
* Exact McNemar test: `mcnemar1947note`
* Wilson interval: `wilson1927probable`

**Used by the code and not mentioned in the manuscript**

* BoxMOT, the tracker implementation: `brostrom2024boxmot` (10.0.52) or `brostrom2026boxmot`
  (21.0.0), whichever is confirmed (the most important omission, because the counts depend
  on it)
* TorchVision: `torchvision2016`
* Decord: `decord2021`
* OpenCV: `bradski2000opencv`
* Pillow: `murray2026pillow`
* Apache Arrow: `apachearrow2025`
* Supervision (a second ByteTrack, used only by the optional store-only tracker; cite only if
  a result from it is reported): `roboflow2026supervision`
* Transformers: `wolf2020transformers`
* Faster-COCO-Eval, PyYAML, TensorBoard, calflops (inside DEIMv2 training):
  `mixaill2026fastercocoeval`, `simonov2025pyyaml`, `tensorboard2025`, `ye2023calflops`
* GeoPandas and Shapely (Fig. 1): `fleischmann2026geopandas`, `gillies2025shapely`
* OpenStreetMap and Natural Earth (Fig. 1): `openstreetmap2026`, `naturalearth`
* The Okabe-Ito palette: `okabe2008color`, `wong2011points`
* Bland and Altman (1999), if the ratio or proportional limits are reported:
  `bland1999measuring`
* Brown, Cai and DasGupta (2001), cited in a code docstring: `brown2001interval`

**Background to the models, worth citing only if the Methods describe that lineage**

* DEIM and D-FINE, which DEIMv2 builds on: `huang2025deim`, `peng2025dfine`
* Torchreid and MSMT17, the source and training data of the OSNet weights:
  `zhou2019torchreid`, `wei2018msmt17`, `zhou2022osnet`
* ImageNet, behind the ResNet-50 comparison classifier: `deng2009imagenet`
* RT-DETRv2, the earlier two-stage detector: `lv2024rtdetrv2`

A reference list that credited every row above would add about 35 entries. The journal does
not require that. A reasonable minimum is the first group plus BoxMOT, TorchVision, OpenCV,
Decord and the OpenStreetMap credit; the full list belongs in `CITATION.cff`, where
`_citation_cff_references.yml` puts it.

## What could not be verified

* **Which BoxMOT version produced the published counts** (10.0.52 in the base environment or
  21.0.0 in the DEIMv2 environment). Both are reported above; this is being confirmed
  separately.
* **Versions of Pandoc, Quarto, Typst and pypandoc** used to build the manuscript. None is on
  this machine's command path. Their citation details were confirmed; the version numbers
  must be read on the machine that builds the manuscript.
* **Source and licence of `lake_ontario.gpkg` and `credit_river.geojson`** (Fig. 1). Not
  recorded in either repository.
* **Usage terms of the MSMT17 dataset.** Its citation is confirmed; its terms page could not
  be reached.
* **OpenCV's reference** is given exactly as the OpenCV project requests it, which has no
  volume or page numbers; none were added.
* **PyArrow, Decord, PyYAML and TensorBoard** publish no preferred citation. Their entries
  are plain software references built from the packages' own metadata (name, version, release
  date, licence, address), all of which were confirmed.

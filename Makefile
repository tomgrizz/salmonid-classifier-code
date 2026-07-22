# Turnkey reproduction targets. `make help` lists them.
# Override the interpreter with e.g. `make PY=python3.11 score`.
PY ?= python

.PHONY: help install compile score figures concordance sensitivity all setup-deimv2 clean

help:
	@echo "salmonid-classifier-code — reproduction targets"
	@echo ""
	@echo "  make install       pip install base env (requirements.txt) + this package (editable)"
	@echo "  make compile       byte-compile every module (fast import smoke test)"
	@echo ""
	@echo "  Tier 1 (needs only committed data/ — no downloads):"
	@echo "    make score        held-out species score -> macro-F1 0.966 (test_set_v6)"
	@echo "    make concordance  Lin's CCC table + concordance/Bland-Altman figures"
	@echo "    make figures      regenerate manuscript figures from canonical_*.csv"
	@echo "    make sensitivity  net-rule sensitivity figure from the committed sweeps"
	@echo "    make all          score + concordance + figures"
	@echo ""
	@echo "  Environment:"
	@echo "    make setup-deimv2 print the steps to reconstruct the pinned external DEIMv2 repo"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

compile:
	$(PY) -m compileall -q salmonid

# --- Tier 1: reproduce from committed data (config.py defaults resolve into data/) ---
score:
	$(PY) -m salmonid.eval.score_testset_v6 --preds data/preds_checkpoint0007.csv

concordance:
	$(PY) -m salmonid.results.compute_concordance --fig-dir figures

figures:
	$(PY) -m salmonid.figures.render_hero_regulation
	$(PY) -m salmonid.figures.render_manuscript_figures
	$(PY) -m salmonid.figures.render_sensitivity_figure

sensitivity:
	$(PY) -m salmonid.figures.render_sensitivity_figure

all: score concordance figures

# --- environment reconstruction (external DEIMv2 repo; see docs/DEIMV2_PIN.md) ---
setup-deimv2:
	@echo "Reconstruct the pinned DEIMv2 training repo (needed only for training / Pass-A detection):"
	@echo "  git clone https://github.com/Intellindust-AI-Lab/DEIMv2"
	@echo "  cd DEIMv2 && git checkout $$(cat docs/DEIMV2_SHA.txt)"
	@echo "  git apply <this-repo>/docs/deimv2_worktree.patch"
	@echo "  cp -r <this-repo>/configs/* configs/          # fish config chain incl. v26"
	@echo "  # place ckpts/deimv2_x_coco.pth + ckpts/dinov3_vits16plus_pretrain_*.pth"
	@echo "  export SALMONID_DEIM_DIR=\$$(pwd)"

clean:
	rm -f figures/*.png figures/*.pdf figures/*.tiff

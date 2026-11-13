# Shortcuts for the reproduction steps. `make help` lists them.
#
# make is OPTIONAL. Every target below is one plain python command, printed when it runs,
# so on a machine without make (most Windows machines) run that command directly. The main
# one is:   python run_tier1.py
#
# Override the interpreter with e.g. `make PY=python3.11 tier1`.
PY ?= python

.PHONY: help install tier1 list score stats figures paper compile clean

help:
	@echo "Reproduction targets (Tier 1 = from the committed data; no downloads, no GPU)"
	@echo ""
	@echo "  make install   install the four Tier-1 packages at the exact published versions"
	@echo "  make tier1     run EVERY Tier-1 step in order, then verify against data/   <- start here"
	@echo "  make list      list the Tier-1 steps and what each one feeds in the manuscript"
	@echo ""
	@echo "  Single steps (all are run by 'make tier1'):"
	@echo "  make score     held-out species score        -> Table 1, Figure 2 (macro-F1 0.961)"
	@echo "  make stats     agreement statistics          -> Tables 1 and 5, Sections 3.3 to 3.5"
	@echo "  make figures   figures drawn from the tables -> Figures 2, 4, 5, 6, 7, S2, S3"
	@echo "  make paper     the remaining manuscript tables -> see docs/PAPER_TO_CODE.md"
	@echo ""
	@echo "  make compile   byte-compile every module (a fast syntax check)"
	@echo "  make clean     delete regenerated outputs (output/ and figures/)"
	@echo ""
	@echo "Tier 2 (model weights, video, stored detections): see docs/REPRODUCE.md"

install:
	$(PY) -m pip install -r environment/tier1.lock.txt

tier1:
	$(PY) run_tier1.py

list:
	$(PY) run_tier1.py --list

score:
	$(PY) run_tier1.py --only 1 2

stats:
	$(PY) run_tier1.py --only 2 3 4 5 6

figures:
	$(PY) run_tier1.py --only 4 7 8 9

paper:
	$(PY) run_tier1.py --only 10 11 12 13 14 15 16 17 18

compile:
	$(PY) -m compileall -q salmonid run_tier1.py

clean:
	rm -rf output figures/*.png figures/*.pdf figures/*.tiff

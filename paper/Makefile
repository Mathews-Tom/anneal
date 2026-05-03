# Makefile for anneal paper
# Run from the repo root: e.g. `make -C paper paper`.
#
# Targets:
#   paper          full build (macros → pdflatex/bibtex cycle)
#   quick          single pdflatex pass (fast, structure check)
#   macros         regenerate generated/macros.tex from benchmark archives
#   architecture   regenerate figures/architecture.png from architecture.html
#   clean          remove generated build artefacts (preserves generated/macros.tex)
#   distclean      additionally remove generated/macros.tex
#   watch          rebuild on file change (requires fswatch)

PAPER_DIR := $(CURDIR)
REPO_ROOT := $(realpath $(PAPER_DIR)/..)
MAIN      := main

GENERATED_MACROS  := generated/macros.tex
ARCHITECTURE_HTML := figures/architecture.html
ARCHITECTURE_PNG  := figures/architecture.png

GATE_RESULTS := \
	$(REPO_ROOT)/benchmarks/results/gate_false_positives.txt \
	$(REPO_ROOT)/benchmarks/results/gate_sa_convergence.txt \
	$(REPO_ROOT)/benchmarks/results/gate_retrieval_precision.txt
E2E_SUMMARY  := $(REPO_ROOT)/benchmarks/results_seeds1-5/summary_table.md

CHROME_BIN ?= $(shell command -v chromium || command -v chrome || command -v google-chrome || echo /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome)

.PHONY: paper quick clean distclean watch macros architecture

paper: $(GENERATED_MACROS)
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

quick: $(GENERATED_MACROS)
	pdflatex -interaction=nonstopmode $(MAIN)

macros: $(GENERATED_MACROS)

$(GENERATED_MACROS): scripts/build_macros.py $(GATE_RESULTS) $(E2E_SUMMARY)
	cd $(REPO_ROOT) && uv run python paper/scripts/build_macros.py

architecture: $(ARCHITECTURE_PNG)

$(ARCHITECTURE_PNG): $(ARCHITECTURE_HTML)
	"$(CHROME_BIN)" --headless --disable-gpu --hide-scrollbars \
		--screenshot=$(ARCHITECTURE_PNG) --window-size=1980,1228 \
		"file://$(PAPER_DIR)/$(ARCHITECTURE_HTML)"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf \
	      *.toc *.lof *.lot *.fls *.fdb_latexmk \
	      *.synctex.gz *.nav *.snm *.vrb \
	      sections/*.aux

distclean: clean
	rm -f $(GENERATED_MACROS)

watch:
	fswatch -o sections/ main.tex references.bib generated/ | \
		xargs -n1 -I{} $(MAKE) quick

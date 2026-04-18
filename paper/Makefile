# Makefile for anneal paper
# Usage:
#   make paper   — full build (runs pdflatex/bibtex cycle)
#   make quick   — single pdflatex pass (fast, for checking structure)
#   make clean   — remove generated files
#   make watch   — rebuild on file change (requires fswatch)

PAPER_DIR = paper
MAIN      = main

.PHONY: paper quick clean watch

paper:
	cd $(PAPER_DIR) && \
	pdflatex $(MAIN) && \
	bibtex $(MAIN) && \
	pdflatex $(MAIN) && \
	pdflatex $(MAIN)

quick:
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode $(MAIN)

clean:
	cd $(PAPER_DIR) && rm -f \
		*.aux *.bbl *.blg *.log *.out *.pdf \
		*.toc *.lof *.lot *.fls *.fdb_latexmk \
		*.synctex.gz *.nav *.snm *.vrb \
		sections/*.aux

watch:
	fswatch -o $(PAPER_DIR)/sections/ $(PAPER_DIR)/main.tex \
		$(PAPER_DIR)/references.bib | \
		xargs -n1 -I{} $(MAKE) quick

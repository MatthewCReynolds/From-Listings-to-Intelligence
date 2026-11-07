# make            draft PDF with writing briefs  -> build/main.pdf
# make final      clean copy, no briefs          -> build/midterm_final.pdf
# make figures    redraw figures from generated/data (no dataset needed)
# make results    recompute every number from the master table (needs data/)
# make all-data   results, then figures, then the PDF

PY      ?= ../../.venv/bin/python
export PATH := /Library/TeX/texbin:$(PATH)
LATEXMK := latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build

.PHONY: pdf final figures results all-data watch clean
pdf:
	$(LATEXMK) main.tex

final:
	$(LATEXMK) -jobname=midterm_final -usepretex -pretex='\def\finalbuild{}' main.tex

figures:
	$(PY) scripts/figures.py

results:
	$(PY) scripts/results.py

all-data: results figures pdf

watch:
	$(LATEXMK) -pvc main.tex

clean:
	rm -rf build

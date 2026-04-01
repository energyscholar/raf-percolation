TEX = pdflatex
BIB = bibtex
MAIN = raf-percolation

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(MAIN).bib
	$(TEX) $(MAIN)
	$(BIB) $(MAIN)
	$(TEX) $(MAIN)
	$(TEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz *.run.xml *.bcf

.PHONY: all clean

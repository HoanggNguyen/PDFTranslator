MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib $(wildcard sections/*.tex) $(wildcard appendix/*.tex)
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean

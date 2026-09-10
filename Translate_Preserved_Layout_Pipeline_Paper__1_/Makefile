MAIN = main
TITLE_PAGE = title_page
SUPPLEMENT = supplementary_material

all: $(MAIN).pdf $(TITLE_PAGE).pdf $(SUPPLEMENT).pdf submission-docs

$(MAIN).pdf: $(MAIN).tex paper_preamble.tex references.bib $(wildcard sections/*.tex) $(wildcard appendix/*.tex)
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

$(TITLE_PAGE).pdf: $(TITLE_PAGE).tex
	pdflatex $(TITLE_PAGE)
	pdflatex $(TITLE_PAGE)

$(SUPPLEMENT).pdf: $(SUPPLEMENT).tex paper_preamble.tex references.bib $(MAIN).pdf $(wildcard appendix/*.tex)
	test -f $(MAIN).aux || $(MAKE) -B $(MAIN).pdf
	pdflatex $(SUPPLEMENT)
	pdflatex $(SUPPLEMENT)
	pdflatex $(SUPPLEMENT)

submission-docs: submission/cover_letter.docx submission/declaration_of_competing_interests.docx submission/author_contributions.docx submission/author_approval_record.docx

submission/%.docx: submission/%.md
	pandoc $< -o $@

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).spl
	rm -f $(TITLE_PAGE).aux $(TITLE_PAGE).log $(TITLE_PAGE).out $(TITLE_PAGE).spl
	rm -f $(SUPPLEMENT).aux $(SUPPLEMENT).bbl $(SUPPLEMENT).blg $(SUPPLEMENT).log $(SUPPLEMENT).out $(SUPPLEMENT).toc $(SUPPLEMENT).spl

distclean: clean
	rm -f $(MAIN).pdf $(TITLE_PAGE).pdf $(SUPPLEMENT).pdf

.PHONY: all clean distclean submission-docs

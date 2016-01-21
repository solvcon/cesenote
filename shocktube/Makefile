MAIN_FN = shocktube
EPS_DIR = eps
SCHEMATIC_DIR = schematic

ALL_TEX = $(wildcard $(SCHEMATIC_DIR)/*.tex)
ALL_EPS = $(patsubst $(SCHEMATIC_DIR)/%.tex,$(EPS_DIR)/%.eps,$(ALL_TEX))

default: $(MAIN_FN).pdf

$(EPS_DIR)/%.eps: $(SCHEMATIC_DIR)/%.tex Makefile
	mkdir -p $(EPS_DIR)
	pstake -q $< $@

$(MAIN_FN).dvi: $(MAIN_FN).tex $(ALL_EPS) Makefile
	echo $(ALL_EPS)
	latex $<

$(MAIN_FN).ps: $(MAIN_FN).dvi Makefile
	dvips $<

$(MAIN_FN).pdf: $(MAIN_FN).ps Makefile
	ps2pdf $<

.PHONY: clean fun

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	GVIM=mvim
	PDFVIEW=open
endif

fun:
	$(GVIM) Makefile `cat funfile` -p -c "set lines=999"
	$(PDFVIEW) shocktube.pdf

clean:
	rm -f $(EPS_DIR)/* \
		$(MAIN_FN).toc $(MAIN_FN).log $(MAIN_FN).out \
		$(MAIN_FN).dvi $(MAIN_FN).pdf

# vim: set sw=8 ts=8 noet:

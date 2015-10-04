# Outside control.
NUM ?= 3 # For LaTeX to correctly cross-reference.  To save time, use 1.
VERBOSE ?= 0

ifeq ($(VERBOSE),0)
	CMDLOG_REDIRECT := > /dev/null
	PSTAKE := pstake -q
else
	CMDLOG_REDIRECT := >&1
	PSTAKE := pstake
endif

# Environmental setting.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	GVIM=mvim
	PDFVIEW=open -a skim
endif # FIXME: other platform?

# Build setting.
MAIN_FN := stcon
OLDMAIN_FN := onedim_old
EPS_DIR := eps
SCHEMATIC_DIR := schematic
LATEX_DERIVED_EXT := toc aux log out dvi ps pdf
NOWDATE := $(shell date +"%Y%m%d")
HANDOVER_DIR := handover
HANDOVER_FN = $(HANDOVER_DIR)/$(basename $<).$(NOWDATE).pdf

# Targets.
ALL_TEX = $(wildcard $(SCHEMATIC_DIR)/*.tex)
ALL_EPS = $(patsubst $(SCHEMATIC_DIR)/%.tex,$(EPS_DIR)/%.eps,$(ALL_TEX))
ALL_SRC = $(MAIN_FN).tex $(ALL_TEX)

default: $(MAIN_FN).pdf

$(EPS_DIR)/%.eps: $(SCHEMATIC_DIR)/%.tex Makefile
	mkdir -p $(EPS_DIR)
	$(PSTAKE) $< $@

$(MAIN_FN).dvi: $(MAIN_FN).tex $(ALL_EPS) Makefile
	@echo "Having EPS files: $(ALL_EPS)"
	num=1 ; while [[ $$num -le $(NUM) ]] ; do \
		latex $< 2>&1 | tee $@.$$num.cmd.log $(CMDLOG_REDIRECT) ; \
		(( num = num + 1 )) ; \
	done

$(OLDMAIN_FN).dvi: $(OLDMAIN_FN).tex $(ALL_EPS) Makefile
	@echo "Having EPS files: $(ALL_EPS)"
	num=1 ; while [[ $$num -le $(NUM) ]] ; do \
		latex $< 2>&1 | tee $@.$$num.cmd.log $(CMDLOG_REDIRECT) ; \
		(( num = num + 1 )) ; \
	done

%.ps: %.dvi Makefile
	dvips $< 2>&1 | tee $@.cmd.log $(CMDLOG_REDIRECT)

%.pdf: %.ps Makefile
	ps2pdf $< 2>&1 | tee $@.cmd.log $(CMDLOG_REDIRECT)

.PRECIOUS: %.ps

# Handy tools.
.PHONY: ho handover clean fun

ho: handover

handover: $(MAIN_FN).pdf
	@echo "Generating today's PDF: $(HANDOVER_FN)"
	mkdir -p $(HANDOVER_DIR)
	cp -f $< $(HANDOVER_FN)

clean:
	rm -f $(EPS_DIR)/* *.cmd.log \
		$(foreach ext,$(LATEX_DERIVED_EXT),$(MAIN_FN).$(ext)) \
		$(foreach ext,$(LATEX_DERIVED_EXT),$(OLDMAIN_FN).$(ext))

fun:
	$(GVIM) Makefile $(ALL_SRC) -p -c "set lines=999"
	$(PDFVIEW) $(MAIN_FN).pdf

# vim: set sw=8 ts=8 noet:

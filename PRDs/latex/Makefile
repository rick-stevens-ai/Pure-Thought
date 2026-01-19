# Makefile for compiling Pure Thought PRD LaTeX documents
#
# Usage:
#   make all          - Compile all documents
#   make clean        - Remove auxiliary files
#   make distclean    - Remove all generated files including PDFs
#   make help         - Show available targets

# LaTeX compiler
LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode -output-directory=.

# Find all .tex files
SOURCES = $(wildcard *.tex)
TARGETS = $(patsubst %.tex,pdfs/%.pdf,$(SOURCES))

# Auxiliary file extensions to clean
AUX_EXTS = aux log out toc lof lot fls fdb_latexmk synctex.gz

.PHONY: all clean distclean help pdfs

# Default target
all: pdfs $(TARGETS)

# Create pdfs directory
pdfs:
	@mkdir -p pdfs

# Pattern rule for compiling .tex to .pdf
pdfs/%.pdf: %.tex | pdfs
	@echo "Compiling $< ..."
	@$(LATEX) $(LATEXFLAGS) $< > /dev/null 2>&1
	@$(LATEX) $(LATEXFLAGS) $< > /dev/null 2>&1
	@if [ -f $*.pdf ]; then \
		mv $*.pdf pdfs/$*.pdf; \
		echo "✓ Created pdfs/$*.pdf"; \
	else \
		echo "✗ Failed to create $*.pdf (see $*.log)"; \
		exit 1; \
	fi

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	@for ext in $(AUX_EXTS); do \
		rm -f *.$$ext; \
	done
	@echo "✓ Cleaned"

# Clean everything including PDFs
distclean: clean
	@echo "Removing all generated files..."
	@rm -rf pdfs
	@echo "✓ All generated files removed"

# Show help
help:
	@echo "Pure Thought PRD LaTeX Compilation"
	@echo "==================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all        - Compile all $(words $(SOURCES)) PRD documents to PDF"
	@echo "  make clean      - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make distclean  - Remove all generated files including PDFs"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Individual compilation:"
	@echo "  make pdfs/01-AdS3-Modular-Bootstrap.pdf"
	@echo "  make pdfs/22-Bell-Inequalities-Nonlocality.pdf"
	@echo "  ..."
	@echo ""
	@echo "Output directory: ./pdfs/"
	@echo ""

# List all PDF targets
list:
	@echo "Available PDF targets:"
	@for target in $(TARGETS); do \
		echo "  make $$target"; \
	done

# Compile count
.PHONY: count
count:
	@echo "Total LaTeX source files: $(words $(SOURCES))"
	@echo "Total PDF targets: $(words $(TARGETS))"

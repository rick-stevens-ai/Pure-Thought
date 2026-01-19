#!/bin/bash
#
# Compile all PRD LaTeX documents to PDF
#
# Usage: ./compile_all.sh [options]
# Options:
#   -c, --clean    Clean auxiliary files after compilation
#   -h, --help     Show this help message

set -e

CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./compile_all.sh [options]"
            echo "Options:"
            echo "  -c, --clean    Clean auxiliary files after compilation"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Compiling all PRD LaTeX documents to PDF"
echo "=========================================="
echo ""

# Count total files
TOTAL=$(ls -1 *.tex 2>/dev/null | wc -l)
echo "Found $TOTAL LaTeX files"
echo ""

# Create output directory for PDFs
mkdir -p pdfs

COMPILED=0
FAILED=0

# Compile each .tex file
for texfile in *.tex; do
    if [ ! -f "$texfile" ]; then
        continue
    fi

    basename="${texfile%.tex}"
    echo "-------------------------------------------"
    echo "Compiling: $texfile"
    echo "-------------------------------------------"

    # Run pdflatex twice for references and TOC
    if pdflatex -interaction=nonstopmode -output-directory=. "$texfile" > /dev/null 2>&1; then
        echo "  First pass: OK"

        # Second pass for references
        if pdflatex -interaction=nonstopmode -output-directory=. "$texfile" > /dev/null 2>&1; then
            echo "  Second pass: OK"

            # Move PDF to pdfs directory
            if [ -f "${basename}.pdf" ]; then
                mv "${basename}.pdf" "pdfs/${basename}.pdf"
                echo "  ✓ PDF created: pdfs/${basename}.pdf"
                ((COMPILED++))
            fi
        else
            echo "  ✗ Second pass failed"
            ((FAILED++))
        fi
    else
        echo "  ✗ Compilation failed"
        echo "  See ${basename}.log for details"
        ((FAILED++))
    fi

    echo ""
done

# Clean auxiliary files if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning auxiliary files..."
    rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
    echo "✓ Cleaned"
    echo ""
fi

echo "=========================================="
echo "Compilation Summary"
echo "=========================================="
echo "Total files: $TOTAL"
echo "Compiled successfully: $COMPILED"
echo "Failed: $FAILED"
echo ""
echo "PDFs are available in: ./pdfs/"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi

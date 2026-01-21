#!/bin/bash

# Compilation script for Pure Thought Challenges Report

cd "/Users/stevens/Dropbox/PURE-THOUGHT-CHALLENGES"

echo "Compiling LaTeX report..."
echo "=========================="
echo ""

# First pass
echo "Running pdflatex (first pass)..."
pdflatex -interaction=nonstopmode 30-Pure-Thought-Challenges-Report.tex

# Second pass for references
echo ""
echo "Running pdflatex (second pass for references)..."
pdflatex -interaction=nonstopmode 30-Pure-Thought-Challenges-Report.tex

# Clean up auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.toc

echo ""
echo "=========================="
echo "Compilation complete!"
echo "PDF file: 30-Pure-Thought-Challenges-Report.pdf"

# Pure Thought AI Challenges - LaTeX Document Index

This directory contains professionally formatted LaTeX versions of all 30 PRDs (Product Requirement Documents) for the Pure Thought AI Challenges project.

## Quick Start

```bash
# Compile all documents at once
./compile_all.sh

# Or using Make
make all

# Compile a single document
pdflatex 01-AdS3-Modular-Bootstrap.tex
```

PDFs will be created in the `pdfs/` directory.

## Document Inventory

### Quantum Gravity & Particle Physics (01-08)

| # | Title | Size | Status |
|---|-------|------|--------|
| 01 | [AdS₃ Pure Gravity via the Modular Bootstrap](01-AdS3-Modular-Bootstrap.tex) | 47 KB | ✓ Ready |
| 02 | [Gravitational Positivity Bounds](02-Gravitational-Positivity-Bounds.tex) | 15 KB | ✓ Ready |
| 03 | [Celestial CFT Bootstrap](03-Celestial-CFT-Bootstrap.tex) | 15 KB | ✓ Ready |
| 04 | [Modular-Lightcone Bootstrap](04-Modular-Lightcone-Bootstrap.tex) | 43 KB | ✓ Ready |
| 05 | [Positive Geometry for Gravity](05-Positive-Geometry-Gravity.tex) | 24 KB | ✓ Ready |
| 06 | [Non-perturbative S-matrix Bootstrap](06-Nonperturbative-S-matrix-Bootstrap.tex) | 22 KB | ✓ Ready |
| 07 | [Extremal Higher-D CFTs with Stress Tensor](07-Extremal-CFTs-Stress-Tensor.tex) | 23 KB | ✓ Ready |
| 08 | [Swampland via Modularity & Higher-Form Symmetries](08-Swampland-Modularity-Symmetries.tex) | 36 KB | ✓ Ready |

### Materials Science (09-15)

| # | Title | Size | Status |
|---|-------|------|--------|
| 09 | [Topological Band Theory Without Materials Data](09-Topological-Band-Theory.tex) | 40 KB | ✓ Ready |
| 10 | [Flat Chern Bands with Provable Geometry](10-Flat-Chern-Bands.tex) | 34 KB | ✓ Ready |
| 11 | [Photonic Topological Crystals from Symmetry](11-Photonic-Topological-Crystals.tex) | 30 KB | ✓ Ready |
| 12 | [Topological Mechanical Metamaterials](12-Topological-Mechanical-Metamaterials.tex) | 28 KB | ✓ Ready |
| 13 | [Higher-Order Topological Insulators](13-Higher-Order-Topological-Insulators.tex) | 28 KB | ✓ Ready |
| 14 | [Topological Semimetals: Weyl and Dirac Points](14-Topological-Semimetals-Weyl-Dirac.tex) | 27 KB | ✓ Ready |
| 15 | [Topological Quantum Chemistry](15-Topological-Quantum-Chemistry.tex) | 29 KB | ✓ Ready |

### Chemistry (16-20)

| # | Title | Size | Status |
|---|-------|------|--------|
| 16 | [N-Representability and the 2-RDM Method](16-N-Representability-2RDM.tex) | 24 KB | ✓ Ready |
| 17 | [Isomer Enumeration via Molecular Graph Theory](17-Isomer-Enumeration-Molecular-Graphs.tex) | 24 KB | ✓ Ready |
| 18 | [Optimal Transport for Molecular Systems](18-Optimal-Transport-Chemistry.tex) | 42 KB | ✓ Ready |
| 19 | [Chemical Reaction Network Theory](19-Chemical-Reaction-Networks.tex) | 49 KB | ✓ Ready |
| 20 | [Ab Initio Path Integral Molecular Dynamics](20-Ab-Initio-Path-Integrals.tex) | 47 KB | ✓ Ready |

### Quantum Information & Many-Body Theory (21-25)

| # | Title | Size | Status |
|---|-------|------|--------|
| 21 | [Quantum LDPC Codes](21-Quantum-LDPC-Codes.tex) | 45 KB | ✓ Ready |
| 22 | [Bell Inequalities and Quantum Nonlocality](22-Bell-Inequalities-Nonlocality.tex) | 44 KB | ✓ Ready |
| 23 | [Entanglement Measures and Witnesses](23-Entanglement-Measures-Witnesses.tex) | 47 KB | ✓ Ready |
| 24 | [Topological Quantum Error Correction](24-Topological-Quantum-Error-Correction.tex) | 42 KB | ✓ Ready |
| 25 | [Quantum Algorithms and Computational Complexity](25-Quantum-Algorithms-Complexity.tex) | 48 KB | ✓ Ready |

### Planetary Systems & Celestial Mechanics (26-28)

| # | Title | Size | Status |
|---|-------|------|--------|
| 26 | [KAM Theory and Planetary Stability](26-KAM-Theory-Planetary-Stability.tex) | 45 KB | ✓ Ready |
| 27 | [N-Body Problem and Central Configurations](27-N-Body-Central-Configurations.tex) | 44 KB | ✓ Ready |
| 28 | [Nekhoroshev Stability Theory](28-Nekhoroshev-Stability-Theory.tex) | 46 KB | ✓ Ready |

### Biology & Origin of Life (29-30)

| # | Title | Size | Status |
|---|-------|------|--------|
| 29 | [Chemical Reaction Networks and the Origin of Life](29-Chemical-Reaction-Networks-Origins.tex) | 41 KB | ✓ Ready |
| 30 | [Genotype-Phenotype Mapping and Evolutionary Landscapes](30-Genotype-Phenotype-Mapping.tex) | 50 KB | ✓ Ready |

## Statistics

- **Total Documents**: 30 (+ 2 duplicate variants)
- **Total Size**: ~1.1 MB LaTeX source
- **Average Document Length**: ~35 KB per file
- **Largest Documents**:
  - 30-Genotype-Phenotype-Mapping.tex (50 KB)
  - 19-Chemical-Reaction-Networks.tex (49 KB)
  - 25-Quantum-Algorithms-Complexity.tex (48 KB)

## Features

Each LaTeX document includes:

✓ Professional formatting with proper section hierarchy
✓ Syntax-highlighted Python code blocks (using `listings` package)
✓ Mathematical equations and formulas
✓ Hyperlinked table of contents
✓ Header with challenge number and page numbers
✓ Custom theorem environments
✓ Formatted lists and checklists
✓ Bibliography-ready structure

## LaTeX Packages Used

**Core packages:**
- `amsmath, amssymb, amsthm` - Mathematical typesetting
- `listings` - Code highlighting
- `hyperref` - Hyperlinks and PDF metadata
- `geometry` - Page layout
- `fancyhdr` - Headers and footers
- `xcolor` - Colors for code highlighting

**Optional enhancements:**
- `algorithm, algpseudocode` - Algorithm formatting
- `booktabs` - Professional tables
- `physics` - Physics notation
- `mathtools` - Extended math features

## Compilation Requirements

**Minimum:**
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- `pdflatex` compiler

**Recommended:**
- Full TeX Live installation for all packages
- 2 GB free disk space for compilation
- ~1-2 minutes per document compilation time

## Usage Examples

### Compile Everything

```bash
# Using shell script
./compile_all.sh --clean

# Using Makefile
make all
make clean
```

### Compile Individual Documents

```bash
# Directly with pdflatex
pdflatex 01-AdS3-Modular-Bootstrap.tex
pdflatex 01-AdS3-Modular-Bootstrap.tex  # Second pass for TOC

# Using Make
make pdfs/01-AdS3-Modular-Bootstrap.pdf
```

### Compile Specific Domain

```bash
# Quantum Gravity documents (01-08)
for i in {01..08}; do
    make pdfs/$(ls ${i}-*.tex | head -1 | sed 's/.tex/.pdf/')
done

# Materials Science documents (09-15)
for i in {09..15}; do
    make pdfs/$(ls ${i}-*.tex | head -1 | sed 's/.tex/.pdf/')
done
```

## Customization Guide

### Change Page Size

Edit preamble in `convert_to_latex.py`:

```latex
\geometry{
    letterpaper,  % or a4paper
    left=1in,
    right=1in,
}
```

### Modify Code Highlighting

Edit the `pythonstyle` definition:

```latex
\lstdefinestyle{pythonstyle}{
    language=Python,
    backgroundcolor=\color{codegray},
    % Add more customizations
}
```

### Add Custom Sections

Insert into individual `.tex` files:

```latex
\section{Additional Material}
Your content here...
```

## Troubleshooting

### Common Issues

**Missing packages:**
```bash
tlmgr install <package-name>  # TeX Live
```

**Compilation errors:**
- Check `.log` files in the latex directory
- Ensure all special characters are properly escaped
- Verify UTF-8 encoding

**Long compilation times:**
- First compilation: 30-60s per document (normal)
- Subsequent: 10-20s per document
- Use `make -j4 all` for parallel compilation

## Files Overview

```
latex/
├── INDEX.md                    # This file
├── README.md                   # Detailed documentation
├── compile_all.sh              # Batch compilation script
├── Makefile                    # Make targets
├── 01-*.tex through 30-*.tex   # PRD LaTeX sources
└── pdfs/                       # Compiled PDFs (after make)
    └── *.pdf
```

## Version Information

- **Generated**: 2026-01-18
- **Converter**: `convert_to_latex.py`
- **Source Format**: Markdown (GitHub-flavored)
- **Target Format**: LaTeX (pdflatex-compatible)
- **Encoding**: UTF-8

## Contributing

To regenerate LaTeX from updated markdown:

```bash
cd ..
python3 convert_to_latex.py
cd latex
```

To improve the conversion script:
1. Edit `convert_to_latex.py`
2. Test on a single PRD
3. Regenerate all documents
4. Verify compilation

## License

All documents are released under MIT License, consistent with the parent project.

---

**Ready to compile?** Run `./compile_all.sh` to generate all PDFs!

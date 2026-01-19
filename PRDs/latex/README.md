# LaTeX Documents for Pure Thought AI Challenges

This directory contains professionally formatted LaTeX versions of all 30 PRDs (Product Requirement Documents) for the Pure Thought AI Challenges project.

## Contents

- **30 LaTeX source files (.tex)**: One for each PRD challenge
- **compile_all.sh**: Shell script to compile all documents to PDF
- **Makefile**: Convenience makefile for compilation
- **pdfs/**: Directory containing compiled PDF documents (created after compilation)

## Features

Each LaTeX document includes:
- Professional formatting with proper section hierarchy
- Syntax-highlighted Python code blocks
- Mathematical equations rendered with proper LaTeX
- Hyperlinked table of contents
- Cross-references and citations
- Custom theorem environments
- Formatted lists and tables

## Prerequisites

To compile the LaTeX documents, you need:
- A LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- `pdflatex` command-line tool
- Required LaTeX packages:
  - amsmath, amssymb, amsthm
  - listings (for code highlighting)
  - hyperref (for hyperlinks)
  - geometry (for page layout)
  - fancyhdr (for headers/footers)
  - xcolor (for colors)

### Installing LaTeX

**macOS:**
```bash
brew install --cask mactex
# Or install BasicTeX for a minimal distribution:
brew install --cask basictex
```

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-full
```

**Windows:**
Download and install [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/)

## Compilation

### Compile All Documents

To compile all 30 PRDs to PDF:

```bash
./compile_all.sh
```

This will:
1. Run pdflatex twice on each document (for references and TOC)
2. Create PDFs in the `pdfs/` directory
3. Display a summary of successful and failed compilations

To also clean auxiliary files after compilation:

```bash
./compile_all.sh --clean
```

### Compile Single Document

To compile a single PRD:

```bash
pdflatex 01-AdS3-Modular-Bootstrap.tex
pdflatex 01-AdS3-Modular-Bootstrap.tex  # Run twice for TOC
```

Or use the Makefile:

```bash
make 01-AdS3-Modular-Bootstrap.pdf
```

### Using Make

The Makefile provides convenient targets:

```bash
# Compile all documents
make all

# Compile a specific document
make pdfs/01-AdS3-Modular-Bootstrap.pdf

# Clean auxiliary files
make clean

# Clean everything including PDFs
make distclean

# View list of available targets
make help
```

## Document Structure

Each LaTeX document follows this structure:

1. **Title Page**: Challenge number, title, author, date, abstract
2. **Table of Contents**: Hyperlinked navigation
3. **Main Content**:
   - Problem Statement
   - Scientific Context
   - Mathematical Formulation
   - Implementation Approach (6 phases)
   - Example Starting Prompt
   - Success Criteria (MVR, Strong, Publication-quality)
   - Verification Protocol
   - Resources & References
   - Common Pitfalls
   - Milestone Checklist

## Customization

### Modifying the Template

To change the styling for all documents, edit the preamble in `convert_to_latex.py` and regenerate:

```bash
cd ..
python3 convert_to_latex.py
```

### Individual Document Changes

You can edit any `.tex` file directly. Common customizations:

**Change page size:**
```latex
\geometry{
    a4paper,  % or letterpaper
    left=25mm,
    right=25mm,
}
```

**Change code style:**
```latex
\lstdefinestyle{pythonstyle}{
    % Modify colors, fonts, etc.
}
```

**Add custom commands:**
```latex
\newcommand{\mycmd}[1]{...}
```

## Troubleshooting

### Missing Packages

If compilation fails with "missing package" errors:

**TeX Live:**
```bash
tlmgr install <package-name>
```

**MiKTeX:**
Packages are usually installed automatically. If not:
```bash
mpm --install <package-name>
```

### Long Compilation Times

For large documents with many code blocks:
- First compilation may take 30-60 seconds per document
- Subsequent compilations are faster
- Use `make -j4 all` to compile 4 documents in parallel

### Encoding Issues

All documents use UTF-8 encoding. Ensure your editor saves files as UTF-8.

## File Organization

```
latex/
├── README.md                              # This file
├── compile_all.sh                         # Batch compilation script
├── Makefile                               # Make targets
├── 01-AdS3-Modular-Bootstrap.tex         # PRD 01
├── 02-Gravitational-Positivity-Bounds.tex # PRD 02
├── ...                                    # PRDs 03-29
├── 30-Genotype-Phenotype-Mapping.tex     # PRD 30
└── pdfs/                                  # Compiled PDFs (after compilation)
    ├── 01-AdS3-Modular-Bootstrap.pdf
    ├── 02-Gravitational-Positivity-Bounds.pdf
    └── ...
```

## License

All LaTeX documents are released under MIT License, consistent with the parent project.

## Citation

When using these documents in research or publications:

```bibtex
@techreport{PureThoughtPRDs2026,
  title={Pure Thought AI Challenges: 30 Fundamental Problems in Mathematical Physics},
  author={Pure Thought AI Challenges Project},
  year={2026},
  institution={},
  note={Available at: https://github.com/[your-repo]/pure-thought-challenges}
}
```

## Support

For issues with:
- **LaTeX compilation**: Check `.log` files in this directory
- **Content errors**: Submit issue to main repository
- **Formatting improvements**: Edit `convert_to_latex.py` and regenerate

---

**Generated by**: `convert_to_latex.py`
**Last updated**: 2026-01-18
**Total documents**: 30 PRDs across 6 scientific domains

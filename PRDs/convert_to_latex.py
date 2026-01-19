#!/usr/bin/env python3
"""
Convert PRD markdown files to LaTeX format.

This script converts all 30 PRD markdown files into nicely formatted
LaTeX documents with proper styling, code highlighting, and annotations.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

class MarkdownToLatexConverter:
    """Convert markdown PRD files to LaTeX format."""

    def __init__(self):
        self.in_code_block = False
        self.code_language = 'python'
        self.in_list = False
        self.list_level = 0

    def escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        # Characters that need escaping in LaTeX
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }

        # Don't escape if we're in a code block or math mode
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text

    def convert_inline_formatting(self, text: str) -> str:
        """Convert inline markdown formatting to LaTeX."""
        # Bold: **text** or __text__ -> \textbf{text}
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'__(.+?)__', r'\\textbf{\1}', text)

        # Italic: *text* or _text_ -> \textit{text}
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        text = re.sub(r'_(.+?)_', r'\\textit{\1}', text)

        # Inline code: `code` -> \texttt{code}
        text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)

        return text

    def generate_latex_preamble(self, title: str, prd_number: int) -> str:
        """Generate LaTeX document preamble with packages and styling."""
        return r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{physics}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{tocloft}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}

% Page geometry
\geometry{
    a4paper,
    left=25mm,
    right=25mm,
    top=30mm,
    bottom=30mm,
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Pure Thought Challenge """ + f"{prd_number:02d}" + r"""}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    pdfauthor={Pure Thought AI Challenges},
    pdftitle={""" + title + r"""},
}

% Code listing style
\definecolor{codegray}{rgb}{0.95,0.95,0.95}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codepurple}{rgb}{0.58,0,0.82}

\lstdefinestyle{pythonstyle}{
    language=Python,
    backgroundcolor=\color{codegray},
    commentstyle=\color{codegreen},
    keywordstyle=\color{blue},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\small,
    breaklines=true,
    breakatwhitespace=true,
    captionpos=b,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{gray},
    tabsize=4,
    showstringspaces=false,
}

\lstset{style=pythonstyle}

% Theorem environments
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% Custom commands
\newcommand{\checklist}[1]{\item[$\square$] #1}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\N}{\mathbb{N}}

% Title information
\title{\textbf{""" + title + r"""} \\
\large Pure Thought AI Challenge """ + f"{prd_number:02d}" + r"""}
\author{Pure Thought AI Challenges Project}
\date{\today}

\begin{document}

\maketitle
\thispagestyle{empty}

\begin{abstract}
This document presents a comprehensive Product Requirement Document (PRD) for implementing a pure-thought computational challenge. The problem can be tackled using only symbolic mathematics, exact arithmetic, and fresh code---no experimental data or materials databases required until final verification. All results must be accompanied by machine-checkable certificates.
\end{abstract}

\clearpage
\tableofcontents
\clearpage

"""

    def generate_latex_footer(self) -> str:
        """Generate LaTeX document footer."""
        return r"""
\end{document}
"""

    def convert_heading(self, line: str) -> str:
        """Convert markdown headings to LaTeX sections."""
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not match:
            return line

        level = len(match.group(1))
        title = match.group(2).strip()

        # Remove markdown formatting from title
        title = re.sub(r'\*\*(.+?)\*\*', r'\1', title)
        title = re.sub(r'`(.+?)`', r'\1', title)

        if level == 1:
            return f"\n\\section{{{title}}}\n"
        elif level == 2:
            return f"\n\\subsection{{{title}}}\n"
        elif level == 3:
            return f"\n\\subsubsection{{{title}}}\n"
        elif level == 4:
            return f"\n\\paragraph{{{title}}}\n"
        else:
            return f"\n\\textbf{{{title}}}\n\n"

    def convert_list_item(self, line: str) -> str:
        """Convert markdown list items to LaTeX."""
        # Bullet list: - item or * item
        if re.match(r'^\s*[-*]\s+', line):
            indent = len(re.match(r'^(\s*)', line).group(1))
            item = re.sub(r'^\s*[-*]\s+', '', line)
            item = self.convert_inline_formatting(item)
            return f"\\item {item}\n"

        # Numbered list: 1. item
        if re.match(r'^\s*\d+\.\s+', line):
            item = re.sub(r'^\s*\d+\.\s+', '', line)
            item = self.convert_inline_formatting(item)
            return f"\\item {item}\n"

        # Checkbox: - [ ] item or - [x] item
        if re.match(r'^\s*-\s+\[([ x])\]\s+', line):
            checked = 'x' in line[:10]
            item = re.sub(r'^\s*-\s+\[([ x])\]\s+', '', line)
            item = self.convert_inline_formatting(item)
            if checked:
                return f"\\item[$\\boxtimes$] {item}\n"
            else:
                return f"\\item[$\\square$] {item}\n"

        return line

    def convert_line(self, line: str) -> str:
        """Convert a single markdown line to LaTeX."""
        # Skip empty lines
        if not line.strip():
            return "\n"

        # Handle code blocks
        if line.strip().startswith('```'):
            if not self.in_code_block:
                # Start code block
                self.in_code_block = True
                lang_match = re.match(r'^```(\w+)?', line.strip())
                self.code_language = lang_match.group(1) if lang_match and lang_match.group(1) else 'python'
                return "\\begin{lstlisting}\n"
            else:
                # End code block
                self.in_code_block = False
                return "\\end{lstlisting}\n"

        # If in code block, return line as-is
        if self.in_code_block:
            return line

        # Convert headings
        if line.startswith('#'):
            return self.convert_heading(line)

        # Handle horizontal rules
        if re.match(r'^[-*_]{3,}$', line.strip()):
            return "\n\\bigskip\\hrule\\bigskip\n"

        # Handle list items
        if re.match(r'^\s*[-*]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            if not self.in_list:
                self.in_list = True
                result = "\\begin{itemize}\n"
            else:
                result = ""
            result += self.convert_list_item(line)
            return result
        else:
            if self.in_list:
                self.in_list = False
                return "\\end{itemize}\n\n" + self.convert_inline_formatting(line) + "\n"

        # Handle blockquotes
        if line.startswith('>'):
            text = line[1:].strip()
            text = self.convert_inline_formatting(text)
            return f"\\begin{{quote}}\n{text}\n\\end{{quote}}\n"

        # Regular paragraph
        text = self.convert_inline_formatting(line)
        return text + "\n"

    def convert_file(self, input_path: str, output_path: str, prd_number: int):
        """Convert a markdown file to LaTeX."""
        print(f"Converting {input_path} -> {output_path}")

        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Extract title from first heading
        title = "Pure Thought Challenge"
        for line in lines[:10]:
            if line.startswith('# '):
                title = line[2:].strip()
                break

        # Generate LaTeX document
        latex_lines = [self.generate_latex_preamble(title, prd_number)]

        self.in_code_block = False
        self.in_list = False

        # Skip the first heading (title) as it's in the preamble
        skip_first_heading = True

        for line in lines:
            if skip_first_heading and line.startswith('# '):
                skip_first_heading = False
                continue

            converted = self.convert_line(line)
            latex_lines.append(converted)

        # Close any open environments
        if self.in_list:
            latex_lines.append("\\end{itemize}\n")
        if self.in_code_block:
            latex_lines.append("\\end{lstlisting}\n")

        latex_lines.append(self.generate_latex_footer())

        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(latex_lines)

        print(f"✓ Converted {prd_number:02d}")


def main():
    """Main conversion function."""
    prds_dir = Path(__file__).parent
    latex_dir = prds_dir / "latex"
    latex_dir.mkdir(exist_ok=True)

    converter = MarkdownToLatexConverter()

    # List of all PRDs
    prd_files = sorted([
        f for f in os.listdir(prds_dir)
        if f.endswith('.md') and re.match(r'^\d{2}-', f)
    ])

    print(f"Found {len(prd_files)} PRD files")
    print("Converting to LaTeX...\n")

    for prd_file in prd_files:
        # Extract PRD number
        prd_num = int(prd_file[:2])

        input_path = prds_dir / prd_file
        output_filename = prd_file.replace('.md', '.tex')
        output_path = latex_dir / output_filename

        converter.convert_file(str(input_path), str(output_path), prd_num)

    print(f"\n✓ All {len(prd_files)} PRDs converted successfully!")
    print(f"LaTeX files saved to: {latex_dir}")
    print("\nTo compile a LaTeX file, run:")
    print("  cd latex")
    print("  pdflatex 01-AdS3-Modular-Bootstrap.tex")


if __name__ == "__main__":
    main()

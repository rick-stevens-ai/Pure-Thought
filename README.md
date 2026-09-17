# Pure Thought: revised scientific problem specifications

Every problem includes a **Plain-language guide** with an everyday explanation, key-term definitions, why the problem matters, and what meaningful progress would establish.

Thirty research challenges framed as precise mathematical or computational tasks, with detailed critiques of the original descriptions. Revised locally on 2026-09-11 from source commit `f854d4e12b55c1f5b5c2a964b75f2d23496d981e`.

## Read the revision

- [Review overview and major findings](reviews/REVIEW.md)
- [Complete problem summaries and formulations: all 42 topics](COMPLETE-PROBLEM-SUMMARIES.md)
- [Complete summaries PDF](latex-reports/Complete-Problem-Summaries.pdf)
- [Revised LaTeX/PDF reports: 30 canonical and 12 supplemental](latex-reports/README.md)
- [Complete reading copy: rewrites and critiques together](reviews/COMPLETE-REVIEW.md)
- [All 30 revised problems and individual critiques](PRDs/README.md)
- [Synthesis crosswalk: all 30 earlier topics](reviews/SYNTHESIS-CROSSWALK.md)
- [12 supplemental revised topics with critiques](supplemental/README.md)
- [Evidence and execution standard](reviews/EVIDENCE-STANDARD.md)
- [Validation report](reviews/VALIDATION.md)

Each revised problem defines a core question, scope, mathematical target, deliverables, verification controls and milestones. Baselines are deliberately bounded; ambitious research extensions remain explicit. Numerical evidence, finite-model certificates and continuum theorems are distinguished.

## Repository layout

| Directory | Role |
|---|---|
| `PRDs/` | Current canonical specifications; two old duplicate filenames redirect |
| `reviews/` | Overview, 30 individual critiques, evidence standard, crosswalk and provenance |
| `supplemental/` | Revised distinct topics from the earlier synthesis |
| `archive/` | Original specifications, long-form reports and synthesis, preserved for comparison |
| `latex-reports/` | Regenerated TeX/PDF research reports and complete summaries |
| `NOTES/` | Revised 30-topic synthesis plus historical drafts and legacy scripts |
| `scripts/` | Reproducible report and summary generation |

The revision is an editorial and scientific-framing review. It does not claim the open problems are solved, the archived research code is validated, or the archived long-form derivations are validated. The active reports were replaced on 2026-09-17 with the reviewed research specifications, displayed formulations and critiques; unchecked historical code sketches are preserved in the archive. See the review overview for coverage.

Original repository: [rick-stevens-ai/Pure-Thought](https://github.com/rick-stevens-ai/Pure-Thought). The source README stated MIT licensing but the reviewed commit contains no LICENSE file; this revision does not add or infer license terms.

## Rebuild and validate

With Python 3, Pandoc and XeLaTeX installed:

```sh
python3 scripts/build_reports.py --pdf
python3 scripts/build_reports.py --check
python3 reviews/validate_revision.py
```

See [report revision and validation](reviews/REPORT-REVISION.md) for coverage, preservation and remaining scientific proof obligations.

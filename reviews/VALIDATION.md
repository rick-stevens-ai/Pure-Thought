# Validation of the revision package

Validated locally on 2026-09-11. These checks assess document coverage, preservation, and selected corrected examples. They do not certify the proposed research algorithms or all scientific claims in the portfolio.

## Coverage and preservation

- 30 canonical revised specifications, each with the required question, assumptions, target, outputs, validation and milestones.
- 30 individual critiques linked to the corresponding source and rewrite.
- Both duplicate filenames for 07 and 08 retained as redirects; their original contents preserved.
- All 30 synthesis entries mapped; 12 distinct objectives have supplemental rewrites and critiques.
- All 33 original PRD-directory Markdown files, including its README, preserved byte for byte.
- Historical TeX/PDF reports and draft TXT/DOCX files checked against their original SHA-256 hashes; no content changes.
- Revised-document relative links and display-math delimiters checked; no broken local targets or accidental control characters.

## Independently checked corrections

The included [validation script](validate_revision.py) uses only the Python standard library. Run it from any directory with `python3 /path/to/Pure-Thought/reviews/validate_revision.py`.

| Check | Result |
|---|---|
| Hamming hypergraph product | n=58, k=16; commutation and GF(2) ranks checked |
| Distance of that product | Both sectors exclude nontrivial logicals of weights 1 and 2; weight-3 witnesses found, so d=3 |
| Reaction-network cubic | Roots 1,2,3 exactly; derivatives −2,+1,−2 give two locally stable scalar equilibria |
| Nekhoroshev arithmetic example | exp(10^0.3)=7.354131873770767, a dimensionless number |
| RNA length-20 enumeration scale | 4^20=1,099,511,627,776 |
| Weyl benchmark | Stated nodes satisfy the model; velocity determinants have opposite signs |

Primary-source checks supporting conceptual corrections are cited in each problem/critique. They are selective checks, not a systematic literature review or a complete audit of all historical references.

## Packaging checks

The accompanying Git patch is checked against a fresh checkout of the recorded source commit. The ZIP contains the revised repository files, including preserved originals and historical reports, but excludes `.git`. Local-link and coverage checks are rerun after assembling the combined reading copy.

Active revised Markdown is checked for whitespace errors. Archived originals deliberately retain their source bytes, including any pre-existing whitespace. The long-form PDFs are historical and have not been rendered or regenerated as part of this revision.

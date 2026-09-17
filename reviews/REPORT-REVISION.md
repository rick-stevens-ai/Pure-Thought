# Report and complete-summary revision

**Date:** 2026-09-17. **Input revision:** `fe2c1a3` (the updated specifications, critiques and plain-language guides). **Historical source:** `f854d4e12b55c1f5b5c2a964b75f2d23496d981e`.

## Delivered coverage

- 30 canonical LaTeX reports and rebuilt PDFs at the existing report paths.
- 12 new supplemental LaTeX reports and PDFs, preserving distinct synthesis objectives.
- [Complete problem summaries](../COMPLETE-PROBLEM-SUMMARIES.md), with all 42 specifications and their guides, scope, equations, outputs and validation gates, plus a [TeX/PDF reading copy](../latex-reports/Complete-Problem-Summaries.pdf).
- [Updated complete review](COMPLETE-REVIEW.md), retaining every specification and detailed critique and adding the displayed formulations.
- [Revised synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md) and rebuilt TeX/PDF, preserving all 30 original synthesis IDs and their canonical/supplemental crosswalk.
- [Mathematical formulation companion](MATHEMATICAL-FORMULATIONS.md), containing 42 explicit formulations and claim boundaries.

Each canonical report includes its full current specification, displayed formulation, individual critique and the shared evidence standard. Supplemental reports include the full specification/critique, formulation and evidence standard. Historical tutorial derivations, unsupported conclusions and unchecked implementation sketches are preserved in the archive rather than reproduced as active guidance. This is a replacement of the report contents with the reviewed research contracts, not a claim that every archived derivation or program has been repaired.

## Mathematical changes

The active report formulations now follow the review: corrected holomorphic characters and finite spectral tests; conditional gravitational sum rules and infrared assumptions; distributional celestial transforms; spinning versus scalar crossing; finite geometry search classes; whole-domain topology certificates; correct variational bound directions for RDM and Coulomb transport; consistent path-integral temperature factors; exact CSS/hypergraph-product dimensions and homology tests; NPA versus tensor-product distinctions; all-integer KAM assumptions and dimensional stability times; and distinct RAF and finite genotype-map targets.

The formulation companion also makes the supplemental questions explicit, including operator-versus-density notation for finite density functionals, cost-constrained channel dual bounds, and finite-to-bulk or finite-to-asymptotic proof obligations. Small source edits repair mathematical delimiters and the stray comma in the projector Chern integral. The generator converts Unicode mathematical notation without dropping superscripts, subscripts or accents.

No new numerical research results or novelty claims are inferred from this editorial work. The existing independent benchmark checks are retained. The original specifications' numerical evidence, finite-model certificate, conditional theorem and formal-proof distinctions apply throughout.

## Preservation and reproducibility

The original 30 TeX/PDF pairs are preserved byte for byte in `archive/original-latex-reports/`; the original synthesis Markdown/TeX/PDF are in `archive/original-synthesis/`. The original SHA-256 manifest is unchanged. The validator resolves the relocated originals before checking their original hashes. Original PRDs and draft DOCX/TXT files remain preserved.

`python3 scripts/build_reports.py --pdf` regenerates the summaries, complete review, synthesis and 44 standalone TeX/PDF pairs from the shared source documents. `--check` verifies deterministic generated-text synchronization. Python 3, Pandoc and XeLaTeX are required; a standard TeX installation supplies the fonts and packages. Compilation uses two passes for contents and cross-references. Logs and rendering intermediates stay in ignored `work/report-build/`.

## Validation performed

- Coverage and local-link checks for 30 canonical specifications, 30 critiques and 12 supplemental specifications; every problem retains its complete plain-language guide.
- Source synchronization for all 47 generated text artifacts.
- Original-file SHA-256 preservation checks, including all historical reports and the synthesis.
- Independent exact benchmark checks for the Hamming hypergraph product, scalar reaction equilibria, Weyl nodes, RNA enumeration scale and dimensionless exponential.
- Compilation of all 44 PDFs; compiler checks for missing glyphs and layout overflow.
- PDF text extraction and page rendering for visual inspection of report layouts, mathematical sections and combined reading copies.

These checks validate packaging, typesetting, synchronization and the listed benchmarks. They do not certify the proposed research algorithms, prove all target theorems, or audit all referenced literature for novelty.

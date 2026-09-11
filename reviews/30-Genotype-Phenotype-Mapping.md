# Critique 30: An exact finite genotype–phenotype map and its mutation graph

[Revised problem](../PRDs/30-Genotype-Phenotype-Mapping.md) · [Preserved original](../archive/original-PRDs/30-Genotype-Phenotype-Mapping.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/30-Genotype-Phenotype-Mapping.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — two folding objectives (16, 61–71, 1083).** Nussinov pair maximization is not nearest-neighbor MFE minimization. Requiring 100% agreement with ViennaRNA’s default energy model is an invalid acceptance test.

2. **Major — nondeterministic ties.** Without an explicit tie-breaking rule, the map is set-valued and neutral-network statistics depend on the chosen representative. The rewrite fixes the map before measuring it.

3. **Major — neighborhood definition (55, 83).** B₁ includes the genotype itself, so the original normalization is not the fraction of one-mutant neighbors. The correct degree is 3L.

4. **Major — feasibility and extrapolation (39, 87, 1102–1105).** 4²⁰=1,099,511,627,776 sequences; exhaustive folding is not a casual short-sequence baseline. A universal L≈30 percolation threshold and fixed robustness correlations do not follow from the model. Start with a complete tractable graph and report sampling separately.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fully specified combinatorial RNA folding rule on short sequences, what are the exact neutral components, mutation robustness, and accessible phenotype counts?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete exact maps through L=8. Strong: increase L within measured resources and characterize component-level robustness/evolvability. Research extension: a proved asymptotic property or a specified noisy GP-channel capacity problem. Do not hard-code a universal percolation threshold or empirical target correlation.

## Sources supporting the corrections

- [ViennaRNA, RNAfold documentation](https://www.tbi.univie.ac.at/RNA/RNAfold) — RNAfold’s energy model and parameter choices differ from a pair-count model.

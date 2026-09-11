# Critique 13: Higher-order topology with explicit boundary and symmetry assumptions

[Revised problem](../PRDs/13-Higher-Order-Topological-Insulators.md) · [Preserved original](../archive/original-PRDs/13-Higher-Order-Topological-Insulators.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/13-Higher-Order-Topological-Insulators.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong observable (73–89).** The proposed quadrupole formula is a Chern-type curvature integral, not a quadrupole moment, and the corner-charge product formula is not universal. Replacing these prevents certifying the wrong invariant.

2. **Blocking — nested loop construction (78–85).** One must construct a Wannier-sector bundle and its holonomy; integrating eigenphases from the first loop is not that construction.

3. **Major — corner charge versus zero mode (101 onward).** Quantized boundary charge does not automatically pin a level to zero energy. Termination, filling, reference charge, and extra spectral symmetry matter.

4. **Major — scope (36–45).** All wallpaper/space-group HOTIs encompass different equivalence notions and boundary phenomena. The rewrite uses one well-specified benchmark and demands a Wannier gap before invoking nested polarization.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed quadrupole-insulator family, can one certify the bulk and Wannier gaps, a quantized nested polarization, and the corresponding boundary charge under a specified termination?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: reproduce one quadrupole benchmark with valid Wilson-loop construction. Strong: certify its invariant and boundary charge on a parameter box. Research extension: classify a bounded family or one additional symmetry setting; no universal HOTI classification is implied.

## Sources supporting the corrections

- [Benalcazar, Bernevig and Hughes, Quantized Electric Multipole Insulators](https://arxiv.org/abs/1611.07987) — Nested Wilson loops and assumptions for quantized multipole response.

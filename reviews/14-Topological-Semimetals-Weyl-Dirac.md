# Critique 14: Certified Weyl nodes and slice topology in a lattice model

[Revised problem](../PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Preserved original](../archive/original-PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/14-Topological-Semimetals-Weyl-Dirac.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — oversimplified symmetry (22–25).** Time reversal and inversion alone are not a universal recipe for a protected fourfold Dirac crossing; representation and additional symmetry constraints matter. They cannot replace an explicit model.

2. **Major — isotropy assumed (73–87).** A general Weyl cone has a velocity matrix, can be tilted, and is not specified by one scalar v_F. Its determinant gives the simple-node chirality in a declared convention.

3. **Major — completeness (99–105).** Finding roots and checking their charge sum does not prove there are no additional zero-charge pairs. A proof on the complement of the isolating boxes is needed.

4. **Major — boundary overclaim.** Fermi-arc connectivity depends on surface projection and termination; opposite nodes can project together. The revision uses slice topology as the robust intermediate claim and separates surface-specific predictions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can all nodes in a specified two-band three-dimensional Hamiltonian be isolated, assigned chirality, and related to the Chern numbers of gapped momentum slices?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete benchmark node and slice audit. Strong: certified parameter-dependent node motion or one bounded model family. Research extension: a symmetry-protected Dirac model with explicit little-group constraints, or nodal-link invariants with their own definitions.

## Sources supporting the corrections

- [Wan et al., Electronic Structure of Pyrochlore Iridates (published as Topological semimetal and Fermi-arc surface states)](https://arxiv.org/abs/1007.0016) — Reference for Weyl nodes and surface arcs; benchmark and proof obligations here are explicitly specified.

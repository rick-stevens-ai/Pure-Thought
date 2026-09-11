# Critique 18: Certified optimal-transport bounds for strictly correlated electrons

[Revised problem](../PRDs/18-Optimal-Transport-Chemistry.md) · [Preserved original](../archive/original-PRDs/18-Optimal-Transport-Chemistry.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/18-Optimal-Transport-Chemistry.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong functional (16, 34, 98–108).** Coulomb OT describes a strong-correlation limit, not the exact exchange functional asserted here. The negative half-cost formula and negative-cost dual are inappropriate for the positive repulsion minimization.

2. **Major — normalization and antisymmetry.** The electron density integrates to N, while each probability marginal integrates to one. Fermionic wavefunction antisymmetry does not make a probability measure antisymmetric.

3. **Major — reaction pathway inference (18, 36).** A Wasserstein geodesic minimizes transport cost, not a molecular potential-energy barrier. The original chemical interpretation lacks a dynamical/energetic bridge.

4. **Major — uniqueness and certification (106, 868–905).** Quadratic-cost Brenier results do not transfer automatically to Coulomb multimarginal problems. Marginal residuals and a regularized solver output are not continuum certificates. The revised formulation distinguishes these errors and removes the inappropriate helium-exchange target.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a supplied normalized electron density, can a discretized Coulomb multi-marginal transport problem yield verified bounds, with continuum error separated from optimization error?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: certified two-marginal discrete bounds. Strong: one continuum enclosure with singularity/tail control. Research extension: N>2 with symmetry reduction and validated bounds, or a demonstrably improved relaxation. Do not promise deterministic Monge maps for arbitrary multimarginal inputs.

## Sources supporting the corrections

- [Cotar, Friesecke and Klüppelberg, Density functional theory and optimal transportation with Coulomb cost](https://arxiv.org/abs/1104.0603) — Strong-correlation/semiclassical connection; theorem hypotheses must be checked for each density.

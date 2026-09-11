# Critique 12: Rigidity and boundary modes of periodic Maxwell frames

[Revised problem](../PRDs/12-Topological-Mechanical-Metamaterials.md) · [Preserved original](../archive/original-PRDs/12-Topological-Mechanical-Metamaterials.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/12-Topological-Mechanical-Metamaterials.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — counting formula (22–27, 100).** The original replaces the Maxwell–Calladine difference between zero modes and self stresses with an incorrect count. It can mislabel even simple rigid structures.

2. **Blocking — mechanical operator (75–78).** The displayed dynamical matrix does not provide a reliable central-force construction. Building C from bond geometry and K=C†kC supplies a checkable definition.

3. **Major — graph-only topology (38, 61–64).** Frames with the same graph can have different rigidity and polarization. Geometric embedding and unit-cell conventions are part of the input.

4. **Major — boundary and stability claims (87–103).** A Berry phase alone does not give a universal zero-mode count, and a passive unstressed stiffness matrix does not have generic negative-frequency-squared spectral flow. Boundary indices and transfer decay replace the unsupported shortcut.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a geometrically specified isostatic frame and boundary termination, can the zero-mode count and localization be proved from its compatibility matrix?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact Maxwell–Calladine counts on finite frames. Strong: one periodic frame with certified winding and termination-dependent edge localization. Research extension: a bounded family with robust mode counts. Elastic beams, prestress, nonlinear stability, and manufacturing tolerance require separate models.

## Sources supporting the corrections

- [Kane and Lubensky, Topological Boundary Modes in Isostatic Lattices](https://arxiv.org/abs/1308.0554) — Compatibility matrices, topological indices, and boundary modes.

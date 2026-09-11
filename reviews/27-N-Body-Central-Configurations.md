# Critique 27: Central configurations in a bounded, symmetry-reduced N-body problem

[Revised problem](../PRDs/27-N-Body-Central-Configurations.md) · [Preserved original](../archive/original-PRDs/27-N-Body-Central-Configurations.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/27-N-Body-Central-Configurations.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong example (26–29, 120).** The figure-eight choreography changes shape and is not a central configuration. It belongs in a periodic-orbit problem, retained separately in the supplement.

2. **Blocking — counts and quotients (112–114).** The text confuses five restricted-problem Lagrange points with Euler central configurations. Counting depends on labels, rotations and reflections; the rewrite states the quotient and a four-shape three-body benchmark.

3. **Blocking — stability shortcut (124–130).** Hessian signs alone do not give rotating-frame orbital stability; Coriolis terms and symmetry modes must be treated.

4. **Major — algebraic and finiteness claims (36, 93–98).** Clearing denominators can add collisions and does not remove square roots without auxiliary variables. Numerical roots do not prove completeness, and the generic-five-body status claim needs qualification against the literature. The new bounded scope avoids an unsupported universal claim.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For fixed positive masses and a declared symmetry quotient, can all collision-free central configurations in the chosen family be certified and their relative-equilibrium linearization analyzed?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete certified three-body catalog. Strong: complete enumeration in one four-body symmetry family or mass instance, with all charts covered. Research extension: a five-body restricted family; no claim to solve the general finiteness problem without an appropriately global proof.

## Sources supporting the corrections

- [Yu and Zhu, On the finiteness of four-body central configurations](https://arxiv.org/abs/2103.08906) — Context for finiteness versus explicit certified enumeration.

# Critique 05: Positive-geometry tests for a fixed gravity integrand

[Revised problem](../PRDs/05-Positive-Geometry-Gravity.md) · [Preserved original](../archive/original-PRDs/05-Positive-Geometry-Gravity.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/05-Positive-Geometry-Gravity.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — undefined search universe (35–45, 91–95).** An unrestricted geometry-existence question cannot be ruled out by an unsuccessful symbol search. The rewrite makes both positive and negative answers mathematically scoped.

2. **Major — form versus integral (53–63).** A symbol is not a wedge of dlog one-forms, and integrated branch-cut data are not interchangeable with poles of an integrand. Canonical-form residue recursion applies to a specified differential form.

3. **Major — kinematics (49).** All physical Mandelstam invariants need not be positive simultaneously. Positivity refers to the chosen real geometry and coordinates.

4. **Major — validation (84–89).** Agreement on selected cuts or numerical integration can miss contact terms, infinity poles, or representation ambiguities. The new deliverable requires an exact equality statement and an exhaustive boundary audit within the chosen ansatz.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified low-loop supergravity integrand, does a precisely bounded class of candidate positive geometries reproduce its differential form, including every boundary and pole at infinity?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact target and pole/residue audit. Strong: one verified representation or one restricted no-go theorem. Research extension: broaden the geometry class or multiplicity; explain how the result changes understanding of gravity amplitudes rather than treating a computational failure as a universal obstruction.

## Sources supporting the corrections

- [Arkani-Hamed, Bai and Lam, Positive Geometries and Canonical Forms](https://arxiv.org/abs/1703.04541) — Definition of logarithmic canonical forms and boundary recursion.

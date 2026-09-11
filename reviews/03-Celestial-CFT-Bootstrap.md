# Critique 03: Celestial amplitudes: distributional crossing and soft-limit consistency

[Revised problem](../PRDs/03-Celestial-CFT-Bootstrap.md) · [Preserved original](../archive/original-PRDs/03-Celestial-CFT-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/03-Celestial-CFT-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — missing distributional specification (36–45, 94–113).** A Mellin transform of a stripped amplitude without momentum conservation and convergence rules is incomplete. The numerical quadrature sketch does not define the multidimensional distribution that is actually needed.

2. **Blocking — imported positivity (18–23, 57–75).** Principal-series celestial data cannot simply inherit the positive OPE-coefficient squares of a unitary Euclidean two-dimensional CFT. An SDP needs a proved positivity structure, not an asserted positive norm.

3. **Major — soft poles (60–64).** The generic Δ→0 double-pole formula does not reproduce the leading graviton soft pole in the stated Mellin convention. Deriving the pole from the energy power prevents convention mistakes.

4. **Major — scope (15–28, 79–86).** “Consistent space” and “islands” assume tools whose foundational prerequisites are themselves research questions. A distributionally correct identity pipeline is a meaningful prerequisite and preserves the original direction.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can a regulated celestial transform of a specified four-graviton tree amplitude reproduce Lorentz covariance, crossing, and the leading conformally soft residue in one consistent convention?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one four-point transform with explicit support and checked leading soft residue. Strong: a second helicity/process example and a verified crossing relation. Research extension: derive a justified positive constraint in a restricted subsector before attempting an SDP island.

## Sources supporting the corrections

- [Adamo et al., Celestial amplitudes and conformal soft theorems](https://www.pure.ed.ac.uk/ws/files/121747399/Adamo_2019_Class._Quantum_Grav._36_205018_1_.pdf) — Conformally soft graviton operators at Δ=1 and Δ=0.

# Critique 11: Certified topological bands in an idealized Maxwell medium

[Revised problem](../PRDs/11-Photonic-Topological-Crystals.md) · [Preserved original](../archive/original-PRDs/11-Photonic-Topological-Crystals.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/11-Photonic-Topological-Crystals.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — missing nonreciprocal constitutive physics (68–98).** A real scalar dielectric model does not implement the claimed gyromagnetic Chern phase. The relevant tensor and inner product must be included.

2. **Major — electronic Z₂ shortcut (95–98).** Ordinary photonic time reversal squares to +1; an electronic parity formula cannot be transplanted without an appropriate additional symmetry structure.

3. **Major — “exact” Maxwell solution (21–25, 56–60).** Exact governing equations do not imply exact numerical eigenvalues. Discretization, spurious longitudinal modes, and continuum coverage need validation.

4. **Major — fabrication and robustness (37–42, 740–754).** Generic dielectric rods and STL files do not establish nonreciprocity, telecom performance, or a universal 15% disorder tolerance. The new problem certifies a mathematical medium first and identifies the extra evidence required for engineering claims.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can a specified lossless periodic electromagnetic model support a rigorously isolated band group with nonzero Chern number, and can its interface modes be validated?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: validated reciprocal eigenproblem. Strong: one certified ideal nonreciprocal topological design. Research extension: a realistic dispersive constitutive model or gap optimization within a bounded geometry family. Fabrication drawings alone do not certify material behavior.

## Sources supporting the corrections

- [Haldane and Raghu, Directional optical waveguides with broken time-reversal symmetry](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.013904) — Nonreciprocal media as the mechanism for photonic quantum-Hall-like edge modes.

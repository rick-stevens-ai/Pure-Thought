# Critique 07: Stress-tensor bootstrap: a sector-specific gap bound in three dimensions

[Revised problem](../PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Preserved original](../archive/original-PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/07-Extremal-CFTs-Stress-Tensor.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect current dimensions (main original 60–64).** Higher-spin conserved currents have Δ=ℓ+d−2. Assigning all of them Δ=d violates the representation-theory input.

2. **Blocking — scalar treatment of spinning crossing (49–75).** TTTT has multiple tensor structures and OPE matrices; a single squared coefficient and G(u,v)=G(v,u) do not specify its crossing system.

3. **Blocking — functional domain/sign (83–91; short duplicate 39–49).** Positivity must hold for allowed operators. The original asks for positivity in the excluded region and alternates lower and upper gap bounds. The rewrite gives an explicit contradiction convention.

4. **Major — ambiguous target and duplicate (both 07 files).** A gap to a conserved current, a scalar gap, and a single-trace holographic gap are different observables. Both originals now resolve to one canonical scalar-sector problem; gravity uniqueness is not inferred from it.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **At fixed stress-tensor two- and three-point data, can crossing exclude a proposed gap in the parity-even scalar sector of T×T in a unitary three-dimensional CFT?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: reproduce one finite benchmark and publish the assumptions. Strong: turn one exclusion into a certified result with omitted-sector control. Research extension: improve a bound, add another sector, or obtain a conditional extremal spectrum. Saturation does not construct a CFT.

## Sources supporting the corrections

- [Dymarsky et al., The 3d Stress-Tensor Bootstrap](https://arxiv.org/abs/1708.05718) — Conserved tensor crossing and sector-specific numerical gap bounds.

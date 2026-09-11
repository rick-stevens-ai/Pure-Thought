# Critique 09: Certified Chern phases in a bounded tight-binding family

[Revised problem](../PRDs/09-Topological-Band-Theory.md) · [Preserved original](../archive/original-PRDs/09-Topological-Band-Theory.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/09-Topological-Band-Theory.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — unbounded classification (37–41, 803–813).** Fixing a space group does not bound the hopping range or continuous parameters. Claims to enumerate every model and prove minimality have no finite search universe.

2. **Blocking — false orbital lower bounds (811–813).** Higher Chern number does not generally require |C|+1 bands; two-band maps can have higher degree when suitable hopping harmonics are allowed. Range restrictions are indispensable to any proposed lower bound.

3. **Major — numerical integer versus certified topology (89–94, 790–793).** Discretized curvature and finite differences are not exact continuum certificates without gap and error control.

4. **Major — physical extrapolation (45–53).** A mathematical model does not imply a stable synthesizable material. The revision makes the model atlas the result and keeps materials relevance conditional.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

## Sources supporting the corrections

- [Chen et al., The impossibility of exactly flat non-trivial Chern bands in strictly local periodic tight binding models](https://arxiv.org/abs/1311.4956) — A concrete example of why range/locality assumptions change realizability claims.

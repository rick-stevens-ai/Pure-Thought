# Critique 06: S-matrix bootstrap with an explicit gravitational infrared prescription

[Revised problem](../PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Preserved original](../archive/original-PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/06-Nonperturbative-S-matrix-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — elastic window with gravity (68–79).** Massless radiation removes the simple massive-theory inelastic threshold assumption. The equality form of elastic unitarity cannot be silently reused. The relaxed inequality and the IR observable must be justified.

2. **Blocking — singular partial waves.** Long-range graviton exchange makes naive four-dimensional angular projection problematic. The proposed all-purpose Roy-like equation (89–92) is not a derivation of the required kernels.

3. **Major — inner versus outer approximation.** Infeasibility of a restricted amplitude ansatz excludes only that ansatz. This distinction is essential for every claimed no-go theorem.

4. **Major — missing continuum coverage.** A finite partial-wave/energy grid cannot establish all-energy unitarity. The new specification requires analytic tail bounds and otherwise reports a finite relaxation, preserving meaningful progress without overstating it.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can one certify an exclusion for a low-energy scalar-scattering parameter using a crossing-symmetric partial-wave relaxation whose infrared and truncation assumptions are explicit?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: certified finite nongravitational relaxation. Strong: justify an IR treatment and certify one conditional gravitational exclusion. Research extension: improved continuum bounds or a constructive amplitude with a precisely delimited consistency claim.

## Sources supporting the corrections

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Infrared and analyticity issues that must be resolved before importing standard bounds.

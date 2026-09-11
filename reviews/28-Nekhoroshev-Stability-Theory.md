# Critique 28: Explicit finite-time action confinement from a normal form

[Revised problem](../PRDs/28-Nekhoroshev-Stability-Theory.md) · [Preserved original](../archive/original-PRDs/28-Nekhoroshev-Stability-Theory.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/28-Nekhoroshev-Stability-Theory.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect steepness (30–31, 78–88).** Steepness and quasi-convexity are not equivalent to a positive Hessian determinant or an auxiliary convex function. The rewrite uses a theorem-specific condition.

2. **Blocking — fabricated timescale (24, 111, 854–861).** The numerical exponential shown is about 7.35, not 10¹³ years, and no physical time unit or constants were supplied. This invalidates the stated solar-system benchmark.

3. **Major — exponents (106–109).** “Super-steep” and ε-dependent exponents are not a substitute for a cited precise theorem. Exponents cannot be optimized merely to meet a desired age target.

4. **Major — global versus local control.** A truncated Fourier series, short integration, or local Hessian check does not certify the full action domain and time interval. A direct normal-form remainder bound supplies a defensible first milestone and exposes exactly what stronger claims require.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified analytic Hamiltonian and action domain, what rigorously computable action excursion and confinement time follow from a certified normal form or quantitative Nekhoroshev theorem?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one explicit finite-time confinement certificate. Strong: optimized truncation with a stronger bound or full quantitative quasi-convex theorem application. Research extension: exponential scaling across a parameter family; do not infer actual solar-system stability from ε alone.

## Sources supporting the corrections

- [Zhang and Zhang, Improved stability for analytic quasi-convex nearly integrable systems](https://arxiv.org/abs/1701.06026) — Theorem-specific quasi-convex exponential stability results.

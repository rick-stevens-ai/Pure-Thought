# 28. Explicit finite-time action confinement from a normal form

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/28-Nekhoroshev-Stability-Theory.md) · [Original description](../archive/original-PRDs/28-Nekhoroshev-Stability-Theory.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a specified analytic Hamiltonian and action domain, what rigorously computable action excursion and confinement time follow from a certified normal form or quantitative Nekhoroshev theorem?

## Scope and assumptions

Use H=h(I)+εf(I,θ), h=ω·I+|I|²/2, in two or three degrees of freedom on a compact action box with an explicit complex extension. Supply ω, a finite Fourier polynomial f, and rational ε. Bound the gradient away from zero where the selected theorem requires it.

The initial target is a valid dimensional time bound, even if modest. Planetary and post-Newtonian models are extensions requiring their own reductions and hypotheses.

## Mathematical target

For a quasi-convex theorem verify its actual condition, for example vᵀD²h(I)v≥m|v|² for v⊥∇h(I), plus the theorem’s gradient, domain and analyticity conditions. A positive Hessian determinant alone is insufficient.

Alternatively construct a canonical Φ with H∘Φ=Z(J)+R(J,θ), bound |I−J|≤η and ||∂_θR||≤r on a validated domain. Then |I(t)−I(0)|≤2η+r|t| while the transformed orbit remains in that domain. If resonant Z depends on angles, include its action drift or restrict the claim to conserved/protected components.

## Required outputs and proof obligations

Export normal-form order, transformations, inverse/domain bounds, Fourier tails, r, η, and a domain-containment proof for the claimed interval. A Nekhoroshev result must include all constants and its smallness threshold in T=C exp(c ε^−a) with stated units; do not choose a from a desired time target.

## Validation and rejection controls

Verify integrable ε=0 behavior and a low-order symbolic normal form. Check symplecticity with validated remainder control. Confirm the action bound stays inside the certified domain. Independently recompute exp(10^0.3)≈7.35: it is a dimensionless factor, not 10¹³ years.

## Milestones and research extension

Baseline: one explicit finite-time confinement certificate. Strong: optimized truncation with a stronger bound or full quantitative quasi-convex theorem application. Research extension: exponential scaling across a parameter family; do not infer actual solar-system stability from ε alone.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Zhang and Zhang, Improved stability for analytic quasi-convex nearly integrable systems](https://arxiv.org/abs/1701.06026) — Theorem-specific quasi-convex exponential stability results.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

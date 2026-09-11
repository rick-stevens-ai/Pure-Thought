# 01. Extremal holomorphic CFT partition functions: certified necessary tests

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/01-AdS3-Modular-Bootstrap.md) · [Original description](../archive/original-PRDs/01-AdS3-Modular-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For c=24k, which extremal holomorphic partition-function candidates pass exact genus-one spectral tests, and can any candidate be excluded by an additional, explicitly stated consistency condition?

## Scope and assumptions

Start with bosonic holomorphic theories, a unique vacuum, c=24k, and k=1,…,4. This is a deliberately restricted version of the original gravity question. Use h for holomorphic weight; do not identify it with the full scaling dimension Δ=h+h̄. The extremal ansatz has no nonvacuum Virasoro primary with h≤k. Treat the nonchiral modular bootstrap as a separate extension with independent (h,h̄) data.

Keep the motivating existence question, but make the first deliverable a necessary-condition test. A modular function with integer coefficients is not a constructed CFT or a demonstrated gravity dual.

## Mathematical target

With q=exp(2πiτ), c>1, use

$$χ_0=q^{-c/24}\prod_{n=2}^{∞}(1-q^n)^{-1},\qquad χ_h=q^{h-(c-1)/24}/η(τ)\quad(h>0).$$

Construct the unique degree-k polynomial in J=j−744 whose expansion agrees with χ₀ through q⁰. Write Zₖ=χ₀+Σ_{h≥k+1}dₕχₕ and extract primary multiplicities exactly through a declared cutoff H, initially H=100. Prove modular invariance from the polynomial representation, not from evaluations at a few τ. A negative or noninteger dₕ excludes the stated candidate; positivity through H is only a finite test.

## Required outputs and proof obligations

Export the polynomial coefficients, exact q-series, primary multiplicities, conventions, cutoff, and a checker that independently expands η and J. An exclusion must name the violated necessary condition. An additional genus-two or OPE test must state its own assumptions and how its constraints follow from CFT consistency.

## Validation and rejection controls

Recover J=q⁻¹+196884q+… at k=1 and d₂=196883 after subtracting the vacuum descendant. Test the checker with a deliberately altered coefficient. Check agreement of two series-construction routes. Do not report all-order coefficient positivity without a tail theorem.

## Milestones and research extension

Baseline: exact k=1,…,4 tables with explicitly finite coverage. Strong: one additional sewing/OPE condition with independently checked consequences. Research extension: a new exclusion theorem or a full CFT construction; neither is a required outcome or a promised deadline.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Witten, Three-Dimensional Gravity Revisited](https://arxiv.org/abs/0706.3359) — Motivation for the holomorphic extremal ansatz, not a proof that every candidate exists.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

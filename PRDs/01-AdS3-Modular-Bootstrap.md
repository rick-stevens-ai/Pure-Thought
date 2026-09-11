# 01. Extremal holomorphic CFT partition functions: certified necessary tests

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/01-AdS3-Modular-Bootstrap.md) · [Original description](../archive/original-PRDs/01-AdS3-Modular-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Imagine trying to identify a musical instrument from a list of the notes it can produce. Some lists are impossible because the notes do not fit the instrument’s mathematical rules. Others pass those checks, but that still does not tell us how to build the instrument. Here the “instrument” is a quantum theory and the “notes” are its allowed states. We ask whether exceptionally sparse lists of states obey the consistency rules. Such sparse theories are candidates for describing a simplified universe containing gravity with no additional matter.

### Key terms

- **Conformal field theory (CFT)** — A quantum theory whose laws respect transformations that preserve angles, including changes of scale; it describes systems with no preferred length scale.

- **AdS₃** — A three-dimensional spacetime with a particular negative curvature, used as a mathematically tractable setting for gravity.

- **Holomorphic** — Depending on one complex coordinate rather than both it and its complex conjugate; this is a special restriction on the theories studied here.

- **Central charge c** — A number characterizing aspects of a CFT’s symmetry and number of degrees of freedom. Here c=24k, with k a positive integer.

- **Partition function** — A generating function that packages the theory’s state energies or scaling weights and the number of states at each level.

- **Primary and descendant** — A primary is a basic type of excitation; descendants are related excitations generated from it by symmetry.

- **Extremal spectrum** — A spectrum with the largest prescribed initial gap before additional primary excitations, under this particular ansatz.

- **Modular invariance** — The requirement that different equivalent descriptions of a torus give the same partition function.

### Why this matters

Quantum gravity is hard partly because we do not know which mathematically plausible theories are internally consistent. These tests can eliminate candidates without experiments or a complete construction. They also connect gravity to the arithmetic of modular functions: counting quantum states becomes a problem about exact number sequences.

### What progress would mean

A useful first result is a trustworthy table of candidates and any exact reason for rejecting one. Passing the table’s tests would identify a candidate worth further study; constructing the full theory would remain a much stronger achievement.

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

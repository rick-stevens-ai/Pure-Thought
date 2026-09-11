# 24. Topological code verification and noise-specific decoder evaluation

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/24-Topological-Quantum-Error-Correction.md) · [Original description](../archive/original-PRDs/24-Topological-Quantum-Error-Correction.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

Can a toric-code family’s algebraic parameters be proved, and can a fixed decoder’s logical failure probability be evaluated under a precisely specified error model?

## Scope and assumptions

Use an L×L square torus with qubits on edges and L≥3. Start with independent Z errors of probability p and perfect syndrome extraction. Specify the decoder, tie-breaking rule, and whether recovery weights are uniform or likelihood based.

Keep exact code properties separate from statistical decoder performance. Planar boundaries, color codes, measurement noise, and circuit-level noise are distinct extensions.

## Mathematical target

Over F₂ use C₂→∂₂ C₁→∂₁ C₀ with ∂₁∂₂=0, H_X=∂₁, H_Z=∂₂ᵀ. Then n=2L², k=2, and prove d=L using noncontractible primal and dual cycles.

For Z error e and recovery r, success means e+r∈im ∂₂, not merely ∂₁(e+r)=0. A threshold assertion concerns the limit of P_fail(L,p) as L→∞ for this fixed noise/decoder family; a finite crossing is an estimate.

## Required outputs and proof obligations

Export lattice incidence data, ranks and logical bases, distance argument, decoder specification, and failure counts with confidence intervals. Exhaustive small-size evaluation can give an exact failure polynomial in p. Monte Carlo output must retain trials and failure counts, including zero-failure upper intervals.

## Validation and rejection controls

Check n,k,d on several small tori. Verify logical anticommutation via primal/dual intersection. Enumerate all low-weight errors on tiny instances and test the stabilizer-membership success criterion. Validate a matching solution’s optimality separately from logical decoding success.

## Milestones and research extension

Baseline: exact construction and distance proof with a tested decoder. Strong: reproducible finite-size failure curves and uncertainty-aware threshold estimates. Research extension: a rigorous threshold bound or noisy-measurement/circuit model, with no transfer of numerical thresholds across noise conventions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Dennis, Kitaev, Landahl and Preskill, Topological quantum memory](https://arxiv.org/abs/quant-ph/0110143) — Homological coding and model-dependent error-correction thresholds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

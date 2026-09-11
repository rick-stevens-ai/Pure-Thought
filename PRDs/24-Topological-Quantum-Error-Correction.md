# 24. Topological code verification and noise-specific decoder evaluation

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/24-Topological-Quantum-Error-Correction.md) · [Original description](../archive/original-PRDs/24-Topological-Quantum-Error-Correction.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Imagine storing information in loops spread across a lattice so that a small local disturbance usually cannot change the stored value. A topological quantum code uses a quantum version of this idea. Local measurements reveal the endpoints of error patterns, and a decoder tries to join those endpoints correctly. The subtlety is that two repairs can have the same endpoints yet differ by a loop that changes the logical information.

### Key terms

- **Toric code** — A quantum error-correcting code on a lattice with periodic boundaries, topologically equivalent to a torus.

- **Surface code** — A related geometric code with carefully chosen open boundaries.

- **Logical operator** — An operation changing the encoded information while remaining consistent with the code’s checks.

- **Noncontractible loop** — A loop winding around the torus that cannot be shrunk to a point within it.

- **Homology** — A mathematical way of distinguishing cycles modulo boundaries; here it identifies whether a closed error pattern acts logically.

- **Syndrome** — Check outcomes locating the boundary of an error chain.

- **Recovery** — An operation chosen to cancel the error; matching the syndrome is necessary but does not ensure logical success.

- **Minimum-weight matching** — A graph optimization method used in certain decoders to pair defects efficiently.

- **Logical failure probability** — The probability that error and recovery together change encoded information.

- **Threshold** — An asymptotic noise boundary for successful suppression of logical errors in a specified code-and-decoder family.

### Why this matters

Geometric codes connect an intuitive picture of local errors to precise algebraic guarantees. They are important models for protecting quantum information using structured measurements. Understanding logical failure, rather than only syndrome cancellation, is essential for assessing whether increasing the code size actually improves reliability.

### What progress would mean

A useful result would prove the code’s parameters and measure a fixed decoder under a declared noise model. An estimated threshold would remain specific to that model and decoder, not a universal device error tolerance.

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

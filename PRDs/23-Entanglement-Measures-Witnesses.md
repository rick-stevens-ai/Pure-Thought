# 23. Entanglement certification with explicit inconclusive outcomes

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/23-Entanglement-Measures-Witnesses.md) · [Original description](../archive/original-PRDs/23-Entanglement-Measures-Witnesses.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two quantum systems can be related in ways that no mixture of independent local states can reproduce. That is entanglement. Given a mathematical description of a joint state, this problem asks for evidence that it is entangled, evidence that it is separable, or an honest statement that the available tests cannot decide. It also asks how to quantify one particular aspect of entanglement without mixing different measures together.

### Key terms

- **Bipartite system** — A quantum system divided into two identified parts, such as two qubits held by different parties.

- **Density matrix ρ** — An operator describing a quantum state, including statistical mixtures.

- **Separable state** — A mixture of product states of the two parts; correlations in such a decomposition can arise from shared classical information.

- **Entangled state** — A state that has no separable decomposition across the chosen division.

- **Entanglement witness** — An observable nonnegative on every separable state but negative on the state being tested.

- **Partial transpose** — A matrix operation transposing the indices of just one subsystem, useful for testing entanglement.

- **PPT** — Positive partial transpose: a necessary separability condition that is sufficient only in certain small dimensions.

- **Negativity** — A quantity measuring the negative part of the partial-transpose spectrum; zero does not generally imply separability.

- **DPS hierarchy** — Tests based on whether the state can be extended to larger symmetric states; failure can certify entanglement.

- **Concurrence** — A particular entanglement measure with a useful explicit formula for two-qubit states.

### Why this matters

Entanglement is an important resource in many quantum-information protocols, but claiming it from an inconclusive test can invalidate a proposed demonstration. Conversely, explicit witnesses show exactly what establishes its presence. Quantitative bounds are useful because different states may possess very different strengths or kinds of quantum correlations.

### What progress would mean

The deliverable would make its verdict inspectable: a witness, a decomposition, or unresolved status. That would provide a reliable component for analyzing quantum states, rather than promise a universally easy entanglement test.

## Core question

For a finite bipartite density matrix, can one produce a checked entanglement witness or a separable decomposition, and bound a specifically chosen entanglement measure?

## Scope and assumptions

Input subsystem dimensions and an exact rational/algebraic density matrix, or a rigorously described uncertainty set. Begin with 2×2 systems; then include 3×3 PPT-entangled examples. Validate Hermiticity, positivity and unit trace before classification.

The primary decision outputs are entangled, separable, or unresolved. Restrict concurrence and its entanglement-of-formation formula to two qubits. Treat general distillable entanglement and multipartite classification as separate research problems.

## Mathematical target

Compute partial transpose in a fixed tensor-index convention. Negativity is N(ρ)=(||ρ^{T_B}||₁−1)/2. For entanglement give W satisfying 〈a⊗b|W|a⊗b〉≥0 for all product vectors and Tr(Wρ)<0. For separability give ρ=Σ_jp_j|a_jb_j〉〈a_jb_j| with p_j≥0 and Σp_j=1, or another sufficient theorem in its valid dimensions.

PPT/DPS feasibility is generally only a necessary separability test. DPS infeasibility with a validated dual can certify entanglement; passing a finite level generally remains unresolved.

## Required outputs and proof obligations

Export input validation, partial-transpose spectrum enclosures, measure intervals, and a witness/decomposition with an independent checker. Normalize witnesses before optimizing them. For an uncertain input, require the detection margin to hold throughout the uncertainty set.

## Validation and rejection controls

Use Bell states, product states, and ρ_p=p|ψ⁻〉〈ψ⁻|+(1−p)I/4, 0≤p≤1. In this declared convention separability holds for p≤1/3. Verify negativity 1/2 and concurrence 1 for a Bell state. Include a 3×3 PPT-entangled control so zero negativity is not mislabeled separability.

## Milestones and research extension

Baseline: checked two-qubit decisions and measure values. Strong: a validated DPS witness beyond PPT and a separable-decomposition example. Research extension: robust uncertainty-set certification or normalized optimal witnesses on a stated family.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Doherty, Parrilo and Spedalieri, A complete family of separability criteria](https://arxiv.org/abs/quant-ph/0308032) — Symmetric-extension hierarchy and dual entanglement witnesses.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

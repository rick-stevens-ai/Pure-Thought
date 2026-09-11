# 23. Entanglement certification with explicit inconclusive outcomes

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/23-Entanglement-Measures-Witnesses.md) · [Original description](../archive/original-PRDs/23-Entanglement-Measures-Witnesses.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

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

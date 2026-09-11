# 14. Certified Weyl nodes and slice topology in a lattice model

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/14-Topological-Semimetals-Weyl-Dirac.md) · [Original description](../archive/original-PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

Can all nodes in a specified two-band three-dimensional Hamiltonian be isolated, assigned chirality, and related to the Chern numbers of gapped momentum slices?

## Scope and assumptions

Begin with H(k)=sin k_x σ_x+sin k_y σ_y+(2−cos k_x−cos k_y−cos k_z)σ_z, k∈T³, lower band occupied away from nodes. Fix orientation and Pauli-matrix conventions. This exact benchmark has nodes at (0,0,±π/2).

Subsequent searches must fix orbitals, hopping support, parameter box, and symmetry representation. Dirac points and nodal lines are separate extensions because their protection criteria differ.

## Mathematical target

For H=d₀I+d·σ, nodes solve d(k)=0. A simple node has invertible velocity matrix V_ij=∂d_i/∂k_j. Define Weyl chirality χ=sgn det V, and explicitly relate this to the lower-band Chern flux under the chosen convention. Prove the complement of all isolating neighborhoods has |d(k)|>0.

For fixed k_z away from nodes, certify the two-dimensional occupied Chern number. Its jumps track node charge. A slab calculation requires a chosen surface normal and termination; arc details are not determined solely by bulk node positions.

## Required outputs and proof obligations

Return all root enclosures, uniqueness/completeness proof, Jacobian signs, slice Chern numbers, and optional slab spectrum with error control. Nodes need not have closed-form coordinates: certified isolating boxes are acceptable. Robustness is scoped to perturbations that prevent annihilation with opposite total charge.

## Validation and rejection controls

Verify the two benchmark roots analytically and opposite determinant signs. Check total charge on the periodic Brillouin zone vanishes. Confirm slice invariants only on gapped slices. Test a perturbation moving nodes and a parameter change permitting pair annihilation.

## Milestones and research extension

Baseline: complete benchmark node and slice audit. Strong: certified parameter-dependent node motion or one bounded model family. Research extension: a symmetry-protected Dirac model with explicit little-group constraints, or nodal-link invariants with their own definitions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Wan et al., Electronic Structure of Pyrochlore Iridates (published as Topological semimetal and Fermi-arc surface states)](https://arxiv.org/abs/1007.0016) — Reference for Weyl nodes and surface arcs; benchmark and proof obligations here are explicitly specified.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

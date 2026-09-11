# 27. Central configurations in a bounded, symmetry-reduced N-body problem

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/27-N-Body-Central-Configurations.md) · [Original description](../archive/original-PRDs/27-N-Body-Central-Configurations.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For fixed positive masses and a declared symmetry quotient, can all collision-free central configurations in the chosen family be certified and their relative-equilibrium linearization analyzed?

## Scope and assumptions

Start with three labeled positive rational masses in the plane. Fix G=1, center of mass zero, moment of inertia I=Σm_i|r_i|²=1, and quotient by O(2), so mirror images count together. Permute only equal masses when explicitly requested. Extend to a specified four-body symmetry family after validating the complete three-body case.

Central configurations generate homographic motions; general periodic choreographies are a separate problem.

## Mathematical target

Use positive U=Σ_{i<j}m_im_j/|r_i−r_j| and

$$\nabla_{r_i}U=-λm_ir_i,\qquad λ=U/I>0.$$

Introduce distance variables d_ij>0 with d_ij²=|r_i−r_j|², and retain positivity/collision exclusions when polynomializing. Use symmetry charts covering every configuration claimed. Completeness requires certified global domain coverage or an exact elimination/root-count argument, not just isolated numerical solutions.

## Required outputs and proof obligations

Return normalized configurations, distance enclosures, residual identities, existence/uniqueness proofs, equivalence representatives, and completeness scope. For planar relative equilibria analyze the reduced rotating-frame first-order system including Coriolis terms. Distinguish spectral stability from nonlinear stability.

## Validation and rejection controls

For three labeled masses under the stated O(2) quotient, recover three collinear orderings (one per middle body) and one equilateral shape. Check λ=U at I=1. Test collision artifacts introduced by denominator clearing. For stability, reproduce a known three-body criterion using identical mass and reduction conventions.

## Milestones and research extension

Baseline: complete certified three-body catalog. Strong: complete enumeration in one four-body symmetry family or mass instance, with all charts covered. Research extension: a five-body restricted family; no claim to solve the general finiteness problem without an appropriately global proof.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Yu and Zhu, On the finiteness of four-body central configurations](https://arxiv.org/abs/2103.08906) — Context for finiteness versus explicit certified enumeration.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

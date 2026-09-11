# 27. Central configurations in a bounded, symmetry-reduced N-body problem

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/27-N-Body-Central-Configurations.md) · [Original description](../archive/original-PRDs/27-N-Body-Central-Configurations.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Most collections of gravitating bodies change their overall shape as they move. Certain special arrangements can rotate or expand and contract while keeping the same shape. These are central configurations. This problem asks for every such arrangement in a carefully limited case, with a proof that the list is complete and a separate analysis of what small disturbances do to the associated rotating motion.

### Key terms

- **N-body problem** — The problem of describing the motion of N mutually gravitating bodies.

- **Central configuration** — An arrangement in which each body’s gravitational acceleration is proportional to its position relative to the center of mass, with one common proportionality factor.

- **Homographic motion** — Motion preserving the arrangement’s shape while its size and orientation may change.

- **Relative equilibrium** — A motion that is stationary in an appropriate rotating reference frame.

- **Center of mass** — The mass-weighted average position, used here to remove overall translation.

- **Normalization** — A fixed choice of scale, such as setting the moment of inertia to one, so scaled copies are not counted separately.

- **Symmetry quotient** — The convention for treating configurations related by rotations, reflections or specified relabelings as equivalent.

- **Collision-free** — Having strictly positive separation between every pair of bodies.

- **Coriolis term** — A velocity-dependent term arising in a rotating frame, essential to the stability calculation.

- **Spectral stability** — A statement about eigenvalues of the linearized dynamics; it is weaker than a full nonlinear stability proof.

### Why this matters

Central configurations are organizing structures in gravitational dynamics and provide exact reference cases amid otherwise complicated motion. A complete small catalog can test numerical methods and reveal how mass choices and symmetry affect possible motions. The problem also links gravity to exact algebraic root finding.

### What progress would mean

Progress would establish a complete list under a stated mass choice and symmetry convention. It would not classify every periodic orbit: a choreography such as the figure-eight changes shape and belongs to a different question.

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

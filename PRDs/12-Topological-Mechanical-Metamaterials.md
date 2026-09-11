# 12. Rigidity and boundary modes of periodic Maxwell frames

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/12-Topological-Mechanical-Metamaterials.md) · [Original description](../archive/original-PRDs/12-Topological-Mechanical-Metamaterials.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a geometrically specified isostatic frame and boundary termination, can the zero-mode count and localization be proved from its compatibility matrix?

## Scope and assumptions

Use unstressed central-force springs with exact rational or algebraic site coordinates, positive spring constants, and explicit periodic bond offsets. Fix the unit-cell gauge, dimension, masses, and boundary termination. Begin with a small frame and a known periodic Maxwell lattice.

Connectivity alone is insufficient: bond directions determine the compatibility matrix. Distinguish rigid-body motions, floppy modes, and states of self stress. Claims concern linearized ideal mechanics.

## Mathematical target

Let C map displacements to bond extensions and Q=C†. For a finite free frame,

$$n_0−n_{ss}=dN−N_b,\qquad K=C^{†}\operatorname{diag}(k_b)C,$$

with dynamical matrix M⁻¹ᐟ²KM⁻¹ᐟ². For a square periodic C(k), define winding (2πi)⁻¹∮d log det C(k) only along loops where the determinant is nonzero. Account for acoustic translation zeros and the local boundary count when translating bulk winding into an edge index.

## Required outputs and proof obligations

Export positions, bonds, C, exact rank/kernel calculations, self-stress basis, winding certificate, and a boundary-specific mode count. For localization, provide a transfer-matrix/root-modulus bound separated from the unit circle. Specify perturbations preserving the nonvanishing condition and positivity of spring constants.

## Validation and rejection controls

Check one elementary underconstrained and one overconstrained frame against the index equation. Verify K is positive semidefinite and translations are zero modes where appropriate. Compare two terminations of the same bulk. Reject winding calculations crossing a zero of det C.

## Milestones and research extension

Baseline: exact Maxwell–Calladine counts on finite frames. Strong: one periodic frame with certified winding and termination-dependent edge localization. Research extension: a bounded family with robust mode counts. Elastic beams, prestress, nonlinear stability, and manufacturing tolerance require separate models.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Kane and Lubensky, Topological Boundary Modes in Isostatic Lattices](https://arxiv.org/abs/1308.0554) — Compatibility matrices, topological indices, and boundary modes.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

# 12. Rigidity and boundary modes of periodic Maxwell frames

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/12-Topological-Mechanical-Metamaterials.md) · [Original description](../archive/original-PRDs/12-Topological-Mechanical-Metamaterials.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A framework of bars or springs can be rigid in one place and easy to deform in another. Sometimes a carefully arranged repeating framework has soft motions concentrated at its boundary. The question is whether we can predict and prove those motions from the exact geometry, rather than simply observing them in a simulation. Counting connections helps, but the direction of each connection matters too.

### Key terms

- **Frame** — A network of sites joined by bars or springs, used as an ideal mechanical model.

- **Degree of freedom** — An independently variable coordinate of the system’s motion.

- **Isostatic or Maxwell frame** — A frame balanced between degrees of freedom and constraints in the relevant counting sense; details of zero modes and self stress still matter.

- **Zero mode** — An infinitesimal displacement requiring no restoring energy in the linearized model.

- **State of self stress** — A pattern of internal bond tensions that balances forces without external loads.

- **Compatibility matrix C** — The linear map from small site displacements to bond extensions.

- **Stiffness matrix K** — The operator relating displacements to restoring forces in the linear approximation.

- **Winding number** — An integer counting how a complex quantity winds around zero along a closed path; here it helps characterize the periodic frame.

- **Termination** — The precise way a periodic structure is cut to make a boundary.

### Why this matters

Geometry-controlled softness could help explain and design structures with directional compliance, boundary motion or localized mechanical response. The scientific value is understanding which behaviors follow from a robust index and which are accidental features of one geometry. That distinction matters before translating an ideal spring model into a physical structure.

### What progress would mean

A useful result would give a frame, an exact count of its allowed soft motions and a proof of where they concentrate. Material failure, nonlinear deformation and manufacturing tolerance would remain additional engineering questions.

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

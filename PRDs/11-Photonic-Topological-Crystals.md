# 11. Certified topological bands in an idealized Maxwell medium

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/11-Photonic-Topological-Crystals.md) · [Original description](../archive/original-PRDs/11-Photonic-Topological-Crystals.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

Can a specified lossless periodic electromagnetic model support a rigorously isolated band group with nonzero Chern number, and can its interface modes be validated?

## Scope and assumptions

Choose a two-dimensional periodic, lossless, frequency-independent constitutive model with Hermitian positive ε(r), μ(r), specified boundary conditions, and a mechanism breaking physical time reversal. Treat the constitutive tensors as mathematical inputs. Real-frequency dispersion, loss, and fabrication feasibility are extensions requiring additional physical data.

Start by validating a reciprocal scalar dielectric benchmark with zero total Chern number for a time-reversal-invariant band group; then introduce an explicitly nonreciprocal tensor.

## Mathematical target

In a formulation supporting the chosen model, solve

$$(∇+ik)×μ^{-1}(r)(∇+ik)×u=ω²ε(r)u,$$

in units with c=1, subject to the appropriate transverse constraint. Use the ε-weighted inner product or an equivalent Hermitian formulation. Specify an isolated positive-frequency band group; photons do not require an electron-like filling assumption. Prove a frequency gap over the entire Brillouin zone and certify the projector invariant in the correct metric.

## Required outputs and proof obligations

Return ideal geometry and constitutive tensors, solver discretization, eigenvalue enclosures, spurious-mode controls, gap and topology certificates. Interface calculations must state both media and termination. Report robustness only for an explicit perturbation class and bound that preserves the required gap/symmetry.

## Validation and rejection controls

Check homogeneous-medium dispersion, mesh/basis convergence, Hermiticity, and transverse modes. Separate Galerkin error from k-space interpolation error. Verify reciprocal controls cannot acquire a net Chern number from numerical artifacts. Compare an interface spectrum with the bulk-index difference without claiming immunity to arbitrary disorder.

## Milestones and research extension

Baseline: validated reciprocal eigenproblem. Strong: one certified ideal nonreciprocal topological design. Research extension: a realistic dispersive constitutive model or gap optimization within a bounded geometry family. Fabrication drawings alone do not certify material behavior.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Haldane and Raghu, Directional optical waveguides with broken time-reversal symmetry](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.013904) — Nonreciprocal media as the mechanism for photonic quantum-Hall-like edge modes.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

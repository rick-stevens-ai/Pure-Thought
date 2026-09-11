# 11. Certified topological bands in an idealized Maxwell medium

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/11-Photonic-Topological-Crystals.md) · [Original description](../archive/original-PRDs/11-Photonic-Topological-Crystals.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A repeating optical structure changes which light waves can travel through it. With suitable nonreciprocal response, it may also support light that travels along an interface in a preferred direction. This problem asks whether a precisely defined ideal electromagnetic medium really has the required band structure and topology. It separates a mathematically guaranteed wave effect from the later task of finding a material that realizes the assumed response.

### Key terms

- **Maxwell equations** — The equations governing classical electric and magnetic fields.

- **Periodic medium** — A medium whose properties repeat from cell to cell.

- **Permittivity ε and permeability μ** — Quantities describing how a medium responds to electric and magnetic fields; they can be direction-dependent tensors.

- **Nonreciprocity** — Response that breaks the usual interchange symmetry between source and receiver; here it provides the required time-reversal-breaking mechanism.

- **Photonic band** — An allowed family of electromagnetic wave frequencies across crystal momenta.

- **Transverse constraint** — The condition selecting physical wave fields consistent with the relevant divergence equation.

- **Chern number** — An integer characterizing the topology of an isolated group of wave bands.

- **Interface mode** — A wave concentrated near the boundary between two media.

- **Lossless and nondispersive** — Idealizations in which energy is not absorbed and the constitutive response is taken independent of frequency.

### Why this matters

Controlling where and in which direction waves travel is a basic aim of optical engineering. Topology offers one mechanism for robust interface behavior under appropriate conditions. A certified Maxwell calculation would also expose whether an apparent effect is caused by numerical artifacts or by assumptions a real material cannot satisfy.

### What progress would mean

The first meaningful achievement would be a validated ideal medium with a proved gap and topological index. It could inform later waveguide designs, but performance in a lossy, frequency-dependent device would require a separate physical model.

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

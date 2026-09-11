# 20. Path-integral thermodynamics with a transparent error budget

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/20-Ab-Initio-Path-Integrals.md) · [Original description](../archive/original-PRDs/20-Ab-Initio-Path-Integrals.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a specified one-dimensional quantum Hamiltonian, can path-integral sampling reproduce equilibrium energy and position moments with separately assessed discretization and statistical errors?

## Scope and assumptions

Start with H=p²/(2m)+V(q), m>0, β>0, and a confining harmonic potential; extend to an explicitly defined anharmonic potential. Nuclei are distinguishable Boltzmann particles. Report only equilibrium observables initially.

An ab initio extension must specify the Born–Oppenheimer surface, electronic method, basis, and force errors. It is not exact quantum chemistry merely because no empirical force field is used. Real-time rates and exchange statistics require distinct methods.

## Mathematical target

Set β_P=β/P and ω_P=P/(βℏ). Sample the coordinate density proportional to

$$\exp\left[-β_P\sum_{s=1}^P\left\{\tfrac12mω_P²(q_s-q_{s+1})²+V(q_s)\right\}\right],\quad q_{P+1}=q_1.$$

Coordinate observables use P⁻¹Σ_s A(q_s), generally not A(q_centroid). For one dimension the primitive energy estimator is P/(2β)−mP/(2β²ℏ²)Σ_s(q_s−q_{s+1})²+P⁻¹Σ_sV(q_s). Derive the chosen estimator from the partition function to fix all temperature factors.

## Required outputs and proof obligations

Return algorithm, seeds, temperatures, bead counts, time-step settings, estimators, and an error ledger separating sampling, integration, finite-P, and potential/model errors. Use “statistical interval” for sampling uncertainty. Reserve “rigorous enclosure” for cases with proved sampling/discretization/tail bounds.

## Validation and rejection controls

Recover E=(ℏω/2)coth(βℏω/2) and 〈q²〉=ℏcoth(βℏω/2)/(2mω) for the oscillator over a declared β range. Compare multiple P and time steps, independent chains, and autocorrelation diagnostics. Validate anharmonic results against a separately converged or certified spectral method. A convergence fit alone cannot certify its asymptote.

## Milestones and research extension

Baseline: oscillator thermodynamics within predeclared statistical and bias tolerances. Strong: one anharmonic error-controlled result. Research extension: a small ab initio system with a full error budget; rigorous certification requires more than SCF convergence and stable trajectories.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Ceriotti et al., Efficient stochastic thermostatting of path integral molecular dynamics](https://arxiv.org/abs/1009.1045) — Normal-mode sampling and stochastic thermostat methods.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

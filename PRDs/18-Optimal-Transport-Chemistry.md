# 18. Certified optimal-transport bounds for strictly correlated electrons

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/18-Optimal-Transport-Chemistry.md) · [Original description](../archive/original-PRDs/18-Optimal-Transport-Chemistry.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a supplied normalized electron density, can a discretized Coulomb multi-marginal transport problem yield verified bounds, with continuum error separated from optimization error?

## Scope and assumptions

Take an electron density ρ≥0 with ∫ρ=N and marginal μ=ρ/N. Begin with N=2 and a compactly supported mathematical density or a finite rational discrete model. Specify treatment of the diagonal Coulomb singularity. A continuous-density result additionally needs domain-tail and discretization estimates.

The main target is the strictly correlated interaction functional. Wasserstein comparison of densities is a separate optional metric task; it is not a minimum-energy chemical reaction path.

## Mathematical target

Define

$$V_{SCE}[ρ]=\inf_{Γ\in\Pi(μ,\ldots,μ)}\int\sum_{i<j}|r_i-r_j|^{-1}\,dΓ.$$

Γ is a probability measure and may be symmetrized; it is not an antisymmetric probability distribution. The symmetric dual uses u with Σ_i u(r_i)≤Σ_{i<j}|r_i−r_j|⁻¹ and objective ∫uρ. A feasible coupling gives an upper bound and a dual-feasible potential gives a lower bound for the same problem. Define W_∞=V_SCE−U_H if subtracting Hartree energy; do not call V_SCE exact exchange.

## Required outputs and proof obligations

Return marginal conventions, cost and singularity policy, primal plan, dual potentials, rigorously checked constraints, and L≤V≤U. State whether V is a finite discrete optimum or the continuum functional. Sinkhorn candidates must be converted to feasible bounds for the unregularized problem, with entropic bias controlled.

## Validation and rejection controls

Verify a small rational-cost transport LP exactly and a one-dimensional quadratic-cost quantile example as a separate solver test. For Coulomb densities, use a case with a known co-motion construction under its actual hypotheses. Reject comparing normalized densities with unequal total mass unless normalization or an unbalanced-OT model is explicitly chosen.

## Milestones and research extension

Baseline: certified two-marginal discrete bounds. Strong: one continuum enclosure with singularity/tail control. Research extension: N>2 with symmetry reduction and validated bounds, or a demonstrably improved relaxation. Do not promise deterministic Monge maps for arbitrary multimarginal inputs.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Cotar, Friesecke and Klüppelberg, Density functional theory and optimal transportation with Coulomb cost](https://arxiv.org/abs/1104.0603) — Strong-correlation/semiclassical connection; theorem hypotheses must be checked for each density.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

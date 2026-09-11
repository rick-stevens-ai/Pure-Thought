# 26. A posteriori certification of an invariant Hamiltonian torus

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/26-KAM-Theory-Planetary-Stability.md) · [Original description](../archive/original-PRDs/26-KAM-Theory-Planetary-Stability.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For an explicitly specified analytic near-integrable Hamiltonian and an approximate torus, can every hypothesis of a quantitative KAM theorem be verified?

## Scope and assumptions

Begin with H(I,θ)=ω·I+|I|²/2+ε[cos θ₁+cos(θ₁−θ₂)] in two degrees of freedom, ω=(1,(1+√5)/2), on a stated complex neighborhood of a real action box. Choose a rational ε only after validating the unperturbed case. Seek an embedding with rotation vector ω.

This is a mathematical model, not a solar-system stability assertion. Planetary extensions require an explicit canonical reduction, handling Kepler degeneracy, collision exclusions, and input uncertainty.

## Mathematical target

For an approximate embedding K:T²→phase space define E(θ)=X_H(K(θ))−DK(θ)ω. Select a published a posteriori KAM theorem and reproduce its hypotheses and computable constants: analytic widths, nondegeneracy/twist inverse, residual norm, and small-divisor bounds.

Prove |k·ω|≥γ/|k|₁^τ for every nonzero integer k using arithmetic properties of the exact chosen ω. Finite checks alone are insufficient. Evaluate the theorem’s smallness inequalities with outward-rounded intervals and certify closeness of a true K_* to K.

## Required outputs and proof obligations

Export H, ε, domain, exact frequency representation, Fourier coefficients, all theorem constants, interval inequalities, and a checker. State the conclusion: existence of a nearby quasiperiodic invariant torus with a quantified embedding error. A failed sufficient condition yields “not certified,” not “unstable.”

## Validation and rejection controls

Verify E=0 for K₀(θ)=(I=0,θ) at ε=0. Derive an all-integer Diophantine bound for the golden-ratio frequency. Check Fourier tails and loss of analytic width. Test a resonant frequency where the stated theorem must refuse certification.

## Milestones and research extension

Baseline: unperturbed certificate and complete arithmetic/theorem setup. Strong: a nonzero-ε torus certificate. Research extension: a larger perturbation range or a reduced planetary model. One invariant torus does not certify a neighborhood of arbitrary observed initial conditions or a phase-space measure bound.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Valvo and Locatelli, Hamiltonian Control of Magnetic Field Lines: Computer Assisted Results Proving the Existence of KAM Barriers](https://arxiv.org/abs/2101.07785) — Example of a complete computer-assisted Hamiltonian KAM workflow; use a theorem matching the chosen formulation.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

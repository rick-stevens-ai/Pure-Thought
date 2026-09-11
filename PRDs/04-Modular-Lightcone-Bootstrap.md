# 04. Lightcone bootstrap with controlled large-spin remainders

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/04-Modular-Lightcone-Bootstrap.md) · [Original description](../archive/original-PRDs/04-Modular-Lightcone-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For one scalar four-point function in a unitary CFT with d>2, what controlled bounds on large-spin double-twist data follow from specified low-twist exchanges?

## Scope and assumptions

Start in d=3 with an identical scalar φ of fixed dimension Δφ. Specify the normalization of C_T, the stress-tensor OPE coefficient fixed by its Ward identity, and an explicit twist gap above the low-twist exchanges retained. Use a large-C_T expansion only when a factorization assumption and its error order are stated.

This challenge studies analytic large-spin behavior. Challenge 07 instead studies the spinning stress-tensor correlator. Torus modular invariance is restricted to a separate d=2 variant.

## Mathematical target

For a fixed scalar block convention, define F_{Δ,ℓ}=v^{Δφ}g_{Δ,ℓ}(u,v)−u^{Δφ}g_{Δ,ℓ}(v,u), and use F_id+Σ λ²F=0. Twist is τ=Δ−ℓ. The double-twist families have Δ_{n,ℓ}=2Δφ+2n+ℓ+γ_{n,ℓ} asymptotically.

Choose n=0 initially and derive a retained contribution γ̂_{0,ℓ}, together with a bound |γ_{0,ℓ}−γ̂_{0,ℓ}|≤R(ℓ) for ℓ≥ℓ₀ under explicit spectral/Regge assumptions. If only an asymptotic series is obtained, label it as such and do not manufacture a finite-spin bound.

## Required outputs and proof obligations

Export block normalizations, low-twist input, the derivation or inversion formula, coefficients, and a decomposition of R into omitted exchanges, expansion terms, and numerical errors. State clearly whether the result is an identity, a conditional bound, or an asymptotic estimate.

## Validation and rejection controls

Use generalized-free-field crossing as an algebraic benchmark, while noting that it is not by itself the target local CFT with a finite stress tensor. Recover double-twist accumulation in a known analytic example. Verify every uniform estimate on the specified cross-ratio domain and all spins covered by the claim.

## Milestones and research extension

Baseline: reproduce an established leading large-spin result with conventions checked. Strong: obtain one explicit remainder estimate, or identify the extra data needed to obtain it. Research extension: a tighter conditional bound on low-twist exchanges or bulk couplings with a stated holographic dictionary.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Fitzpatrick et al., The Analytic Bootstrap and AdS Superhorizon Locality](https://arxiv.org/abs/1212.3616) — Large-spin double-twist structure and analytic bootstrap.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

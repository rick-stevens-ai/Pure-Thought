# 02. Conditional positivity bounds for a specified gravitational amplitude

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/02-Gravitational-Positivity-Bounds.md) · [Original description](../archive/original-PRDs/02-Gravitational-Positivity-Bounds.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

Under a fully stated infrared prescription and high-energy assumption, what certified inequality constrains one selected low-energy coefficient of scalar scattering coupled to gravity?

## Scope and assumptions

Use four-dimensional scattering of one identical real massive scalar coupled to Einstein gravity. This replaces the redundant schematic pure-gravity R² coefficient list with an amplitude-defined observable. Set m=1 as the mass unit; specify κ, the EFT subtraction scale, perturbative order, and whether the target is a tree coefficient or a renormalized coefficient. Graviton helicity amplitudes are a later extension.

The first task is to establish a valid sum rule. Do not run an optimization until the pole subtraction, massless cuts, crossed channel, and large-contour contribution have been derived.

## Mathematical target

Let ν=s−2m²+t/2, so s↔u sends ν↦−ν. Define B(ν,t) by subtracting the explicitly calculated light poles and the chosen low-energy cut contributions from A(ν,t). Define b₂(t)=(1/2)∂²_νB(0,t).

Derive b₂(t)=I_UV(t)+R_∞(t) with all kernels and subtractions written out. Choose either a finite-t smeared positive sum rule or a justified regulated forward limit. Nonnegativity must be established for the resulting partial-wave kernel; Im A(s,t)≥0 at arbitrary t<0 is not an admissible substitute. State a bound on R_∞ or retain it as an explicit unknown.

## Required outputs and proof obligations

Return a derivation dossier, normalized coefficient definition, assumptions, and either a conditional inequality b₂≥L or an explanation of the missing hypothesis. If optimizing, export a dual functional with certified kernel positivity and all truncation errors. A surviving coefficient is “not excluded by these tests,” not “UV complete.”

## Validation and rejection controls

Check s↔u symmetry, dimensional consistency, and pole residues. Reproduce the nongravitational forward positivity argument in the κ→0 limit. Use a fully specified example amplitude to test the subtraction algebra. Perturb a certificate beyond its margin and verify rejection.

## Milestones and research extension

Baseline: one correct sum rule and its gravity-free limit. Strong: a numerically useful certified conditional bound with a finite positive margin. Research extension: improve the bound or relax a high-energy assumption; label any sharpness claim with both matching constructions and upper/lower evidence.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Tokuda, Aoki and Hirano, Gravitational positivity bounds](https://arxiv.org/abs/2007.15009) — Regge assumptions and finite gravitational corrections.

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Why naive pole-subtracted forward positivity can fail.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

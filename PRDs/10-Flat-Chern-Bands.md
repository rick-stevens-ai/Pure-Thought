# 10. Flatness–locality–geometry tradeoffs for Chern bands

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/10-Flat-Chern-Bands.md) · [Original description](../archive/original-PRDs/10-Flat-Chern-Bands.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For fixed orbital count, hopping range, and gap normalization, how small can a Chern band’s bandwidth and geometric nonuniformity be, with certified bounds?

## Scope and assumptions

Choose a two-dimensional isolated rank-one band, target C=1, and a compact coefficient family with range R and fixed orbital positions. Initially use a specified two-band family extending Challenge 09. Impose a positive gap floor and a norm bound to prevent trivial energy rescaling.

Maintain two distinct tracks: finite-range approximate flatness, and exact spectral flattening with generally non-finite-range hopping. Do not require exact nonzero-Chern flatness with strictly finite-range hopping under the no-go theorem’s hypotheses.

## Mathematical target

Define W=max_k E_n−min_k E_n and Δ=min_k min_{j≠n}|E_j−E_n|. Optimize W/Δ. For the fixed Brillouin-zone coordinates define g_ij=Re〈∂_iu|(1−P)|∂_ju〉 and F with the same Chern convention as Challenge 09. Then C=(1/2π)∫F and F̄=2πC/Area(BZ).

Report separately Var(F)=Area⁻¹∫(F−F̄)² and D_g=Area⁻¹∫(tr g−|F|). These are diagnostics under a specified Euclidean coordinate metric, not a universal fractional-Chern-insulator stability score.

## Required outputs and proof obligations

Export coefficient/domain constraints, W upper and Δ lower bounds, certified C, geometric enclosures, and optimization gaps. An achieved objective gives an upper bound on the minimum; a global lower bound needs a covering/relaxation proof. For infinite-range flattening, give decay and truncation estimates.

## Validation and rejection controls

Verify normalization ∫F=2πC and the pointwise metric inequality in the chosen convention. Check that truncating a spectrally flattened model generally reintroduces bandwidth. Include a trivial exactly flat band as a negative topology test. Compare candidate objectives using identical units, embedding, and norm constraints.

## Milestones and research extension

Baseline: one certified near-flat Chern model and a correct locality audit. Strong: global objective bounds on one compact family or a rigorous truncation tradeoff. Research extension: optimal geometry within a specified class; many-body FCI stability requires an added interacting Hamiltonian and filling.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Chen et al., Flat Chern-band locality obstruction](https://arxiv.org/abs/1311.4956) — No-go theorem for simultaneous exact flatness, nonzero Chern number, and strictly local hopping.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

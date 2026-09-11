# 10. Flatness–locality–geometry tradeoffs for Chern bands

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/10-Flat-Chern-Bands.md) · [Original description](../archive/original-PRDs/10-Flat-Chern-Bands.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

In an ordinary energy band, particles can have quite different energies depending on how they move. A nearly flat band reduces those energy differences, potentially making interactions between particles more influential. We want that band to also have a nontrivial topological structure. The problem asks how well these goals can be combined when particles are allowed to hop only a limited distance and the model has a limited number of orbitals.

### Key terms

- **Flat band** — A band whose energy is constant across momentum space; a nearly flat band has a small but nonzero variation.

- **Bandwidth W** — The difference between the largest and smallest energy in the selected band.

- **Gap Δ** — The minimum energy separation between that band and other bands.

- **Flatness ratio W/Δ** — A way to compare bandwidth to isolation from other bands; smaller values mean a flatter, well-separated band.

- **Hopping range** — The greatest lattice distance over which the model allows direct hopping.

- **Chern band** — A band with a nonzero Chern number, indicating nontrivial global topology.

- **Berry curvature** — A local measure of how quantum states twist as momentum changes; its integral determines the Chern number in the stated convention.

- **Quantum metric** — A measure of how distinguishable nearby momentum-dependent quantum states are.

- **Spectral flattening** — A mathematical change to band energies that preserves the chosen states but generally changes the spatial range of hopping.

### Why this matters

Flat topological bands are useful settings for studying collective quantum behavior driven by interactions. But there are tradeoffs: restricting the model’s spatial range changes which flatness properties are possible. Quantifying those tradeoffs can show whether a proposed improvement is attainable or requires a more complicated model.

### What progress would mean

Progress would provide certified examples and limits within a defined design family. Favorable single-particle geometry would make a model interesting to investigate further; it would not by itself prove an interacting fractional topological phase.

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

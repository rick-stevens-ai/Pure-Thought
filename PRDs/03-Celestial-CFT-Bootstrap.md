# 03. Celestial amplitudes: distributional crossing and soft-limit consistency

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/03-Celestial-CFT-Bootstrap.md) · [Original description](../archive/original-PRDs/03-Celestial-CFT-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Instead of describing a collision by particle energies and momenta, imagine recording the directions in which particles enter or leave on a sphere surrounding the experiment. Celestial methods reorganize scattering data into quantities associated with that sphere. The ambition is a new language for gravity. Before using that language to discover new theories, this problem checks that it faithfully preserves familiar facts about a simple graviton collision.

### Key terms

- **Celestial sphere** — The sphere of directions seen from a point, used here to label asymptotic particle directions rather than a physical shell.

- **Graviton** — The hypothetical quantum excitation of the gravitational field used in perturbative quantum-gravity calculations.

- **Helicity** — The component of a particle’s spin along its direction of motion.

- **Mellin transform** — A mathematical transformation that reorganizes dependence on energy into dependence on scaling behavior.

- **Conformal weight Δ** — A label describing scaling behavior in the celestial description; it can be complex.

- **Distribution** — A generalized function, such as a delta function, understood through integration against test functions rather than ordinary pointwise values.

- **Crossing** — The relationship between different scattering processes obtained by reinterpreting an incoming particle as an outgoing antiparticle, with appropriate continuation.

- **Soft limit** — The limit in which one particle’s energy approaches zero; a soft theorem controls the resulting leading behavior.

- **Regulator** — A temporary modification that makes a singular expression well defined, accompanied by a rule for removing or interpreting it.

### Why this matters

A successful reformulation could expose symmetries or constraints that are difficult to see in ordinary scattering variables. That would give theorists another way to investigate gravity. But a change of mathematical language is useful only if singularities, conservation laws and limiting cases survive the translation correctly.

### What progress would mean

A verified example would supply a dependable foundation for later celestial calculations. It would establish that specific identities translate correctly, rather than establish a complete two-dimensional theory of the universe.

## Core question

Can a regulated celestial transform of a specified four-graviton tree amplitude reproduce Lorentz covariance, crossing, and the leading conformally soft residue in one consistent convention?

## Scope and assumptions

Fix four-dimensional Einstein gravity at tree level, a nonzero four-graviton helicity amplitude, incoming/outgoing signs εᵢ, and pᵢ=εᵢωᵢq(zᵢ,z̄ᵢ). Declare metric, helicity, normalization, and analytic-continuation conventions. The target is an identity pipeline for amplitudes, not a classification of celestial CFTs.

Use Δᵢ=1+iλᵢ on the principal series where appropriate and hᵢ=(Δᵢ+Jᵢ)/2, h̄ᵢ=(Δᵢ−Jᵢ)/2. Identify explicitly which continuations away from that contour are used for soft residues.

## Mathematical target

Define the transformed distribution by

$$\widetilde{\mathcal A}=\prod_i\int_0^∞dω_i\,ω_i^{Δ_i-1}\;δ^{(4)}\!\left(\sum_i ε_iω_iq_i\right)\mathcal M_4.$$

State a regulator or test-function space making the integral meaningful, and specify regulator removal or meromorphic continuation. From a leading momentum-space term proportional to ω_s⁻¹, derive the Mellin pole at Δ_s=1 in this convention. Subleading residues require their own derivation. Crossing must transform helicities, signs, and support, not just permute ordinary Euclidean functions.

## Required outputs and proof obligations

Produce exact momentum-space expressions, regulated transform formulas, support conditions, and symbolic covariance/crossing/soft identities. If a transform cannot be defined under the chosen assumptions, return a precise obstruction at that step. A positive bootstrap is an extension only after deriving an applicable inner product and positivity statement.

## Validation and rejection controls

Check the original amplitude’s factorization first. Test Mellin formulas against analytic one-variable examples in their convergence strips. Verify soft residues before and after transformation under stated limit-interchange hypotheses. Use smeared distributions for numerical checks; never evaluate a delta distribution as an ordinary function.

## Milestones and research extension

Baseline: one four-point transform with explicit support and checked leading soft residue. Strong: a second helicity/process example and a verified crossing relation. Research extension: derive a justified positive constraint in a restricted subsector before attempting an SDP island.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Adamo et al., Celestial amplitudes and conformal soft theorems](https://www.pure.ed.ac.uk/ws/files/121747399/Adamo_2019_Class._Quantum_Grav._36_205018_1_.pdf) — Conformally soft graviton operators at Δ=1 and Δ=0.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

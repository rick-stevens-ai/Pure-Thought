# 06. S-matrix bootstrap with an explicit gravitational infrared prescription

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/06-Nonperturbative-S-matrix-Bootstrap.md) · [Original description](../archive/original-PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Suppose we do not know a theory’s microscopic machinery, but do know that collision probabilities must be sensible and different descriptions of the same process must agree. Can those rules alone rule out a proposed interaction strength? This problem searches the space of scattering possibilities rather than choosing one detailed microscopic theory. Because gravity acts over long distances and allows very low-energy radiation, defining the scattering question correctly is itself part of the work.

### Key terms

- **S-matrix** — The object relating incoming quantum states to outgoing states in a scattering experiment.

- **Partial wave** — A contribution to scattering organized by angular momentum, which makes some probability constraints easier to state.

- **Unitarity** — Conservation of total probability; an allowed scattering process cannot produce a total probability greater than one.

- **Elastic scattering** — Scattering that stays within the chosen particle channel without producing additional particles or excitations.

- **Crossing symmetry** — Relations between amplitudes for processes with differently assigned incoming and outgoing particles.

- **Infrared prescription** — A stated way to handle low-energy radiation and long-distance singularities.

- **Outer relaxation** — A larger, easier-to-test set containing all valid possibilities under the assumptions. Excluding a point from it can exclude the original possibility.

- **Inner ansatz** — A selected subset of candidate amplitudes; failing to find a solution there need not exclude other amplitudes.

- **Dual certificate** — An independently checkable mathematical witness establishing a bound or inconsistency.

### Why this matters

The method could identify restrictions shared by many possible microscopic theories. That is useful when no single theory is established. Its reliability depends on not mistaking a limitation of the search formula, grid or cutoff for a limitation imposed by physics.

### What progress would mean

A certified exclusion would eliminate a parameter choice under the stated scattering assumptions. The gravity-free reference case first checks the machinery; adding a defensible treatment of gravity would be the important next step.

## Core question

Can one certify an exclusion for a low-energy scalar-scattering parameter using a crossing-symmetric partial-wave relaxation whose infrared and truncation assumptions are explicit?

## Scope and assumptions

Begin with identical massive scalars in four dimensions and no gravity to validate the machinery. The gravitational stage must first define an infrared-finite observable or a justified regulator/pole treatment, including massless radiation. Keep one scalar mass and one amplitude coefficient fixed; optimize a second coefficient in a declared normalization.

This is complementary to Challenge 02: the focus is amplitude feasibility and partial-wave constraints rather than deriving a low-energy positivity sum rule. A nongravitational baseline does not count as solving the gravitational stage.

## Mathematical target

Choose A=16πΣ_{ℓ≥0}(2ℓ+1)a_ℓ(s)P_ℓ(cosθ), s+t+u=4m², ρ=√(1−4m²/s), and S_ℓ=1+2iρa_ℓ. Then |S_ℓ|≤1 is equivalent to Im a_ℓ≥ρ|a_ℓ|². Equality needs an actually elastic sector.

State how crossing is imposed and distinguish an outer relaxation of necessary constraints from an inner finite amplitude ansatz. A dual exclusion from an outer relaxation transfers to the full problem only when every physical amplitude maps into that relaxation. Bound energy, angular-momentum, and basis tails; checking a finite mesh is insufficient.

## Required outputs and proof obligations

Export the amplitude conventions, IR prescription, finite optimization problem, mapping from the continuum problem, and dual certificate with its positivity/error margins. Feasible ansatz points are examples satisfying the verified constraints only. If the IR construction or tail control fails, state which conclusion remains unavailable.

## Validation and rejection controls

Check free scattering and a specified analytic nongravitational benchmark. Verify the unitarity disk algebra independently. Test between grid points and beyond cutoffs using proved bounds. Demonstrate that known admissible examples are not falsely excluded. Increasing precision is a diagnostic; rigorous enclosures are needed for certification.

## Milestones and research extension

Baseline: certified finite nongravitational relaxation. Strong: justify an IR treatment and certify one conditional gravitational exclusion. Research extension: improved continuum bounds or a constructive amplitude with a precisely delimited consistency claim.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Infrared and analyticity issues that must be resolved before importing standard bounds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

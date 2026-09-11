# 13. Higher-order topology with explicit boundary and symmetry assumptions

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/13-Higher-Order-Topological-Insulators.md) · [Original description](../archive/original-PRDs/13-Higher-Order-Topological-Insulators.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A topological material can have interesting boundary behavior even when its broad edges do not conduct. In certain models, charge accumulates in a quantized way near corners instead. This problem asks how to establish that behavior carefully: what property belongs to the bulk, what depends on the way the sample is cut, and what extra symmetry is needed to place a corner state at exactly zero energy?

### Key terms

- **Higher-order topology** — Topology associated with lower-dimensional boundaries, such as corners of a two-dimensional sample or hinges of a three-dimensional sample.

- **Quadrupole** — A pattern of electric charge organization beyond total charge and ordinary dipole polarization; the selected model supports a quantized version under particular assumptions.

- **Bulk** — The interior of a system, far from boundaries.

- **Wilson loop** — A matrix describing how a chosen set of quantum states changes when transported around a closed path in momentum space.

- **Wannier sector** — A selected group of Wilson-loop eigenstates, separated from the others by a gap in that loop’s spectrum.

- **Nested polarization** — A polarization calculated within such a Wannier sector, using a second transport calculation.

- **Corner excess charge** — Charge near a corner measured relative to a specified background, after accounting for other contributions.

- **Chiral symmetry** — An additional spectral symmetry that can relate positive and negative energies; it can matter for pinning special states to zero.

- **Termination** — The actual boundary cut and local boundary structure of the sample.

### Why this matters

This problem broadens our understanding of how a material’s interior organizes behavior at its boundaries. It also corrects a common interpretive shortcut: a quantized corner charge and a zero-energy corner state are not the same observation. Reliable distinctions are needed before such features can be used as design principles.

### What progress would mean

Progress would produce a model with verified gaps, a well-defined bulk invariant and a carefully measured boundary consequence. It would establish a controlled example, rather than guarantee protected corner states for every sample shape or boundary.

## Core question

For a fixed quadrupole-insulator family, can one certify the bulk and Wannier gaps, a quantized nested polarization, and the corresponding boundary charge under a specified termination?

## Scope and assumptions

Start with a four-band Benalcazar–Bernevig–Hughes-type square-lattice model at half filling. Supply all hopping signs, orbital positions, mirror representations, and a rectangular termination. State whether chiral symmetry is imposed. Restrict parameters to a compact region away from bulk and Wannier-gap closings.

Separate fractional corner charge from an exactly zero-energy corner eigenstate. The latter requires additional spectral symmetry and boundary assumptions.

## Mathematical target

Let P(k) be the rank-two occupied projector. Build the path-ordered Wilson loop W_x(k_y) from occupied-state overlaps. If its spectrum has a certified separation into Wannier sectors, form a smooth sector projector and its Berry holonomy along y. Define the nested polarization from that holonomy modulo one.

Do not replace a nested Wilson loop by the integral of the first Wilson-loop eigenphases. Define corner excess charge relative to an explicit ionic/reference background and a stated spatial window, with finite-size and edge contributions controlled.

## Required outputs and proof obligations

Return H, symmetry operators and identities, bulk and Wannier-gap enclosures, nested invariant, boundary charge definition, and finite-size bounds. Any robustness statement must preserve the protecting symmetry and relevant gaps. State whether conclusions are intrinsic bulk statements or termination-dependent observables.

## Validation and rejection controls

Recover topological and trivial limits of the selected model. Verify zero total Chern number where required while the nested invariant is nontrivial. Break a protecting symmetry as a control; the checker must stop claiming quantization. Compare two terminations and explain any boundary difference.

## Milestones and research extension

Baseline: reproduce one quadrupole benchmark with valid Wilson-loop construction. Strong: certify its invariant and boundary charge on a parameter box. Research extension: classify a bounded family or one additional symmetry setting; no universal HOTI classification is implied.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Benalcazar, Bernevig and Hughes, Quantized Electric Multipole Insulators](https://arxiv.org/abs/1611.07987) — Nested Wilson loops and assumptions for quantized multipole response.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

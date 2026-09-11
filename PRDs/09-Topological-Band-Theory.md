# 09. Certified Chern phases in a bounded tight-binding family

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/09-Topological-Band-Theory.md) · [Original description](../archive/original-PRDs/09-Topological-Band-Theory.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two solids can both resist ordinary electrical conduction in their interiors yet differ in how their quantum states fit together. In a topological phase, that global organization cannot be smoothly changed without passing through a gap closing or leaving the assumptions that protect it. Here we build simple lattice models and prove which parameter ranges belong to which phase. The task is closer to drawing an exact map of a model than predicting a particular material.

### Key terms

- **Hamiltonian H** — The mathematical operator specifying a quantum system’s energy and dynamics.

- **Tight-binding model** — A lattice model in which particles occupy localized orbitals and move between them through specified hopping terms.

- **Orbital** — One of the local quantum states included at a lattice site or within a unit cell.

- **Band** — A range of allowed energies labeled by crystal momentum in a periodic system.

- **Band gap** — An energy separation between the selected occupied bands and the next bands.

- **Brillouin zone** — The space of distinct crystal momenta for a periodic lattice.

- **Chern number** — An integer measuring a global twist in the selected quantum states across a two-dimensional Brillouin zone.

- **Occupied projector** — An operator selecting the band states being treated as occupied.

- **Phase diagram** — A map showing which kind of behavior occurs for each model-parameter range.

### Why this matters

Topological models explain why some properties depend on global structure rather than microscopic details. A certified phase map provides reliable examples for theory and simulation, and it helps identify which ingredients a proposed design actually needs. Proving a gap everywhere is important because a small missed gap closing can invalidate a computed topological label.

### What progress would mean

A useful result would give explicit models, guaranteed parameter ranges and checked Chern numbers. These could guide later physical designs, while leaving material stability, fabrication and experimental behavior to additional work.

## Core question

Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?

## Scope and assumptions

Begin with the two-band square-lattice model H_m(k)=sin(k_x)σ_x+sin(k_y)σ_y+[m+cos(k_x)+cos(k_y)]σ_z, k∈[−π,π]². Use one occupied band and a fixed orientation. Analyze rational parameter intervals separated from m=−2,0,2.

For subsequent searches fix dimension, orbitals, filling, hopping support, coefficient domain, and symmetry representation. A rational grid is finite; a continuous coefficient box requires a different completeness argument. “All Hamiltonians for a space group” is not an enumerable specification.

## Mathematical target

For a general family H(k)=Σ_R t_Re^{ik·R}, impose t_−R=t_R† and the stated symmetry equations. Prove E_{r+1}(k)−E_r(k)≥δ>0 for all k and all parameters claimed. With P the occupied projector, choose

$$C=\frac{1}{2πi}\int_{BZ}\operatorname{Tr}P[∂_xP,∂_yP],d²k.$$

Certify C through an analytic degree/homotopy argument or validated integration with an error enclosure containing a unique integer, including discretization and projector errors.

## Required outputs and proof obligations

Return model coefficients, gap lower bounds, invariant and sign convention, certified parameter coverage, and uncovered boxes. A minimality claim must specify whether it minimizes orbitals, hopping range, or coefficients and must exclude every smaller object in that declared class.

## Validation and rejection controls

Recover gap closings at −2,0,2 and trivial/nontrivial intervals on either side. Reverse orientation and verify C changes sign. A fine-grid lattice Chern calculation is a useful cross-check, but must not substitute for the continuous gap proof. Check a time-reversal-invariant occupied bundle has zero total Chern number.

## Milestones and research extension

Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Chen et al., The impossibility of exactly flat non-trivial Chern bands in strictly local periodic tight binding models](https://arxiv.org/abs/1311.4956) — A concrete example of why range/locality assumptions change realizability claims.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

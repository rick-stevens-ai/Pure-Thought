# 05. Positive-geometry tests for a fixed gravity integrand

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/05-Positive-Geometry-Gravity.md) · [Original description](../archive/original-PRDs/05-Positive-Geometry-Gravity.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Particle-collision formulas often contain many terms whose cancellations hide a simpler structure. In some theories, a geometric shape organizes the answer: its boundaries encode the ways a collision can split into simpler events. This problem asks whether a tightly specified class of shapes can do the same for one gravity calculation. It treats geometry as a possible explanation for the structure of a formula, not as a literal picture of spacetime.

### Key terms

- **Integrand** — The mathematical expression to be integrated to obtain a quantity such as a scattering amplitude.

- **Loop** — A feature of a quantum correction involving internal momenta that must be integrated over; one loop is the first such level here.

- **Supergravity** — A theory combining gravity with supersymmetry, a proposed relation between types of particles; N=8 specifies the chosen four-dimensional model.

- **Positive geometry** — A region with mathematical boundary properties that determine a special differential form.

- **Canonical form** — That distinguished differential form, whose singularities and boundary residues obey precise recursive rules.

- **Residue** — The coefficient describing the leading contribution of a simple pole; here it encodes boundary information.

- **Factorization** — The way a scattering process separates into simpler processes near certain physical singularities.

- **Ansatz** — A deliberately restricted class of candidate expressions or geometric constructions searched for a solution.

- **Pole at infinity** — A singularity revealed when a coordinate is taken to infinity, which must be included in a global comparison.

### Why this matters

A geometric organization could explain cancellations and suggest more efficient calculations or new identities in quantum gravity. Even a carefully limited impossibility theorem is informative: it tells us which geometric proposals are too restrictive and where a different idea is needed.

### What progress would mean

Success would mean either a verified shape-to-form correspondence or a proof that the declared candidate class cannot work. Failure to find a shape would not rule out every possible geometric description of gravity.

## Core question

For a specified low-loop supergravity integrand, does a precisely bounded class of candidate positive geometries reproduce its differential form, including every boundary and pole at infinity?

## Scope and assumptions

Start with the four-point one-loop N=8 supergravity integrand, in a declared four-dimensional representation and helicity sector. Record any regulator and the limits of four-dimensional cut reconstruction. Derive or reproduce the target before searching. Tree-level and scalar-box canonical forms are calibration examples.

Define the search class before claiming an obstruction: ambient coordinates, dimension, allowed rational maps, inequalities, permitted external prefactors, and bounds on degree/number of boundaries. “No geometry exists” is not a finite computational question without such restrictions.

## Mathematical target

Compare the top-form Ω_target=I(ℓ;k)d⁴ℓ to f(k)π_*Ω_G for a declared map π and external factor f. State whether equality is literal rational-form equality or equality modulo a specified class of terms integrating to zero. Work on a defined real slice with orientations.

For a standard positive geometry, Ω_G has logarithmic boundary poles and recursively normalized residues. Check all finite boundaries, their intersections, and infinity. Distinguish a canonical differential form from the symbol of an integrated polylogarithmic function.

## Required outputs and proof obligations

Return either an explicit G, map, canonical form, and exact equality check, or a proof excluding the declared ansatz class. A failed search returns “unresolved within budget.” A dlog decomposition alone is a candidate diagnostic, not a geometry certificate.

## Validation and rejection controls

Verify simplex/box normalization independently. Check factorization and generalized cuts of the target. Rational reconstruction must be followed by an exact polynomial identity after denominators are cleared, with singular loci accounted for. Check orientation signs and residues at infinity explicitly.

## Milestones and research extension

Baseline: exact target and pole/residue audit. Strong: one verified representation or one restricted no-go theorem. Research extension: broaden the geometry class or multiplicity; explain how the result changes understanding of gravity amplitudes rather than treating a computational failure as a universal obstruction.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Arkani-Hamed, Bai and Lam, Positive Geometries and Canonical Forms](https://arxiv.org/abs/1703.04541) — Definition of logarithmic canonical forms and boundary recursion.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

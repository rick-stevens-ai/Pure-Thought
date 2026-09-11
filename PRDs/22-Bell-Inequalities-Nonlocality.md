# 22. Bell inequalities with certified local and quantum value brackets

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/22-Bell-Inequalities-Nonlocality.md) · [Original description](../archive/original-PRDs/22-Bell-Inequalities-Nonlocality.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two distant experimenters choose measurements and compare their outcomes. If the outcomes were produced only by shared classical instructions, their correlations would obey particular inequalities. Quantum systems can exceed some of those limits. This problem calculates both the classical ceiling and a reliable quantum range for a specified test, so that the size of the difference is established mathematically.

### Key terms

- **Bell scenario** — A specification of the parties, their measurement choices and the possible outcomes.

- **Local hidden-variable model** — A model in which outcomes depend on local choices and shared classical information, with measurement-choice independence among the assumptions.

- **Bell inequality** — A linear restriction satisfied by the correlations of such local models.

- **CHSH** — The simplest standard two-party test with two binary measurement choices per party.

- **Quantum strategy** — A specified shared quantum state and local measurements producing correlations.

- **POVM** — A set of positive operators summing to the identity that describes a general quantum measurement.

- **Tsirelson bound** — A quantum upper limit on a Bell expression, with 2√2 the familiar CHSH value in its standard normalization.

- **NPA hierarchy** — A sequence of matrix-positivity tests providing outer bounds on quantum correlations.

- **Commuting-operator model** — A framework in which different parties’ measurements commute; its general correlation set differs from the finite-dimensional tensor-product setting.

- **Facet** — A full boundary face of the appropriate dimension of the classical correlation polytope.

### Why this matters

Bell inequalities make the difference between classical local explanations and quantum correlations precise. They also underlie ways to certify quantum behavior from observed input-output relationships. Reliable quantum bounds are essential: a numerical optimization value without a valid upper certificate or achieving strategy may misstate what quantum theory permits.

### What progress would mean

A complete small test would include an exact classical maximum, an explicit quantum strategy and a matching or enclosing upper bound. Applications to randomness or cryptography would need additional statistical and security assumptions beyond this mathematical benchmark.

## Core question

For a fixed rational Bell functional in a small bipartite scenario, can its exact local value and a certified quantum lower/upper bracket be produced?

## Scope and assumptions

Start with two parties, two binary settings each, and CHSH. Fix normalization and no-signaling affine coordinates. The target quantum value is the supremum over finite-dimensional tensor-product strategies; identify its closure when discussing limits. Finite NPA levels provide outer bounds through the commuting-operator framework.

Facet discovery and randomness certification are extensions with separate definitions. Do not equate a finite SDP solution with a realizable strategy.

## Mathematical target

For β(P)=Σ_abxy c_abxyP(ab|xy), compute β_L by evaluating every deterministic local strategy. A verified finite-dimensional state and POVMs give L_Q≤β_Q. A certified NPA dual or noncommutative SOS gives β_Q≤U_Q. Report β_L and [L_Q,U_Q].

The NPA limit characterizes commuting-operator correlations, which need not equal the closure of finite-dimensional tensor-product correlations. Finite-level tightness needs an explicit achieving strategy or a valid extraction theorem.

## Required outputs and proof obligations

Export coefficients, local vertices, state/measurements, moment-word list, algebraic relations, dual matrices/SOS, and exact or interval objective bounds. To certify a facet in affine dimension D, verify the inequality for every vertex and show saturating vertices have affine dimension D−1. Declare relabeling symmetries when counting classes.

## Validation and rejection controls

Check all 16 CHSH deterministic strategies and β_L=2. Reproduce 2√2 with an explicit two-qubit strategy and an operator upper-bound identity. Reject an invalid dual PSD matrix or a wrong commutation relation. Distinguish eight CHSH variants from one class under relabeling; positivity facets must be handled separately.

## Milestones and research extension

Baseline: exact CHSH lower and upper certificates. Strong: a preselected additional rational functional with a certified bracket, whether or not it closes. Research extension: a new inequality with normalized comparison, or device-independent entropy bounds with an explicit adversary and statistical model.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Navascués, Pironio and Acín, A convergent hierarchy of semidefinite programs](https://arxiv.org/abs/0803.4290) — NPA convergence to a commuting-measurement representation.

- [Ji et al., MIP*=RE](https://arxiv.org/abs/2001.04383) — Strict separation of the relevant correlation sets.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

# 08. Exact modular-data and symmetry-anomaly consistency tests

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/08-Swampland-Modularity-Symmetries.md) · [Original description](../archive/original-PRDs/08-Swampland-Modularity-Symmetries.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A proposed quantum theory comes with mathematical rules for combining excitations and transforming its descriptions. These rules should fit together like the pieces of a puzzle. This problem builds exact checks for that fit and then asks a second question: can a chosen global symmetry be made into a gauge symmetry? A theory can be consistent even when this second operation is obstructed, so the two verdicts must remain separate.

### Key terms

- **Rational CFT** — A conformal field theory with finitely many basic representation types relative to a specified chiral algebra.

- **Character** — A generating function counting states within one representation of that algebra.

- **Modular S and T matrices** — Matrices describing how the characters change under two basic transformations of the torus.

- **Fusion rules** — Rules specifying which excitation types can occur when two types are combined.

- **Global symmetry** — A transformation applied consistently throughout a system that leaves its laws unchanged.

- **Gauging** — Promoting an appropriate symmetry to a local redundancy and adding the corresponding gauge structure.

- **Anomaly** — An obstruction to implementing a classical symmetry or a proposed gauging consistently in the quantum setting; an ’t Hooft anomaly can exist in a valid theory.

- **Cocycle** — Algebraic data satisfying a compatibility identity, used here to encode an anomaly.

- **Bulk and boundary** — The interior gravitational theory and its lower-dimensional boundary description; symmetries can have different interpretations in the two.

### Why this matters

Exact consistency checks can detect false candidate data before researchers invest in a full construction. They also clarify which symmetries can be gauged and which require additional structure, such as anomaly cancellation or inflow from another system. This is useful both for studying quantum field theories and for interpreting holographic gravity proposals.

### What progress would mean

The first contribution would be a dependable checker with known passing and failing examples. A failure would identify a particular algebraic inconsistency or gauging obstruction, rather than issuing an unsupported verdict about all of quantum gravity.

## Core question

For supplied rational-CFT modular data and a specified finite internal symmetry, which algebraic consistency tests can be certified, and which failures obstruct a proposed gauging?

## Scope and assumptions

Use a bosonic rational CFT with a declared chiral algebra and finitely many characters. Input exact algebraic S,T matrices, vacuum label, and a nonnegative integer multiplicity matrix M. Start with the Ising model. An optional finite 0-form group G requires separate defect/twisted-sector data and an anomaly cocycle; these cannot be inferred from the untwisted partition function alone.

The result concerns the supplied CFT data and gauging problem. A general bulk quantum-gravity compatibility verdict is outside the baseline.

## Mathematical target

Check S†S=I, symmetry of S where assumed, S²=C, (ST)³=C in the declared T convention, M₀₀=1, MS=SM, MT=TM, and Mᵢⱼ∈Z≥0. Check Verlinde coefficients Σ_a S_ia S_ja S*_ka/S_0a are nonnegative integers.

For a finite symmetry cocycle ω∈Z³(G,U(1)), check the cocycle identity exactly. To certify anomaly-free gauging in the stated setting, supply an explicit 2-cochain β with δβ=ω, or prove its nonexistence in a complete cohomology computation. Distinguish this obstruction from inconsistency of an anomalous but valid boundary theory.

## Required outputs and proof obligations

Return pass/fail/unresolved for each named axiom, with exact arithmetic traces. Include the number field and embeddings. Even passing every modular-data test gives necessary algebraic consistency only; full CFT sewing and realizability are separate obligations.

## Validation and rejection controls

Verify Ising S entries including 1/√2 in algebraic arithmetic. Reject altered modular relations, noninteger multiplicities, and a failed cocycle identity. Accept the existence of a boundary global symmetry rather than flagging it as automatically inconsistent. Cross-check a small cyclic-group cohomology example.

## Milestones and research extension

Baseline: exact modular-data checker on Ising and one second supplied model. Strong: one finite-group twisted-sector/gauging analysis. Research extension: classify a explicitly finite family of data or derive a conditional bulk consequence using a stated holographic theorem.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Harlow and Ooguri, Symmetries in quantum field theory and quantum gravity](https://arxiv.org/abs/1810.05338) — Bulk gauge versus boundary global symmetry, including higher-form extensions.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

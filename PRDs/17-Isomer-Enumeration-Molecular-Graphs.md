# 17. Complete enumeration of a precisely defined molecular-graph class

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/17-Isomer-Enumeration-Molecular-Graphs.md) · [Original description](../archive/original-PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A molecular formula tells us how many atoms of each kind are present, but not how they are connected. The same counts can describe different structures: butane and isobutane are a familiar example. This problem asks for every distinct connection pattern allowed by an explicit set of bonding rules, with a proof that none was counted twice and none was missed.

### Key terms

- **Molecular formula** — A count of atoms by element, such as C₄H₁₀.

- **Constitutional isomer** — A structure with the same molecular formula as another but a different pattern of atom connections.

- **Molecular graph** — A mathematical representation with atoms as vertices and bonds as edges carrying bond-order labels.

- **Valence grammar** — The exact bonding rules permitted in this enumeration; it deliberately defines a restricted chemical model.

- **Bond order** — The bond label distinguishing single, double and triple bonds in this graph model.

- **Graph isomorphism** — A relabeling of atoms that preserves element types and bonds, showing two drawings represent the same graph.

- **Canonical labeling** — A consistent representative label for a graph, used to identify duplicates.

- **Stereoisomer** — A distinction in spatial arrangement not captured merely by the basic connection graph.

- **Completeness certificate** — Evidence that the enumeration covers every object in the specified graph class.

### Why this matters

Complete enumeration turns an intuitive picture of “possible structures” into a checkable mathematical set. Such sets are useful for testing structure-generation methods and exploring how bonding constraints shape chemical possibilities. The guarantee is especially valuable when a search is used to support a claim that no alternative structure exists.

### What progress would mean

Success would supply exact graph counts and explicit structures under the chosen rules. A graph’s presence in the list would not imply that the corresponding molecule is stable, easily synthesized or realizable in a particular three-dimensional shape.

## Core question

For a fixed formula and explicit valence grammar, can every connected constitutional graph be generated exactly once, with a checkable completeness argument?

## Scope and assumptions

Initially allow neutral closed-shell C,H,O graphs with bond orders 1,2,3, no self-bonds, and valences 4,1,2. Hydrogens may be implicit if reconstructed uniquely from residual valence. Specify whether different Kekulé graphs are distinct; the baseline uses literal bond-order graphs and no aromatic equivalence.

Constitutional graph validity is not chemical stability or synthesizability. Stereoisomers, conformers, ions, radicals, isotopes, and 3D geometries require added definitions and are not baseline outputs.

## Mathematical target

For labeled heavy atoms set symmetric b_ij∈{0,1,2,3}, b_ii=0, h_i=v_i−Σ_j b_ij≥0, and Σ_i h_i=n_H. Require connectivity and the requested counts of element labels. Quotient by permutations preserving element labels and bond orders.

Prove completeness via canonical augmentation with a proved parent rule, or exhaustive labeled enumeration plus verified orbit reduction. A SAT route needs the exact encoding and orbit-blocking/symmetry-breaking justification; the final UNSAT proof must cover all remaining admissible graphs.

## Required outputs and proof obligations

Export canonical graphs, reconstructed hydrogens, counts, labeling conventions, and generation/coverage proof. Provide a validator independent of the optimized generator. If a resource limit stops enumeration, output a partial catalog and coverage status rather than “all isomers.”

## Validation and rejection controls

Recover alkane constitutional counts for C₄H₁₀ and C₅H₁₂, namely 2 and 3. Cross-check all tiny instances with brute force. Random relabelings must preserve canonical codes. Verify each graph’s formula and weighted valence; reject disconnected mixtures.

## Milestones and research extension

Baseline: complete small-formula catalog with independent counts. Strong: complete C₆H₁₂O under the stated grammar. Research extension: stereochemical orbit enumeration with meso cases and explicit stereocenter/double-bond conventions; keep conformational continua separate.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [McKay, Isomorph-free exhaustive generation](https://users.cecs.anu.edu.au/~bdm/papers/orderly.pdf) — Canonical generation and completeness methodology.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

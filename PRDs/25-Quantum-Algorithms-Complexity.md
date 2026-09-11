# 25. Quantum query complexity with explicit oracle and output models

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/25-Quantum-Algorithms-Complexity.md) · [Original description](../archive/original-PRDs/25-Quantum-Algorithms-Complexity.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For unstructured search with a specified promise, can an implementation and an analytic lower bound establish a matched quantum query complexity?

## Scope and assumptions

Use N=2^n items with exactly one marked item and phase-oracle access O_f|x〉=(−1)^f(x)|x〉. State success probability at least 2/3 and count oracle uses separately from elementary gates, state preparation, and output verification. Begin with small exact simulations and a proof valid for all N.

Retain quantum walks, linear systems, QAOA and oracle separations as independent extensions, each with its own input/output and cost model. Classical simulation wall time is not quantum runtime.

## Mathematical target

Let θ=arcsin(N⁻¹ᐟ²). After r Grover iterations the success probability is sin²((2r+1)θ). Choose r as a nearest nonnegative integer to π/(4θ)−1/2, evaluating neighboring integers and handling small N explicitly. Derive the query upper bound and pair it with a hybrid/adversary/polynomial lower bound Ω(√N) under the same promise.

An oracle separation must be written BQP^O≠BPP^O for a specified O; it does not establish BQP≠BPP. A linear-system extension must state matrix access, κ, precision, state preparation, and whether the output is a quantum state or classical vector.

## Required outputs and proof obligations

Return oracle/circuit specification, exact small-instance results, an all-N success derivation, query/gate/resource accounting, and the lower-bound proof with assumptions. Any QAOA extension must distinguish a certified bound on a finite graph from a performance theorem for a graph family.

## Validation and rejection controls

Check the analytic success formula on n=1,…,8 and all marked positions using symmetry or exhaustive checks. Test neighboring iteration counts to expose overshoot. Verify unitarity and normalization. Compare with randomized classical search under identical oracle access and error tolerance.

## Milestones and research extension

Baseline: Grover construction and matched query bound. Strong: one second promise problem with matched assumptions and a certified upper/lower bracket. Research extension: a new restricted query bound or oracle result. Reimplementation of established algorithms is replication, not a new complexity separation.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Bennett et al., Strengths and Weaknesses of Quantum Computing](https://arxiv.org/abs/quant-ph/9701001) — Oracle lower-bound methodology for quantum search.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

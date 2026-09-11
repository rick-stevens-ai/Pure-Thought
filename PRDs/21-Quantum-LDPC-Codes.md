# 21. Quantum LDPC instances with independently certified code parameters

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/21-Quantum-LDPC-Codes.md) · [Original description](../archive/original-PRDs/21-Quantum-LDPC-Codes.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a specified CSS construction, can code dimension and distance bounds be verified independently, and can decoder performance be measured under one explicit noise model?

## Scope and assumptions

Start with hypergraph-product codes from supplied binary parity-check matrices H₁ and H₂. Include the product of two full-rank 3×7 Hamming parity-check matrices as a small benchmark. Separate finite instances, asymptotic families, and noisy syndrome extraction.

Use independent X errors with perfect syndrome measurements for the first decoder experiment. A family is LDPC only when both row and column weights stay bounded as n grows.

## Mathematical target

For H_i of size m_i×n_i define

$$H_X=[H_1\otimes I_{n_2}\mid I_{m_1}\otimes H_2^{\mathsf T}],\quad H_Z=[I_{n_1}\otimes H_2\mid H_1^{\mathsf T}\otimes I_{m_2}].$$

Check H_XH_Zᵀ=0 over F₂, n=n₁n₂+m₁m₂, k=n−rank H_X−rank H_Z. Define d_X=min{|x|:x∈ker H_Z∖row H_X} and d_Z analogously, d=min(d_X,d_Z). A found logical operator upper-bounds distance; ruling out all smaller logical operators lower-bounds it.

## Required outputs and proof obligations

Export sparse matrices, GF(2) row reductions, logical bases, explicit upper witnesses and lower-bound proofs. SAT certificates need a verified encoding of nontrivial homology and weight. Decoder experiments report physical noise, recovery rule, logical-failure test, sample counts, confidence intervals, and runtime.

## Validation and rejection controls

For the two 3×7 inputs verify n=58 and k=16; certify the claimed distance independently rather than assuming it. Inject all small-weight errors on tiny codes. A matching syndrome is not success unless error plus recovery is a stabilizer. Compare optimized checks with an independent GF(2) implementation.

## Milestones and research extension

Baseline: exact small product code and certified distance bracket. Strong: several instances plus honest finite-size decoder curves. Research extension: a new family tradeoff or proved decoder guarantee. Reproducing asymptotically good codes is not itself a new existence result.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Panteleev and Kalachev, Asymptotically Good Quantum and Locally Testable Classical LDPC Codes](https://arxiv.org/abs/2111.03654) — Existing asymptotically good quantum LDPC constructions; novelty baseline.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

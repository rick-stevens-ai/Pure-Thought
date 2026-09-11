# 07. Stress-tensor bootstrap: a sector-specific gap bound in three dimensions

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/07-Extremal-CFTs-Stress-Tensor.md) · [Original description](../archive/original-PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

At fixed stress-tensor two- and three-point data, can crossing exclude a proposed gap in the parity-even scalar sector of T×T in a unitary three-dimensional CFT?

## Scope and assumptions

Fix d=3, parity invariance, a unique conserved stress tensor, a normalization for C_T, and the remaining independent TTT parameter. The initial observable is the first parity-even nonidentity scalar appearing with nonzero coupling in T×T. It is not a bound on every operator in the theory. State separate gaps for any other restricted spin/parity sectors.

Use the full conserved spinning correlator. A scalar block toy implementation is permitted for learning but is not the target stress-tensor bootstrap.

## Mathematical target

Write the crossing equations in a nonredundant tensor basis as

$$V_{id}+V_T(C_T,t_4)+\sum_{\mathcal O}\lambda_{\mathcal O}^{\mathsf T}V_{Δ,ℓ,\pi}\lambda_{\mathcal O}=0.$$

The OPE vectors account for multiple tensor structures. In d=3, a conserved symmetric-traceless spin-ℓ primary has Δ=ℓ+1, not Δ=3 for all spins. For a proposed scalar gap g, seek α with α(V_id+V_T)>0 and α(V_{Δ,ℓ,π}) positive semidefinite on every allowed exchange. This yields a contradiction and therefore excludes g under the specified assumptions.

## Required outputs and proof obligations

Export tensor/block conventions, conservation and Ward identities, fixed input data, derivative basis, spin treatment, and functional coefficients. Attach rigorous block-approximation and high-spin control. If only the finite approximant is checked, label the result accordingly.

## Validation and rejection controls

Recover a selected published stress-tensor benchmark using exactly its conventions and assumptions. Check the allowed region rather than the excluded region. Verify matrix positivity, not just individual scalar components. Free scalar/fermion data provide normalization tests; numerical agreement alone is not a proof of the continuum bound.

## Milestones and research extension

Baseline: reproduce one finite benchmark and publish the assumptions. Strong: turn one exclusion into a certified result with omitted-sector control. Research extension: improve a bound, add another sector, or obtain a conditional extremal spectrum. Saturation does not construct a CFT.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Dymarsky et al., The 3d Stress-Tensor Bootstrap](https://arxiv.org/abs/1708.05718) — Conserved tensor crossing and sector-specific numerical gap bounds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

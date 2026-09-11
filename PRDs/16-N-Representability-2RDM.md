# 16. Certified energy brackets from reduced-density-matrix relaxations

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/16-N-Representability-2RDM.md) · [Original description](../archive/original-PRDs/16-N-Representability-2RDM.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Describing every possible configuration of many electrons becomes overwhelmingly expensive as the system grows. Yet the energy of a system with one- and two-particle interactions depends only on certain one-electron and two-electron summaries. The challenge is that not every plausible summary can come from a real many-electron state. We use necessary consistency tests to calculate a guaranteed lower energy limit and an actual trial state to obtain an upper limit.

### Key terms

- **Fermion** — A particle, such as an electron, whose quantum states obey antisymmetry under exchange.

- **Reduced density matrix (RDM)** — A summary of selected particles’ quantum statistics obtained from a larger system’s state.

- **2-RDM** — The two-particle reduced density matrix, sufficient with the appropriate one-particle data to compute the chosen two-body Hamiltonian’s energy.

- **N-representability** — The requirement that the reduced data arise from some valid state of N fermions.

- **P, Q and G conditions** — Positivity tests involving particle pairs, hole pairs and particle–hole combinations; they are necessary representability conditions.

- **Hole** — An unoccupied orbital state, treated as an absence of a particle relative to the finite orbital description.

- **Semidefinite program (SDP)** — An optimization problem with linear conditions and positive-semidefinite matrix constraints.

- **Ground-state energy** — The lowest allowed energy of the chosen quantum system.

- **Energy bracket** — A pair of verified lower and upper numbers containing the true ground-state energy.

### Why this matters

Reliable energy calculations underpin the theoretical study of bonding and interacting electrons. A bracket tells us how much uncertainty remains, rather than merely reporting a convincing decimal number. The approach may extract useful information without storing a complete many-electron wavefunction, although general representability remains computationally difficult.

### What progress would mean

Progress would mean tighter verified brackets at an affordable cost for a specified finite model. That would show how much the extra constraints help; extending the conclusion to an actual molecule would also require controlling orbital-basis and Hamiltonian errors.

## Core question

For a specified finite fermionic Hamiltonian, how tight an independently certified ground-state energy interval can P,Q,G and selected higher-order RDM constraints produce?

## Scope and assumptions

Fix N≥2 fermions in M spin orbitals with an exact rational/algebraic, number-conserving one- and two-body Hamiltonian. Start with a small Hubbard-type system where exact diagonalization in the same N-particle sector is feasible. Molecular integrals are a later input with an explicit basis and numerical error enclosure.

Use ensemble N-representability. The task is to certify bounds within a finite orbital model, not solve general electronic structure in polynomial time.

## Mathematical target

Set γ_ij=〈a_i†a_j〉 and Γ_ij,kl=〈a_i†a_j†a_l a_k〉. Require Hermiticity, fermionic antisymmetry, Tr γ=N, Σ_ij Γ_ij,ij=N(N−1), and Σ_j Γ_ij,kj=(N−1)γ_ik. Define Q and G using two-hole and particle-hole operator moments and derive their affine forms from anticommutation identities.

Minimizing the linear energy over a necessary-condition outer relaxation gives E_relax≤E₀. A verified dual feasible point gives L≤E_relax; an explicit normalized N-fermion state gives U≥E₀. Report L≤E₀≤U, not an unsupported accuracy estimate from Γ alone.

## Required outputs and proof obligations

Export the Hamiltonian and indexing convention, moment constraints, certified dual objective L, state witness for U, and gap U−L. Separate solver error, relaxation error, and finite-basis/model error. Report any property bounds via separately optimized observables; a relaxed Γ need not be physical.

## Validation and rejection controls

Compare with exact diagonalization in identical conventions. Check contractions using explicit Slater determinants and correlated states. Adding valid necessary constraints must not lower the exact relaxed minimum. Verify dual PSD matrices and equality residual corrections rigorously, rather than interpreting the raw solver objective as a bound.

## Milestones and research extension

Baseline: valid L,U brackets for small fixed models. Strong: quantify improvement from G or T constraints across a specified instance set. Research extension: a new valid constraint with demonstrated tightening, or a theorem on a tractable family; novelty needs comparison with existing hierarchies.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Liu, Christandl and Verstraete, N-representability is QMA-complete](https://arxiv.org/abs/quant-ph/0609125) — Complexity barrier to general exact representability.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

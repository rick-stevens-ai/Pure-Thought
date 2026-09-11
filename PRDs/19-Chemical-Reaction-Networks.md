# 19. Certified equilibria and conditional persistence in mass-action networks

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/19-Chemical-Reaction-Networks.md) · [Original description](../archive/original-PRDs/19-Chemical-Reaction-Networks.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A list of chemical reactions can settle into different long-term concentration patterns. Some networks have one resting state; others can behave like a switch, with two stable states separated by an unstable one. This problem asks whether we can find every positive equilibrium of a small, explicitly specified network and prove which are locally stable. It also distinguishes never reaching zero at a finite time from genuinely avoiding extinction in the long run.

### Key terms

- **Reaction network** — A set of chemical species and rules converting specified combinations of reactants into products.

- **Mass-action kinetics** — Rate laws in which each reaction rate is proportional to powers of its reactant concentrations, multiplied by a rate constant.

- **Stoichiometry matrix S** — A table describing how each reaction changes the amount of each species.

- **Conservation law** — A combination of species amounts that remains constant under the network’s reactions.

- **Compatibility class** — The concentration states accessible under the same conserved quantities and stoichiometric constraints.

- **Equilibrium or steady state** — A concentration pattern where the net rate of change is zero.

- **Local stability** — The tendency of states starting sufficiently near an equilibrium to remain nearby, with asymptotic stability also implying return toward it.

- **Multistationarity and multistability** — Having multiple equilibria, and having multiple stable equilibria, respectively.

- **Persistence** — A long-time condition preventing each initially present species from approaching zero along the trajectory.

### Why this matters

Stable switches and coexistence are important organizing ideas in chemistry and biological regulation. An exact small-network analysis can establish whether a proposed mechanism even supports those behaviors. It also makes numerical simulation more interpretable: a trajectory can miss another equilibrium or appear stable simply because it has not been followed long enough.

### What progress would mean

Progress would give complete equilibrium information for the stated network, rates and conserved totals. A theorem covering many rate choices would be stronger; identifying a switch in a model would not by itself establish its role in a living system.

## Core question

For a finite mass-action network with specified rates and conservation totals, can all positive equilibria in one compatibility class be isolated and their local stability certified?

## Scope and assumptions

Input integer reactant/product complexes, positive rational rate constants, and positive rational initial totals. Use small networks first, initially at most four species and eight reactions. Distinguish fixed-parameter root analysis from theorems quantified over all rates.

Keep persistence as a separate theorem-driven module. Catalytic RAF structure belongs primarily to Challenge 29 and does not by itself determine mass-action stability.

## Mathematical target

Let S be the stoichiometric matrix, f(x)=Sv(x), v_r=κ_r x^{y_r}, and let rows of L span ker Sᵀ. Solve f(x)=0, Lx=Lx₀, x>0. Remove dependent equations before root isolation. Deficiency is δ=number of complexes−number of linkage classes−rank S.

For an isolated equilibrium, test the Jacobian on im S, not on the full space with conservation-induced zero eigenvalues. Multistationarity means at least two positive equilibria in the same class; multistability additionally needs stability. Persistence means liminf_{t→∞}x_i(t)>0 and requires global hypotheses and boundedness checks.

## Required outputs and proof obligations

Return exact network data, deficiency/reversibility facts, conservation laws, isolating boxes with existence/uniqueness proofs, and coverage of the admissible domain. A positive-dimensional equilibrium set is a separate valid outcome. Persistence output must name a sufficient theorem and check every hypothesis, or return unresolved.

## Validation and rejection controls

Use A⇌B as a deficiency-zero benchmark. Use the one-species open network ∅⇌X and 2X⇌3X with rates giving f(x)=6−11x+6x²−x³; isolate x=1,2,3 and distinguish two stable roots from three equilibria. Include a boundary-approaching trajectory example to distinguish positivity from persistence.

## Milestones and research extension

Baseline: exact structural facts and complete small fixed-parameter root enumeration. Strong: certified local stability and one applicable persistence theorem. Research extension: parameter-region proofs or a new structural criterion, with quantifiers stated explicitly.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Angeli, De Leenheer and Sontag, A Petri Net approach to persistence](https://arxiv.org/abs/q-bio/0608019) — Checkable sufficient conditions for persistence, with hypotheses.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

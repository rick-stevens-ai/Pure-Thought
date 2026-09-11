# 29. Autocatalytic sets: maximality, irreducibility and dynamical viability

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/29-Chemical-Reaction-Networks-Origins.md) · [Original description](../archive/original-PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Could a collection of reactions collectively help produce the substances that make those same reactions happen? Autocatalytic-set theory turns part of that question into a precise network test. Starting from an externally supplied food set, we ask which reactions can be supported by products of the network and which smaller subsets already have that property. This describes a structural prerequisite for self-sustaining chemistry, not a complete definition of life.

### Key terms

- **Catalyst** — A substance participating in a reaction mechanism that increases its rate while being regenerated overall; it does not change the equilibrium free-energy difference.

- **Autocatalysis** — Production that directly or collectively helps enable more production of the relevant system.

- **Food set** — The species assumed to be available from outside the network.

- **Closure** — All species producible from the food set using the selected reaction rules; standard RAF closure temporarily ignores catalysis.

- **RAF** — A reflexively autocatalytic and food-generated reaction set: its reactants are reachable from food and each reaction has a catalyst available within the closure.

- **Maximal RAF** — The largest RAF contained in the supplied network, when a RAF exists.

- **Irreducible RAF** — A RAF containing no smaller nonempty RAF subset.

- **Minimum-cardinality RAF** — A RAF with the fewest reactions among all RAFs in the specified network; this is stronger than irreducibility.

- **Chemostat** — An idealized externally maintained concentration or chemical reservoir.

- **Dynamical viability** — The additional question of whether specified reaction rates and conditions support sustained activity or growth.

### Why this matters

Collectively supported chemistry is relevant to thinking about how organized reaction systems can arise and persist. Exact minimality results help distinguish essential reactions from dispensable ones in a model. Just as importantly, separating structural closure from energy supply and kinetics prevents a network diagram from being mistaken for proof of a viable replicating system.

### What progress would mean

Success would identify supported reaction sets and prove the stated kind of minimality. Demonstrating sustained growth, heredity or evolution would require extra chemical and dynamical assumptions beyond the RAF certificate.

## Core question

For a finite catalytic reaction system, can its maximal RAF and smallest RAF subsets be certified, while keeping structural autocatalysis distinct from sustained growth?

## Scope and assumptions

Input species S, reactions R with integer stoichiometry, food set F, and catalysis relation C⊂S×R. Begin with at most 12 reactions for exhaustive cross-checks. Catalysis is supplied model data, not inferred from a stoichiometric cycle.

Define three separate targets: unique maximal RAF, an inclusion-minimal (irreducible) RAF, and a minimum-cardinality RAF. Thermodynamic and kinetic viability require extra inputs and are separate extensions.

## Mathematical target

For R′⊂R compute cl_{R′}(F) by repeatedly adding products whose reactants are already present, ignoring catalysis in this closure step. R′ is a nonempty RAF when every reactant lies in this closure and every reaction has at least one catalyst in it.

An irreducible RAF has no proper nonempty RAF subset. A minimum-cardinality RAF minimizes |R′| over all RAF subsets. Ordinary RAF closure permits catalysts to arise later; a catalysis-respecting startup ordering is a stronger condition and must be tested separately.

## Required outputs and proof obligations

Export closure rounds, catalyst witnesses, maxRAF elimination trace, and minimality/cardinality proof with explicit scope. For thermodynamic extension specify chemical potentials, chemostats, and driving. For kinetic extension provide rates, dilution, initial conditions and a positive sustained-state/growth criterion; RAF existence alone is insufficient.

## Validation and rejection controls

Cross-check all subsets for tiny networks. Include food-catalyzed, self-catalyzed, catalyst-free, and mutually dependent examples. Verify a maximum RAF can contain several irreducible RAFs of different sizes. Deletion tests for irreducibility must use a complete RAF detector on each remaining network, not merely test whether the entire remainder is itself RAF.

## Milestones and research extension

Baseline: checked maxRAF/RAF witnesses. Strong: complete minimum-cardinality results for a bounded network family. Research extension: characterize which RAFs additionally support a declared driven mass-action regime. Biological self-replication and evolution require their own operational definitions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Hordijk and Steel, Autocatalytic sets in a partitioned biochemical network](https://pmc.ncbi.nlm.nih.gov/articles/PMC4034171/) — RAF closure and the distinction between maximal and irreducible RAFs.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

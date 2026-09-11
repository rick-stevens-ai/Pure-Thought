# 30. An exact finite genotype–phenotype map and its mutation graph

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/30-Genotype-Phenotype-Mapping.md) · [Original description](../archive/original-PRDs/30-Genotype-Phenotype-Mapping.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Core question

For a fully specified combinatorial RNA folding rule on short sequences, what are the exact neutral components, mutation robustness, and accessible phenotype counts?

## Scope and assumptions

Use alphabet {A,C,G,U}, sequence length L=4,…,8 initially, allowed pairs AU/UA, GC/CG and GU/UG, minimum hairpin length three unpaired bases, no pseudoknots, and one pair per base. Maximize pair count and break ties by lexicographically smallest dot-bracket string under a declared character order.

This defines a mathematical GP map, not a thermodynamic RNA predictor. A nearest-neighbor free-energy extension must name its parameter set; empirical parameters are then acknowledged model inputs.

## Mathematical target

Let Φ map each sequence to its deterministic selected structure. On the Hamming graph H(L,4), exact-one-mutation neighbors satisfy d_H(s,t)=1 and each sequence has 3L neighbors. Define

$$r(s)=\frac{|\{t:d_H(s,t)=1,\ Φ(t)=Φ(s)\}|}{3L}.$$

For each phenotype analyze the induced neutral graph and every connected component; a phenotype preimage need not be connected. Define evolvability as the number of distinct other phenotypes reachable by one mutation, stating whether it is measured per genotype, component or whole preimage.

## Required outputs and proof obligations

Export folding rules, tie-breaking, sequence-to-structure table, neutral components, exact counts and rational robustness. A search restricted to a component must not report the full phenotype frequency. Fitness is a separately supplied function; evolutionary dynamics require population size, mutation kernel, selection rule and initial state.

## Validation and rejection controls

Cross-check dynamic programming with exhaustive legal-structure enumeration on short sequences, especially ties and prohibited pairs. Verify Σ_φ|Φ⁻¹(φ)|=4^L and every Hamming vertex degree is 3L. Count neutral edges independently and recover average robustness from twice the edge count. Treat any larger-L sample as statistical evidence.

## Milestones and research extension

Baseline: complete exact maps through L=8. Strong: increase L within measured resources and characterize component-level robustness/evolvability. Research extension: a proved asymptotic property or a specified noisy GP-channel capacity problem. Do not hard-code a universal percolation threshold or empirical target correlation.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [ViennaRNA, RNAfold documentation](https://www.tbi.univie.ac.at/RNA/RNAfold) — RNAfold’s energy model and parameter choices differ from a pair-count model.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.

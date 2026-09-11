# Critique 15: Symmetry-indicator algebra and the limits of band diagnosis

[Revised problem](../PRDs/15-Topological-Quantum-Chemistry.md) · [Preserved original](../archive/original-PRDs/15-Topological-Quantum-Chemistry.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/15-Topological-Quantum-Chemistry.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — “complete classification” (14–22, 36–44).** Symmetry indicators generally do not distinguish every topological bundle. Indicator-trivial phases can still be topological.

2. **Blocking — cone versus group (79–105).** The text interchanges nonnegative EBR sums, integer differences, and a group quotient. Fragile topology and stable indicator obstructions require these to be kept separate.

3. **Major — zero indicator implies positive decomposition (103–105).** Integer-lattice membership does not guarantee a nonnegative solution. Even matching an atomic irrep vector is not a construction of localized Wannier functions for a particular Hamiltonian.

4. **Major — crystallography input.** Not every site representation induction is automatically elementary; site symmetry, maximality and spin/time-reversal conventions must be stated. The revised algebra produces auditable limited conclusions instead of an all-phase label.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For one fixed symmetry setting, can compatibility relations and atomic band representations be generated exactly, and what topology can their quotient actually detect?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact quotient and decompositions for one setting. Strong: connect the diagnosis to explicit Hamiltonians and identify its kernel. Research extension: another symmetry group or a rigorous fragile obstruction with a definition of allowed added bands.

## Sources supporting the corrections

- [Po, Vishwanath and Watanabe, Symmetry-based Indicators of Band Topology](https://arxiv.org/abs/1703.00911) — Quotient of compatible symmetry data by atomic data; symmetry-based diagnosis.

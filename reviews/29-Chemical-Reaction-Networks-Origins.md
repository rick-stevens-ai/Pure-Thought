# Critique 29: Autocatalytic sets: maximality, irreducibility and dynamical viability

[Revised problem](../PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Preserved original](../archive/original-PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/29-Chemical-Reaction-Networks-Origins.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — reversed terminology (81).** maxRAF is maximal, not minimal. Inclusion-minimal and minimum-cardinality are also different optimization targets, so the existing output flag is ambiguous.

2. **Major — closure/startup (30–31, 73–79).** Standard RAF food closure ignores catalysis during reachability; requiring every intermediate step to be catalyzed is stronger. The rewrite keeps these questions separate.

3. **Major — thermodynamic viability (99–108).** A reaction list plus arbitrary negative ΔG values is not a consistent driven chemical model. Chemostats, activities and cycle constraints matter; catalysts do not alter equilibrium free-energy differences.

4. **Major — biology and information (97, 112–116).** Generic hypercycle stability depends on the actual equations, and concentrations are not automatically probabilities or a joint distribution for mutual information. Structural RAF certificates cannot alone establish Darwinian evolution, stability against parasites, or information emergence.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite catalytic reaction system, can its maximal RAF and smallest RAF subsets be certified, while keeping structural autocatalysis distinct from sustained growth?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: checked maxRAF/RAF witnesses. Strong: complete minimum-cardinality results for a bounded network family. Research extension: characterize which RAFs additionally support a declared driven mass-action regime. Biological self-replication and evolution require their own operational definitions.

## Sources supporting the corrections

- [Hordijk and Steel, Autocatalytic sets in a partitioned biochemical network](https://pmc.ncbi.nlm.nih.gov/articles/PMC4034171/) — RAF closure and the distinction between maximal and irreducible RAFs.

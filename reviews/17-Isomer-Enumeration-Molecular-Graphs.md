# Critique 17: Complete enumeration of a precisely defined molecular-graph class

[Revised problem](../PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Preserved original](../archive/original-PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/17-Isomer-Enumeration-Molecular-Graphs.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — chemical universe (39–47).** Fixed valences do not describe every molecule of a formula. Charge, radicals, aromatic equivalence, stereochemistry and disconnected structures change the answer. The grammar now defines what “all” means.

2. **Major — canonicity is not completeness (96–99).** No duplicate outputs does not prove no missing outputs. A coverage argument must audit pruning and symmetry breaking.

3. **Major — Pólya shortcut (101–108).** Burnside/Pólya counting requires a specified group action and correct fixed-point counts under the graph constraints. Writing a cycle-index sum alone does not enumerate connected valence-constrained graphs.

4. **Major — physical and spatial claims.** SMILES and a valence-correct graph do not certify a stable 3D structure. The rewrite preserves an exact combinatorial task and treats stereochemistry as a separately verified extension.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed formula and explicit valence grammar, can every connected constitutional graph be generated exactly once, with a checkable completeness argument?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete small-formula catalog with independent counts. Strong: complete C₆H₁₂O under the stated grammar. Research extension: stereochemical orbit enumeration with meso cases and explicit stereocenter/double-bond conventions; keep conformational continua separate.

## Sources supporting the corrections

- [McKay, Isomorph-free exhaustive generation](https://users.cecs.anu.edu.au/~bdm/papers/orderly.pdf) — Canonical generation and completeness methodology.

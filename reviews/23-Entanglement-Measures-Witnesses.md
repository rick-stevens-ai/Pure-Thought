# Critique 23: Entanglement certification with explicit inconclusive outcomes

[Revised problem](../PRDs/23-Entanglement-Measures-Witnesses.md) · [Preserved original](../archive/original-PRDs/23-Entanglement-Measures-Witnesses.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/23-Entanglement-Measures-Witnesses.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — relaxation direction (36–40).** Passing PPT or a finite DPS level does not generally prove separability. The replacement requires a sufficient certificate or reports unresolved.

2. **Major — “optimal witness” unspecified.** Multiplying a witness by a positive constant scales its objective without changing detection. An optimization needs a normalization and a declared class of witnesses.

3. **Major — measures conflated (23–29, 87–107).** Different entanglement measures do not give a complete interchangeable characterization, and the two-qubit concurrence formula is not a general mixed-state solution.

4. **Major — benchmark convention (817).** A Werner-state threshold such as p=2/3 is meaningless without the mixing definition; for the singlet-weight convention used here the threshold is 1/3. Explicit inputs remove that ambiguity and expose false positives in the checker.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite bipartite density matrix, can one produce a checked entanglement witness or a separable decomposition, and bound a specifically chosen entanglement measure?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: checked two-qubit decisions and measure values. Strong: a validated DPS witness beyond PPT and a separable-decomposition example. Research extension: robust uncertainty-set certification or normalized optimal witnesses on a stated family.

## Sources supporting the corrections

- [Doherty, Parrilo and Spedalieri, A complete family of separability criteria](https://arxiv.org/abs/quant-ph/0308032) — Symmetric-extension hierarchy and dual entanglement witnesses.

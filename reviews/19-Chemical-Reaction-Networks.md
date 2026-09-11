# Critique 19: Certified equilibria and conditional persistence in mass-action networks

[Revised problem](../PRDs/19-Chemical-Reaction-Networks.md) · [Preserved original](../archive/original-PRDs/19-Chemical-Reaction-Networks.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/19-Chemical-Reaction-Networks.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — persistence shortcut (32, 125–131).** Remaining positive at finite time is not persistence, and the stated unrestricted siphon “if and only if” is not a general algorithmic criterion. A sufficient theorem must carry its kinetic and boundedness assumptions.

2. **Major — multistationarity versus stability (27, 1102).** Multiple equilibria need not all be stable. The bistable Schlögl example typically has three positive equilibria, two stable, rather than the required two equilibria. The new exact polynomial benchmark makes this distinction checkable.

3. **Major — algebraic completeness (41, 55).** A Gröbner basis does not automatically isolate all positive real roots or eliminate continuous solution sets. Conservation classes and positivity restrictions are essential.

4. **Major — parameter independence (57–58).** Network deficiency is structural, but most dynamical conclusions depend on rates and totals unless a theorem quantifies over them. The rewrite separates those claims and avoids promised scalability to hundreds of species.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite mass-action network with specified rates and conservation totals, can all positive equilibria in one compatibility class be isolated and their local stability certified?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact structural facts and complete small fixed-parameter root enumeration. Strong: certified local stability and one applicable persistence theorem. Research extension: parameter-region proofs or a new structural criterion, with quantifiers stated explicitly.

## Sources supporting the corrections

- [Angeli, De Leenheer and Sontag, A Petri Net approach to persistence](https://arxiv.org/abs/q-bio/0608019) — Checkable sufficient conditions for persistence, with hypotheses.

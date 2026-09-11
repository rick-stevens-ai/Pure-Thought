# Critique 22: Bell inequalities with certified local and quantum value brackets

[Revised problem](../PRDs/22-Bell-Inequalities-Nonlocality.md) · [Preserved original](../archive/original-PRDs/22-Bell-Inequalities-Nonlocality.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/22-Bell-Inequalities-Nonlocality.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect convergence target (18, 41, 103–105).** NPA convergence is to commuting measurements. Treating this as general convergence to the finite-dimensional tensor-product set ignores a substantive distinction exposed by MIP*=RE.

2. **Major — optimal strategies (24–27, 909–911).** Moment feasibility alone does not supply states and measurements; extraction requires extra conditions. A constructive lower bound is now mandatory for optimality claims.

3. **Major — facet counts/rank (909, 922).** The eight CHSH variants are not eight inequivalent classes; a D-dimensional polytope facet has affine dimension D−1, so D+1 affinely independent saturating vertices would be impossible for a proper facet.

4. **Major — randomness and “exact” numbers.** Floating-point agreement and solver timing are not certificates. Bell violation alone does not specify a full secure randomness protocol. The replacement focuses on auditable value brackets.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed rational Bell functional in a small bipartite scenario, can its exact local value and a certified quantum lower/upper bracket be produced?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact CHSH lower and upper certificates. Strong: a preselected additional rational functional with a certified bracket, whether or not it closes. Research extension: a new inequality with normalized comparison, or device-independent entropy bounds with an explicit adversary and statistical model.

## Sources supporting the corrections

- [Navascués, Pironio and Acín, A convergent hierarchy of semidefinite programs](https://arxiv.org/abs/0803.4290) — NPA convergence to a commuting-measurement representation.

- [Ji et al., MIP*=RE](https://arxiv.org/abs/2001.04383) — Strict separation of the relevant correlation sets.

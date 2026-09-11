# Critique 08: Exact modular-data and symmetry-anomaly consistency tests

[Revised problem](../PRDs/08-Swampland-Modularity-Symmetries.md) · [Preserved original](../archive/original-PRDs/08-Swampland-Modularity-Symmetries.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/08-Swampland-Modularity-Symmetries.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — boundary and bulk symmetries confused (95–105, 618–620).** Boundary global symmetries can correspond to bulk gauge symmetries. Flagging the Ising global symmetry as a gravity inconsistency reverses the relevant distinction.

2. **Blocking — anomaly logic (101–115).** A nontrivial ’t Hooft anomaly can obstruct gauging without making the original theory inconsistent. Ω₂^Spin=Z₂ does not itself say that every nontrivial Arf theory has the asserted gravitational anomaly. The rewrite confines the check to explicit symmetry data.

3. **Major — rational versus algebraic arithmetic (88–93, 613–615).** Modular matrices need algebraic numbers, not only rationals. Finite modular matrices also presuppose a suitable rational chiral algebra.

4. **Major — unbounded classification and duplicates.** A central-charge ceiling alone does not define a finite RCFT search. The two 08 descriptions are consolidated, and higher-form/cobordism extensions require their spacetime dimension, symmetry type, and inflow theory before computation.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For supplied rational-CFT modular data and a specified finite internal symmetry, which algebraic consistency tests can be certified, and which failures obstruct a proposed gauging?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact modular-data checker on Ising and one second supplied model. Strong: one finite-group twisted-sector/gauging analysis. Research extension: classify a explicitly finite family of data or derive a conditional bulk consequence using a stated holographic theorem.

## Sources supporting the corrections

- [Harlow and Ooguri, Symmetries in quantum field theory and quantum gravity](https://arxiv.org/abs/1810.05338) — Bulk gauge versus boundary global symmetry, including higher-form extensions.

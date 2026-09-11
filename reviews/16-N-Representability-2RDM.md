# Critique 16: Certified energy brackets from reduced-density-matrix relaxations

[Revised problem](../PRDs/16-N-Representability-2RDM.md) · [Preserved original](../archive/original-PRDs/16-N-Representability-2RDM.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/16-N-Representability-2RDM.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — bound direction (43 versus 106).** Outer RDM relaxations provide lower bounds; physical wavefunctions provide upper bounds. The original states both directions without distinguishing feasible sets.

2. **Blocking — Q definition (26–30, 97–100).** Q is two-hole positivity, not a generic contracted 3-RDM. Its correct affine expression is essential to the SDP.

3. **Major — accuracy certificate (104–109).** A small optimization residual cannot bound distance to the exact ground-state energy without an upper witness or a proved relaxation-error bound.

4. **Major — exponential-wall claim (48–50).** Polynomially many RDM entries do not make representability easy; the general problem is QMA-complete. The revised goal is a rigorous bracket on a declared finite model, with model error separately acknowledged.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified finite fermionic Hamiltonian, how tight an independently certified ground-state energy interval can P,Q,G and selected higher-order RDM constraints produce?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: valid L,U brackets for small fixed models. Strong: quantify improvement from G or T constraints across a specified instance set. Research extension: a new valid constraint with demonstrated tightening, or a theorem on a tractable family; novelty needs comparison with existing hierarchies.

## Sources supporting the corrections

- [Liu, Christandl and Verstraete, N-representability is QMA-complete](https://arxiv.org/abs/quant-ph/0609125) — Complexity barrier to general exact representability.

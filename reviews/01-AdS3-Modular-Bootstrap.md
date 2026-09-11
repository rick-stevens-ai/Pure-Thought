# Critique 01: Extremal holomorphic CFT partition functions: certified necessary tests

[Revised problem](../PRDs/01-AdS3-Modular-Bootstrap.md) · [Preserved original](../archive/original-PRDs/01-AdS3-Modular-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/01-AdS3-Modular-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incompatible sectors and gaps (original lines 16–29, 49–100).** The text alternates between holomorphic and diagonal nonchiral partition functions, then uses c/12 for both. For the chosen holomorphic extremal problem, the first permitted primary is h=k+1. The rewrite fixes one sector and separates h from Δ.

2. **Blocking — incorrect character identity (49–57).** The η expression drops the q^(1/24) correction; the vacuum also requires removal of L₋₁. These errors change the spectrum being optimized. The purported large-c S-matrix at 84–86 cannot serve as a universal Virasoro transformation law and is removed.

3. **Blocking — feasibility is not construction (18–23, 104 onward).** Continuous LP/SDP feasibility does not impose integer multiplicities, and neither a truncated spectrum nor genus-one consistency proves OPE associativity. The replacement identifies exactly what a successful computation establishes.

4. **Major — certificate coverage.** A positive grid or finite series does not control the omitted spectrum. The finite cutoff is an output, while all-order claims require a theorem. This also makes a useful baseline possible without solving extremal-CFT existence.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For c=24k, which extremal holomorphic partition-function candidates pass exact genus-one spectral tests, and can any candidate be excluded by an additional, explicitly stated consistency condition?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact k=1,…,4 tables with explicitly finite coverage. Strong: one additional sewing/OPE condition with independently checked consequences. Research extension: a new exclusion theorem or a full CFT construction; neither is a required outcome or a promised deadline.

## Sources supporting the corrections

- [Witten, Three-Dimensional Gravity Revisited](https://arxiv.org/abs/0706.3359) — Motivation for the holomorphic extremal ansatz, not a proof that every candidate exists.

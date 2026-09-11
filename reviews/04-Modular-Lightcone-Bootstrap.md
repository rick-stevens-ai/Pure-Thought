# Critique 04: Lightcone bootstrap with controlled large-spin remainders

[Revised problem](../PRDs/04-Modular-Lightcone-Bootstrap.md) · [Preserved original](../archive/original-PRDs/04-Modular-Lightcone-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/04-Modular-Lightcone-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — modular invariance outside its domain (23, 45).** Thermal correlators on S^(d−1)×S¹ do not generally have the torus SL(2,Z) invariance asserted here. The revised d=3 problem uses crossing and Lorentzian analyticity only.

2. **Major — spectrum terminology (16, 31–41).** A large holographic gap normally concerns a selected single-trace sector, not all nonconserved operators; double-twist operators remain. Δ_gap−J_max is not a definition of a minimum twist. The scalar unitarity lower bound is also not a holographic gap prediction.

3. **Major — asymptotics versus finite bounds.** Lightcone expansions alone do not supply certified finite-spin exclusion plots. A remainder or additional input is required.

4. **Major — overlap and overclaim (29, 53–60).** The original mixes analytic scalar bootstrap, spinning bootstrap, modular constraints, and uniqueness of Einstein gravity. The replacement isolates a calculable question and preserves gravity interpretations as conditional extensions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For one scalar four-point function in a unitary CFT with d>2, what controlled bounds on large-spin double-twist data follow from specified low-twist exchanges?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: reproduce an established leading large-spin result with conventions checked. Strong: obtain one explicit remainder estimate, or identify the extra data needed to obtain it. Research extension: a tighter conditional bound on low-twist exchanges or bulk couplings with a stated holographic dictionary.

## Sources supporting the corrections

- [Fitzpatrick et al., The Analytic Bootstrap and AdS Superhorizon Locality](https://arxiv.org/abs/1212.3616) — Large-spin double-twist structure and analytic bootstrap.

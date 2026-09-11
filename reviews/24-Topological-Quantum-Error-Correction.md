# Critique 24: Topological code verification and noise-specific decoder evaluation

[Revised problem](../PRDs/24-Topological-Quantum-Error-Correction.md) · [Preserved original](../archive/original-PRDs/24-Topological-Quantum-Error-Correction.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/24-Topological-Quantum-Error-Correction.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — noise mismatch (17, 88, 1003).** A ~10.9% figure is not a universal depolarizing/circuit threshold. Noise channel, measurement assumptions, and decoder determine the comparison.

2. **Major — coefficient field (97–107).** CSS homology is computed over F₂; writing H₁=Z² without identifying the coefficient change confuses topological and binary-code data.

3. **Blocking — syndrome versus correction (110–113, 133–136).** Two chains with the same boundary may differ by a logical loop. Success requires trivial homology of error plus recovery.

4. **Major — certification boundary (49–52, 136).** Monte Carlo threshold fits are not algebraic proofs, and prohibiting numerical optimization conflicts with matching and simulation plans. The revision allows useful numerical work while reserving proof language for justified conclusions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can a toric-code family’s algebraic parameters be proved, and can a fixed decoder’s logical failure probability be evaluated under a precisely specified error model?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact construction and distance proof with a tested decoder. Strong: reproducible finite-size failure curves and uncertainty-aware threshold estimates. Research extension: a rigorous threshold bound or noisy-measurement/circuit model, with no transfer of numerical thresholds across noise conventions.

## Sources supporting the corrections

- [Dennis, Kitaev, Landahl and Preskill, Topological quantum memory](https://arxiv.org/abs/quant-ph/0110143) — Homological coding and model-dependent error-correction thresholds.

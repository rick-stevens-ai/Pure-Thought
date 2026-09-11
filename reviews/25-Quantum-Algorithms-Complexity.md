# Critique 25: Quantum query complexity with explicit oracle and output models

[Revised problem](../PRDs/25-Quantum-Algorithms-Complexity.md) · [Preserved original](../archive/original-PRDs/25-Quantum-Algorithms-Complexity.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/25-Quantum-Algorithms-Complexity.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — unrelativized separation (25, 41).** An oracle separation does not prove BQP≠BPP. The superscript and access model belong in every theorem statement.

2. **Major — mixed objectives (23–27).** Grover, HHL, walks and QAOA solve different input/output problems; a single “quantum advantage” benchmark cannot fairly compare them without cost models.

3. **Major — Grover rounding (79–81).** The optimal iteration rule must account for the −1/2 shift and nearest-integer choice; the stated floor can overshoot. The exact success formula is the acceptance test.

4. **Major — HHL/QAOA scaling (16, 1035, 1052).** Logarithmic dimension dependence alone omits input access, κ, error and readout costs. A finite sample of QAOA graphs does not establish an approximation guarantee. The rewritten core provides one well-posed theorem target and modular extensions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For unstructured search with a specified promise, can an implementation and an analytic lower bound establish a matched quantum query complexity?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: Grover construction and matched query bound. Strong: one second promise problem with matched assumptions and a certified upper/lower bracket. Research extension: a new restricted query bound or oracle result. Reimplementation of established algorithms is replication, not a new complexity separation.

## Sources supporting the corrections

- [Bennett et al., Strengths and Weaknesses of Quantum Computing](https://arxiv.org/abs/quant-ph/9701001) — Oracle lower-bound methodology for quantum search.

# Critique 21: Quantum LDPC instances with independently certified code parameters

[Revised problem](../PRDs/21-Quantum-LDPC-Codes.md) · [Preserved original](../archive/original-PRDs/21-Quantum-LDPC-Codes.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/21-Quantum-LDPC-Codes.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong concrete size (974–986).** Two 3×7 parity-check inputs give 7²+3²=58 qubits, not 90. The construction and ranks should determine the parameters.

2. **Major — distance shortcut (92–96, 60).** Dimension follows from linear algebra; minimum nontrivial logical weight generally requires harder optimization. The displayed dual-code distance shortcut is not an adequate universal CSS certificate.

3. **Major — finite versus asymptotic (38–42, 995–1005).** A few codes with good ratios cannot prove constant rate, linear distance, or a threshold. Ordinary hypergraph products do not automatically yield linear distance.

4. **Major — noise and overhead.** Pauli labels are discrete even though general quantum noise is richer. Decoder threshold and fault-tolerant overhead require a noise and measurement model. The rewrite removes unsupported hardware-overhead comparisons and recognizes existing asymptotically good constructions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified CSS construction, can code dimension and distance bounds be verified independently, and can decoder performance be measured under one explicit noise model?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact small product code and certified distance bracket. Strong: several instances plus honest finite-size decoder curves. Research extension: a new family tradeoff or proved decoder guarantee. Reproducing asymptotically good codes is not itself a new existence result.

## Sources supporting the corrections

- [Panteleev and Kalachev, Asymptotically Good Quantum and Locally Testable Classical LDPC Codes](https://arxiv.org/abs/2111.03654) — Existing asymptotically good quantum LDPC constructions; novelty baseline.

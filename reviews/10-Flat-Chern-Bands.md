# Critique 10: Flatness–locality–geometry tradeoffs for Chern bands

[Revised problem](../PRDs/10-Flat-Chern-Bands.md) · [Preserved original](../archive/original-PRDs/10-Flat-Chern-Bands.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/10-Flat-Chern-Bands.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incompatible design requirements (33–40; synthesis finite-range search).** Exact flatness, nonzero Chern number, and strictly local finite-range hopping cannot all be required under the cited theorem’s assumptions. The rewrite splits the problem into two valid tracks.

2. **Blocking — normalization (38, 86).** The two displayed constant-curvature formulas disagree and omit/misplace 2π. Defining C by its integral fixes F̄ uniquely.

3. **Major — unsupported objective (22–24, 39–40).** The stated stability ratio is not a general criterion proving an FCI ground state. Geometry, bandwidth, and an interacting many-body gap are different claims.

4. **Major — exact versus approximate (787–805).** A tiny numerical bandwidth or curvature variance does not establish exact flatness or uniformity. The revision asks for certified enclosures and distinguishes achievable values from global optima.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For fixed orbital count, hopping range, and gap normalization, how small can a Chern band’s bandwidth and geometric nonuniformity be, with certified bounds?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one certified near-flat Chern model and a correct locality audit. Strong: global objective bounds on one compact family or a rigorous truncation tradeoff. Research extension: optimal geometry within a specified class; many-body FCI stability requires an added interacting Hamiltonian and filling.

## Sources supporting the corrections

- [Chen et al., Flat Chern-band locality obstruction](https://arxiv.org/abs/1311.4956) — No-go theorem for simultaneous exact flatness, nonzero Chern number, and strictly local hopping.

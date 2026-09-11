# Critique 26: A posteriori certification of an invariant Hamiltonian torus

[Revised problem](../PRDs/26-KAM-Theory-Planetary-Stability.md) · [Preserved original](../archive/original-PRDs/26-KAM-Theory-Planetary-Stability.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/26-KAM-Theory-Planetary-Stability.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — finite Diophantine “certificate” (126).** Checking |k|≤K_max cannot prove an infinite small-divisor condition. Rational decimal frequencies are not exact irrational inputs, and a full uncertainty box generally contains resonances.

2. **Major — missing nondegeneracy and reduction (21–25, 74).** The Kepler action formula and derivative are inconsistent, and planetary Kepler Hamiltonians are properly degenerate. A generic nondegenerate theorem cannot be applied without reduction/averaging work.

3. **Major — torus versus actual trajectory (29–40, 113–120).** Surviving tori are deformed embeddings; existence does not place the measured solar system on one. Measure and time claims require additional arguments.

4. **Major — iteration as proof (713–718).** A shrinking residual over a few steps does not establish convergence. The new task ties every computational quantity to one quantitative theorem and allows honest failure of its sufficient conditions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For an explicitly specified analytic near-integrable Hamiltonian and an approximate torus, can every hypothesis of a quantitative KAM theorem be verified?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: unperturbed certificate and complete arithmetic/theorem setup. Strong: a nonzero-ε torus certificate. Research extension: a larger perturbation range or a reduced planetary model. One invariant torus does not certify a neighborhood of arbitrary observed initial conditions or a phase-space measure bound.

## Sources supporting the corrections

- [Valvo and Locatelli, Hamiltonian Control of Magnetic Field Lines: Computer Assisted Results Proving the Existence of KAM Barriers](https://arxiv.org/abs/2101.07785) — Example of a complete computer-assisted Hamiltonian KAM workflow; use a theorem matching the chosen formulation.

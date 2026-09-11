# Critique 20: Path-integral thermodynamics with a transparent error budget

[Revised problem](../PRDs/20-Ab-Initio-Path-Integrals.md) · [Preserved original](../archive/original-PRDs/20-Ab-Initio-Path-Integrals.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/20-Ab-Initio-Path-Integrals.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — inconsistent temperature/action (70–92).** The exact identity for exp(−βH) is confused with a Trotter approximation, and the action/ring-polymer formulas mix β and β/P. The revised density states one consistent convention.

2. **Blocking — estimators (110–120).** The original primitive energy omits the spring contribution, while nonlinear observables at the centroid do not equal bead-averaged observables. The oscillator provides a decisive regression benchmark.

3. **Major — proof claims (45, 56–59, 132–140).** P-extrapolation, energy conservation, and estimated autocorrelation time do not prove ergodicity or a rigorous total error bound. Statistical evidence is useful but must be labeled.

4. **Major — exactness and applications.** Finite-P PIMD with an approximate electronic surface has several approximations; equilibrium imaginary-time sampling does not directly certify real-time tunneling rates. The rewrite makes those limits part of the task definition.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified one-dimensional quantum Hamiltonian, can path-integral sampling reproduce equilibrium energy and position moments with separately assessed discretization and statistical errors?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: oscillator thermodynamics within predeclared statistical and bias tolerances. Strong: one anharmonic error-controlled result. Research extension: a small ab initio system with a full error budget; rigorous certification requires more than SCF convergence and stable trajectories.

## Sources supporting the corrections

- [Ceriotti et al., Efficient stochastic thermostatting of path integral molecular dynamics](https://arxiv.org/abs/1009.1045) — Normal-mode sampling and stochastic thermostat methods.

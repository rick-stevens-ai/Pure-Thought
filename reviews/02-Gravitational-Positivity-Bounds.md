# Critique 02: Conditional positivity bounds for a specified gravitational amplitude

[Revised problem](../PRDs/02-Gravitational-Positivity-Bounds.md) · [Preserved original](../archive/original-PRDs/02-Gravitational-Positivity-Bounds.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/02-Gravitational-Positivity-Bounds.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — operator basis (18–27).** In four-dimensional pure-gravity on-shell scattering, the displayed curvature-squared coefficients are not independent observables: field redefinitions and the Gauss–Bonnet combination matter. An amplitude coefficient avoids pretending that three basis-dependent numbers are separately bounded.

2. **Blocking — positivity and dispersion (44–68).** Fixed-t imaginary parts are not generally positive; unitarity acts on partial waves. The displayed dispersion formula omits essential crossed-channel/subtraction structure. The spin-2 pole makes the forward limit especially delicate, as the cited primary papers demonstrate.

3. **Major — placeholder physics (94–113).** The schematic stu/M_pl² expression is not a graviton helicity amplitude and lacks its pole structure. It must not seed a physical optimization.

4. **Major — interpretation (73–84).** Numerical unitarity checks do not establish all-energy consistency or a UV completion. The revised goal is a conditional necessary bound, with the Regge and infrared assumptions attached to the result.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Under a fully stated infrared prescription and high-energy assumption, what certified inequality constrains one selected low-energy coefficient of scalar scattering coupled to gravity?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one correct sum rule and its gravity-free limit. Strong: a numerically useful certified conditional bound with a finite positive margin. Research extension: improve the bound or relax a high-energy assumption; label any sharpness claim with both matching constructions and upper/lower evidence.

## Sources supporting the corrections

- [Tokuda, Aoki and Hirano, Gravitational positivity bounds](https://arxiv.org/abs/2007.15009) — Regge assumptions and finite gravitational corrections.

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Why naive pole-subtracted forward positivity can fail.

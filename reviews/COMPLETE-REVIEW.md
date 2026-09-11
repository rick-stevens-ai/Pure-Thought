# Complete revised problems and detailed critiques

2026-09-11. This reading copy combines all 30 revised canonical problems, their critiques, and 12 supplemental topics. Every problem includes an everyday explanation, key-term definitions, motivation and a scoped account of what progress would mean. [Review overview](REVIEW.md) · [Catalog](../PRDs/README.md) · [Evidence standard](EVIDENCE-STANDARD.md) · [Synthesis crosswalk](SYNTHESIS-CROSSWALK.md)

## Contents

- [01. Extremal holomorphic CFT partition functions: certified necessary tests](#problem-01)
- [02. Conditional positivity bounds for a specified gravitational amplitude](#problem-02)
- [03. Celestial amplitudes: distributional crossing and soft-limit consistency](#problem-03)
- [04. Lightcone bootstrap with controlled large-spin remainders](#problem-04)
- [05. Positive-geometry tests for a fixed gravity integrand](#problem-05)
- [06. S-matrix bootstrap with an explicit gravitational infrared prescription](#problem-06)
- [07. Stress-tensor bootstrap: a sector-specific gap bound in three dimensions](#problem-07)
- [08. Exact modular-data and symmetry-anomaly consistency tests](#problem-08)
- [09. Certified Chern phases in a bounded tight-binding family](#problem-09)
- [10. Flatness–locality–geometry tradeoffs for Chern bands](#problem-10)
- [11. Certified topological bands in an idealized Maxwell medium](#problem-11)
- [12. Rigidity and boundary modes of periodic Maxwell frames](#problem-12)
- [13. Higher-order topology with explicit boundary and symmetry assumptions](#problem-13)
- [14. Certified Weyl nodes and slice topology in a lattice model](#problem-14)
- [15. Symmetry-indicator algebra and the limits of band diagnosis](#problem-15)
- [16. Certified energy brackets from reduced-density-matrix relaxations](#problem-16)
- [17. Complete enumeration of a precisely defined molecular-graph class](#problem-17)
- [18. Certified optimal-transport bounds for strictly correlated electrons](#problem-18)
- [19. Certified equilibria and conditional persistence in mass-action networks](#problem-19)
- [20. Path-integral thermodynamics with a transparent error budget](#problem-20)
- [21. Quantum LDPC instances with independently certified code parameters](#problem-21)
- [22. Bell inequalities with certified local and quantum value brackets](#problem-22)
- [23. Entanglement certification with explicit inconclusive outcomes](#problem-23)
- [24. Topological code verification and noise-specific decoder evaluation](#problem-24)
- [25. Quantum query complexity with explicit oracle and output models](#problem-25)
- [26. A posteriori certification of an invariant Hamiltonian torus](#problem-26)
- [27. Central configurations in a bounded, symmetry-reduced N-body problem](#problem-27)
- [28. Explicit finite-time action confinement from a normal form](#problem-28)
- [29. Autocatalytic sets: maximality, irreducibility and dynamical viability](#problem-29)
- [30. An exact finite genotype–phenotype map and its mutation graph](#problem-30)

---

<a id="problem-01"></a>

# 01. Extremal holomorphic CFT partition functions: certified necessary tests

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/01-AdS3-Modular-Bootstrap.md) · [Original description](../archive/original-PRDs/01-AdS3-Modular-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Imagine trying to identify a musical instrument from a list of the notes it can produce. Some lists are impossible because the notes do not fit the instrument’s mathematical rules. Others pass those checks, but that still does not tell us how to build the instrument. Here the “instrument” is a quantum theory and the “notes” are its allowed states. We ask whether exceptionally sparse lists of states obey the consistency rules. Such sparse theories are candidates for describing a simplified universe containing gravity with no additional matter.

### Key terms

- **Conformal field theory (CFT)** — A quantum theory whose laws respect transformations that preserve angles, including changes of scale; it describes systems with no preferred length scale.

- **AdS₃** — A three-dimensional spacetime with a particular negative curvature, used as a mathematically tractable setting for gravity.

- **Holomorphic** — Depending on one complex coordinate rather than both it and its complex conjugate; this is a special restriction on the theories studied here.

- **Central charge c** — A number characterizing aspects of a CFT’s symmetry and number of degrees of freedom. Here c=24k, with k a positive integer.

- **Partition function** — A generating function that packages the theory’s state energies or scaling weights and the number of states at each level.

- **Primary and descendant** — A primary is a basic type of excitation; descendants are related excitations generated from it by symmetry.

- **Extremal spectrum** — A spectrum with the largest prescribed initial gap before additional primary excitations, under this particular ansatz.

- **Modular invariance** — The requirement that different equivalent descriptions of a torus give the same partition function.

### Why this matters

Quantum gravity is hard partly because we do not know which mathematically plausible theories are internally consistent. These tests can eliminate candidates without experiments or a complete construction. They also connect gravity to the arithmetic of modular functions: counting quantum states becomes a problem about exact number sequences.

### What progress would mean

A useful first result is a trustworthy table of candidates and any exact reason for rejecting one. Passing the table’s tests would identify a candidate worth further study; constructing the full theory would remain a much stronger achievement.

## Core question

For c=24k, which extremal holomorphic partition-function candidates pass exact genus-one spectral tests, and can any candidate be excluded by an additional, explicitly stated consistency condition?

## Scope and assumptions

Start with bosonic holomorphic theories, a unique vacuum, c=24k, and k=1,…,4. This is a deliberately restricted version of the original gravity question. Use h for holomorphic weight; do not identify it with the full scaling dimension Δ=h+h̄. The extremal ansatz has no nonvacuum Virasoro primary with h≤k. Treat the nonchiral modular bootstrap as a separate extension with independent (h,h̄) data.

Keep the motivating existence question, but make the first deliverable a necessary-condition test. A modular function with integer coefficients is not a constructed CFT or a demonstrated gravity dual.

## Mathematical target

With q=exp(2πiτ), c>1, use

$$χ_0=q^{-c/24}\prod_{n=2}^{∞}(1-q^n)^{-1},\qquad χ_h=q^{h-(c-1)/24}/η(τ)\quad(h>0).$$

Construct the unique degree-k polynomial in J=j−744 whose expansion agrees with χ₀ through q⁰. Write Zₖ=χ₀+Σ_{h≥k+1}dₕχₕ and extract primary multiplicities exactly through a declared cutoff H, initially H=100. Prove modular invariance from the polynomial representation, not from evaluations at a few τ. A negative or noninteger dₕ excludes the stated candidate; positivity through H is only a finite test.

## Required outputs and proof obligations

Export the polynomial coefficients, exact q-series, primary multiplicities, conventions, cutoff, and a checker that independently expands η and J. An exclusion must name the violated necessary condition. An additional genus-two or OPE test must state its own assumptions and how its constraints follow from CFT consistency.

## Validation and rejection controls

Recover J=q⁻¹+196884q+… at k=1 and d₂=196883 after subtracting the vacuum descendant. Test the checker with a deliberately altered coefficient. Check agreement of two series-construction routes. Do not report all-order coefficient positivity without a tail theorem.

## Milestones and research extension

Baseline: exact k=1,…,4 tables with explicitly finite coverage. Strong: one additional sewing/OPE condition with independently checked consequences. Research extension: a new exclusion theorem or a full CFT construction; neither is a required outcome or a promised deadline.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Witten, Three-Dimensional Gravity Revisited](https://arxiv.org/abs/0706.3359) — Motivation for the holomorphic extremal ansatz, not a proof that every candidate exists.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-02"></a>

# 02. Conditional positivity bounds for a specified gravitational amplitude

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/02-Gravitational-Positivity-Bounds.md) · [Original description](../archive/original-PRDs/02-Gravitational-Positivity-Bounds.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A low-energy theory is like a map drawn at a coarse scale: it describes what we can resolve while summarizing finer details in a few adjustable numbers. Not every choice of those numbers can come from a consistent microscopic world. This problem asks whether basic rules about probabilities, cause and effect, and scattering restrict one such number when gravity is present. Gravity’s long reach makes the usual argument harder, so part of the task is making the assumptions precise.

### Key terms

- **Effective field theory (EFT)** — An approximate description valid over a stated range of energies, with unresolved physics represented by additional terms.

- **Wilson coefficient** — The number multiplying one of those additional terms; it measures the strength of a correction to the simpler theory.

- **Scalar particle** — A particle with zero intrinsic spin, chosen here to simplify the scattering calculation.

- **Scattering amplitude** — A quantum quantity used to calculate probabilities for incoming particles to emerge in particular outgoing states.

- **Positivity bound** — An inequality on theory parameters derived from a quantity that must be nonnegative under specified assumptions.

- **Infrared (IR) and ultraviolet (UV)** — The low-energy or long-distance regime, and the high-energy or short-distance regime, respectively.

- **Pole and subtraction** — A pole is a singular term in a mathematical expression; a subtraction removes a specified contribution so the remaining quantity can be analyzed.

- **Dispersion relation** — A mathematical relation connecting an amplitude’s behavior in different energy regions.

- **Regge assumption** — An assumption about how scattering behaves at very high energy with momentum transfer held fixed.

### Why this matters

These bounds can tell us that an apparently sensible low-energy model cannot fit particular assumptions about its underlying physics. That is valuable when direct access to microscopic gravity is unavailable. A reliable bound also prevents researchers from interpreting an artifact of a singular calculation as a physical restriction.

### What progress would mean

The first success would be one defensible inequality with its assumptions attached. It would narrow the parameter space of candidate theories, without claiming that every surviving theory has a complete microscopic realization.

## Core question

Under a fully stated infrared prescription and high-energy assumption, what certified inequality constrains one selected low-energy coefficient of scalar scattering coupled to gravity?

## Scope and assumptions

Use four-dimensional scattering of one identical real massive scalar coupled to Einstein gravity. This replaces the redundant schematic pure-gravity R² coefficient list with an amplitude-defined observable. Set m=1 as the mass unit; specify κ, the EFT subtraction scale, perturbative order, and whether the target is a tree coefficient or a renormalized coefficient. Graviton helicity amplitudes are a later extension.

The first task is to establish a valid sum rule. Do not run an optimization until the pole subtraction, massless cuts, crossed channel, and large-contour contribution have been derived.

## Mathematical target

Let ν=s−2m²+t/2, so s↔u sends ν↦−ν. Define B(ν,t) by subtracting the explicitly calculated light poles and the chosen low-energy cut contributions from A(ν,t). Define b₂(t)=(1/2)∂²_νB(0,t).

Derive b₂(t)=I_UV(t)+R_∞(t) with all kernels and subtractions written out. Choose either a finite-t smeared positive sum rule or a justified regulated forward limit. Nonnegativity must be established for the resulting partial-wave kernel; Im A(s,t)≥0 at arbitrary t<0 is not an admissible substitute. State a bound on R_∞ or retain it as an explicit unknown.

## Required outputs and proof obligations

Return a derivation dossier, normalized coefficient definition, assumptions, and either a conditional inequality b₂≥L or an explanation of the missing hypothesis. If optimizing, export a dual functional with certified kernel positivity and all truncation errors. A surviving coefficient is “not excluded by these tests,” not “UV complete.”

## Validation and rejection controls

Check s↔u symmetry, dimensional consistency, and pole residues. Reproduce the nongravitational forward positivity argument in the κ→0 limit. Use a fully specified example amplitude to test the subtraction algebra. Perturb a certificate beyond its margin and verify rejection.

## Milestones and research extension

Baseline: one correct sum rule and its gravity-free limit. Strong: a numerically useful certified conditional bound with a finite positive margin. Research extension: improve the bound or relax a high-energy assumption; label any sharpness claim with both matching constructions and upper/lower evidence.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Tokuda, Aoki and Hirano, Gravitational positivity bounds](https://arxiv.org/abs/2007.15009) — Regge assumptions and finite gravitational corrections.

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Why naive pole-subtracted forward positivity can fail.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-03"></a>

# 03. Celestial amplitudes: distributional crossing and soft-limit consistency

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/03-Celestial-CFT-Bootstrap.md) · [Original description](../archive/original-PRDs/03-Celestial-CFT-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Instead of describing a collision by particle energies and momenta, imagine recording the directions in which particles enter or leave on a sphere surrounding the experiment. Celestial methods reorganize scattering data into quantities associated with that sphere. The ambition is a new language for gravity. Before using that language to discover new theories, this problem checks that it faithfully preserves familiar facts about a simple graviton collision.

### Key terms

- **Celestial sphere** — The sphere of directions seen from a point, used here to label asymptotic particle directions rather than a physical shell.

- **Graviton** — The hypothetical quantum excitation of the gravitational field used in perturbative quantum-gravity calculations.

- **Helicity** — The component of a particle’s spin along its direction of motion.

- **Mellin transform** — A mathematical transformation that reorganizes dependence on energy into dependence on scaling behavior.

- **Conformal weight Δ** — A label describing scaling behavior in the celestial description; it can be complex.

- **Distribution** — A generalized function, such as a delta function, understood through integration against test functions rather than ordinary pointwise values.

- **Crossing** — The relationship between different scattering processes obtained by reinterpreting an incoming particle as an outgoing antiparticle, with appropriate continuation.

- **Soft limit** — The limit in which one particle’s energy approaches zero; a soft theorem controls the resulting leading behavior.

- **Regulator** — A temporary modification that makes a singular expression well defined, accompanied by a rule for removing or interpreting it.

### Why this matters

A successful reformulation could expose symmetries or constraints that are difficult to see in ordinary scattering variables. That would give theorists another way to investigate gravity. But a change of mathematical language is useful only if singularities, conservation laws and limiting cases survive the translation correctly.

### What progress would mean

A verified example would supply a dependable foundation for later celestial calculations. It would establish that specific identities translate correctly, rather than establish a complete two-dimensional theory of the universe.

## Core question

Can a regulated celestial transform of a specified four-graviton tree amplitude reproduce Lorentz covariance, crossing, and the leading conformally soft residue in one consistent convention?

## Scope and assumptions

Fix four-dimensional Einstein gravity at tree level, a nonzero four-graviton helicity amplitude, incoming/outgoing signs εᵢ, and pᵢ=εᵢωᵢq(zᵢ,z̄ᵢ). Declare metric, helicity, normalization, and analytic-continuation conventions. The target is an identity pipeline for amplitudes, not a classification of celestial CFTs.

Use Δᵢ=1+iλᵢ on the principal series where appropriate and hᵢ=(Δᵢ+Jᵢ)/2, h̄ᵢ=(Δᵢ−Jᵢ)/2. Identify explicitly which continuations away from that contour are used for soft residues.

## Mathematical target

Define the transformed distribution by

$$\widetilde{\mathcal A}=\prod_i\int_0^∞dω_i\,ω_i^{Δ_i-1}\;δ^{(4)}\!\left(\sum_i ε_iω_iq_i\right)\mathcal M_4.$$

State a regulator or test-function space making the integral meaningful, and specify regulator removal or meromorphic continuation. From a leading momentum-space term proportional to ω_s⁻¹, derive the Mellin pole at Δ_s=1 in this convention. Subleading residues require their own derivation. Crossing must transform helicities, signs, and support, not just permute ordinary Euclidean functions.

## Required outputs and proof obligations

Produce exact momentum-space expressions, regulated transform formulas, support conditions, and symbolic covariance/crossing/soft identities. If a transform cannot be defined under the chosen assumptions, return a precise obstruction at that step. A positive bootstrap is an extension only after deriving an applicable inner product and positivity statement.

## Validation and rejection controls

Check the original amplitude’s factorization first. Test Mellin formulas against analytic one-variable examples in their convergence strips. Verify soft residues before and after transformation under stated limit-interchange hypotheses. Use smeared distributions for numerical checks; never evaluate a delta distribution as an ordinary function.

## Milestones and research extension

Baseline: one four-point transform with explicit support and checked leading soft residue. Strong: a second helicity/process example and a verified crossing relation. Research extension: derive a justified positive constraint in a restricted subsector before attempting an SDP island.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Adamo et al., Celestial amplitudes and conformal soft theorems](https://www.pure.ed.ac.uk/ws/files/121747399/Adamo_2019_Class._Quantum_Grav._36_205018_1_.pdf) — Conformally soft graviton operators at Δ=1 and Δ=0.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 03: Celestial amplitudes: distributional crossing and soft-limit consistency

[Revised problem](../PRDs/03-Celestial-CFT-Bootstrap.md) · [Preserved original](../archive/original-PRDs/03-Celestial-CFT-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/03-Celestial-CFT-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — missing distributional specification (36–45, 94–113).** A Mellin transform of a stripped amplitude without momentum conservation and convergence rules is incomplete. The numerical quadrature sketch does not define the multidimensional distribution that is actually needed.

2. **Blocking — imported positivity (18–23, 57–75).** Principal-series celestial data cannot simply inherit the positive OPE-coefficient squares of a unitary Euclidean two-dimensional CFT. An SDP needs a proved positivity structure, not an asserted positive norm.

3. **Major — soft poles (60–64).** The generic Δ→0 double-pole formula does not reproduce the leading graviton soft pole in the stated Mellin convention. Deriving the pole from the energy power prevents convention mistakes.

4. **Major — scope (15–28, 79–86).** “Consistent space” and “islands” assume tools whose foundational prerequisites are themselves research questions. A distributionally correct identity pipeline is a meaningful prerequisite and preserves the original direction.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can a regulated celestial transform of a specified four-graviton tree amplitude reproduce Lorentz covariance, crossing, and the leading conformally soft residue in one consistent convention?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one four-point transform with explicit support and checked leading soft residue. Strong: a second helicity/process example and a verified crossing relation. Research extension: derive a justified positive constraint in a restricted subsector before attempting an SDP island.

## Sources supporting the corrections

- [Adamo et al., Celestial amplitudes and conformal soft theorems](https://www.pure.ed.ac.uk/ws/files/121747399/Adamo_2019_Class._Quantum_Grav._36_205018_1_.pdf) — Conformally soft graviton operators at Δ=1 and Δ=0.


---

<a id="problem-04"></a>

# 04. Lightcone bootstrap with controlled large-spin remainders

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/04-Modular-Lightcone-Bootstrap.md) · [Original description](../archive/original-PRDs/04-Modular-Lightcone-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A quantum correlation can be calculated by grouping the participating excitations in different ways, and all valid groupings must agree. That agreement can force some excitations to exist and restrict their properties. This problem studies the pattern at large spin, where useful approximations become available. The central challenge is to say not only what the leading approximation predicts, but also how large its error could be.

### Key terms

- **Correlation function** — A quantity describing how field measurements at different spacetime points are related.

- **Four-point function** — A correlation involving four field insertions; it can be reorganized into different pairings.

- **Bootstrap** — A method that constrains a theory by requiring its own consistency rules to agree, rather than starting from a complete microscopic model.

- **Scaling dimension Δ** — A measure of how a field or excitation changes when all lengths are rescaled.

- **Spin ℓ** — A label describing how an excitation transforms under rotations.

- **Lightcone limit** — A limit in which two spacetime points approach a separation that light could traverse; it organizes which contributions dominate the correlation function.

- **Twist** — The combination Δ−ℓ, useful for ordering contributions near a lightlike separation.

- **Operator product expansion (OPE)** — A rule expressing the effect of nearby field insertions as a sum of other operators with specified coefficients.

- **Double-twist family** — A sequence of large-spin operators whose dimensions approach a characteristic pattern related to two external operators.

- **Remainder bound** — A guaranteed limit on the terms omitted from an approximation.

### Why this matters

Consistency can reveal features shared by many strongly interacting theories, even when their detailed dynamics cannot be solved. Large-spin patterns are also useful in theories proposed to describe gravity through a lower-dimensional boundary. Controlled errors turn suggestive asymptotic patterns into statements that can actually constrain a finite calculation.

### What progress would mean

The baseline would reproduce an established large-spin pattern with unambiguous conventions. A stronger result would show how far that pattern can deviate at specified spins, making later comparisons and exclusions more trustworthy.

## Core question

For one scalar four-point function in a unitary CFT with d>2, what controlled bounds on large-spin double-twist data follow from specified low-twist exchanges?

## Scope and assumptions

Start in d=3 with an identical scalar φ of fixed dimension Δφ. Specify the normalization of C_T, the stress-tensor OPE coefficient fixed by its Ward identity, and an explicit twist gap above the low-twist exchanges retained. Use a large-C_T expansion only when a factorization assumption and its error order are stated.

This challenge studies analytic large-spin behavior. Challenge 07 instead studies the spinning stress-tensor correlator. Torus modular invariance is restricted to a separate d=2 variant.

## Mathematical target

For a fixed scalar block convention, define F_{Δ,ℓ}=v^{Δφ}g_{Δ,ℓ}(u,v)−u^{Δφ}g_{Δ,ℓ}(v,u), and use F_id+Σ λ²F=0. Twist is τ=Δ−ℓ. The double-twist families have Δ_{n,ℓ}=2Δφ+2n+ℓ+γ_{n,ℓ} asymptotically.

Choose n=0 initially and derive a retained contribution γ̂_{0,ℓ}, together with a bound |γ_{0,ℓ}−γ̂_{0,ℓ}|≤R(ℓ) for ℓ≥ℓ₀ under explicit spectral/Regge assumptions. If only an asymptotic series is obtained, label it as such and do not manufacture a finite-spin bound.

## Required outputs and proof obligations

Export block normalizations, low-twist input, the derivation or inversion formula, coefficients, and a decomposition of R into omitted exchanges, expansion terms, and numerical errors. State clearly whether the result is an identity, a conditional bound, or an asymptotic estimate.

## Validation and rejection controls

Use generalized-free-field crossing as an algebraic benchmark, while noting that it is not by itself the target local CFT with a finite stress tensor. Recover double-twist accumulation in a known analytic example. Verify every uniform estimate on the specified cross-ratio domain and all spins covered by the claim.

## Milestones and research extension

Baseline: reproduce an established leading large-spin result with conventions checked. Strong: obtain one explicit remainder estimate, or identify the extra data needed to obtain it. Research extension: a tighter conditional bound on low-twist exchanges or bulk couplings with a stated holographic dictionary.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Fitzpatrick et al., The Analytic Bootstrap and AdS Superhorizon Locality](https://arxiv.org/abs/1212.3616) — Large-spin double-twist structure and analytic bootstrap.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-05"></a>

# 05. Positive-geometry tests for a fixed gravity integrand

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/05-Positive-Geometry-Gravity.md) · [Original description](../archive/original-PRDs/05-Positive-Geometry-Gravity.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Particle-collision formulas often contain many terms whose cancellations hide a simpler structure. In some theories, a geometric shape organizes the answer: its boundaries encode the ways a collision can split into simpler events. This problem asks whether a tightly specified class of shapes can do the same for one gravity calculation. It treats geometry as a possible explanation for the structure of a formula, not as a literal picture of spacetime.

### Key terms

- **Integrand** — The mathematical expression to be integrated to obtain a quantity such as a scattering amplitude.

- **Loop** — A feature of a quantum correction involving internal momenta that must be integrated over; one loop is the first such level here.

- **Supergravity** — A theory combining gravity with supersymmetry, a proposed relation between types of particles; N=8 specifies the chosen four-dimensional model.

- **Positive geometry** — A region with mathematical boundary properties that determine a special differential form.

- **Canonical form** — That distinguished differential form, whose singularities and boundary residues obey precise recursive rules.

- **Residue** — The coefficient describing the leading contribution of a simple pole; here it encodes boundary information.

- **Factorization** — The way a scattering process separates into simpler processes near certain physical singularities.

- **Ansatz** — A deliberately restricted class of candidate expressions or geometric constructions searched for a solution.

- **Pole at infinity** — A singularity revealed when a coordinate is taken to infinity, which must be included in a global comparison.

### Why this matters

A geometric organization could explain cancellations and suggest more efficient calculations or new identities in quantum gravity. Even a carefully limited impossibility theorem is informative: it tells us which geometric proposals are too restrictive and where a different idea is needed.

### What progress would mean

Success would mean either a verified shape-to-form correspondence or a proof that the declared candidate class cannot work. Failure to find a shape would not rule out every possible geometric description of gravity.

## Core question

For a specified low-loop supergravity integrand, does a precisely bounded class of candidate positive geometries reproduce its differential form, including every boundary and pole at infinity?

## Scope and assumptions

Start with the four-point one-loop N=8 supergravity integrand, in a declared four-dimensional representation and helicity sector. Record any regulator and the limits of four-dimensional cut reconstruction. Derive or reproduce the target before searching. Tree-level and scalar-box canonical forms are calibration examples.

Define the search class before claiming an obstruction: ambient coordinates, dimension, allowed rational maps, inequalities, permitted external prefactors, and bounds on degree/number of boundaries. “No geometry exists” is not a finite computational question without such restrictions.

## Mathematical target

Compare the top-form Ω_target=I(ℓ;k)d⁴ℓ to f(k)π_*Ω_G for a declared map π and external factor f. State whether equality is literal rational-form equality or equality modulo a specified class of terms integrating to zero. Work on a defined real slice with orientations.

For a standard positive geometry, Ω_G has logarithmic boundary poles and recursively normalized residues. Check all finite boundaries, their intersections, and infinity. Distinguish a canonical differential form from the symbol of an integrated polylogarithmic function.

## Required outputs and proof obligations

Return either an explicit G, map, canonical form, and exact equality check, or a proof excluding the declared ansatz class. A failed search returns “unresolved within budget.” A dlog decomposition alone is a candidate diagnostic, not a geometry certificate.

## Validation and rejection controls

Verify simplex/box normalization independently. Check factorization and generalized cuts of the target. Rational reconstruction must be followed by an exact polynomial identity after denominators are cleared, with singular loci accounted for. Check orientation signs and residues at infinity explicitly.

## Milestones and research extension

Baseline: exact target and pole/residue audit. Strong: one verified representation or one restricted no-go theorem. Research extension: broaden the geometry class or multiplicity; explain how the result changes understanding of gravity amplitudes rather than treating a computational failure as a universal obstruction.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Arkani-Hamed, Bai and Lam, Positive Geometries and Canonical Forms](https://arxiv.org/abs/1703.04541) — Definition of logarithmic canonical forms and boundary recursion.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 05: Positive-geometry tests for a fixed gravity integrand

[Revised problem](../PRDs/05-Positive-Geometry-Gravity.md) · [Preserved original](../archive/original-PRDs/05-Positive-Geometry-Gravity.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/05-Positive-Geometry-Gravity.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — undefined search universe (35–45, 91–95).** An unrestricted geometry-existence question cannot be ruled out by an unsuccessful symbol search. The rewrite makes both positive and negative answers mathematically scoped.

2. **Major — form versus integral (53–63).** A symbol is not a wedge of dlog one-forms, and integrated branch-cut data are not interchangeable with poles of an integrand. Canonical-form residue recursion applies to a specified differential form.

3. **Major — kinematics (49).** All physical Mandelstam invariants need not be positive simultaneously. Positivity refers to the chosen real geometry and coordinates.

4. **Major — validation (84–89).** Agreement on selected cuts or numerical integration can miss contact terms, infinity poles, or representation ambiguities. The new deliverable requires an exact equality statement and an exhaustive boundary audit within the chosen ansatz.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified low-loop supergravity integrand, does a precisely bounded class of candidate positive geometries reproduce its differential form, including every boundary and pole at infinity?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact target and pole/residue audit. Strong: one verified representation or one restricted no-go theorem. Research extension: broaden the geometry class or multiplicity; explain how the result changes understanding of gravity amplitudes rather than treating a computational failure as a universal obstruction.

## Sources supporting the corrections

- [Arkani-Hamed, Bai and Lam, Positive Geometries and Canonical Forms](https://arxiv.org/abs/1703.04541) — Definition of logarithmic canonical forms and boundary recursion.


---

<a id="problem-06"></a>

# 06. S-matrix bootstrap with an explicit gravitational infrared prescription

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/06-Nonperturbative-S-matrix-Bootstrap.md) · [Original description](../archive/original-PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Suppose we do not know a theory’s microscopic machinery, but do know that collision probabilities must be sensible and different descriptions of the same process must agree. Can those rules alone rule out a proposed interaction strength? This problem searches the space of scattering possibilities rather than choosing one detailed microscopic theory. Because gravity acts over long distances and allows very low-energy radiation, defining the scattering question correctly is itself part of the work.

### Key terms

- **S-matrix** — The object relating incoming quantum states to outgoing states in a scattering experiment.

- **Partial wave** — A contribution to scattering organized by angular momentum, which makes some probability constraints easier to state.

- **Unitarity** — Conservation of total probability; an allowed scattering process cannot produce a total probability greater than one.

- **Elastic scattering** — Scattering that stays within the chosen particle channel without producing additional particles or excitations.

- **Crossing symmetry** — Relations between amplitudes for processes with differently assigned incoming and outgoing particles.

- **Infrared prescription** — A stated way to handle low-energy radiation and long-distance singularities.

- **Outer relaxation** — A larger, easier-to-test set containing all valid possibilities under the assumptions. Excluding a point from it can exclude the original possibility.

- **Inner ansatz** — A selected subset of candidate amplitudes; failing to find a solution there need not exclude other amplitudes.

- **Dual certificate** — An independently checkable mathematical witness establishing a bound or inconsistency.

### Why this matters

The method could identify restrictions shared by many possible microscopic theories. That is useful when no single theory is established. Its reliability depends on not mistaking a limitation of the search formula, grid or cutoff for a limitation imposed by physics.

### What progress would mean

A certified exclusion would eliminate a parameter choice under the stated scattering assumptions. The gravity-free reference case first checks the machinery; adding a defensible treatment of gravity would be the important next step.

## Core question

Can one certify an exclusion for a low-energy scalar-scattering parameter using a crossing-symmetric partial-wave relaxation whose infrared and truncation assumptions are explicit?

## Scope and assumptions

Begin with identical massive scalars in four dimensions and no gravity to validate the machinery. The gravitational stage must first define an infrared-finite observable or a justified regulator/pole treatment, including massless radiation. Keep one scalar mass and one amplitude coefficient fixed; optimize a second coefficient in a declared normalization.

This is complementary to Challenge 02: the focus is amplitude feasibility and partial-wave constraints rather than deriving a low-energy positivity sum rule. A nongravitational baseline does not count as solving the gravitational stage.

## Mathematical target

Choose A=16πΣ_{ℓ≥0}(2ℓ+1)a_ℓ(s)P_ℓ(cosθ), s+t+u=4m², ρ=√(1−4m²/s), and S_ℓ=1+2iρa_ℓ. Then |S_ℓ|≤1 is equivalent to Im a_ℓ≥ρ|a_ℓ|². Equality needs an actually elastic sector.

State how crossing is imposed and distinguish an outer relaxation of necessary constraints from an inner finite amplitude ansatz. A dual exclusion from an outer relaxation transfers to the full problem only when every physical amplitude maps into that relaxation. Bound energy, angular-momentum, and basis tails; checking a finite mesh is insufficient.

## Required outputs and proof obligations

Export the amplitude conventions, IR prescription, finite optimization problem, mapping from the continuum problem, and dual certificate with its positivity/error margins. Feasible ansatz points are examples satisfying the verified constraints only. If the IR construction or tail control fails, state which conclusion remains unavailable.

## Validation and rejection controls

Check free scattering and a specified analytic nongravitational benchmark. Verify the unitarity disk algebra independently. Test between grid points and beyond cutoffs using proved bounds. Demonstrate that known admissible examples are not falsely excluded. Increasing precision is a diagnostic; rigorous enclosures are needed for certification.

## Milestones and research extension

Baseline: certified finite nongravitational relaxation. Strong: justify an IR treatment and certify one conditional gravitational exclusion. Research extension: improved continuum bounds or a constructive amplitude with a precisely delimited consistency claim.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Infrared and analyticity issues that must be resolved before importing standard bounds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 06: S-matrix bootstrap with an explicit gravitational infrared prescription

[Revised problem](../PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Preserved original](../archive/original-PRDs/06-Nonperturbative-S-matrix-Bootstrap.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/06-Nonperturbative-S-matrix-Bootstrap.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — elastic window with gravity (68–79).** Massless radiation removes the simple massive-theory inelastic threshold assumption. The equality form of elastic unitarity cannot be silently reused. The relaxed inequality and the IR observable must be justified.

2. **Blocking — singular partial waves.** Long-range graviton exchange makes naive four-dimensional angular projection problematic. The proposed all-purpose Roy-like equation (89–92) is not a derivation of the required kernels.

3. **Major — inner versus outer approximation.** Infeasibility of a restricted amplitude ansatz excludes only that ansatz. This distinction is essential for every claimed no-go theorem.

4. **Major — missing continuum coverage.** A finite partial-wave/energy grid cannot establish all-energy unitarity. The new specification requires analytic tail bounds and otherwise reports a finite relaxation, preserving meaningful progress without overstating it.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can one certify an exclusion for a low-energy scalar-scattering parameter using a crossing-symmetric partial-wave relaxation whose infrared and truncation assumptions are explicit?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: certified finite nongravitational relaxation. Strong: justify an IR treatment and certify one conditional gravitational exclusion. Research extension: improved continuum bounds or a constructive amplitude with a precisely delimited consistency claim.

## Sources supporting the corrections

- [Alberte et al., Positivity Bounds and the Massless Spin-2 Pole](https://arxiv.org/abs/2007.12667) — Infrared and analyticity issues that must be resolved before importing standard bounds.


---

<a id="problem-07"></a>

# 07. Stress-tensor bootstrap: a sector-specific gap bound in three dimensions

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/07-Extremal-CFTs-Stress-Tensor.md) · [Original description](../archive/original-PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Energy and momentum must be accounted for consistently in a quantum theory. The stress tensor is the mathematical object that tracks them. By examining how four stress-tensor insertions relate to one another, we can test whether a proposed theory is missing an excitation it ought to contain. The question here is how large a particular gap in the spectrum can be before the consistency rules contradict each other.

### Key terms

- **Stress tensor T** — The field that describes the local flow and distribution of energy and momentum.

- **Conformal field theory** — A quantum theory with symmetry under angle-preserving transformations, including rescaling lengths.

- **Central-charge parameter C_T** — The normalization of the stress tensor’s two-point correlation, used to characterize the theory in a fixed convention.

- **TTT data** — Coefficients describing a three-stress-tensor correlation; they contain information beyond C_T.

- **Scalar sector** — The set of exchanged operators with zero spin, chosen as the focus of the gap question.

- **Parity-even** — Unchanged in sign under the specified spatial-reflection transformation.

- **Spectral gap** — Here, the threshold in scaling dimension before the first nonidentity operator in the selected sector appears, not necessarily a material’s energy gap.

- **Tensor structure** — One of the distinct ways the directional indices of spinning fields can enter a correlation function.

- **Positive semidefinite matrix** — A matrix whose quadratic form is nonnegative for every vector; it expresses the needed positivity for multiple coupled tensor structures.

### Why this matters

The stress tensor is present in a broad class of local theories, so restrictions derived from it can have wide reach. Such restrictions help map which strongly interacting quantum theories are possible. They can also constrain proposed boundary descriptions of gravity, where the stress tensor has a special role.

### What progress would mean

A verified gap bound would show that a theory with the declared inputs must contain a certain kind of excitation below a threshold. It would not determine the entire spectrum or prove that a gravity theory is unique.

## Core question

At fixed stress-tensor two- and three-point data, can crossing exclude a proposed gap in the parity-even scalar sector of T×T in a unitary three-dimensional CFT?

## Scope and assumptions

Fix d=3, parity invariance, a unique conserved stress tensor, a normalization for C_T, and the remaining independent TTT parameter. The initial observable is the first parity-even nonidentity scalar appearing with nonzero coupling in T×T. It is not a bound on every operator in the theory. State separate gaps for any other restricted spin/parity sectors.

Use the full conserved spinning correlator. A scalar block toy implementation is permitted for learning but is not the target stress-tensor bootstrap.

## Mathematical target

Write the crossing equations in a nonredundant tensor basis as

$$V_{id}+V_T(C_T,t_4)+\sum_{\mathcal O}\lambda_{\mathcal O}^{\mathsf T}V_{Δ,ℓ,\pi}\lambda_{\mathcal O}=0.$$

The OPE vectors account for multiple tensor structures. In d=3, a conserved symmetric-traceless spin-ℓ primary has Δ=ℓ+1, not Δ=3 for all spins. For a proposed scalar gap g, seek α with α(V_id+V_T)>0 and α(V_{Δ,ℓ,π}) positive semidefinite on every allowed exchange. This yields a contradiction and therefore excludes g under the specified assumptions.

## Required outputs and proof obligations

Export tensor/block conventions, conservation and Ward identities, fixed input data, derivative basis, spin treatment, and functional coefficients. Attach rigorous block-approximation and high-spin control. If only the finite approximant is checked, label the result accordingly.

## Validation and rejection controls

Recover a selected published stress-tensor benchmark using exactly its conventions and assumptions. Check the allowed region rather than the excluded region. Verify matrix positivity, not just individual scalar components. Free scalar/fermion data provide normalization tests; numerical agreement alone is not a proof of the continuum bound.

## Milestones and research extension

Baseline: reproduce one finite benchmark and publish the assumptions. Strong: turn one exclusion into a certified result with omitted-sector control. Research extension: improve a bound, add another sector, or obtain a conditional extremal spectrum. Saturation does not construct a CFT.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Dymarsky et al., The 3d Stress-Tensor Bootstrap](https://arxiv.org/abs/1708.05718) — Conserved tensor crossing and sector-specific numerical gap bounds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 07: Stress-tensor bootstrap: a sector-specific gap bound in three dimensions

[Revised problem](../PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Preserved original](../archive/original-PRDs/07-Extremal-CFTs-Stress-Tensor.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/07-Extremal-CFTs-Stress-Tensor.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect current dimensions (main original 60–64).** Higher-spin conserved currents have Δ=ℓ+d−2. Assigning all of them Δ=d violates the representation-theory input.

2. **Blocking — scalar treatment of spinning crossing (49–75).** TTTT has multiple tensor structures and OPE matrices; a single squared coefficient and G(u,v)=G(v,u) do not specify its crossing system.

3. **Blocking — functional domain/sign (83–91; short duplicate 39–49).** Positivity must hold for allowed operators. The original asks for positivity in the excluded region and alternates lower and upper gap bounds. The rewrite gives an explicit contradiction convention.

4. **Major — ambiguous target and duplicate (both 07 files).** A gap to a conserved current, a scalar gap, and a single-trace holographic gap are different observables. Both originals now resolve to one canonical scalar-sector problem; gravity uniqueness is not inferred from it.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **At fixed stress-tensor two- and three-point data, can crossing exclude a proposed gap in the parity-even scalar sector of T×T in a unitary three-dimensional CFT?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: reproduce one finite benchmark and publish the assumptions. Strong: turn one exclusion into a certified result with omitted-sector control. Research extension: improve a bound, add another sector, or obtain a conditional extremal spectrum. Saturation does not construct a CFT.

## Sources supporting the corrections

- [Dymarsky et al., The 3d Stress-Tensor Bootstrap](https://arxiv.org/abs/1708.05718) — Conserved tensor crossing and sector-specific numerical gap bounds.


---

<a id="problem-08"></a>

# 08. Exact modular-data and symmetry-anomaly consistency tests

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/08-Swampland-Modularity-Symmetries.md) · [Original description](../archive/original-PRDs/08-Swampland-Modularity-Symmetries.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A proposed quantum theory comes with mathematical rules for combining excitations and transforming its descriptions. These rules should fit together like the pieces of a puzzle. This problem builds exact checks for that fit and then asks a second question: can a chosen global symmetry be made into a gauge symmetry? A theory can be consistent even when this second operation is obstructed, so the two verdicts must remain separate.

### Key terms

- **Rational CFT** — A conformal field theory with finitely many basic representation types relative to a specified chiral algebra.

- **Character** — A generating function counting states within one representation of that algebra.

- **Modular S and T matrices** — Matrices describing how the characters change under two basic transformations of the torus.

- **Fusion rules** — Rules specifying which excitation types can occur when two types are combined.

- **Global symmetry** — A transformation applied consistently throughout a system that leaves its laws unchanged.

- **Gauging** — Promoting an appropriate symmetry to a local redundancy and adding the corresponding gauge structure.

- **Anomaly** — An obstruction to implementing a classical symmetry or a proposed gauging consistently in the quantum setting; an ’t Hooft anomaly can exist in a valid theory.

- **Cocycle** — Algebraic data satisfying a compatibility identity, used here to encode an anomaly.

- **Bulk and boundary** — The interior gravitational theory and its lower-dimensional boundary description; symmetries can have different interpretations in the two.

### Why this matters

Exact consistency checks can detect false candidate data before researchers invest in a full construction. They also clarify which symmetries can be gauged and which require additional structure, such as anomaly cancellation or inflow from another system. This is useful both for studying quantum field theories and for interpreting holographic gravity proposals.

### What progress would mean

The first contribution would be a dependable checker with known passing and failing examples. A failure would identify a particular algebraic inconsistency or gauging obstruction, rather than issuing an unsupported verdict about all of quantum gravity.

## Core question

For supplied rational-CFT modular data and a specified finite internal symmetry, which algebraic consistency tests can be certified, and which failures obstruct a proposed gauging?

## Scope and assumptions

Use a bosonic rational CFT with a declared chiral algebra and finitely many characters. Input exact algebraic S,T matrices, vacuum label, and a nonnegative integer multiplicity matrix M. Start with the Ising model. An optional finite 0-form group G requires separate defect/twisted-sector data and an anomaly cocycle; these cannot be inferred from the untwisted partition function alone.

The result concerns the supplied CFT data and gauging problem. A general bulk quantum-gravity compatibility verdict is outside the baseline.

## Mathematical target

Check S†S=I, symmetry of S where assumed, S²=C, (ST)³=C in the declared T convention, M₀₀=1, MS=SM, MT=TM, and Mᵢⱼ∈Z≥0. Check Verlinde coefficients Σ_a S_ia S_ja S*_ka/S_0a are nonnegative integers.

For a finite symmetry cocycle ω∈Z³(G,U(1)), check the cocycle identity exactly. To certify anomaly-free gauging in the stated setting, supply an explicit 2-cochain β with δβ=ω, or prove its nonexistence in a complete cohomology computation. Distinguish this obstruction from inconsistency of an anomalous but valid boundary theory.

## Required outputs and proof obligations

Return pass/fail/unresolved for each named axiom, with exact arithmetic traces. Include the number field and embeddings. Even passing every modular-data test gives necessary algebraic consistency only; full CFT sewing and realizability are separate obligations.

## Validation and rejection controls

Verify Ising S entries including 1/√2 in algebraic arithmetic. Reject altered modular relations, noninteger multiplicities, and a failed cocycle identity. Accept the existence of a boundary global symmetry rather than flagging it as automatically inconsistent. Cross-check a small cyclic-group cohomology example.

## Milestones and research extension

Baseline: exact modular-data checker on Ising and one second supplied model. Strong: one finite-group twisted-sector/gauging analysis. Research extension: classify a explicitly finite family of data or derive a conditional bulk consequence using a stated holographic theorem.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Harlow and Ooguri, Symmetries in quantum field theory and quantum gravity](https://arxiv.org/abs/1810.05338) — Bulk gauge versus boundary global symmetry, including higher-form extensions.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 08: Exact modular-data and symmetry-anomaly consistency tests

[Revised problem](../PRDs/08-Swampland-Modularity-Symmetries.md) · [Preserved original](../archive/original-PRDs/08-Swampland-Modularity-Symmetries.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/08-Swampland-Modularity-Symmetries.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — boundary and bulk symmetries confused (95–105, 618–620).** Boundary global symmetries can correspond to bulk gauge symmetries. Flagging the Ising global symmetry as a gravity inconsistency reverses the relevant distinction.

2. **Blocking — anomaly logic (101–115).** A nontrivial ’t Hooft anomaly can obstruct gauging without making the original theory inconsistent. Ω₂^Spin=Z₂ does not itself say that every nontrivial Arf theory has the asserted gravitational anomaly. The rewrite confines the check to explicit symmetry data.

3. **Major — rational versus algebraic arithmetic (88–93, 613–615).** Modular matrices need algebraic numbers, not only rationals. Finite modular matrices also presuppose a suitable rational chiral algebra.

4. **Major — unbounded classification and duplicates.** A central-charge ceiling alone does not define a finite RCFT search. The two 08 descriptions are consolidated, and higher-form/cobordism extensions require their spacetime dimension, symmetry type, and inflow theory before computation.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For supplied rational-CFT modular data and a specified finite internal symmetry, which algebraic consistency tests can be certified, and which failures obstruct a proposed gauging?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact modular-data checker on Ising and one second supplied model. Strong: one finite-group twisted-sector/gauging analysis. Research extension: classify a explicitly finite family of data or derive a conditional bulk consequence using a stated holographic theorem.

## Sources supporting the corrections

- [Harlow and Ooguri, Symmetries in quantum field theory and quantum gravity](https://arxiv.org/abs/1810.05338) — Bulk gauge versus boundary global symmetry, including higher-form extensions.


---

<a id="problem-09"></a>

# 09. Certified Chern phases in a bounded tight-binding family

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/09-Topological-Band-Theory.md) · [Original description](../archive/original-PRDs/09-Topological-Band-Theory.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two solids can both resist ordinary electrical conduction in their interiors yet differ in how their quantum states fit together. In a topological phase, that global organization cannot be smoothly changed without passing through a gap closing or leaving the assumptions that protect it. Here we build simple lattice models and prove which parameter ranges belong to which phase. The task is closer to drawing an exact map of a model than predicting a particular material.

### Key terms

- **Hamiltonian H** — The mathematical operator specifying a quantum system’s energy and dynamics.

- **Tight-binding model** — A lattice model in which particles occupy localized orbitals and move between them through specified hopping terms.

- **Orbital** — One of the local quantum states included at a lattice site or within a unit cell.

- **Band** — A range of allowed energies labeled by crystal momentum in a periodic system.

- **Band gap** — An energy separation between the selected occupied bands and the next bands.

- **Brillouin zone** — The space of distinct crystal momenta for a periodic lattice.

- **Chern number** — An integer measuring a global twist in the selected quantum states across a two-dimensional Brillouin zone.

- **Occupied projector** — An operator selecting the band states being treated as occupied.

- **Phase diagram** — A map showing which kind of behavior occurs for each model-parameter range.

### Why this matters

Topological models explain why some properties depend on global structure rather than microscopic details. A certified phase map provides reliable examples for theory and simulation, and it helps identify which ingredients a proposed design actually needs. Proving a gap everywhere is important because a small missed gap closing can invalidate a computed topological label.

### What progress would mean

A useful result would give explicit models, guaranteed parameter ranges and checked Chern numbers. These could guide later physical designs, while leaving material stability, fabrication and experimental behavior to additional work.

## Core question

Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?

## Scope and assumptions

Begin with the two-band square-lattice model H_m(k)=sin(k_x)σ_x+sin(k_y)σ_y+[m+cos(k_x)+cos(k_y)]σ_z, k∈[−π,π]². Use one occupied band and a fixed orientation. Analyze rational parameter intervals separated from m=−2,0,2.

For subsequent searches fix dimension, orbitals, filling, hopping support, coefficient domain, and symmetry representation. A rational grid is finite; a continuous coefficient box requires a different completeness argument. “All Hamiltonians for a space group” is not an enumerable specification.

## Mathematical target

For a general family H(k)=Σ_R t_Re^{ik·R}, impose t_−R=t_R† and the stated symmetry equations. Prove E_{r+1}(k)−E_r(k)≥δ>0 for all k and all parameters claimed. With P the occupied projector, choose

$$C=\frac{1}{2πi}\int_{BZ}\operatorname{Tr}P[∂_xP,∂_yP],d²k.$$

Certify C through an analytic degree/homotopy argument or validated integration with an error enclosure containing a unique integer, including discretization and projector errors.

## Required outputs and proof obligations

Return model coefficients, gap lower bounds, invariant and sign convention, certified parameter coverage, and uncovered boxes. A minimality claim must specify whether it minimizes orbitals, hopping range, or coefficients and must exclude every smaller object in that declared class.

## Validation and rejection controls

Recover gap closings at −2,0,2 and trivial/nontrivial intervals on either side. Reverse orientation and verify C changes sign. A fine-grid lattice Chern calculation is a useful cross-check, but must not substitute for the continuous gap proof. Check a time-reversal-invariant occupied bundle has zero total Chern number.

## Milestones and research extension

Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Chen et al., The impossibility of exactly flat non-trivial Chern bands in strictly local periodic tight binding models](https://arxiv.org/abs/1311.4956) — A concrete example of why range/locality assumptions change realizability claims.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 09: Certified Chern phases in a bounded tight-binding family

[Revised problem](../PRDs/09-Topological-Band-Theory.md) · [Preserved original](../archive/original-PRDs/09-Topological-Band-Theory.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/09-Topological-Band-Theory.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — unbounded classification (37–41, 803–813).** Fixing a space group does not bound the hopping range or continuous parameters. Claims to enumerate every model and prove minimality have no finite search universe.

2. **Blocking — false orbital lower bounds (811–813).** Higher Chern number does not generally require |C|+1 bands; two-band maps can have higher degree when suitable hopping harmonics are allowed. Range restrictions are indispensable to any proposed lower bound.

3. **Major — numerical integer versus certified topology (89–94, 790–793).** Discretized curvature and finite differences are not exact continuum certificates without gap and error control.

4. **Major — physical extrapolation (45–53).** A mathematical model does not imply a stable synthesizable material. The revision makes the model atlas the result and keeps materials relevance conditional.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

## Sources supporting the corrections

- [Chen et al., The impossibility of exactly flat non-trivial Chern bands in strictly local periodic tight binding models](https://arxiv.org/abs/1311.4956) — A concrete example of why range/locality assumptions change realizability claims.


---

<a id="problem-10"></a>

# 10. Flatness–locality–geometry tradeoffs for Chern bands

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/10-Flat-Chern-Bands.md) · [Original description](../archive/original-PRDs/10-Flat-Chern-Bands.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

In an ordinary energy band, particles can have quite different energies depending on how they move. A nearly flat band reduces those energy differences, potentially making interactions between particles more influential. We want that band to also have a nontrivial topological structure. The problem asks how well these goals can be combined when particles are allowed to hop only a limited distance and the model has a limited number of orbitals.

### Key terms

- **Flat band** — A band whose energy is constant across momentum space; a nearly flat band has a small but nonzero variation.

- **Bandwidth W** — The difference between the largest and smallest energy in the selected band.

- **Gap Δ** — The minimum energy separation between that band and other bands.

- **Flatness ratio W/Δ** — A way to compare bandwidth to isolation from other bands; smaller values mean a flatter, well-separated band.

- **Hopping range** — The greatest lattice distance over which the model allows direct hopping.

- **Chern band** — A band with a nonzero Chern number, indicating nontrivial global topology.

- **Berry curvature** — A local measure of how quantum states twist as momentum changes; its integral determines the Chern number in the stated convention.

- **Quantum metric** — A measure of how distinguishable nearby momentum-dependent quantum states are.

- **Spectral flattening** — A mathematical change to band energies that preserves the chosen states but generally changes the spatial range of hopping.

### Why this matters

Flat topological bands are useful settings for studying collective quantum behavior driven by interactions. But there are tradeoffs: restricting the model’s spatial range changes which flatness properties are possible. Quantifying those tradeoffs can show whether a proposed improvement is attainable or requires a more complicated model.

### What progress would mean

Progress would provide certified examples and limits within a defined design family. Favorable single-particle geometry would make a model interesting to investigate further; it would not by itself prove an interacting fractional topological phase.

## Core question

For fixed orbital count, hopping range, and gap normalization, how small can a Chern band’s bandwidth and geometric nonuniformity be, with certified bounds?

## Scope and assumptions

Choose a two-dimensional isolated rank-one band, target C=1, and a compact coefficient family with range R and fixed orbital positions. Initially use a specified two-band family extending Challenge 09. Impose a positive gap floor and a norm bound to prevent trivial energy rescaling.

Maintain two distinct tracks: finite-range approximate flatness, and exact spectral flattening with generally non-finite-range hopping. Do not require exact nonzero-Chern flatness with strictly finite-range hopping under the no-go theorem’s hypotheses.

## Mathematical target

Define W=max_k E_n−min_k E_n and Δ=min_k min_{j≠n}|E_j−E_n|. Optimize W/Δ. For the fixed Brillouin-zone coordinates define g_ij=Re〈∂_iu|(1−P)|∂_ju〉 and F with the same Chern convention as Challenge 09. Then C=(1/2π)∫F and F̄=2πC/Area(BZ).

Report separately Var(F)=Area⁻¹∫(F−F̄)² and D_g=Area⁻¹∫(tr g−|F|). These are diagnostics under a specified Euclidean coordinate metric, not a universal fractional-Chern-insulator stability score.

## Required outputs and proof obligations

Export coefficient/domain constraints, W upper and Δ lower bounds, certified C, geometric enclosures, and optimization gaps. An achieved objective gives an upper bound on the minimum; a global lower bound needs a covering/relaxation proof. For infinite-range flattening, give decay and truncation estimates.

## Validation and rejection controls

Verify normalization ∫F=2πC and the pointwise metric inequality in the chosen convention. Check that truncating a spectrally flattened model generally reintroduces bandwidth. Include a trivial exactly flat band as a negative topology test. Compare candidate objectives using identical units, embedding, and norm constraints.

## Milestones and research extension

Baseline: one certified near-flat Chern model and a correct locality audit. Strong: global objective bounds on one compact family or a rigorous truncation tradeoff. Research extension: optimal geometry within a specified class; many-body FCI stability requires an added interacting Hamiltonian and filling.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Chen et al., Flat Chern-band locality obstruction](https://arxiv.org/abs/1311.4956) — No-go theorem for simultaneous exact flatness, nonzero Chern number, and strictly local hopping.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-11"></a>

# 11. Certified topological bands in an idealized Maxwell medium

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/11-Photonic-Topological-Crystals.md) · [Original description](../archive/original-PRDs/11-Photonic-Topological-Crystals.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A repeating optical structure changes which light waves can travel through it. With suitable nonreciprocal response, it may also support light that travels along an interface in a preferred direction. This problem asks whether a precisely defined ideal electromagnetic medium really has the required band structure and topology. It separates a mathematically guaranteed wave effect from the later task of finding a material that realizes the assumed response.

### Key terms

- **Maxwell equations** — The equations governing classical electric and magnetic fields.

- **Periodic medium** — A medium whose properties repeat from cell to cell.

- **Permittivity ε and permeability μ** — Quantities describing how a medium responds to electric and magnetic fields; they can be direction-dependent tensors.

- **Nonreciprocity** — Response that breaks the usual interchange symmetry between source and receiver; here it provides the required time-reversal-breaking mechanism.

- **Photonic band** — An allowed family of electromagnetic wave frequencies across crystal momenta.

- **Transverse constraint** — The condition selecting physical wave fields consistent with the relevant divergence equation.

- **Chern number** — An integer characterizing the topology of an isolated group of wave bands.

- **Interface mode** — A wave concentrated near the boundary between two media.

- **Lossless and nondispersive** — Idealizations in which energy is not absorbed and the constitutive response is taken independent of frequency.

### Why this matters

Controlling where and in which direction waves travel is a basic aim of optical engineering. Topology offers one mechanism for robust interface behavior under appropriate conditions. A certified Maxwell calculation would also expose whether an apparent effect is caused by numerical artifacts or by assumptions a real material cannot satisfy.

### What progress would mean

The first meaningful achievement would be a validated ideal medium with a proved gap and topological index. It could inform later waveguide designs, but performance in a lossy, frequency-dependent device would require a separate physical model.

## Core question

Can a specified lossless periodic electromagnetic model support a rigorously isolated band group with nonzero Chern number, and can its interface modes be validated?

## Scope and assumptions

Choose a two-dimensional periodic, lossless, frequency-independent constitutive model with Hermitian positive ε(r), μ(r), specified boundary conditions, and a mechanism breaking physical time reversal. Treat the constitutive tensors as mathematical inputs. Real-frequency dispersion, loss, and fabrication feasibility are extensions requiring additional physical data.

Start by validating a reciprocal scalar dielectric benchmark with zero total Chern number for a time-reversal-invariant band group; then introduce an explicitly nonreciprocal tensor.

## Mathematical target

In a formulation supporting the chosen model, solve

$$(∇+ik)×μ^{-1}(r)(∇+ik)×u=ω²ε(r)u,$$

in units with c=1, subject to the appropriate transverse constraint. Use the ε-weighted inner product or an equivalent Hermitian formulation. Specify an isolated positive-frequency band group; photons do not require an electron-like filling assumption. Prove a frequency gap over the entire Brillouin zone and certify the projector invariant in the correct metric.

## Required outputs and proof obligations

Return ideal geometry and constitutive tensors, solver discretization, eigenvalue enclosures, spurious-mode controls, gap and topology certificates. Interface calculations must state both media and termination. Report robustness only for an explicit perturbation class and bound that preserves the required gap/symmetry.

## Validation and rejection controls

Check homogeneous-medium dispersion, mesh/basis convergence, Hermiticity, and transverse modes. Separate Galerkin error from k-space interpolation error. Verify reciprocal controls cannot acquire a net Chern number from numerical artifacts. Compare an interface spectrum with the bulk-index difference without claiming immunity to arbitrary disorder.

## Milestones and research extension

Baseline: validated reciprocal eigenproblem. Strong: one certified ideal nonreciprocal topological design. Research extension: a realistic dispersive constitutive model or gap optimization within a bounded geometry family. Fabrication drawings alone do not certify material behavior.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Haldane and Raghu, Directional optical waveguides with broken time-reversal symmetry](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.013904) — Nonreciprocal media as the mechanism for photonic quantum-Hall-like edge modes.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 11: Certified topological bands in an idealized Maxwell medium

[Revised problem](../PRDs/11-Photonic-Topological-Crystals.md) · [Preserved original](../archive/original-PRDs/11-Photonic-Topological-Crystals.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/11-Photonic-Topological-Crystals.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — missing nonreciprocal constitutive physics (68–98).** A real scalar dielectric model does not implement the claimed gyromagnetic Chern phase. The relevant tensor and inner product must be included.

2. **Major — electronic Z₂ shortcut (95–98).** Ordinary photonic time reversal squares to +1; an electronic parity formula cannot be transplanted without an appropriate additional symmetry structure.

3. **Major — “exact” Maxwell solution (21–25, 56–60).** Exact governing equations do not imply exact numerical eigenvalues. Discretization, spurious longitudinal modes, and continuum coverage need validation.

4. **Major — fabrication and robustness (37–42, 740–754).** Generic dielectric rods and STL files do not establish nonreciprocity, telecom performance, or a universal 15% disorder tolerance. The new problem certifies a mathematical medium first and identifies the extra evidence required for engineering claims.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can a specified lossless periodic electromagnetic model support a rigorously isolated band group with nonzero Chern number, and can its interface modes be validated?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: validated reciprocal eigenproblem. Strong: one certified ideal nonreciprocal topological design. Research extension: a realistic dispersive constitutive model or gap optimization within a bounded geometry family. Fabrication drawings alone do not certify material behavior.

## Sources supporting the corrections

- [Haldane and Raghu, Directional optical waveguides with broken time-reversal symmetry](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.013904) — Nonreciprocal media as the mechanism for photonic quantum-Hall-like edge modes.


---

<a id="problem-12"></a>

# 12. Rigidity and boundary modes of periodic Maxwell frames

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/12-Topological-Mechanical-Metamaterials.md) · [Original description](../archive/original-PRDs/12-Topological-Mechanical-Metamaterials.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A framework of bars or springs can be rigid in one place and easy to deform in another. Sometimes a carefully arranged repeating framework has soft motions concentrated at its boundary. The question is whether we can predict and prove those motions from the exact geometry, rather than simply observing them in a simulation. Counting connections helps, but the direction of each connection matters too.

### Key terms

- **Frame** — A network of sites joined by bars or springs, used as an ideal mechanical model.

- **Degree of freedom** — An independently variable coordinate of the system’s motion.

- **Isostatic or Maxwell frame** — A frame balanced between degrees of freedom and constraints in the relevant counting sense; details of zero modes and self stress still matter.

- **Zero mode** — An infinitesimal displacement requiring no restoring energy in the linearized model.

- **State of self stress** — A pattern of internal bond tensions that balances forces without external loads.

- **Compatibility matrix C** — The linear map from small site displacements to bond extensions.

- **Stiffness matrix K** — The operator relating displacements to restoring forces in the linear approximation.

- **Winding number** — An integer counting how a complex quantity winds around zero along a closed path; here it helps characterize the periodic frame.

- **Termination** — The precise way a periodic structure is cut to make a boundary.

### Why this matters

Geometry-controlled softness could help explain and design structures with directional compliance, boundary motion or localized mechanical response. The scientific value is understanding which behaviors follow from a robust index and which are accidental features of one geometry. That distinction matters before translating an ideal spring model into a physical structure.

### What progress would mean

A useful result would give a frame, an exact count of its allowed soft motions and a proof of where they concentrate. Material failure, nonlinear deformation and manufacturing tolerance would remain additional engineering questions.

## Core question

For a geometrically specified isostatic frame and boundary termination, can the zero-mode count and localization be proved from its compatibility matrix?

## Scope and assumptions

Use unstressed central-force springs with exact rational or algebraic site coordinates, positive spring constants, and explicit periodic bond offsets. Fix the unit-cell gauge, dimension, masses, and boundary termination. Begin with a small frame and a known periodic Maxwell lattice.

Connectivity alone is insufficient: bond directions determine the compatibility matrix. Distinguish rigid-body motions, floppy modes, and states of self stress. Claims concern linearized ideal mechanics.

## Mathematical target

Let C map displacements to bond extensions and Q=C†. For a finite free frame,

$$n_0−n_{ss}=dN−N_b,\qquad K=C^{†}\operatorname{diag}(k_b)C,$$

with dynamical matrix M⁻¹ᐟ²KM⁻¹ᐟ². For a square periodic C(k), define winding (2πi)⁻¹∮d log det C(k) only along loops where the determinant is nonzero. Account for acoustic translation zeros and the local boundary count when translating bulk winding into an edge index.

## Required outputs and proof obligations

Export positions, bonds, C, exact rank/kernel calculations, self-stress basis, winding certificate, and a boundary-specific mode count. For localization, provide a transfer-matrix/root-modulus bound separated from the unit circle. Specify perturbations preserving the nonvanishing condition and positivity of spring constants.

## Validation and rejection controls

Check one elementary underconstrained and one overconstrained frame against the index equation. Verify K is positive semidefinite and translations are zero modes where appropriate. Compare two terminations of the same bulk. Reject winding calculations crossing a zero of det C.

## Milestones and research extension

Baseline: exact Maxwell–Calladine counts on finite frames. Strong: one periodic frame with certified winding and termination-dependent edge localization. Research extension: a bounded family with robust mode counts. Elastic beams, prestress, nonlinear stability, and manufacturing tolerance require separate models.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Kane and Lubensky, Topological Boundary Modes in Isostatic Lattices](https://arxiv.org/abs/1308.0554) — Compatibility matrices, topological indices, and boundary modes.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 12: Rigidity and boundary modes of periodic Maxwell frames

[Revised problem](../PRDs/12-Topological-Mechanical-Metamaterials.md) · [Preserved original](../archive/original-PRDs/12-Topological-Mechanical-Metamaterials.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/12-Topological-Mechanical-Metamaterials.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — counting formula (22–27, 100).** The original replaces the Maxwell–Calladine difference between zero modes and self stresses with an incorrect count. It can mislabel even simple rigid structures.

2. **Blocking — mechanical operator (75–78).** The displayed dynamical matrix does not provide a reliable central-force construction. Building C from bond geometry and K=C†kC supplies a checkable definition.

3. **Major — graph-only topology (38, 61–64).** Frames with the same graph can have different rigidity and polarization. Geometric embedding and unit-cell conventions are part of the input.

4. **Major — boundary and stability claims (87–103).** A Berry phase alone does not give a universal zero-mode count, and a passive unstressed stiffness matrix does not have generic negative-frequency-squared spectral flow. Boundary indices and transfer decay replace the unsupported shortcut.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a geometrically specified isostatic frame and boundary termination, can the zero-mode count and localization be proved from its compatibility matrix?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact Maxwell–Calladine counts on finite frames. Strong: one periodic frame with certified winding and termination-dependent edge localization. Research extension: a bounded family with robust mode counts. Elastic beams, prestress, nonlinear stability, and manufacturing tolerance require separate models.

## Sources supporting the corrections

- [Kane and Lubensky, Topological Boundary Modes in Isostatic Lattices](https://arxiv.org/abs/1308.0554) — Compatibility matrices, topological indices, and boundary modes.


---

<a id="problem-13"></a>

# 13. Higher-order topology with explicit boundary and symmetry assumptions

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/13-Higher-Order-Topological-Insulators.md) · [Original description](../archive/original-PRDs/13-Higher-Order-Topological-Insulators.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A topological material can have interesting boundary behavior even when its broad edges do not conduct. In certain models, charge accumulates in a quantized way near corners instead. This problem asks how to establish that behavior carefully: what property belongs to the bulk, what depends on the way the sample is cut, and what extra symmetry is needed to place a corner state at exactly zero energy?

### Key terms

- **Higher-order topology** — Topology associated with lower-dimensional boundaries, such as corners of a two-dimensional sample or hinges of a three-dimensional sample.

- **Quadrupole** — A pattern of electric charge organization beyond total charge and ordinary dipole polarization; the selected model supports a quantized version under particular assumptions.

- **Bulk** — The interior of a system, far from boundaries.

- **Wilson loop** — A matrix describing how a chosen set of quantum states changes when transported around a closed path in momentum space.

- **Wannier sector** — A selected group of Wilson-loop eigenstates, separated from the others by a gap in that loop’s spectrum.

- **Nested polarization** — A polarization calculated within such a Wannier sector, using a second transport calculation.

- **Corner excess charge** — Charge near a corner measured relative to a specified background, after accounting for other contributions.

- **Chiral symmetry** — An additional spectral symmetry that can relate positive and negative energies; it can matter for pinning special states to zero.

- **Termination** — The actual boundary cut and local boundary structure of the sample.

### Why this matters

This problem broadens our understanding of how a material’s interior organizes behavior at its boundaries. It also corrects a common interpretive shortcut: a quantized corner charge and a zero-energy corner state are not the same observation. Reliable distinctions are needed before such features can be used as design principles.

### What progress would mean

Progress would produce a model with verified gaps, a well-defined bulk invariant and a carefully measured boundary consequence. It would establish a controlled example, rather than guarantee protected corner states for every sample shape or boundary.

## Core question

For a fixed quadrupole-insulator family, can one certify the bulk and Wannier gaps, a quantized nested polarization, and the corresponding boundary charge under a specified termination?

## Scope and assumptions

Start with a four-band Benalcazar–Bernevig–Hughes-type square-lattice model at half filling. Supply all hopping signs, orbital positions, mirror representations, and a rectangular termination. State whether chiral symmetry is imposed. Restrict parameters to a compact region away from bulk and Wannier-gap closings.

Separate fractional corner charge from an exactly zero-energy corner eigenstate. The latter requires additional spectral symmetry and boundary assumptions.

## Mathematical target

Let P(k) be the rank-two occupied projector. Build the path-ordered Wilson loop W_x(k_y) from occupied-state overlaps. If its spectrum has a certified separation into Wannier sectors, form a smooth sector projector and its Berry holonomy along y. Define the nested polarization from that holonomy modulo one.

Do not replace a nested Wilson loop by the integral of the first Wilson-loop eigenphases. Define corner excess charge relative to an explicit ionic/reference background and a stated spatial window, with finite-size and edge contributions controlled.

## Required outputs and proof obligations

Return H, symmetry operators and identities, bulk and Wannier-gap enclosures, nested invariant, boundary charge definition, and finite-size bounds. Any robustness statement must preserve the protecting symmetry and relevant gaps. State whether conclusions are intrinsic bulk statements or termination-dependent observables.

## Validation and rejection controls

Recover topological and trivial limits of the selected model. Verify zero total Chern number where required while the nested invariant is nontrivial. Break a protecting symmetry as a control; the checker must stop claiming quantization. Compare two terminations and explain any boundary difference.

## Milestones and research extension

Baseline: reproduce one quadrupole benchmark with valid Wilson-loop construction. Strong: certify its invariant and boundary charge on a parameter box. Research extension: classify a bounded family or one additional symmetry setting; no universal HOTI classification is implied.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Benalcazar, Bernevig and Hughes, Quantized Electric Multipole Insulators](https://arxiv.org/abs/1611.07987) — Nested Wilson loops and assumptions for quantized multipole response.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 13: Higher-order topology with explicit boundary and symmetry assumptions

[Revised problem](../PRDs/13-Higher-Order-Topological-Insulators.md) · [Preserved original](../archive/original-PRDs/13-Higher-Order-Topological-Insulators.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/13-Higher-Order-Topological-Insulators.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong observable (73–89).** The proposed quadrupole formula is a Chern-type curvature integral, not a quadrupole moment, and the corner-charge product formula is not universal. Replacing these prevents certifying the wrong invariant.

2. **Blocking — nested loop construction (78–85).** One must construct a Wannier-sector bundle and its holonomy; integrating eigenphases from the first loop is not that construction.

3. **Major — corner charge versus zero mode (101 onward).** Quantized boundary charge does not automatically pin a level to zero energy. Termination, filling, reference charge, and extra spectral symmetry matter.

4. **Major — scope (36–45).** All wallpaper/space-group HOTIs encompass different equivalence notions and boundary phenomena. The rewrite uses one well-specified benchmark and demands a Wannier gap before invoking nested polarization.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed quadrupole-insulator family, can one certify the bulk and Wannier gaps, a quantized nested polarization, and the corresponding boundary charge under a specified termination?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: reproduce one quadrupole benchmark with valid Wilson-loop construction. Strong: certify its invariant and boundary charge on a parameter box. Research extension: classify a bounded family or one additional symmetry setting; no universal HOTI classification is implied.

## Sources supporting the corrections

- [Benalcazar, Bernevig and Hughes, Quantized Electric Multipole Insulators](https://arxiv.org/abs/1611.07987) — Nested Wilson loops and assumptions for quantized multipole response.


---

<a id="problem-14"></a>

# 14. Certified Weyl nodes and slice topology in a lattice model

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/14-Topological-Semimetals-Weyl-Dirac.md) · [Original description](../archive/original-PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Energy bands in a three-dimensional crystal can meet at isolated points instead of remaining separated everywhere. Near certain crossings, the equations resemble those of a Weyl particle. Each crossing carries a handedness-like topological label. This problem asks whether we can find every such crossing in one lattice model, prove their labels, and connect them to the behavior of two-dimensional momentum slices.

### Key terms

- **Semimetal** — A system in which the occupied and unoccupied electronic states are not separated by a full insulating gap; the selected model has isolated band crossings.

- **Weyl node** — A simple crossing of two bands with a characteristic linear structure and nonzero topological charge.

- **Chirality** — The signed handedness of a simple Weyl crossing, defined here from its velocity matrix.

- **Velocity matrix** — The derivatives describing how the Hamiltonian changes in each momentum direction near a node.

- **Berry flux** — The integrated Berry curvature through a closed surface around a crossing.

- **Momentum slice** — A two-dimensional section of the three-dimensional Brillouin zone, obtained by holding one momentum coordinate fixed.

- **Fermi arc** — An open surface-state contour whose relation to bulk nodes depends on the chosen surface and energy.

- **Root isolation** — A mathematical procedure proving that a small region contains a solution, often uniquely.

- **Completeness** — A proof that no additional nodes have been missed anywhere in the claimed domain.

### Why this matters

Weyl nodes connect a local band crossing to a global topological constraint. They help explain unusual electronic structures and surface states, and provide controlled settings for studying the relation between particle-like excitations and lattice physics. Missing even a pair of nodes can change the interpretation of a phase diagram.

### What progress would mean

A certified catalog would establish all crossings and slice invariants of the selected model. It would give a reliable starting point for surface and response calculations, without pretending that bulk nodes uniquely determine every surface arc.

## Core question

Can all nodes in a specified two-band three-dimensional Hamiltonian be isolated, assigned chirality, and related to the Chern numbers of gapped momentum slices?

## Scope and assumptions

Begin with H(k)=sin k_x σ_x+sin k_y σ_y+(2−cos k_x−cos k_y−cos k_z)σ_z, k∈T³, lower band occupied away from nodes. Fix orientation and Pauli-matrix conventions. This exact benchmark has nodes at (0,0,±π/2).

Subsequent searches must fix orbitals, hopping support, parameter box, and symmetry representation. Dirac points and nodal lines are separate extensions because their protection criteria differ.

## Mathematical target

For H=d₀I+d·σ, nodes solve d(k)=0. A simple node has invertible velocity matrix V_ij=∂d_i/∂k_j. Define Weyl chirality χ=sgn det V, and explicitly relate this to the lower-band Chern flux under the chosen convention. Prove the complement of all isolating neighborhoods has |d(k)|>0.

For fixed k_z away from nodes, certify the two-dimensional occupied Chern number. Its jumps track node charge. A slab calculation requires a chosen surface normal and termination; arc details are not determined solely by bulk node positions.

## Required outputs and proof obligations

Return all root enclosures, uniqueness/completeness proof, Jacobian signs, slice Chern numbers, and optional slab spectrum with error control. Nodes need not have closed-form coordinates: certified isolating boxes are acceptable. Robustness is scoped to perturbations that prevent annihilation with opposite total charge.

## Validation and rejection controls

Verify the two benchmark roots analytically and opposite determinant signs. Check total charge on the periodic Brillouin zone vanishes. Confirm slice invariants only on gapped slices. Test a perturbation moving nodes and a parameter change permitting pair annihilation.

## Milestones and research extension

Baseline: complete benchmark node and slice audit. Strong: certified parameter-dependent node motion or one bounded model family. Research extension: a symmetry-protected Dirac model with explicit little-group constraints, or nodal-link invariants with their own definitions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Wan et al., Electronic Structure of Pyrochlore Iridates (published as Topological semimetal and Fermi-arc surface states)](https://arxiv.org/abs/1007.0016) — Reference for Weyl nodes and surface arcs; benchmark and proof obligations here are explicitly specified.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 14: Certified Weyl nodes and slice topology in a lattice model

[Revised problem](../PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Preserved original](../archive/original-PRDs/14-Topological-Semimetals-Weyl-Dirac.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/14-Topological-Semimetals-Weyl-Dirac.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — oversimplified symmetry (22–25).** Time reversal and inversion alone are not a universal recipe for a protected fourfold Dirac crossing; representation and additional symmetry constraints matter. They cannot replace an explicit model.

2. **Major — isotropy assumed (73–87).** A general Weyl cone has a velocity matrix, can be tilted, and is not specified by one scalar v_F. Its determinant gives the simple-node chirality in a declared convention.

3. **Major — completeness (99–105).** Finding roots and checking their charge sum does not prove there are no additional zero-charge pairs. A proof on the complement of the isolating boxes is needed.

4. **Major — boundary overclaim.** Fermi-arc connectivity depends on surface projection and termination; opposite nodes can project together. The revision uses slice topology as the robust intermediate claim and separates surface-specific predictions.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **Can all nodes in a specified two-band three-dimensional Hamiltonian be isolated, assigned chirality, and related to the Chern numbers of gapped momentum slices?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete benchmark node and slice audit. Strong: certified parameter-dependent node motion or one bounded model family. Research extension: a symmetry-protected Dirac model with explicit little-group constraints, or nodal-link invariants with their own definitions.

## Sources supporting the corrections

- [Wan et al., Electronic Structure of Pyrochlore Iridates (published as Topological semimetal and Fermi-arc surface states)](https://arxiv.org/abs/1007.0016) — Reference for Weyl nodes and surface arcs; benchmark and proof obligations here are explicitly specified.


---

<a id="problem-15"></a>

# 15. Symmetry-indicator algebra and the limits of band diagnosis

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/15-Topological-Quantum-Chemistry.md) · [Original description](../archive/original-PRDs/15-Topological-Quantum-Chemistry.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Symmetry can act like a fingerprint for quantum bands: it provides a compact set of labels that can sometimes reveal nontrivial topology. But different objects can share a fingerprint. This problem asks exactly what can be inferred from symmetry labels, how to compute that inference reliably, and when the correct answer is that more information about the quantum states is needed.

### Key terms

- **Crystal symmetry** — A translation, rotation, reflection or related operation preserving the lattice structure and the stated physical setting.

- **Representation** — Matrices describing how quantum states transform under a symmetry.

- **Irreducible representation (irrep)** — A basic representation that cannot be decomposed into smaller independent representations of the same group.

- **Compatibility relations** — Rules requiring symmetry labels at different momenta to fit into continuous bands.

- **Atomic band representation** — Band data generated from suitably localized, symmetry-compatible orbitals.

- **Wannier function** — A localized real-space representation of band states; the existence of an appropriate symmetry-respecting basis is central to atomicity.

- **Symmetry indicator** — An algebraic label detecting certain obstructions to atomic band data, but not every possible topological distinction.

- **Smith normal form** — An exact integer-matrix decomposition used to calculate the quotient of symmetry-data lattices.

- **Fragile topology** — An obstruction that can disappear when suitable trivial bands are added, under the chosen definition.

### Why this matters

Symmetry labels are often much simpler to work with than the full wavefunctions across momentum space. Knowing what those labels do and do not prove can make topological diagnosis more efficient and prevent false declarations that a system is ordinary. The distinction is essential for turning a classification table into a trustworthy scientific tool.

### What progress would mean

A useful result would be an exact indicator calculation for one symmetry setting, together with examples showing its reach and blind spots. Stronger conclusions about a particular Hamiltonian would use additional information, such as Berry phases or a Wannier construction.

## Core question

For one fixed symmetry setting, can compatibility relations and atomic band representations be generated exactly, and what topology can their quotient actually detect?

## Scope and assumptions

Choose one small symmetry setting, initially a spinless two-dimensional inversion-symmetric lattice without time reversal. Specify the space-group generators, orbital/site representations, and high-symmetry momenta. Derive all input representations from group operations, with external tables used only as checks.

The result is an algebraic classification of symmetry data. A zero indicator is not a certificate of a trivial occupied bundle.

## Mathematical target

Let B⊂Z^r be the lattice of integer irrep vectors satisfying compatibility constraints, and let A⊂B be the integer lattice generated by atomic band-representation vectors. Compute X=B/A by Smith normal form, including basis-change matrices.

For a physical nonnegative irrep vector b, distinguish integer-lattice membership from nonnegative decomposition b=A_mat n, n≥0. Nonnegative decomposition establishes compatibility with atomic symmetry data, not by itself a Wannier construction for a supplied Hamiltonian. Conversely, an actual symmetry-preserving localized Wannier basis is constructive atomic evidence.

## Required outputs and proof obligations

Export group conventions, irreps, compatibility matrix, atomic generators, Smith decomposition, indicator mapping, and witnesses for any integer/nonnegative decomposition. Report “indicator trivial; topology unresolved” when symmetry data are insufficient. Use Berry/Wilson-loop information when making a stronger claim about a specific H.

## Validation and rejection controls

Verify all representation identities and dimension counts. Reconstruct the Smith identity U A_mat V=D in the chosen B basis, with U,V unimodular. Test known atomic data, incompatible data, and models sharing symmetry labels but differing in an additional invariant. Keep stable and fragile notions distinct.

## Milestones and research extension

Baseline: exact quotient and decompositions for one setting. Strong: connect the diagnosis to explicit Hamiltonians and identify its kernel. Research extension: another symmetry group or a rigorous fragile obstruction with a definition of allowed added bands.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Po, Vishwanath and Watanabe, Symmetry-based Indicators of Band Topology](https://arxiv.org/abs/1703.00911) — Quotient of compatible symmetry data by atomic data; symmetry-based diagnosis.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 15: Symmetry-indicator algebra and the limits of band diagnosis

[Revised problem](../PRDs/15-Topological-Quantum-Chemistry.md) · [Preserved original](../archive/original-PRDs/15-Topological-Quantum-Chemistry.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/15-Topological-Quantum-Chemistry.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — “complete classification” (14–22, 36–44).** Symmetry indicators generally do not distinguish every topological bundle. Indicator-trivial phases can still be topological.

2. **Blocking — cone versus group (79–105).** The text interchanges nonnegative EBR sums, integer differences, and a group quotient. Fragile topology and stable indicator obstructions require these to be kept separate.

3. **Major — zero indicator implies positive decomposition (103–105).** Integer-lattice membership does not guarantee a nonnegative solution. Even matching an atomic irrep vector is not a construction of localized Wannier functions for a particular Hamiltonian.

4. **Major — crystallography input.** Not every site representation induction is automatically elementary; site symmetry, maximality and spin/time-reversal conventions must be stated. The revised algebra produces auditable limited conclusions instead of an all-phase label.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For one fixed symmetry setting, can compatibility relations and atomic band representations be generated exactly, and what topology can their quotient actually detect?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact quotient and decompositions for one setting. Strong: connect the diagnosis to explicit Hamiltonians and identify its kernel. Research extension: another symmetry group or a rigorous fragile obstruction with a definition of allowed added bands.

## Sources supporting the corrections

- [Po, Vishwanath and Watanabe, Symmetry-based Indicators of Band Topology](https://arxiv.org/abs/1703.00911) — Quotient of compatible symmetry data by atomic data; symmetry-based diagnosis.


---

<a id="problem-16"></a>

# 16. Certified energy brackets from reduced-density-matrix relaxations

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/16-N-Representability-2RDM.md) · [Original description](../archive/original-PRDs/16-N-Representability-2RDM.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Describing every possible configuration of many electrons becomes overwhelmingly expensive as the system grows. Yet the energy of a system with one- and two-particle interactions depends only on certain one-electron and two-electron summaries. The challenge is that not every plausible summary can come from a real many-electron state. We use necessary consistency tests to calculate a guaranteed lower energy limit and an actual trial state to obtain an upper limit.

### Key terms

- **Fermion** — A particle, such as an electron, whose quantum states obey antisymmetry under exchange.

- **Reduced density matrix (RDM)** — A summary of selected particles’ quantum statistics obtained from a larger system’s state.

- **2-RDM** — The two-particle reduced density matrix, sufficient with the appropriate one-particle data to compute the chosen two-body Hamiltonian’s energy.

- **N-representability** — The requirement that the reduced data arise from some valid state of N fermions.

- **P, Q and G conditions** — Positivity tests involving particle pairs, hole pairs and particle–hole combinations; they are necessary representability conditions.

- **Hole** — An unoccupied orbital state, treated as an absence of a particle relative to the finite orbital description.

- **Semidefinite program (SDP)** — An optimization problem with linear conditions and positive-semidefinite matrix constraints.

- **Ground-state energy** — The lowest allowed energy of the chosen quantum system.

- **Energy bracket** — A pair of verified lower and upper numbers containing the true ground-state energy.

### Why this matters

Reliable energy calculations underpin the theoretical study of bonding and interacting electrons. A bracket tells us how much uncertainty remains, rather than merely reporting a convincing decimal number. The approach may extract useful information without storing a complete many-electron wavefunction, although general representability remains computationally difficult.

### What progress would mean

Progress would mean tighter verified brackets at an affordable cost for a specified finite model. That would show how much the extra constraints help; extending the conclusion to an actual molecule would also require controlling orbital-basis and Hamiltonian errors.

## Core question

For a specified finite fermionic Hamiltonian, how tight an independently certified ground-state energy interval can P,Q,G and selected higher-order RDM constraints produce?

## Scope and assumptions

Fix N≥2 fermions in M spin orbitals with an exact rational/algebraic, number-conserving one- and two-body Hamiltonian. Start with a small Hubbard-type system where exact diagonalization in the same N-particle sector is feasible. Molecular integrals are a later input with an explicit basis and numerical error enclosure.

Use ensemble N-representability. The task is to certify bounds within a finite orbital model, not solve general electronic structure in polynomial time.

## Mathematical target

Set γ_ij=〈a_i†a_j〉 and Γ_ij,kl=〈a_i†a_j†a_l a_k〉. Require Hermiticity, fermionic antisymmetry, Tr γ=N, Σ_ij Γ_ij,ij=N(N−1), and Σ_j Γ_ij,kj=(N−1)γ_ik. Define Q and G using two-hole and particle-hole operator moments and derive their affine forms from anticommutation identities.

Minimizing the linear energy over a necessary-condition outer relaxation gives E_relax≤E₀. A verified dual feasible point gives L≤E_relax; an explicit normalized N-fermion state gives U≥E₀. Report L≤E₀≤U, not an unsupported accuracy estimate from Γ alone.

## Required outputs and proof obligations

Export the Hamiltonian and indexing convention, moment constraints, certified dual objective L, state witness for U, and gap U−L. Separate solver error, relaxation error, and finite-basis/model error. Report any property bounds via separately optimized observables; a relaxed Γ need not be physical.

## Validation and rejection controls

Compare with exact diagonalization in identical conventions. Check contractions using explicit Slater determinants and correlated states. Adding valid necessary constraints must not lower the exact relaxed minimum. Verify dual PSD matrices and equality residual corrections rigorously, rather than interpreting the raw solver objective as a bound.

## Milestones and research extension

Baseline: valid L,U brackets for small fixed models. Strong: quantify improvement from G or T constraints across a specified instance set. Research extension: a new valid constraint with demonstrated tightening, or a theorem on a tractable family; novelty needs comparison with existing hierarchies.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Liu, Christandl and Verstraete, N-representability is QMA-complete](https://arxiv.org/abs/quant-ph/0609125) — Complexity barrier to general exact representability.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-17"></a>

# 17. Complete enumeration of a precisely defined molecular-graph class

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/17-Isomer-Enumeration-Molecular-Graphs.md) · [Original description](../archive/original-PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A molecular formula tells us how many atoms of each kind are present, but not how they are connected. The same counts can describe different structures: butane and isobutane are a familiar example. This problem asks for every distinct connection pattern allowed by an explicit set of bonding rules, with a proof that none was counted twice and none was missed.

### Key terms

- **Molecular formula** — A count of atoms by element, such as C₄H₁₀.

- **Constitutional isomer** — A structure with the same molecular formula as another but a different pattern of atom connections.

- **Molecular graph** — A mathematical representation with atoms as vertices and bonds as edges carrying bond-order labels.

- **Valence grammar** — The exact bonding rules permitted in this enumeration; it deliberately defines a restricted chemical model.

- **Bond order** — The bond label distinguishing single, double and triple bonds in this graph model.

- **Graph isomorphism** — A relabeling of atoms that preserves element types and bonds, showing two drawings represent the same graph.

- **Canonical labeling** — A consistent representative label for a graph, used to identify duplicates.

- **Stereoisomer** — A distinction in spatial arrangement not captured merely by the basic connection graph.

- **Completeness certificate** — Evidence that the enumeration covers every object in the specified graph class.

### Why this matters

Complete enumeration turns an intuitive picture of “possible structures” into a checkable mathematical set. Such sets are useful for testing structure-generation methods and exploring how bonding constraints shape chemical possibilities. The guarantee is especially valuable when a search is used to support a claim that no alternative structure exists.

### What progress would mean

Success would supply exact graph counts and explicit structures under the chosen rules. A graph’s presence in the list would not imply that the corresponding molecule is stable, easily synthesized or realizable in a particular three-dimensional shape.

## Core question

For a fixed formula and explicit valence grammar, can every connected constitutional graph be generated exactly once, with a checkable completeness argument?

## Scope and assumptions

Initially allow neutral closed-shell C,H,O graphs with bond orders 1,2,3, no self-bonds, and valences 4,1,2. Hydrogens may be implicit if reconstructed uniquely from residual valence. Specify whether different Kekulé graphs are distinct; the baseline uses literal bond-order graphs and no aromatic equivalence.

Constitutional graph validity is not chemical stability or synthesizability. Stereoisomers, conformers, ions, radicals, isotopes, and 3D geometries require added definitions and are not baseline outputs.

## Mathematical target

For labeled heavy atoms set symmetric b_ij∈{0,1,2,3}, b_ii=0, h_i=v_i−Σ_j b_ij≥0, and Σ_i h_i=n_H. Require connectivity and the requested counts of element labels. Quotient by permutations preserving element labels and bond orders.

Prove completeness via canonical augmentation with a proved parent rule, or exhaustive labeled enumeration plus verified orbit reduction. A SAT route needs the exact encoding and orbit-blocking/symmetry-breaking justification; the final UNSAT proof must cover all remaining admissible graphs.

## Required outputs and proof obligations

Export canonical graphs, reconstructed hydrogens, counts, labeling conventions, and generation/coverage proof. Provide a validator independent of the optimized generator. If a resource limit stops enumeration, output a partial catalog and coverage status rather than “all isomers.”

## Validation and rejection controls

Recover alkane constitutional counts for C₄H₁₀ and C₅H₁₂, namely 2 and 3. Cross-check all tiny instances with brute force. Random relabelings must preserve canonical codes. Verify each graph’s formula and weighted valence; reject disconnected mixtures.

## Milestones and research extension

Baseline: complete small-formula catalog with independent counts. Strong: complete C₆H₁₂O under the stated grammar. Research extension: stereochemical orbit enumeration with meso cases and explicit stereocenter/double-bond conventions; keep conformational continua separate.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [McKay, Isomorph-free exhaustive generation](https://users.cecs.anu.edu.au/~bdm/papers/orderly.pdf) — Canonical generation and completeness methodology.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 17: Complete enumeration of a precisely defined molecular-graph class

[Revised problem](../PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Preserved original](../archive/original-PRDs/17-Isomer-Enumeration-Molecular-Graphs.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/17-Isomer-Enumeration-Molecular-Graphs.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Major — chemical universe (39–47).** Fixed valences do not describe every molecule of a formula. Charge, radicals, aromatic equivalence, stereochemistry and disconnected structures change the answer. The grammar now defines what “all” means.

2. **Major — canonicity is not completeness (96–99).** No duplicate outputs does not prove no missing outputs. A coverage argument must audit pruning and symmetry breaking.

3. **Major — Pólya shortcut (101–108).** Burnside/Pólya counting requires a specified group action and correct fixed-point counts under the graph constraints. Writing a cycle-index sum alone does not enumerate connected valence-constrained graphs.

4. **Major — physical and spatial claims.** SMILES and a valence-correct graph do not certify a stable 3D structure. The rewrite preserves an exact combinatorial task and treats stereochemistry as a separately verified extension.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed formula and explicit valence grammar, can every connected constitutional graph be generated exactly once, with a checkable completeness argument?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete small-formula catalog with independent counts. Strong: complete C₆H₁₂O under the stated grammar. Research extension: stereochemical orbit enumeration with meso cases and explicit stereocenter/double-bond conventions; keep conformational continua separate.

## Sources supporting the corrections

- [McKay, Isomorph-free exhaustive generation](https://users.cecs.anu.edu.au/~bdm/papers/orderly.pdf) — Canonical generation and completeness methodology.


---

<a id="problem-18"></a>

# 18. Certified optimal-transport bounds for strictly correlated electrons

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/18-Optimal-Transport-Chemistry.md) · [Original description](../archive/original-PRDs/18-Optimal-Transport-Chemistry.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Electrons repel one another, but a prescribed electron density also constrains where they spend their time. Imagine arranging their joint positions so they avoid one another as effectively as possible while preserving that average density. This problem makes that arrangement question precise and seeks reliable upper and lower bounds on its repulsion energy. It represents a limiting strong-correlation model, not the full energy of ordinary electrons.

### Key terms

- **Electron density ρ** — The average spatial distribution of electrons, integrating to their total number N.

- **Coulomb cost** — The repulsion contribution between charged particles, proportional here to the inverse of their separation.

- **Optimal transport** — Optimization over ways of distributing or coupling quantities while satisfying prescribed distribution constraints.

- **Marginal distribution** — The distribution of one variable after the other variables in a joint distribution are ignored.

- **Coupling Γ** — A joint distribution of all electron positions consistent with the prescribed individual marginals.

- **Multi-marginal transport** — Transport optimization with more than two prescribed marginals.

- **Strictly correlated electrons (SCE)** — A limiting model in which electron positions are organized to minimize interaction energy at fixed density.

- **Dual potential** — An auxiliary function providing a certified lower bound when it satisfies the required inequality everywhere.

- **Discretization error** — The difference introduced by replacing a continuous spatial problem with a finite representation.

### Why this matters

Strongly correlated electrons are difficult to approximate because their motions are strongly linked. This limiting problem gives a mathematically structured reference for studying that linkage and for developing density-functional ideas. Verified bounds help distinguish a true property of the continuum model from an artifact of a spatial grid or optimization tolerance.

### What progress would mean

A useful result would enclose the SCE interaction energy for a declared density. It could inform further electronic-structure theory, while leaving kinetic energy, finite-correlation effects and actual molecular reaction paths to additional modeling.

## Core question

For a supplied normalized electron density, can a discretized Coulomb multi-marginal transport problem yield verified bounds, with continuum error separated from optimization error?

## Scope and assumptions

Take an electron density ρ≥0 with ∫ρ=N and marginal μ=ρ/N. Begin with N=2 and a compactly supported mathematical density or a finite rational discrete model. Specify treatment of the diagonal Coulomb singularity. A continuous-density result additionally needs domain-tail and discretization estimates.

The main target is the strictly correlated interaction functional. Wasserstein comparison of densities is a separate optional metric task; it is not a minimum-energy chemical reaction path.

## Mathematical target

Define

$$V_{SCE}[ρ]=\inf_{Γ\in\Pi(μ,\ldots,μ)}\int\sum_{i<j}|r_i-r_j|^{-1}\,dΓ.$$

Γ is a probability measure and may be symmetrized; it is not an antisymmetric probability distribution. The symmetric dual uses u with Σ_i u(r_i)≤Σ_{i<j}|r_i−r_j|⁻¹ and objective ∫uρ. A feasible coupling gives an upper bound and a dual-feasible potential gives a lower bound for the same problem. Define W_∞=V_SCE−U_H if subtracting Hartree energy; do not call V_SCE exact exchange.

## Required outputs and proof obligations

Return marginal conventions, cost and singularity policy, primal plan, dual potentials, rigorously checked constraints, and L≤V≤U. State whether V is a finite discrete optimum or the continuum functional. Sinkhorn candidates must be converted to feasible bounds for the unregularized problem, with entropic bias controlled.

## Validation and rejection controls

Verify a small rational-cost transport LP exactly and a one-dimensional quadratic-cost quantile example as a separate solver test. For Coulomb densities, use a case with a known co-motion construction under its actual hypotheses. Reject comparing normalized densities with unequal total mass unless normalization or an unbalanced-OT model is explicitly chosen.

## Milestones and research extension

Baseline: certified two-marginal discrete bounds. Strong: one continuum enclosure with singularity/tail control. Research extension: N>2 with symmetry reduction and validated bounds, or a demonstrably improved relaxation. Do not promise deterministic Monge maps for arbitrary multimarginal inputs.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Cotar, Friesecke and Klüppelberg, Density functional theory and optimal transportation with Coulomb cost](https://arxiv.org/abs/1104.0603) — Strong-correlation/semiclassical connection; theorem hypotheses must be checked for each density.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 18: Certified optimal-transport bounds for strictly correlated electrons

[Revised problem](../PRDs/18-Optimal-Transport-Chemistry.md) · [Preserved original](../archive/original-PRDs/18-Optimal-Transport-Chemistry.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/18-Optimal-Transport-Chemistry.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong functional (16, 34, 98–108).** Coulomb OT describes a strong-correlation limit, not the exact exchange functional asserted here. The negative half-cost formula and negative-cost dual are inappropriate for the positive repulsion minimization.

2. **Major — normalization and antisymmetry.** The electron density integrates to N, while each probability marginal integrates to one. Fermionic wavefunction antisymmetry does not make a probability measure antisymmetric.

3. **Major — reaction pathway inference (18, 36).** A Wasserstein geodesic minimizes transport cost, not a molecular potential-energy barrier. The original chemical interpretation lacks a dynamical/energetic bridge.

4. **Major — uniqueness and certification (106, 868–905).** Quadratic-cost Brenier results do not transfer automatically to Coulomb multimarginal problems. Marginal residuals and a regularized solver output are not continuum certificates. The revised formulation distinguishes these errors and removes the inappropriate helium-exchange target.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a supplied normalized electron density, can a discretized Coulomb multi-marginal transport problem yield verified bounds, with continuum error separated from optimization error?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: certified two-marginal discrete bounds. Strong: one continuum enclosure with singularity/tail control. Research extension: N>2 with symmetry reduction and validated bounds, or a demonstrably improved relaxation. Do not promise deterministic Monge maps for arbitrary multimarginal inputs.

## Sources supporting the corrections

- [Cotar, Friesecke and Klüppelberg, Density functional theory and optimal transportation with Coulomb cost](https://arxiv.org/abs/1104.0603) — Strong-correlation/semiclassical connection; theorem hypotheses must be checked for each density.


---

<a id="problem-19"></a>

# 19. Certified equilibria and conditional persistence in mass-action networks

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/19-Chemical-Reaction-Networks.md) · [Original description](../archive/original-PRDs/19-Chemical-Reaction-Networks.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A list of chemical reactions can settle into different long-term concentration patterns. Some networks have one resting state; others can behave like a switch, with two stable states separated by an unstable one. This problem asks whether we can find every positive equilibrium of a small, explicitly specified network and prove which are locally stable. It also distinguishes never reaching zero at a finite time from genuinely avoiding extinction in the long run.

### Key terms

- **Reaction network** — A set of chemical species and rules converting specified combinations of reactants into products.

- **Mass-action kinetics** — Rate laws in which each reaction rate is proportional to powers of its reactant concentrations, multiplied by a rate constant.

- **Stoichiometry matrix S** — A table describing how each reaction changes the amount of each species.

- **Conservation law** — A combination of species amounts that remains constant under the network’s reactions.

- **Compatibility class** — The concentration states accessible under the same conserved quantities and stoichiometric constraints.

- **Equilibrium or steady state** — A concentration pattern where the net rate of change is zero.

- **Local stability** — The tendency of states starting sufficiently near an equilibrium to remain nearby, with asymptotic stability also implying return toward it.

- **Multistationarity and multistability** — Having multiple equilibria, and having multiple stable equilibria, respectively.

- **Persistence** — A long-time condition preventing each initially present species from approaching zero along the trajectory.

### Why this matters

Stable switches and coexistence are important organizing ideas in chemistry and biological regulation. An exact small-network analysis can establish whether a proposed mechanism even supports those behaviors. It also makes numerical simulation more interpretable: a trajectory can miss another equilibrium or appear stable simply because it has not been followed long enough.

### What progress would mean

Progress would give complete equilibrium information for the stated network, rates and conserved totals. A theorem covering many rate choices would be stronger; identifying a switch in a model would not by itself establish its role in a living system.

## Core question

For a finite mass-action network with specified rates and conservation totals, can all positive equilibria in one compatibility class be isolated and their local stability certified?

## Scope and assumptions

Input integer reactant/product complexes, positive rational rate constants, and positive rational initial totals. Use small networks first, initially at most four species and eight reactions. Distinguish fixed-parameter root analysis from theorems quantified over all rates.

Keep persistence as a separate theorem-driven module. Catalytic RAF structure belongs primarily to Challenge 29 and does not by itself determine mass-action stability.

## Mathematical target

Let S be the stoichiometric matrix, f(x)=Sv(x), v_r=κ_r x^{y_r}, and let rows of L span ker Sᵀ. Solve f(x)=0, Lx=Lx₀, x>0. Remove dependent equations before root isolation. Deficiency is δ=number of complexes−number of linkage classes−rank S.

For an isolated equilibrium, test the Jacobian on im S, not on the full space with conservation-induced zero eigenvalues. Multistationarity means at least two positive equilibria in the same class; multistability additionally needs stability. Persistence means liminf_{t→∞}x_i(t)>0 and requires global hypotheses and boundedness checks.

## Required outputs and proof obligations

Return exact network data, deficiency/reversibility facts, conservation laws, isolating boxes with existence/uniqueness proofs, and coverage of the admissible domain. A positive-dimensional equilibrium set is a separate valid outcome. Persistence output must name a sufficient theorem and check every hypothesis, or return unresolved.

## Validation and rejection controls

Use A⇌B as a deficiency-zero benchmark. Use the one-species open network ∅⇌X and 2X⇌3X with rates giving f(x)=6−11x+6x²−x³; isolate x=1,2,3 and distinguish two stable roots from three equilibria. Include a boundary-approaching trajectory example to distinguish positivity from persistence.

## Milestones and research extension

Baseline: exact structural facts and complete small fixed-parameter root enumeration. Strong: certified local stability and one applicable persistence theorem. Research extension: parameter-region proofs or a new structural criterion, with quantifiers stated explicitly.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Angeli, De Leenheer and Sontag, A Petri Net approach to persistence](https://arxiv.org/abs/q-bio/0608019) — Checkable sufficient conditions for persistence, with hypotheses.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 19: Certified equilibria and conditional persistence in mass-action networks

[Revised problem](../PRDs/19-Chemical-Reaction-Networks.md) · [Preserved original](../archive/original-PRDs/19-Chemical-Reaction-Networks.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/19-Chemical-Reaction-Networks.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — persistence shortcut (32, 125–131).** Remaining positive at finite time is not persistence, and the stated unrestricted siphon “if and only if” is not a general algorithmic criterion. A sufficient theorem must carry its kinetic and boundedness assumptions.

2. **Major — multistationarity versus stability (27, 1102).** Multiple equilibria need not all be stable. The bistable Schlögl example typically has three positive equilibria, two stable, rather than the required two equilibria. The new exact polynomial benchmark makes this distinction checkable.

3. **Major — algebraic completeness (41, 55).** A Gröbner basis does not automatically isolate all positive real roots or eliminate continuous solution sets. Conservation classes and positivity restrictions are essential.

4. **Major — parameter independence (57–58).** Network deficiency is structural, but most dynamical conclusions depend on rates and totals unless a theorem quantifies over them. The rewrite separates those claims and avoids promised scalability to hundreds of species.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite mass-action network with specified rates and conservation totals, can all positive equilibria in one compatibility class be isolated and their local stability certified?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact structural facts and complete small fixed-parameter root enumeration. Strong: certified local stability and one applicable persistence theorem. Research extension: parameter-region proofs or a new structural criterion, with quantifiers stated explicitly.

## Sources supporting the corrections

- [Angeli, De Leenheer and Sontag, A Petri Net approach to persistence](https://arxiv.org/abs/q-bio/0608019) — Checkable sufficient conditions for persistence, with hypotheses.


---

<a id="problem-20"></a>

# 20. Path-integral thermodynamics with a transparent error budget

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/20-Ab-Initio-Path-Integrals.md) · [Original description](../archive/original-PRDs/20-Ab-Initio-Path-Integrals.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

At small scales, particles cannot always be treated as tiny classical balls sitting at definite positions. Their quantum spreading affects thermal averages, especially for light particles. Path-integral methods replace part of this quantum calculation with a sampling problem involving an artificial ring of linked copies of a particle. This challenge checks that the sampling produces the right equilibrium quantities and that each source of error is reported separately.

### Key terms

- **Thermodynamics** — The study of quantities such as energy and temperature describing equilibrium behavior.

- **Path integral** — A formulation of quantum mechanics involving a sum over possible paths; here it is used in imaginary time for equilibrium properties.

- **Imaginary time** — A mathematical reformulation of time used to connect quantum calculations with thermal statistics, not a physical clock reading.

- **Ring polymer and beads** — The artificial closed chain and its particle copies used to approximate the imaginary-time path integral.

- **PIMD** — Path-integral molecular dynamics, a method using auxiliary dynamics to sample the ring-polymer equilibrium distribution.

- **Estimator** — A formula calculated from sampled configurations to estimate a desired physical average.

- **Discretization bias** — Systematic error caused here by using a finite number of beads or a finite integration step.

- **Sampling uncertainty** — Uncertainty from using only a finite sample of configurations.

- **Born–Oppenheimer surface** — An approximate potential energy for nuclear motion obtained by treating electronic behavior separately.

### Why this matters

Quantum thermal effects matter for understanding how light particles behave in molecules and materials. But an accurate-looking simulation can combine several hidden errors: too few beads, incomplete sampling, time-step effects and an approximate potential. A transparent error budget makes the calculation useful as a scientific measurement of the model rather than just a plausible plot.

### What progress would mean

The initial result would reproduce known oscillator averages and then handle one controlled anharmonic case. It would validate equilibrium predictions; it would not automatically give exact real-time reaction rates or remove approximations in an electronic-structure model.

## Core question

For a specified one-dimensional quantum Hamiltonian, can path-integral sampling reproduce equilibrium energy and position moments with separately assessed discretization and statistical errors?

## Scope and assumptions

Start with H=p²/(2m)+V(q), m>0, β>0, and a confining harmonic potential; extend to an explicitly defined anharmonic potential. Nuclei are distinguishable Boltzmann particles. Report only equilibrium observables initially.

An ab initio extension must specify the Born–Oppenheimer surface, electronic method, basis, and force errors. It is not exact quantum chemistry merely because no empirical force field is used. Real-time rates and exchange statistics require distinct methods.

## Mathematical target

Set β_P=β/P and ω_P=P/(βℏ). Sample the coordinate density proportional to

$$\exp\left[-β_P\sum_{s=1}^P\left\{\tfrac12mω_P²(q_s-q_{s+1})²+V(q_s)\right\}\right],\quad q_{P+1}=q_1.$$

Coordinate observables use P⁻¹Σ_s A(q_s), generally not A(q_centroid). For one dimension the primitive energy estimator is P/(2β)−mP/(2β²ℏ²)Σ_s(q_s−q_{s+1})²+P⁻¹Σ_sV(q_s). Derive the chosen estimator from the partition function to fix all temperature factors.

## Required outputs and proof obligations

Return algorithm, seeds, temperatures, bead counts, time-step settings, estimators, and an error ledger separating sampling, integration, finite-P, and potential/model errors. Use “statistical interval” for sampling uncertainty. Reserve “rigorous enclosure” for cases with proved sampling/discretization/tail bounds.

## Validation and rejection controls

Recover E=(ℏω/2)coth(βℏω/2) and 〈q²〉=ℏcoth(βℏω/2)/(2mω) for the oscillator over a declared β range. Compare multiple P and time steps, independent chains, and autocorrelation diagnostics. Validate anharmonic results against a separately converged or certified spectral method. A convergence fit alone cannot certify its asymptote.

## Milestones and research extension

Baseline: oscillator thermodynamics within predeclared statistical and bias tolerances. Strong: one anharmonic error-controlled result. Research extension: a small ab initio system with a full error budget; rigorous certification requires more than SCF convergence and stable trajectories.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Ceriotti et al., Efficient stochastic thermostatting of path integral molecular dynamics](https://arxiv.org/abs/1009.1045) — Normal-mode sampling and stochastic thermostat methods.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-21"></a>

# 21. Quantum LDPC instances with independently certified code parameters

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/21-Quantum-LDPC-Codes.md) · [Original description](../archive/original-PRDs/21-Quantum-LDPC-Codes.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Quantum information is fragile, but simply copying an unknown quantum state is not allowed. An error-correcting code instead stores information jointly across many physical qubits so that selected measurements reveal errors without directly reading the encoded information. This problem builds sparse-check codes and verifies how much information they store, how many errors their structure can withstand, and how a chosen recovery algorithm performs.

### Key terms

- **Qubit** — A quantum information unit whose state can involve superpositions of two basis states.

- **Physical and logical qubits** — The individual qubits used by the device, and the protected information units encoded across them.

- **Quantum error-correcting code** — A structured encoding that permits detection and correction of specified errors.

- **LDPC** — Low-density parity-check: a code family with checks involving bounded numbers of qubits and each qubit participating in bounded numbers of checks.

- **CSS code** — A code built from compatible binary checks for two types of quantum error, named after Calderbank, Shor and Steane.

- **Stabilizer** — An operator that leaves every valid encoded state unchanged; measuring checks can reveal violations.

- **Syndrome** — The collection of check outcomes indicating what kind of error may have occurred.

- **Code distance d** — The smallest number of qubits on which an undetectable nontrivial logical operation can act.

- **Decoder** — An algorithm that uses a syndrome to choose a recovery operation.

- **Hypergraph product** — A construction combining two classical parity-check matrices into a quantum code.

### Why this matters

Useful quantum computation requires reliable logical information despite noisy components. Sparse checks offer a route to distributing error correction efficiently, but code parameters and decoder behavior must be demonstrated rather than assumed. Exact small examples are valuable for detecting mistakes that large simulations can conceal.

### What progress would mean

Success would provide verified code parameters and decoder results under a stated error model. It would help compare constructions fairly, while leaving hardware connectivity, noisy measurements and the full cost of fault-tolerant computation as separate requirements.

## Core question

For a specified CSS construction, can code dimension and distance bounds be verified independently, and can decoder performance be measured under one explicit noise model?

## Scope and assumptions

Start with hypergraph-product codes from supplied binary parity-check matrices H₁ and H₂. Include the product of two full-rank 3×7 Hamming parity-check matrices as a small benchmark. Separate finite instances, asymptotic families, and noisy syndrome extraction.

Use independent X errors with perfect syndrome measurements for the first decoder experiment. A family is LDPC only when both row and column weights stay bounded as n grows.

## Mathematical target

For H_i of size m_i×n_i define

$$H_X=[H_1\otimes I_{n_2}\mid I_{m_1}\otimes H_2^{\mathsf T}],\quad H_Z=[I_{n_1}\otimes H_2\mid H_1^{\mathsf T}\otimes I_{m_2}].$$

Check H_XH_Zᵀ=0 over F₂, n=n₁n₂+m₁m₂, k=n−rank H_X−rank H_Z. Define d_X=min{|x|:x∈ker H_Z∖row H_X} and d_Z analogously, d=min(d_X,d_Z). A found logical operator upper-bounds distance; ruling out all smaller logical operators lower-bounds it.

## Required outputs and proof obligations

Export sparse matrices, GF(2) row reductions, logical bases, explicit upper witnesses and lower-bound proofs. SAT certificates need a verified encoding of nontrivial homology and weight. Decoder experiments report physical noise, recovery rule, logical-failure test, sample counts, confidence intervals, and runtime.

## Validation and rejection controls

For the two 3×7 inputs verify n=58 and k=16; certify the claimed distance independently rather than assuming it. Inject all small-weight errors on tiny codes. A matching syndrome is not success unless error plus recovery is a stabilizer. Compare optimized checks with an independent GF(2) implementation.

## Milestones and research extension

Baseline: exact small product code and certified distance bracket. Strong: several instances plus honest finite-size decoder curves. Research extension: a new family tradeoff or proved decoder guarantee. Reproducing asymptotically good codes is not itself a new existence result.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Panteleev and Kalachev, Asymptotically Good Quantum and Locally Testable Classical LDPC Codes](https://arxiv.org/abs/2111.03654) — Existing asymptotically good quantum LDPC constructions; novelty baseline.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-22"></a>

# 22. Bell inequalities with certified local and quantum value brackets

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/22-Bell-Inequalities-Nonlocality.md) · [Original description](../archive/original-PRDs/22-Bell-Inequalities-Nonlocality.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two distant experimenters choose measurements and compare their outcomes. If the outcomes were produced only by shared classical instructions, their correlations would obey particular inequalities. Quantum systems can exceed some of those limits. This problem calculates both the classical ceiling and a reliable quantum range for a specified test, so that the size of the difference is established mathematically.

### Key terms

- **Bell scenario** — A specification of the parties, their measurement choices and the possible outcomes.

- **Local hidden-variable model** — A model in which outcomes depend on local choices and shared classical information, with measurement-choice independence among the assumptions.

- **Bell inequality** — A linear restriction satisfied by the correlations of such local models.

- **CHSH** — The simplest standard two-party test with two binary measurement choices per party.

- **Quantum strategy** — A specified shared quantum state and local measurements producing correlations.

- **POVM** — A set of positive operators summing to the identity that describes a general quantum measurement.

- **Tsirelson bound** — A quantum upper limit on a Bell expression, with 2√2 the familiar CHSH value in its standard normalization.

- **NPA hierarchy** — A sequence of matrix-positivity tests providing outer bounds on quantum correlations.

- **Commuting-operator model** — A framework in which different parties’ measurements commute; its general correlation set differs from the finite-dimensional tensor-product setting.

- **Facet** — A full boundary face of the appropriate dimension of the classical correlation polytope.

### Why this matters

Bell inequalities make the difference between classical local explanations and quantum correlations precise. They also underlie ways to certify quantum behavior from observed input-output relationships. Reliable quantum bounds are essential: a numerical optimization value without a valid upper certificate or achieving strategy may misstate what quantum theory permits.

### What progress would mean

A complete small test would include an exact classical maximum, an explicit quantum strategy and a matching or enclosing upper bound. Applications to randomness or cryptography would need additional statistical and security assumptions beyond this mathematical benchmark.

## Core question

For a fixed rational Bell functional in a small bipartite scenario, can its exact local value and a certified quantum lower/upper bracket be produced?

## Scope and assumptions

Start with two parties, two binary settings each, and CHSH. Fix normalization and no-signaling affine coordinates. The target quantum value is the supremum over finite-dimensional tensor-product strategies; identify its closure when discussing limits. Finite NPA levels provide outer bounds through the commuting-operator framework.

Facet discovery and randomness certification are extensions with separate definitions. Do not equate a finite SDP solution with a realizable strategy.

## Mathematical target

For β(P)=Σ_abxy c_abxyP(ab|xy), compute β_L by evaluating every deterministic local strategy. A verified finite-dimensional state and POVMs give L_Q≤β_Q. A certified NPA dual or noncommutative SOS gives β_Q≤U_Q. Report β_L and [L_Q,U_Q].

The NPA limit characterizes commuting-operator correlations, which need not equal the closure of finite-dimensional tensor-product correlations. Finite-level tightness needs an explicit achieving strategy or a valid extraction theorem.

## Required outputs and proof obligations

Export coefficients, local vertices, state/measurements, moment-word list, algebraic relations, dual matrices/SOS, and exact or interval objective bounds. To certify a facet in affine dimension D, verify the inequality for every vertex and show saturating vertices have affine dimension D−1. Declare relabeling symmetries when counting classes.

## Validation and rejection controls

Check all 16 CHSH deterministic strategies and β_L=2. Reproduce 2√2 with an explicit two-qubit strategy and an operator upper-bound identity. Reject an invalid dual PSD matrix or a wrong commutation relation. Distinguish eight CHSH variants from one class under relabeling; positivity facets must be handled separately.

## Milestones and research extension

Baseline: exact CHSH lower and upper certificates. Strong: a preselected additional rational functional with a certified bracket, whether or not it closes. Research extension: a new inequality with normalized comparison, or device-independent entropy bounds with an explicit adversary and statistical model.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Navascués, Pironio and Acín, A convergent hierarchy of semidefinite programs](https://arxiv.org/abs/0803.4290) — NPA convergence to a commuting-measurement representation.

- [Ji et al., MIP*=RE](https://arxiv.org/abs/2001.04383) — Strict separation of the relevant correlation sets.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 22: Bell inequalities with certified local and quantum value brackets

[Revised problem](../PRDs/22-Bell-Inequalities-Nonlocality.md) · [Preserved original](../archive/original-PRDs/22-Bell-Inequalities-Nonlocality.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/22-Bell-Inequalities-Nonlocality.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect convergence target (18, 41, 103–105).** NPA convergence is to commuting measurements. Treating this as general convergence to the finite-dimensional tensor-product set ignores a substantive distinction exposed by MIP*=RE.

2. **Major — optimal strategies (24–27, 909–911).** Moment feasibility alone does not supply states and measurements; extraction requires extra conditions. A constructive lower bound is now mandatory for optimality claims.

3. **Major — facet counts/rank (909, 922).** The eight CHSH variants are not eight inequivalent classes; a D-dimensional polytope facet has affine dimension D−1, so D+1 affinely independent saturating vertices would be impossible for a proper facet.

4. **Major — randomness and “exact” numbers.** Floating-point agreement and solver timing are not certificates. Bell violation alone does not specify a full secure randomness protocol. The replacement focuses on auditable value brackets.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fixed rational Bell functional in a small bipartite scenario, can its exact local value and a certified quantum lower/upper bracket be produced?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: exact CHSH lower and upper certificates. Strong: a preselected additional rational functional with a certified bracket, whether or not it closes. Research extension: a new inequality with normalized comparison, or device-independent entropy bounds with an explicit adversary and statistical model.

## Sources supporting the corrections

- [Navascués, Pironio and Acín, A convergent hierarchy of semidefinite programs](https://arxiv.org/abs/0803.4290) — NPA convergence to a commuting-measurement representation.

- [Ji et al., MIP*=RE](https://arxiv.org/abs/2001.04383) — Strict separation of the relevant correlation sets.


---

<a id="problem-23"></a>

# 23. Entanglement certification with explicit inconclusive outcomes

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/23-Entanglement-Measures-Witnesses.md) · [Original description](../archive/original-PRDs/23-Entanglement-Measures-Witnesses.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Two quantum systems can be related in ways that no mixture of independent local states can reproduce. That is entanglement. Given a mathematical description of a joint state, this problem asks for evidence that it is entangled, evidence that it is separable, or an honest statement that the available tests cannot decide. It also asks how to quantify one particular aspect of entanglement without mixing different measures together.

### Key terms

- **Bipartite system** — A quantum system divided into two identified parts, such as two qubits held by different parties.

- **Density matrix ρ** — An operator describing a quantum state, including statistical mixtures.

- **Separable state** — A mixture of product states of the two parts; correlations in such a decomposition can arise from shared classical information.

- **Entangled state** — A state that has no separable decomposition across the chosen division.

- **Entanglement witness** — An observable nonnegative on every separable state but negative on the state being tested.

- **Partial transpose** — A matrix operation transposing the indices of just one subsystem, useful for testing entanglement.

- **PPT** — Positive partial transpose: a necessary separability condition that is sufficient only in certain small dimensions.

- **Negativity** — A quantity measuring the negative part of the partial-transpose spectrum; zero does not generally imply separability.

- **DPS hierarchy** — Tests based on whether the state can be extended to larger symmetric states; failure can certify entanglement.

- **Concurrence** — A particular entanglement measure with a useful explicit formula for two-qubit states.

### Why this matters

Entanglement is an important resource in many quantum-information protocols, but claiming it from an inconclusive test can invalidate a proposed demonstration. Conversely, explicit witnesses show exactly what establishes its presence. Quantitative bounds are useful because different states may possess very different strengths or kinds of quantum correlations.

### What progress would mean

The deliverable would make its verdict inspectable: a witness, a decomposition, or unresolved status. That would provide a reliable component for analyzing quantum states, rather than promise a universally easy entanglement test.

## Core question

For a finite bipartite density matrix, can one produce a checked entanglement witness or a separable decomposition, and bound a specifically chosen entanglement measure?

## Scope and assumptions

Input subsystem dimensions and an exact rational/algebraic density matrix, or a rigorously described uncertainty set. Begin with 2×2 systems; then include 3×3 PPT-entangled examples. Validate Hermiticity, positivity and unit trace before classification.

The primary decision outputs are entangled, separable, or unresolved. Restrict concurrence and its entanglement-of-formation formula to two qubits. Treat general distillable entanglement and multipartite classification as separate research problems.

## Mathematical target

Compute partial transpose in a fixed tensor-index convention. Negativity is N(ρ)=(||ρ^{T_B}||₁−1)/2. For entanglement give W satisfying 〈a⊗b|W|a⊗b〉≥0 for all product vectors and Tr(Wρ)<0. For separability give ρ=Σ_jp_j|a_jb_j〉〈a_jb_j| with p_j≥0 and Σp_j=1, or another sufficient theorem in its valid dimensions.

PPT/DPS feasibility is generally only a necessary separability test. DPS infeasibility with a validated dual can certify entanglement; passing a finite level generally remains unresolved.

## Required outputs and proof obligations

Export input validation, partial-transpose spectrum enclosures, measure intervals, and a witness/decomposition with an independent checker. Normalize witnesses before optimizing them. For an uncertain input, require the detection margin to hold throughout the uncertainty set.

## Validation and rejection controls

Use Bell states, product states, and ρ_p=p|ψ⁻〉〈ψ⁻|+(1−p)I/4, 0≤p≤1. In this declared convention separability holds for p≤1/3. Verify negativity 1/2 and concurrence 1 for a Bell state. Include a 3×3 PPT-entangled control so zero negativity is not mislabeled separability.

## Milestones and research extension

Baseline: checked two-qubit decisions and measure values. Strong: a validated DPS witness beyond PPT and a separable-decomposition example. Research extension: robust uncertainty-set certification or normalized optimal witnesses on a stated family.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Doherty, Parrilo and Spedalieri, A complete family of separability criteria](https://arxiv.org/abs/quant-ph/0308032) — Symmetric-extension hierarchy and dual entanglement witnesses.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 23: Entanglement certification with explicit inconclusive outcomes

[Revised problem](../PRDs/23-Entanglement-Measures-Witnesses.md) · [Preserved original](../archive/original-PRDs/23-Entanglement-Measures-Witnesses.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/23-Entanglement-Measures-Witnesses.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — relaxation direction (36–40).** Passing PPT or a finite DPS level does not generally prove separability. The replacement requires a sufficient certificate or reports unresolved.

2. **Major — “optimal witness” unspecified.** Multiplying a witness by a positive constant scales its objective without changing detection. An optimization needs a normalization and a declared class of witnesses.

3. **Major — measures conflated (23–29, 87–107).** Different entanglement measures do not give a complete interchangeable characterization, and the two-qubit concurrence formula is not a general mixed-state solution.

4. **Major — benchmark convention (817).** A Werner-state threshold such as p=2/3 is meaningless without the mixing definition; for the singlet-weight convention used here the threshold is 1/3. Explicit inputs remove that ambiguity and expose false positives in the checker.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite bipartite density matrix, can one produce a checked entanglement witness or a separable decomposition, and bound a specifically chosen entanglement measure?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: checked two-qubit decisions and measure values. Strong: a validated DPS witness beyond PPT and a separable-decomposition example. Research extension: robust uncertainty-set certification or normalized optimal witnesses on a stated family.

## Sources supporting the corrections

- [Doherty, Parrilo and Spedalieri, A complete family of separability criteria](https://arxiv.org/abs/quant-ph/0308032) — Symmetric-extension hierarchy and dual entanglement witnesses.


---

<a id="problem-24"></a>

# 24. Topological code verification and noise-specific decoder evaluation

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/24-Topological-Quantum-Error-Correction.md) · [Original description](../archive/original-PRDs/24-Topological-Quantum-Error-Correction.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Imagine storing information in loops spread across a lattice so that a small local disturbance usually cannot change the stored value. A topological quantum code uses a quantum version of this idea. Local measurements reveal the endpoints of error patterns, and a decoder tries to join those endpoints correctly. The subtlety is that two repairs can have the same endpoints yet differ by a loop that changes the logical information.

### Key terms

- **Toric code** — A quantum error-correcting code on a lattice with periodic boundaries, topologically equivalent to a torus.

- **Surface code** — A related geometric code with carefully chosen open boundaries.

- **Logical operator** — An operation changing the encoded information while remaining consistent with the code’s checks.

- **Noncontractible loop** — A loop winding around the torus that cannot be shrunk to a point within it.

- **Homology** — A mathematical way of distinguishing cycles modulo boundaries; here it identifies whether a closed error pattern acts logically.

- **Syndrome** — Check outcomes locating the boundary of an error chain.

- **Recovery** — An operation chosen to cancel the error; matching the syndrome is necessary but does not ensure logical success.

- **Minimum-weight matching** — A graph optimization method used in certain decoders to pair defects efficiently.

- **Logical failure probability** — The probability that error and recovery together change encoded information.

- **Threshold** — An asymptotic noise boundary for successful suppression of logical errors in a specified code-and-decoder family.

### Why this matters

Geometric codes connect an intuitive picture of local errors to precise algebraic guarantees. They are important models for protecting quantum information using structured measurements. Understanding logical failure, rather than only syndrome cancellation, is essential for assessing whether increasing the code size actually improves reliability.

### What progress would mean

A useful result would prove the code’s parameters and measure a fixed decoder under a declared noise model. An estimated threshold would remain specific to that model and decoder, not a universal device error tolerance.

## Core question

Can a toric-code family’s algebraic parameters be proved, and can a fixed decoder’s logical failure probability be evaluated under a precisely specified error model?

## Scope and assumptions

Use an L×L square torus with qubits on edges and L≥3. Start with independent Z errors of probability p and perfect syndrome extraction. Specify the decoder, tie-breaking rule, and whether recovery weights are uniform or likelihood based.

Keep exact code properties separate from statistical decoder performance. Planar boundaries, color codes, measurement noise, and circuit-level noise are distinct extensions.

## Mathematical target

Over F₂ use C₂→∂₂ C₁→∂₁ C₀ with ∂₁∂₂=0, H_X=∂₁, H_Z=∂₂ᵀ. Then n=2L², k=2, and prove d=L using noncontractible primal and dual cycles.

For Z error e and recovery r, success means e+r∈im ∂₂, not merely ∂₁(e+r)=0. A threshold assertion concerns the limit of P_fail(L,p) as L→∞ for this fixed noise/decoder family; a finite crossing is an estimate.

## Required outputs and proof obligations

Export lattice incidence data, ranks and logical bases, distance argument, decoder specification, and failure counts with confidence intervals. Exhaustive small-size evaluation can give an exact failure polynomial in p. Monte Carlo output must retain trials and failure counts, including zero-failure upper intervals.

## Validation and rejection controls

Check n,k,d on several small tori. Verify logical anticommutation via primal/dual intersection. Enumerate all low-weight errors on tiny instances and test the stabilizer-membership success criterion. Validate a matching solution’s optimality separately from logical decoding success.

## Milestones and research extension

Baseline: exact construction and distance proof with a tested decoder. Strong: reproducible finite-size failure curves and uncertainty-aware threshold estimates. Research extension: a rigorous threshold bound or noisy-measurement/circuit model, with no transfer of numerical thresholds across noise conventions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Dennis, Kitaev, Landahl and Preskill, Topological quantum memory](https://arxiv.org/abs/quant-ph/0110143) — Homological coding and model-dependent error-correction thresholds.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-25"></a>

# 25. Quantum query complexity with explicit oracle and output models

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/25-Quantum-Algorithms-Complexity.md) · [Original description](../archive/original-PRDs/25-Quantum-Algorithms-Complexity.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A claim that a quantum computer is faster is incomplete until we specify what information it receives, what answer it must return, and which operations we count. Here the problem is deliberately simple: find one marked item in a list when the only way to learn about an item is to query it. We compare a quantum search procedure with a proof of how many queries any procedure of the chosen kind must use.

### Key terms

- **Oracle** — A precisely defined black-box operation providing access to input information; it is a mathematical access model, not a prediction device.

- **Query** — One use of the oracle, counted separately from other computation.

- **Promise problem** — A problem whose inputs are guaranteed to satisfy a stated restriction, here exactly one marked item.

- **Grover search** — A quantum algorithm that amplifies the probability of observing a marked item using repeated oracle and interference operations.

- **Amplitude** — A quantum coefficient whose squared magnitude contributes to an outcome probability.

- **Query complexity** — The number of oracle uses required to solve a problem with the specified success probability.

- **Upper and lower complexity bounds** — An algorithm showing that a cost suffices, and a proof that no allowed algorithm can beat a stated cost.

- **BPP and BQP** — Classes of problems efficiently solvable with bounded error by classical randomized and quantum computation, respectively.

- **Oracle separation** — A proof that two classes differ when both have access to a particular oracle; it does not by itself separate the ordinary classes.

- **Gate complexity** — The number of elementary circuit operations, a different resource from oracle queries.

### Why this matters

This is a clean setting in which a quantum advantage can be proved rather than inferred from a small simulation. It also teaches how to assess broader speedup claims fairly. Input preparation, precision and output requirements can dominate a real application even when a query bound looks attractive.

### What progress would mean

Success would combine a working small implementation with an all-size argument matching upper and lower query bounds. That establishes an advantage in this access model, not a blanket claim that quantum computers speed up every search task.

## Core question

For unstructured search with a specified promise, can an implementation and an analytic lower bound establish a matched quantum query complexity?

## Scope and assumptions

Use N=2^n items with exactly one marked item and phase-oracle access O_f|x〉=(−1)^f(x)|x〉. State success probability at least 2/3 and count oracle uses separately from elementary gates, state preparation, and output verification. Begin with small exact simulations and a proof valid for all N.

Retain quantum walks, linear systems, QAOA and oracle separations as independent extensions, each with its own input/output and cost model. Classical simulation wall time is not quantum runtime.

## Mathematical target

Let θ=arcsin(N⁻¹ᐟ²). After r Grover iterations the success probability is sin²((2r+1)θ). Choose r as a nearest nonnegative integer to π/(4θ)−1/2, evaluating neighboring integers and handling small N explicitly. Derive the query upper bound and pair it with a hybrid/adversary/polynomial lower bound Ω(√N) under the same promise.

An oracle separation must be written BQP^O≠BPP^O for a specified O; it does not establish BQP≠BPP. A linear-system extension must state matrix access, κ, precision, state preparation, and whether the output is a quantum state or classical vector.

## Required outputs and proof obligations

Return oracle/circuit specification, exact small-instance results, an all-N success derivation, query/gate/resource accounting, and the lower-bound proof with assumptions. Any QAOA extension must distinguish a certified bound on a finite graph from a performance theorem for a graph family.

## Validation and rejection controls

Check the analytic success formula on n=1,…,8 and all marked positions using symmetry or exhaustive checks. Test neighboring iteration counts to expose overshoot. Verify unitarity and normalization. Compare with randomized classical search under identical oracle access and error tolerance.

## Milestones and research extension

Baseline: Grover construction and matched query bound. Strong: one second promise problem with matched assumptions and a certified upper/lower bracket. Research extension: a new restricted query bound or oracle result. Reimplementation of established algorithms is replication, not a new complexity separation.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Bennett et al., Strengths and Weaknesses of Quantum Computing](https://arxiv.org/abs/quant-ph/9701001) — Oracle lower-bound methodology for quantum search.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-26"></a>

# 26. A posteriori certification of an invariant Hamiltonian torus

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/26-KAM-Theory-Planetary-Stability.md) · [Original description](../archive/original-PRDs/26-KAM-Theory-Planetary-Stability.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A weakly disturbed mechanical system can sometimes keep moving in an orderly pattern forever, even though the motion never exactly repeats. KAM theory explains how certain such motions survive a perturbation. This problem starts with a proposed orderly motion and checks, with rigorous error bounds, whether a true one must exist nearby. The goal is a guarantee about a specified mathematical orbit family, not a prediction that every nearby trajectory is stable.

### Key terms

- **Hamiltonian** — An energy function that generates the system’s equations of motion.

- **Integrable system** — A system with enough conserved structure to describe its motion using constant actions and evolving angles in the setting considered here.

- **Action–angle variables** — Coordinates in which the actions label the motion and the angles describe its cyclic phases.

- **Invariant torus** — A torus-shaped set in phase space that trajectories starting on it never leave; it need not be a physical doughnut in ordinary space.

- **Quasiperiodic motion** — Motion combining incommensurate frequencies, so it remains organized without repeating exactly.

- **KAM theory** — Kolmogorov–Arnold–Moser theory, describing persistence of suitable quasiperiodic motions under sufficiently small perturbations.

- **Diophantine condition** — An arithmetic bound keeping integer combinations of the frequencies from becoming too close to zero.

- **A posteriori theorem** — A theorem that uses a computed approximate solution and its error to establish a nearby exact solution.

- **Residual** — The amount by which the approximate solution fails to satisfy the exact equation.

- **Interval arithmetic** — Computation with enclosing intervals, using controlled rounding to bound numerical errors.

### Why this matters

Long-term stability is difficult to infer from a finite trajectory simulation. A theorem can instead prove that a particular organized motion exists despite perturbations. This offers a foundation for studying stability mechanisms in celestial and other Hamiltonian systems, provided the model and theorem assumptions are appropriate.

### What progress would mean

A certificate for a nonzero perturbation would convert an approximate torus into an existence result. Applying the method to planetary observations would additionally require reduction of the planetary model and a careful treatment of measured initial-condition uncertainty.

## Core question

For an explicitly specified analytic near-integrable Hamiltonian and an approximate torus, can every hypothesis of a quantitative KAM theorem be verified?

## Scope and assumptions

Begin with H(I,θ)=ω·I+|I|²/2+ε[cos θ₁+cos(θ₁−θ₂)] in two degrees of freedom, ω=(1,(1+√5)/2), on a stated complex neighborhood of a real action box. Choose a rational ε only after validating the unperturbed case. Seek an embedding with rotation vector ω.

This is a mathematical model, not a solar-system stability assertion. Planetary extensions require an explicit canonical reduction, handling Kepler degeneracy, collision exclusions, and input uncertainty.

## Mathematical target

For an approximate embedding K:T²→phase space define E(θ)=X_H(K(θ))−DK(θ)ω. Select a published a posteriori KAM theorem and reproduce its hypotheses and computable constants: analytic widths, nondegeneracy/twist inverse, residual norm, and small-divisor bounds.

Prove |k·ω|≥γ/|k|₁^τ for every nonzero integer k using arithmetic properties of the exact chosen ω. Finite checks alone are insufficient. Evaluate the theorem’s smallness inequalities with outward-rounded intervals and certify closeness of a true K_* to K.

## Required outputs and proof obligations

Export H, ε, domain, exact frequency representation, Fourier coefficients, all theorem constants, interval inequalities, and a checker. State the conclusion: existence of a nearby quasiperiodic invariant torus with a quantified embedding error. A failed sufficient condition yields “not certified,” not “unstable.”

## Validation and rejection controls

Verify E=0 for K₀(θ)=(I=0,θ) at ε=0. Derive an all-integer Diophantine bound for the golden-ratio frequency. Check Fourier tails and loss of analytic width. Test a resonant frequency where the stated theorem must refuse certification.

## Milestones and research extension

Baseline: unperturbed certificate and complete arithmetic/theorem setup. Strong: a nonzero-ε torus certificate. Research extension: a larger perturbation range or a reduced planetary model. One invariant torus does not certify a neighborhood of arbitrary observed initial conditions or a phase-space measure bound.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Valvo and Locatelli, Hamiltonian Control of Magnetic Field Lines: Computer Assisted Results Proving the Existence of KAM Barriers](https://arxiv.org/abs/2101.07785) — Example of a complete computer-assisted Hamiltonian KAM workflow; use a theorem matching the chosen formulation.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


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


---

<a id="problem-27"></a>

# 27. Central configurations in a bounded, symmetry-reduced N-body problem

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/27-N-Body-Central-Configurations.md) · [Original description](../archive/original-PRDs/27-N-Body-Central-Configurations.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Most collections of gravitating bodies change their overall shape as they move. Certain special arrangements can rotate or expand and contract while keeping the same shape. These are central configurations. This problem asks for every such arrangement in a carefully limited case, with a proof that the list is complete and a separate analysis of what small disturbances do to the associated rotating motion.

### Key terms

- **N-body problem** — The problem of describing the motion of N mutually gravitating bodies.

- **Central configuration** — An arrangement in which each body’s gravitational acceleration is proportional to its position relative to the center of mass, with one common proportionality factor.

- **Homographic motion** — Motion preserving the arrangement’s shape while its size and orientation may change.

- **Relative equilibrium** — A motion that is stationary in an appropriate rotating reference frame.

- **Center of mass** — The mass-weighted average position, used here to remove overall translation.

- **Normalization** — A fixed choice of scale, such as setting the moment of inertia to one, so scaled copies are not counted separately.

- **Symmetry quotient** — The convention for treating configurations related by rotations, reflections or specified relabelings as equivalent.

- **Collision-free** — Having strictly positive separation between every pair of bodies.

- **Coriolis term** — A velocity-dependent term arising in a rotating frame, essential to the stability calculation.

- **Spectral stability** — A statement about eigenvalues of the linearized dynamics; it is weaker than a full nonlinear stability proof.

### Why this matters

Central configurations are organizing structures in gravitational dynamics and provide exact reference cases amid otherwise complicated motion. A complete small catalog can test numerical methods and reveal how mass choices and symmetry affect possible motions. The problem also links gravity to exact algebraic root finding.

### What progress would mean

Progress would establish a complete list under a stated mass choice and symmetry convention. It would not classify every periodic orbit: a choreography such as the figure-eight changes shape and belongs to a different question.

## Core question

For fixed positive masses and a declared symmetry quotient, can all collision-free central configurations in the chosen family be certified and their relative-equilibrium linearization analyzed?

## Scope and assumptions

Start with three labeled positive rational masses in the plane. Fix G=1, center of mass zero, moment of inertia I=Σm_i|r_i|²=1, and quotient by O(2), so mirror images count together. Permute only equal masses when explicitly requested. Extend to a specified four-body symmetry family after validating the complete three-body case.

Central configurations generate homographic motions; general periodic choreographies are a separate problem.

## Mathematical target

Use positive U=Σ_{i<j}m_im_j/|r_i−r_j| and

$$\nabla_{r_i}U=-λm_ir_i,\qquad λ=U/I>0.$$

Introduce distance variables d_ij>0 with d_ij²=|r_i−r_j|², and retain positivity/collision exclusions when polynomializing. Use symmetry charts covering every configuration claimed. Completeness requires certified global domain coverage or an exact elimination/root-count argument, not just isolated numerical solutions.

## Required outputs and proof obligations

Return normalized configurations, distance enclosures, residual identities, existence/uniqueness proofs, equivalence representatives, and completeness scope. For planar relative equilibria analyze the reduced rotating-frame first-order system including Coriolis terms. Distinguish spectral stability from nonlinear stability.

## Validation and rejection controls

For three labeled masses under the stated O(2) quotient, recover three collinear orderings (one per middle body) and one equilateral shape. Check λ=U at I=1. Test collision artifacts introduced by denominator clearing. For stability, reproduce a known three-body criterion using identical mass and reduction conventions.

## Milestones and research extension

Baseline: complete certified three-body catalog. Strong: complete enumeration in one four-body symmetry family or mass instance, with all charts covered. Research extension: a five-body restricted family; no claim to solve the general finiteness problem without an appropriately global proof.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Yu and Zhu, On the finiteness of four-body central configurations](https://arxiv.org/abs/2103.08906) — Context for finiteness versus explicit certified enumeration.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 27: Central configurations in a bounded, symmetry-reduced N-body problem

[Revised problem](../PRDs/27-N-Body-Central-Configurations.md) · [Preserved original](../archive/original-PRDs/27-N-Body-Central-Configurations.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/27-N-Body-Central-Configurations.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — wrong example (26–29, 120).** The figure-eight choreography changes shape and is not a central configuration. It belongs in a periodic-orbit problem, retained separately in the supplement.

2. **Blocking — counts and quotients (112–114).** The text confuses five restricted-problem Lagrange points with Euler central configurations. Counting depends on labels, rotations and reflections; the rewrite states the quotient and a four-shape three-body benchmark.

3. **Blocking — stability shortcut (124–130).** Hessian signs alone do not give rotating-frame orbital stability; Coriolis terms and symmetry modes must be treated.

4. **Major — algebraic and finiteness claims (36, 93–98).** Clearing denominators can add collisions and does not remove square roots without auxiliary variables. Numerical roots do not prove completeness, and the generic-five-body status claim needs qualification against the literature. The new bounded scope avoids an unsupported universal claim.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For fixed positive masses and a declared symmetry quotient, can all collision-free central configurations in the chosen family be certified and their relative-equilibrium linearization analyzed?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete certified three-body catalog. Strong: complete enumeration in one four-body symmetry family or mass instance, with all charts covered. Research extension: a five-body restricted family; no claim to solve the general finiteness problem without an appropriately global proof.

## Sources supporting the corrections

- [Yu and Zhu, On the finiteness of four-body central configurations](https://arxiv.org/abs/2103.08906) — Context for finiteness versus explicit certified enumeration.


---

<a id="problem-28"></a>

# 28. Explicit finite-time action confinement from a normal form

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/28-Nekhoroshev-Stability-Theory.md) · [Original description](../archive/original-PRDs/28-Nekhoroshev-Stability-Theory.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

A system need not move on a perfectly orderly orbit to remain near its initial state for a very long time. Small disturbances can produce only a slow drift in certain quantities. This problem asks for an explicit guarantee: how far can those quantities change, and for how long is that bound valid? It focuses on a finite time horizon rather than proving that a special motion persists forever.

### Key terms

- **Near-integrable Hamiltonian** — An energy function consisting of a well-understood integrable part plus a small perturbation.

- **Action** — A coordinate that is constant in the integrable reference system and may drift when a perturbation is added.

- **Confinement bound** — A guaranteed maximum change of the specified actions over a stated time interval.

- **Normal form** — A transformed expression of the equations separating manageable terms from a remainder.

- **Canonical transformation** — A change of variables preserving the Hamiltonian structure of the dynamics.

- **Remainder** — Terms not absorbed into the simplified normal form; bounding their size helps bound drift.

- **Resonance** — An integer relation among frequencies that can change how perturbations affect the motion.

- **Nekhoroshev theorem** — A theorem giving very long action-confinement times for small perturbations under specified regularity and nondegeneracy assumptions.

- **Quasi-convexity** — A particular curvature condition on the integrable energy function used by some such theorems.

- **Dimensional time** — A time expressed in declared physical or model units, rather than an unexplained numerical factor.

### Why this matters

Finite-time guarantees answer a question that simulations alone cannot settle beyond their run length. They also distinguish a valid stability argument from an impressive-looking exponential formula with missing constants or units. Understanding slow drift is relevant to the mathematical study of planetary and other nearly integrable systems.

### What progress would mean

A useful first result could be a modest but fully justified confinement time. Improving that time with controlled remainders would be real progress; asserting billions of years would require an appropriate physical model and every conversion and bound needed to support it.

## Core question

For a specified analytic Hamiltonian and action domain, what rigorously computable action excursion and confinement time follow from a certified normal form or quantitative Nekhoroshev theorem?

## Scope and assumptions

Use H=h(I)+εf(I,θ), h=ω·I+|I|²/2, in two or three degrees of freedom on a compact action box with an explicit complex extension. Supply ω, a finite Fourier polynomial f, and rational ε. Bound the gradient away from zero where the selected theorem requires it.

The initial target is a valid dimensional time bound, even if modest. Planetary and post-Newtonian models are extensions requiring their own reductions and hypotheses.

## Mathematical target

For a quasi-convex theorem verify its actual condition, for example vᵀD²h(I)v≥m|v|² for v⊥∇h(I), plus the theorem’s gradient, domain and analyticity conditions. A positive Hessian determinant alone is insufficient.

Alternatively construct a canonical Φ with H∘Φ=Z(J)+R(J,θ), bound |I−J|≤η and ||∂_θR||≤r on a validated domain. Then |I(t)−I(0)|≤2η+r|t| while the transformed orbit remains in that domain. If resonant Z depends on angles, include its action drift or restrict the claim to conserved/protected components.

## Required outputs and proof obligations

Export normal-form order, transformations, inverse/domain bounds, Fourier tails, r, η, and a domain-containment proof for the claimed interval. A Nekhoroshev result must include all constants and its smallness threshold in T=C exp(c ε^−a) with stated units; do not choose a from a desired time target.

## Validation and rejection controls

Verify integrable ε=0 behavior and a low-order symbolic normal form. Check symplecticity with validated remainder control. Confirm the action bound stays inside the certified domain. Independently recompute exp(10^0.3)≈7.35: it is a dimensionless factor, not 10¹³ years.

## Milestones and research extension

Baseline: one explicit finite-time confinement certificate. Strong: optimized truncation with a stronger bound or full quantitative quasi-convex theorem application. Research extension: exponential scaling across a parameter family; do not infer actual solar-system stability from ε alone.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Zhang and Zhang, Improved stability for analytic quasi-convex nearly integrable systems](https://arxiv.org/abs/1701.06026) — Theorem-specific quasi-convex exponential stability results.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 28: Explicit finite-time action confinement from a normal form

[Revised problem](../PRDs/28-Nekhoroshev-Stability-Theory.md) · [Preserved original](../archive/original-PRDs/28-Nekhoroshev-Stability-Theory.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/28-Nekhoroshev-Stability-Theory.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — incorrect steepness (30–31, 78–88).** Steepness and quasi-convexity are not equivalent to a positive Hessian determinant or an auxiliary convex function. The rewrite uses a theorem-specific condition.

2. **Blocking — fabricated timescale (24, 111, 854–861).** The numerical exponential shown is about 7.35, not 10¹³ years, and no physical time unit or constants were supplied. This invalidates the stated solar-system benchmark.

3. **Major — exponents (106–109).** “Super-steep” and ε-dependent exponents are not a substitute for a cited precise theorem. Exponents cannot be optimized merely to meet a desired age target.

4. **Major — global versus local control.** A truncated Fourier series, short integration, or local Hessian check does not certify the full action domain and time interval. A direct normal-form remainder bound supplies a defensible first milestone and exposes exactly what stronger claims require.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a specified analytic Hamiltonian and action domain, what rigorously computable action excursion and confinement time follow from a certified normal form or quantitative Nekhoroshev theorem?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: one explicit finite-time confinement certificate. Strong: optimized truncation with a stronger bound or full quantitative quasi-convex theorem application. Research extension: exponential scaling across a parameter family; do not infer actual solar-system stability from ε alone.

## Sources supporting the corrections

- [Zhang and Zhang, Improved stability for analytic quasi-convex nearly integrable systems](https://arxiv.org/abs/1701.06026) — Theorem-specific quasi-convex exponential stability results.


---

<a id="problem-29"></a>

# 29. Autocatalytic sets: maximality, irreducibility and dynamical viability

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/29-Chemical-Reaction-Networks-Origins.md) · [Original description](../archive/original-PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Could a collection of reactions collectively help produce the substances that make those same reactions happen? Autocatalytic-set theory turns part of that question into a precise network test. Starting from an externally supplied food set, we ask which reactions can be supported by products of the network and which smaller subsets already have that property. This describes a structural prerequisite for self-sustaining chemistry, not a complete definition of life.

### Key terms

- **Catalyst** — A substance participating in a reaction mechanism that increases its rate while being regenerated overall; it does not change the equilibrium free-energy difference.

- **Autocatalysis** — Production that directly or collectively helps enable more production of the relevant system.

- **Food set** — The species assumed to be available from outside the network.

- **Closure** — All species producible from the food set using the selected reaction rules; standard RAF closure temporarily ignores catalysis.

- **RAF** — A reflexively autocatalytic and food-generated reaction set: its reactants are reachable from food and each reaction has a catalyst available within the closure.

- **Maximal RAF** — The largest RAF contained in the supplied network, when a RAF exists.

- **Irreducible RAF** — A RAF containing no smaller nonempty RAF subset.

- **Minimum-cardinality RAF** — A RAF with the fewest reactions among all RAFs in the specified network; this is stronger than irreducibility.

- **Chemostat** — An idealized externally maintained concentration or chemical reservoir.

- **Dynamical viability** — The additional question of whether specified reaction rates and conditions support sustained activity or growth.

### Why this matters

Collectively supported chemistry is relevant to thinking about how organized reaction systems can arise and persist. Exact minimality results help distinguish essential reactions from dispensable ones in a model. Just as importantly, separating structural closure from energy supply and kinetics prevents a network diagram from being mistaken for proof of a viable replicating system.

### What progress would mean

Success would identify supported reaction sets and prove the stated kind of minimality. Demonstrating sustained growth, heredity or evolution would require extra chemical and dynamical assumptions beyond the RAF certificate.

## Core question

For a finite catalytic reaction system, can its maximal RAF and smallest RAF subsets be certified, while keeping structural autocatalysis distinct from sustained growth?

## Scope and assumptions

Input species S, reactions R with integer stoichiometry, food set F, and catalysis relation C⊂S×R. Begin with at most 12 reactions for exhaustive cross-checks. Catalysis is supplied model data, not inferred from a stoichiometric cycle.

Define three separate targets: unique maximal RAF, an inclusion-minimal (irreducible) RAF, and a minimum-cardinality RAF. Thermodynamic and kinetic viability require extra inputs and are separate extensions.

## Mathematical target

For R′⊂R compute cl_{R′}(F) by repeatedly adding products whose reactants are already present, ignoring catalysis in this closure step. R′ is a nonempty RAF when every reactant lies in this closure and every reaction has at least one catalyst in it.

An irreducible RAF has no proper nonempty RAF subset. A minimum-cardinality RAF minimizes |R′| over all RAF subsets. Ordinary RAF closure permits catalysts to arise later; a catalysis-respecting startup ordering is a stronger condition and must be tested separately.

## Required outputs and proof obligations

Export closure rounds, catalyst witnesses, maxRAF elimination trace, and minimality/cardinality proof with explicit scope. For thermodynamic extension specify chemical potentials, chemostats, and driving. For kinetic extension provide rates, dilution, initial conditions and a positive sustained-state/growth criterion; RAF existence alone is insufficient.

## Validation and rejection controls

Cross-check all subsets for tiny networks. Include food-catalyzed, self-catalyzed, catalyst-free, and mutually dependent examples. Verify a maximum RAF can contain several irreducible RAFs of different sizes. Deletion tests for irreducibility must use a complete RAF detector on each remaining network, not merely test whether the entire remainder is itself RAF.

## Milestones and research extension

Baseline: checked maxRAF/RAF witnesses. Strong: complete minimum-cardinality results for a bounded network family. Research extension: characterize which RAFs additionally support a declared driven mass-action regime. Biological self-replication and evolution require their own operational definitions.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [Hordijk and Steel, Autocatalytic sets in a partitioned biochemical network](https://pmc.ncbi.nlm.nih.gov/articles/PMC4034171/) — RAF closure and the distinction between maximal and irreducible RAFs.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 29: Autocatalytic sets: maximality, irreducibility and dynamical viability

[Revised problem](../PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Preserved original](../archive/original-PRDs/29-Chemical-Reaction-Networks-Origins.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/29-Chemical-Reaction-Networks-Origins.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — reversed terminology (81).** maxRAF is maximal, not minimal. Inclusion-minimal and minimum-cardinality are also different optimization targets, so the existing output flag is ambiguous.

2. **Major — closure/startup (30–31, 73–79).** Standard RAF food closure ignores catalysis during reachability; requiring every intermediate step to be catalyzed is stronger. The rewrite keeps these questions separate.

3. **Major — thermodynamic viability (99–108).** A reaction list plus arbitrary negative ΔG values is not a consistent driven chemical model. Chemostats, activities and cycle constraints matter; catalysts do not alter equilibrium free-energy differences.

4. **Major — biology and information (97, 112–116).** Generic hypercycle stability depends on the actual equations, and concentrations are not automatically probabilities or a joint distribution for mutual information. Structural RAF certificates cannot alone establish Darwinian evolution, stability against parasites, or information emergence.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a finite catalytic reaction system, can its maximal RAF and smallest RAF subsets be certified, while keeping structural autocatalysis distinct from sustained growth?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: checked maxRAF/RAF witnesses. Strong: complete minimum-cardinality results for a bounded network family. Research extension: characterize which RAFs additionally support a declared driven mass-action regime. Biological self-replication and evolution require their own operational definitions.

## Sources supporting the corrections

- [Hordijk and Steel, Autocatalytic sets in a partitioned biochemical network](https://pmc.ncbi.nlm.nih.gov/articles/PMC4034171/) — RAF closure and the distinction between maximal and irreducible RAFs.


---

<a id="problem-30"></a>

# 30. An exact finite genotype–phenotype map and its mutation graph

**Revised specification — 2026-09-11.** [Detailed critique](../reviews/30-Genotype-Phenotype-Mapping.md) · [Original description](../archive/original-PRDs/30-Genotype-Phenotype-Mapping.md) · [Evidence standard](../reviews/EVIDENCE-STANDARD.md)

## Plain-language guide

### The problem in everyday terms

Different sequences of genetic letters can produce the same modeled structure. That creates paths through sequence space along which mutations leave the chosen outcome unchanged. This problem constructs a small RNA-inspired map exactly and asks how those paths connect, how often one mutation changes the outcome, and which new structures are nearby. The aim is to study a controlled model of robustness and variation.

### Key terms

- **Genotype** — The input sequence, here a string over the four RNA letters A, C, G and U.

- **Phenotype** — The selected output of the model, here a secondary-structure pattern rather than an organism’s full observable traits.

- **Secondary structure** — A pattern of paired positions along an RNA sequence, with the allowed pairings and restrictions stated explicitly.

- **Nussinov-style model** — A combinatorial folding model maximizing the number of allowed pairs, not a full free-energy calculation.

- **Tie-breaking rule** — A deterministic rule choosing one output when several structures achieve the same score.

- **Hamming graph** — A graph whose vertices are sequences and whose edges join sequences differing at exactly one position.

- **Neutral component** — A connected group of sequences with the same modeled phenotype.

- **Mutational robustness** — The fraction of a genotype’s one-letter changes that preserve its phenotype under the chosen map.

- **Evolvability** — Here, a specified count of different phenotypes accessible by mutation, not a complete measure of biological adaptability.

- **Fitness** — A separately defined measure of reproductive success or a model score; it is not determined automatically by the folding map.

### Why this matters

The structure of a genotype–phenotype map helps explain how variation can accumulate without immediately changing a selected outcome, and how new outcomes become accessible. Exact small maps provide benchmarks for sampling methods and expose how conclusions depend on folding rules and tie choices.

### What progress would mean

A complete short-sequence graph would establish precise robustness and connectivity facts for the model. Those facts could motivate biological hypotheses, but would not by themselves predict RNA function, population evolution or a universal transition at some sequence length.

## Core question

For a fully specified combinatorial RNA folding rule on short sequences, what are the exact neutral components, mutation robustness, and accessible phenotype counts?

## Scope and assumptions

Use alphabet {A,C,G,U}, sequence length L=4,…,8 initially, allowed pairs AU/UA, GC/CG and GU/UG, minimum hairpin length three unpaired bases, no pseudoknots, and one pair per base. Maximize pair count and break ties by lexicographically smallest dot-bracket string under a declared character order.

This defines a mathematical GP map, not a thermodynamic RNA predictor. A nearest-neighbor free-energy extension must name its parameter set; empirical parameters are then acknowledged model inputs.

## Mathematical target

Let Φ map each sequence to its deterministic selected structure. On the Hamming graph H(L,4), exact-one-mutation neighbors satisfy d_H(s,t)=1 and each sequence has 3L neighbors. Define

$$r(s)=\frac{|\{t:d_H(s,t)=1,\ Φ(t)=Φ(s)\}|}{3L}.$$

For each phenotype analyze the induced neutral graph and every connected component; a phenotype preimage need not be connected. Define evolvability as the number of distinct other phenotypes reachable by one mutation, stating whether it is measured per genotype, component or whole preimage.

## Required outputs and proof obligations

Export folding rules, tie-breaking, sequence-to-structure table, neutral components, exact counts and rational robustness. A search restricted to a component must not report the full phenotype frequency. Fitness is a separately supplied function; evolutionary dynamics require population size, mutation kernel, selection rule and initial state.

## Validation and rejection controls

Cross-check dynamic programming with exhaustive legal-structure enumeration on short sequences, especially ties and prohibited pairs. Verify Σ_φ|Φ⁻¹(φ)|=4^L and every Hamming vertex degree is 3L. Count neutral edges independently and recover average robustness from twice the edge count. Treat any larger-L sample as statistical evidence.

## Milestones and research extension

Baseline: complete exact maps through L=8. Strong: increase L within measured resources and characterize component-level robustness/evolvability. Research extension: a proved asymptotic property or a specified noisy GP-channel capacity problem. Do not hard-code a universal percolation threshold or empirical target correlation.

Milestones are acceptance gates, not calendar promises. Before a run, freeze the concrete instance/domain, tolerance, compute budget, and permitted dependencies. A bounded search that fails to find a result must preserve an unresolved outcome.

## Starting instruction

Work on the baseline above. First make every input and convention explicit, then validate the reference case. Build the checker alongside the calculation. Label numerical evidence, finite-model certificates, and continuum theorems separately; return the strongest justified result and identify the exact unmet proof obligation.

## Primary references and their role

- [ViennaRNA, RNAfold documentation](https://www.tbi.univie.ac.at/RNA/RNAfold) — RNAfold’s energy model and parameter choices differ from a pair-count model.

References support definitions or limitations; the scope choices and proposed milestones are editorial recommendations, not claims that these results have already been obtained.


# Critique 30: An exact finite genotype–phenotype map and its mutation graph

[Revised problem](../PRDs/30-Genotype-Phenotype-Mapping.md) · [Preserved original](../archive/original-PRDs/30-Genotype-Phenotype-Mapping.md) · [Pinned upstream source](https://github.com/rick-stevens-ai/Pure-Thought/blob/f854d4e12b55c1f5b5c2a964b75f2d23496d981e/PRDs/30-Genotype-Phenotype-Mapping.md)

Line references below refer to the preserved original at commit `f854d4e12b55`, not the rewritten file. “Blocking” means the original issue can invalidate the scientific target or claimed certificate; “major” means it materially changes scope, inference, or acceptance criteria. These are critiques of specifications, not results of running the embedded research-code sketches.

## Detailed findings

1. **Blocking — two folding objectives (16, 61–71, 1083).** Nussinov pair maximization is not nearest-neighbor MFE minimization. Requiring 100% agreement with ViennaRNA’s default energy model is an invalid acceptance test.

2. **Major — nondeterministic ties.** Without an explicit tie-breaking rule, the map is set-valued and neutral-network statistics depend on the chosen representative. The rewrite fixes the map before measuring it.

3. **Major — neighborhood definition (55, 83).** B₁ includes the genotype itself, so the original normalization is not the fraction of one-mutant neighbors. The correct degree is 3L.

4. **Major — feasibility and extrapolation (39, 87, 1102–1105).** 4²⁰=1,099,511,627,776 sequences; exhaustive folding is not a casual short-sequence baseline. A universal L≈30 percolation threshold and fixed robustness correlations do not follow from the model. Start with a complete tractable graph and report sampling separately.

## What the revision preserves and changes

The central motivation is retained, but the executable question is now: **For a fully specified combinatorial RNA folding rule on short sequences, what are the exact neutral components, mutation robustness, and accessible phenotype counts?** The revised scope chooses a concrete starting model; those choices are proposed research-design decisions, not facts inferred from the original. The baseline can succeed without resolving the full open research program.

The replacement also removes the original implementation sketches from the active specification. They remain in the archive for comparison, but have not been made executable or certified. New implementations should derive their equations from the revised definitions rather than copy unchecked prototypes.

## Acceptance criteria to use instead

Baseline: complete exact maps through L=8. Strong: increase L within measured resources and characterize component-level robustness/evolvability. Research extension: a proved asymptotic property or a specified noisy GP-channel capacity problem. Do not hard-code a universal percolation threshold or empirical target correlation.

## Sources supporting the corrections

- [ViennaRNA, RNAfold documentation](https://www.tbi.univie.ac.at/RNA/RNAfold) — RNAfold’s energy model and parameter choices differ from a pair-count model.


---

# Supplemental synthesis topics

# S01. Crystal-field and spin–orbit anisotropy in a finite ion model

**Source:** synthesis challenge 11; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

An ion inside a crystal can respond differently to a magnetic field pointing in different directions. Its neighboring atoms shape the electron’s available states, while the electron’s spin interacts with its orbital motion. This problem asks whether a small, explicitly specified quantum model can establish those directional differences with reliable numerical bounds.

### Key terms

- **Ion** — An atom or group of atoms carrying net electric charge.

- **Crystal field** — The influence of surrounding charges on an ion’s electronic energy levels.

- **Spin** — An electron’s intrinsic quantum angular momentum.

- **Orbital angular momentum** — Quantum angular momentum associated with spatial electronic motion.

- **Spin–orbit coupling** — An interaction linking spin and orbital angular momentum.

- **Hamiltonian** — The operator specifying the model’s energies and quantum evolution.

- **Kramers doublet** — A pair of equal-energy states protected by time-reversal symmetry in this half-integer-spin setting at zero magnetic field.

- **g tensor** — A matrix describing the directional magnetic response of the selected effective doublet.

- **Excitation gap** — The energy separation from the chosen low-energy states to the next states.

### Why this matters

Directional magnetic response is a basic ingredient in understanding magnetic materials and controlling spin states. A finite ion model lets us separate the roles of local geometry and spin–orbit coupling before tackling interactions among many ions.

### What progress would mean

Progress would mean certified response ranges and excitation gaps over the specified parameter region, with clear limits on when the effective doublet description applies. This would establish a local mechanism; predicting a bulk magnet would require additional interactions and material information.

## Revised core question

For a fixed d-shell occupancy and point-group representation, which anisotropy values are possible in a bounded crystal-field/spin–orbit Hamiltonian family?

## Precise scope, target and outputs

Fix one ion model (start with d¹), a chosen point group such as C₄v, and a finite orbital-spin basis. Supply exact matrices for L,S and the group action. Define H(B)=H_CF+λL·S+μ_B B·(L+2S), with crystal-field coefficients and λ in a compact normalized box. Specify the temperature or isolated low-energy multiplet before defining susceptibility or the effective g tensor. For a Kramers doublet, extract a g tensor from the projected magnetic moment and use its singular values, which remove doublet-basis ambiguity. Require a certified excitation gap above that doublet. Optimize a bounded ratio or difference with a denominator bounded away from zero. Export matrices, symmetry identities, interval eigenvalue bounds, and global objective brackets.

## Detailed critique of the synthesis framing

The synthesis’s “all d^n configurations across every point group” has no interaction model, coefficient bounds, or objective normalization. Symmetry permits tensor components but does not determine their magnitudes; arbitrary crystal-field or spin–orbit scaling can make an optimization meaningless. “Rare-earth-like” is a motivation, not an observable. Many-electron d^n ions also require Coulomb/Hund parameters and a specified approximation. The revision retains an anisotropy atlas but begins with a finite, normalized spectral problem and separates a single-ion response from bulk magnetocrystalline anisotropy.

## Validation and milestones

Baseline: exact symmetry-allowed Hamiltonian basis and one doublet g tensor. Strong: certified parameter-box extrema with a doublet-gap condition. Extend to d² only after specifying interactions. Check the isotropic limit, group covariance, and basis invariance of g singular values.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S02. Bandgap optimization in a bounded wave-operator family

**Source:** synthesis challenge 12; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

Some repeating structures prevent waves in certain frequency ranges from traveling through their interior. The challenge is to choose a structure that makes this forbidden range as large as possible within stated material and geometric limits. A gap must survive every propagation direction, including directions missed by a convenient plotting path.

### Key terms

- **Wave operator** — The mathematical rule governing the waves in the chosen physical model.

- **Periodic medium** — A material pattern repeated through space.

- **Unit cell** — One repeating piece of that pattern.

- **Bloch wave** — A wave adapted to a periodic medium, labeled by a wave vector.

- **Brillouin zone** — The set of inequivalent wave vectors needed to describe a periodic medium.

- **Band** — A family of allowed wave frequencies or corresponding operator eigenvalues.

- **Complete bandgap** — A frequency interval containing no bulk wave modes for any wave vector in the specified model.

- **Contrast** — The permitted ratio or variation between material coefficients.

- **Global optimum** — The best value across the entire declared design family, rather than just nearby designs.

### Why this matters

Controlling which waves propagate supports ideas for sound isolation, vibration management and optical filtering. The same mathematical challenge arises across these applications: a promising numerical design needs reliable evidence that its gap exists everywhere and that any optimality claim covers the stated design space.

### What progress would mean

A successful result would certify a nonzero gap, bound the best achievable objective, or establish an optimum within the bounded family. A scalar-wave benchmark would test the method; applying it to electromagnetic devices requires the appropriate Maxwell model and physical constraints.

## Revised core question

Within a fixed finite family of periodic media, what lower bound on a complete bandgap and what upper bound on the best possible gap can be certified?

## Precise scope, target and outputs

Choose either a scalar acoustic operator or a full Maxwell operator, never an unspecified mixture. Initially choose a periodic scalar elliptic operator −∇·a(r)∇ with 0<a_min≤a(r)≤a_max, a fixed unit cell and a finite geometry parameter box. For adjacent Bloch eigenvalues define the complete gap g=min_k λ_{n+1}(k)−max_k λ_n(k), with a declared normalized objective. A grid-local separation is not a complete gap. Use validated eigenvalue bounds plus k-space variation estimates to cover the Brillouin zone. Branch-and-bound over parameters yields a candidate lower bound and an optimization upper bound, keeping the inequality direction explicit.

## Detailed critique of the synthesis framing

The old description mixes photonic and phononic operators, index contrast and generic gap optimization without fixing constitutive laws, geometry regularity, or constraints. “Lowest possible contrast” and “largest possible gap” are two objectives needing an ordering or Pareto definition. A converged plane-wave plot does not establish a complete gap, and a locally optimized geometry does not prove global optimality. This is distinct from PRD 11’s topological invariant: a wide ordinary bandgap can be topologically trivial.

## Validation and milestones

Baseline: one certified complete gap for a supplied geometry. Strong: bound the optimum to a declared tolerance over the entire compact parameter box, or report unresolved boxes. Validate a homogeneous medium and a simple one-dimensional layered case before multidimensional optimization.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S03. Effective conductivity bounds for two-phase composites

**Source:** synthesis challenge 13; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

Mix two materials that conduct heat or electricity differently. The overall conductivity depends on how the materials are arranged, not just how much of each is present. This problem asks for reliable upper and lower bounds on the conductivity of a specified repeating arrangement, and a careful distinction between its performance and limits that apply to every arrangement.

### Key terms

- **Conductivity** — A coefficient relating a driving gradient to the resulting current or heat flux in the selected linear model.

- **Two-phase composite** — A material made from two constituent material types.

- **Volume fraction** — The fraction of space occupied by one constituent.

- **Effective conductivity** — The large-scale response obtained after averaging the microscopic structure.

- **Conductivity tensor** — A matrix allowing that response to depend on direction.

- **Potential** — A scalar field, such as electric potential, whose gradient drives transport.

- **Flux** — The local flow of the transported quantity per unit area.

- **Divergence-free flux** — A steady flow with no internal accumulation or sources.

- **Variational bound** — An inequality obtained by optimizing or testing an energy principle.

### Why this matters

Composite design seeks useful bulk properties from available ingredients. Bounds reveal whether a proposed arrangement performs well, whether a computation is accurate, and how much improvement remains possible. Simple layered structures also provide exact reference cases for checking more complicated methods.

### What progress would mean

Progress would produce a certified interval for a chosen directional response and clarify which bounds are geometry-specific or universal. The result would concern the stated positive, linear conductivities; nonlinear effects, interfaces and manufacturing constraints would need separate treatment.

## Revised core question

For fixed positive phase conductivities and volume fraction, can rigorous effective-conductivity bounds and matching constructions be obtained in a specified microstructure class?

## Precise scope, target and outputs

Choose scalar conductivities a₁,a₂>0, periodic unit cell Y, phase fraction f, and a unit direction e. Define eᵀA_eff e=min_{periodic u}〈a|e+∇u|²〉. Use an admissible potential trial field for an upper bound and a divergence-free flux variational principle for a lower bound. Distinguish bounds valid for every arrangement from optimization over a finite geometry family. Fix isotropy assumptions before comparing isotropic bounds. Cross-property elasticity/thermal results are separate extensions with both constitutive laws specified.

## Detailed critique of the synthesis framing

The original “universal bounds” problem spans conductivity, permittivity and elasticity, including tensor and frequency effects with different variational principles. “Beyond Hashin–Shtrikman” is not a measurable novelty requirement until the symmetry class and available information are fixed. A finite set of candidate microstructures cannot characterize the unrestricted G-closure. The revision makes one scalar static problem precise and makes the bound direction and construction obligation explicit.

## Validation and milestones

Baseline: verify arithmetic and harmonic directional bounds with layered composites. Strong: matching bounds for one declared geometry class or a certified gap to the unrestricted bound. Check volume fraction and ellipticity exactly; report discretization and geometry errors separately.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S04. Finite-volume certification of a disordered topological index

**Source:** synthesis challenge 15; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

A perfectly repeating lattice is convenient for calculating topological properties, but disorder breaks that repetition. This problem asks when a calculation on a finite, disordered patch can prove a topological property of the corresponding infinite system. The key issue is whether the patch is large enough and the model controlled enough for its answer to be trustworthy.

### Key terms

- **Hamiltonian** — The operator describing the lattice system’s energies and couplings.

- **Disorder** — Spatial variation in parameters that need not repeat periodically.

- **Locality** — A restriction on how far or how strongly sites can interact.

- **Fermi energy** — The reference energy separating occupied and unoccupied states in the specified zero-temperature filling model.

- **Spectral gap** — An open energy interval with no spectrum of the bulk Hamiltonian.

- **Topological index** — A discrete quantity that distinguishes phases and stays fixed under allowed gap-preserving changes.

- **Finite volume** — A bounded patch of a larger or infinite lattice.

- **Spectral localizer** — A matrix combining Hamiltonian and position information to compute an index under stated conditions.

- **Mobility gap** — A regime characterized by localization rather than necessarily an empty spectral interval; it requires different hypotheses.

### Why this matters

Topological behavior is interesting partly because it can survive certain imperfections. Establishing that robustness requires methods that work without a perfect crystal and distinguish a finite-sample observation from a bulk conclusion. A rigorous bridge between the two makes numerical evidence more informative.

### What progress would mean

Progress would give checkable size, gap and locality conditions under which a finite calculation equals the bulk index. It would not automatically establish robustness to every disorder strength, a mobility-gap theorem, or the transport response of an experimental device.

## Revised core question

For a local disordered lattice Hamiltonian with a specified bulk spectral-gap hypothesis, when does a finite-volume index provably equal its infinite-volume counterpart?

## Precise scope, target and outputs

Fix a finite-range two-dimensional tight-binding Hamiltonian on Z², bounded disorder, Fermi energy E_F and a bulk spectral gap. Initially use bounded perturbations small enough to preserve a known clean gap. Choose one index construction, such as a spectral localizer, and one theorem relating finite and infinite volume. Define position operators, truncation radius and boundary convention; compute the finite Hermitian signature or index with interval spectral bounds. Verify every locality, commutator, radius and scale inequality in the selected theorem. Mobility-gap cases require a different theorem and are an extension.

## Detailed critique of the synthesis framing

The synthesis lists Bott indices, noncommutative Chern numbers and localizers as if they shared one universal finite-size error bound. They have different hypotheses and finite-volume behavior. An integer from a finite matrix is not automatically the bulk invariant, and one disorder realization does not establish robustness for all disorder. The revised task certifies the finite-to-infinite transfer under a stated gap/locality theorem; it is distinct from the periodic model classification in PRD 09.

## Validation and milestones

Baseline: recover a clean model’s known index with a validated finite-to-bulk theorem. Strong: certify a full bounded disorder class using gap preservation. Include trivial controls and a gap-closing case that the checker declines to certify.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S05. Finite-basis density functionals and exact convex constraints

**Source:** synthesis challenge 17; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

A many-electron state contains vastly more information than the average number of electrons at each site. Can we still describe the minimum energy using only those average occupancies? This problem studies that question in a finite lattice, where the exact relationship can be defined and bounded without pretending that a convenient approximate formula is exact.

### Key terms

- **Density** — Here, the expected electron occupancy at each lattice site.

- **Density functional** — A rule assigning an energy contribution to a full density profile.

- **Finite basis** — A fixed finite collection of states or orbitals used to define the model.

- **Ensemble** — A statistical mixture of allowed quantum states.

- **Constrained search** — Minimization over all states with a prescribed density.

- **Kinetic energy** — The energy associated with electron motion or hopping in the model.

- **Interaction energy** — The energy arising from interactions between electrons.

- **Convexity** — The property that mixing two density profiles cannot raise the functional above the corresponding mixture of functional values.

- **Duality** — An equivalent optimization viewpoint using external potentials to constrain energy values.

- **Exchange–correlation energy** — A remainder defined only after specifying the noninteracting kinetic and Hartree reference contributions.

### Why this matters

Density-based methods reduce a difficult many-electron description to a much smaller set of variables. Exact finite examples reveal which constraints an approximation should respect and provide benchmarks for evaluating proposed energy formulas.

### What progress would mean

A useful result would bracket the functional on the declared density domain and verify its convex and dual structure. It would give a controlled model benchmark, with a separate argument needed before extending conclusions to continuum chemistry or real materials.

## Revised core question

For a fixed finite lattice Hamiltonian, can the ensemble constrained-search density functional be bracketed and its convexity/duality verified?

## Precise scope, target and outputs

Fix a finite N-particle Hilbert space, kinetic T, interaction W, and commuting site-density operators n_i. Define F(n)=min_{ρ≥0,Trρ=1,Trρn_i=n_i}(Trρ(T+W)) on ensemble-representable densities. Derive the dual F(n)=sup_v[E(v)−v·n] with E(v)=min_ρTrρ(T+W+Σv_i n_i), under the finite-dimensional convex formulation. Use exact states for primal upper bounds and potentials with certified E(v) lower bounds for dual lower bounds. If defining an exchange-correlation remainder, specify T_s and Hartree conventions separately.

## Detailed critique of the synthesis framing

“The cone of exchange-correlation functionals satisfying all exact constraints” is not justified: normalization and affine conditions need not define a cone, and convexity of the ensemble universal functional does not imply convexity of E_xc after subtracting other functionals. Constraint satisfaction alone does not identify the exact physical functional. The broad continuum task also leaves admissible density spaces and spin scaling unspecified. The revision first tests exact convex duality in a finite model, preserving nonempirical functional development as an extension.

## Validation and milestones

Baseline: exact functional values and dual brackets on a two-site model. Strong: a certified piecewise characterization or a new valid inequality for a larger finite model. Verify particle-number bounds, convexity inequalities, and agreement with direct constrained search.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S06. Inverse pair-potential design over an explicit competitor class

**Source:** synthesis challenge 20; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

Instead of asking which arrangement particles prefer under known forces, turn the question around: which interaction would favor an arrangement we want? Here, candidate pair interactions are tuned so that a target repeating structure has lower energy than an explicitly listed set of alternatives across a range of densities.

### Key terms

- **Pair potential** — A function giving the interaction energy of two particles as a function of their separation.

- **Inverse design** — Choosing model parameters to produce a desired outcome.

- **Target lattice** — The repeating particle arrangement the design aims to favor.

- **Competitor class** — The precise collection of alternative arrangements against which the target is compared.

- **Number density** — The number of particles per unit volume or area.

- **Energy per particle** — Total interaction energy normalized by the number of particles in the periodic limit.

- **Lattice sum** — A sum of interactions over particles in a repeating structure.

- **Tail bound** — A guaranteed bound on contributions omitted when truncating an infinite sum.

- **Ground state** — A lowest-energy configuration within a stated admissible class.

### Why this matters

Interaction design is a way to explore self-assembly: simple local rules may favor organized structures. Reliable comparisons expose whether apparent success comes from a useful mechanism or from overlooking a competing arrangement or a long-range contribution.

### What progress would mean

Progress would prove the target’s energy advantage, or identify a defeating competitor, within the declared family and density interval. Beating that family alone does not establish the ground state among every possible configuration or show that particles will assemble into it at finite temperature.

## Revised core question

Can a parameterized pair potential make a target periodic structure an energy minimizer among a precisely specified set of competitors over a density interval?

## Precise scope, target and outputs

Fix dimension, density interval, and v_θ(r)=Σ_jθ_jφ_j(r) with bounded coefficients and convergent lattice sums. Define E_θ(Λ)=1/2 Σ_{x∈Λ∖{0}}v_θ(|x|). Begin with a finite list of normalized lattice competitors and use interval sums plus tail bounds to certify E_target≤E_competitor−δ uniformly on a density interval. Continuous competitors require a compact parameterization and global coverage. State whether particle displacements or variable cell size are allowed. General configuration ground states require additional universal lower bounds beyond lattice comparisons.

## Detailed critique of the synthesis framing

The original jumps from optimizing lattice energies to provable crystals over both densities and temperatures. Beating selected lattices does not exclude aperiodic competitors or infinitesimal deformations, and a zero-temperature energy gap is not a finite-temperature free-energy theorem. A strictly uniform energy gap against all arbitrarily close deformations may be impossible under continuity. The rewrite gives a restricted inverse-design result whose proof scope is honest; unrestricted crystallization and thermal stability remain separate research extensions.

## Validation and milestones

Baseline: exact or interval-verified ordering for a finite competitor list. Strong: uniform density/parameter-region certificate. Validate a direct long-cutoff sum against the certified tail bound; label the result “best among these competitors” unless a universal proof is supplied.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S07. Finite-size criteria for parent-Hamiltonian spectral gaps

**Source:** synthesis challenge 23; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

A small quantum chain can have a clear energy separation above its lowest state while that separation shrinks away as the chain grows. This problem asks when an exact calculation on a short segment can rule out that collapse for an entire specified family of longer chains. A theorem connecting local and global behavior is the essential step.

### Key terms

- **Quantum chain** — A one-dimensional sequence of interacting quantum degrees of freedom.

- **Matrix product state** — A structured representation of a many-body state using products of small matrices.

- **Parent Hamiltonian** — A Hamiltonian constructed so that a specified state belongs to its ground-state space.

- **Ground-state space** — The collection of states attaining the minimum energy.

- **Frustration-free** — Having ground states that simultaneously minimize every local interaction term.

- **Spectral gap** — The energy separation above the entire ground-state space.

- **Uniform gap** — A positive lower bound on that gap that does not vanish as chain length increases.

- **Thermodynamic limit** — The limit of arbitrarily large systems.

- **Finite-size criterion** — A theorem translating a suitable local calculation into a conclusion for larger systems.

### Why this matters

The gap helps distinguish low-energy behavior and enters rigorous statements about correlations and stability under suitable conditions. Proving a uniform gap turns a finite numerical observation into a statement about a many-body family, which is often the scientifically relevant object.

### What progress would mean

Progress would verify all assumptions of a chosen finite-size theorem and obtain a positive size-independent lower bound. Failing its threshold would leave the question unresolved rather than prove gaplessness. The result would apply to the specified family, not solve the general spectral-gap problem.

## Revised core question

For a specified frustration-free one-dimensional parent-Hamiltonian family, can a validated local spectral calculation imply a uniform many-body gap?

## Precise scope, target and outputs

Fix exact MPS tensors, on-site dimension, interaction range, boundary convention, and local positive projectors h_i. Verify frustration freeness and ground-space assumptions. Select a precise Knabe-type or martingale theorem appropriate to that model. Compute interval enclosures for the required finite-segment nonzero gaps and verify the theorem’s threshold to derive a uniform lower bound for the stated sizes/boundaries. Degenerate ground spaces require identifying the first strictly positive eigenvalue rather than assuming a unique ground state.

## Detailed critique of the synthesis framing

The synthesis’s jump from MPS to arbitrary PEPS makes a tractable finite-size criterion look like a general spectral-gap decision method. Such general algorithms are obstructed by undecidability results. A positive finite-chain gap alone says nothing uniform as size grows; one must meet a theorem’s hypotheses and threshold. The revision targets certificate automation for an explicit restricted family and preserves unresolved outcomes when the chosen sufficient criterion fails.

## Validation and milestones

Baseline: certify a standard parent model with a valid finite-size theorem. Strong: a uniform parameter interval with an explicit gap lower bound. Test a deliberately gapless/threshold-failing control without claiming that criterion failure proves gaplessness.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Cubitt, Pérez-García and Wolf, Undecidability of the Spectral Gap](https://arxiv.org/abs/1502.04573) motivates the restriction to a specified certifiable family.


---

# S08. Gapped boundaries from exact anyon data

**Source:** synthesis challenge 24; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

In some two-dimensional quantum phases, collective excitations behave neither like ordinary bosons nor fermions. Their possible combinations and exchanges restrict how the phase can end at an edge. This problem asks which proposed ways of absorbing these excitations at a boundary are mathematically consistent and can describe an edge with a nonzero energy gap.

### Key terms

- **Topological order** — A type of quantum organization characterized by nonlocal structure and, here, anyon excitation data.

- **Anyon** — A two-dimensional excitation with exchange or braiding behavior beyond ordinary bosonic and fermionic possibilities.

- **Fusion** — The possible outcomes when excitations are combined.

- **Braiding** — The transformation associated with moving excitations around one another.

- **Topological twist** — The phase associated with a full rotation of an anyon.

- **F and R data** — Numbers or matrices specifying how fusion regrouping and exchanges act in chosen conventions.

- **Condensate** — A collection of excitations treated as absorbable into the boundary vacuum, together with consistent combination rules.

- **Lagrangian algebra** — The mathematical structure expressing a sufficiently complete compatible condensate for the boundary considered here.

- **Gapped boundary** — An edge whose excitations retain a positive energy threshold in the appropriate large-system limit.

### Why this matters

Boundaries determine how a topological phase interfaces with its surroundings and which excitations can disappear or remain distinguishable there. Understanding these options supports the theory of topological phases and the organization of quantum information in such systems.

### What progress would mean

Progress would verify a boundary candidate using complete consistency data, or reject it for a precise algebraic reason. Matching a list of fusion outcomes alone is insufficient. Turning the algebraic result into a particular microscopic boundary Hamiltonian is an additional construction.

## Revised core question

For one fully specified two-dimensional topological order, can candidate condensates and their gapped boundaries be verified from exact categorical data?

## Precise scope, target and outputs

Start with the toric-code modular category, explicit simple objects, fusion, twists, F and R symbols and conventions. Enumerate candidate condensable algebra objects within a declared multiplicity bound. Check multiplication/unit, associativity, commutativity and separability conditions, and the Lagrangian dimension condition for a boundary to vacuum. For the abelian toric-code benchmark recover electric and magnetic condensates. If constructing a lattice boundary, supply its Hamiltonian and check commuting-projector and gap claims separately.

## Detailed critique of the synthesis framing

Fusion rules and modular S,T tables alone do not in general specify or prove all categorical coherence data. The synthesis also mixes 2D anyons with 3D topological orders, which require different structures. A listed condensate is not automatically a microscopic boundary realization. The rewrite retains an exact categorical task, bounds the enumeration and separates algebraic classification from lattice realization and anomaly/inflow interpretation.

## Validation and milestones

Baseline: verify the toric-code coherence data and two standard vacuum boundaries. Strong: a second fully specified category or a completeness proof within a bounded class. Invalid F/R symbols and a nonbosonic condensate are rejection controls.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Li, Yang and Dong, Gapped Boundaries of Kitaev’s Quantum Double Models](https://arxiv.org/abs/2504.19512) provides a primary example of linking Lagrangian algebras to explicit lattice boundary terms.


---

# S09. Stoquasticity under a bounded class of basis changes

**Source:** synthesis challenge 25; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

Quantum calculations can involve large positive and negative contributions that almost cancel, making some sampling methods inefficient. Sometimes a change in the mathematical basis removes troublesome signs. This problem asks whether that is possible within a finite, explicitly allowed set of local basis changes, so that both success and failure can be checked completely.

### Key terms

- **Basis** — The reference states used to write a quantum operator as a matrix.

- **Hamiltonian** — The matrix describing the system’s energy and couplings.

- **Off-diagonal entry** — A matrix element linking two different basis states.

- **Stoquastic** — Having real, nonpositive off-diagonal Hamiltonian entries in the specified basis.

- **Sign problem** — A sampling difficulty caused by signed or complex weights and cancellation in a chosen representation.

- **Pauli operator** — One of the standard single-qubit matrices used to express spin interactions.

- **Clifford transformation** — A basis transformation that maps Pauli operators to Pauli operators under conjugation.

- **Product transformation** — A change of basis applied independently to the individual qubits.

- **Satisfiability certificate** — A checkable assignment meeting encoded constraints, or an appropriate proof that none exists.

### Why this matters

Knowing when signs can be removed helps separate difficulties caused by representation from those that survive a proposed remedy. A bounded search also creates small, exact benchmarks for algorithms intended to find useful basis changes.

### What progress would mean

A successful result would supply and verify a transformation, or exclude every transformation in the allowed finite class. The complete Hamiltonian must be checked because contributions can cancel. Stoquasticity alone does not guarantee fast sampling, and failure for product Clifford changes does not exclude all possible bases.

## Revised core question

For a finite rational Pauli Hamiltonian, is there an allowed product Clifford transformation making all off-diagonal entries real and nonpositive?

## Precise scope, target and outputs

Fix n qubits, rational coefficients, and allowed U=⊗_iU_i with U_i from the finite single-qubit Clifford group modulo phase. Define stoquasticity in the computational basis for the full transformed H, not automatically term by term. Exhaustively enumerate or use an exactly equivalent constraint encoding. Export U and an exact matrix/sign check if successful; otherwise export complete enumeration or a checked UNSAT proof. Continuous local unitaries require a semialgebraic formulation and are a distinct extension.

## Detailed critique of the synthesis framing

The original omits which local basis changes are allowed, so “unavoidable sign problem” has no definite quantifier. A SAT search over finitely many transformations cannot exclude all continuous local rotations. Requiring every term to be stoquastic can also be stronger than requiring their sum to be stoquastic because entries can cancel. Finally, stoquasticity does not guarantee efficient sampling or mixing. The rewrite makes a finite, falsifiable basis-cure problem and limits every no-go conclusion to its transformation class.

## Validation and milestones

Baseline: exhaustive small-n decisions with explicit transformations or coverage. Strong: verified SAT encoding checked against brute force. Extend to a continuous rotation class only with exact equations, inequalities, and global proof machinery.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Marvian, Lidar and Hen, On the computational complexity of curing non-stoquastic Hamiltonians](https://pmc.ncbi.nlm.nih.gov/articles/PMC6450938/) supports the need to specify the allowed basis class.


---

# S10. Certified periodic orbits of a specified three-body model

**Source:** synthesis challenge 27; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

A simulated three-body trajectory may nearly return to where it started, but a small numerical error can disguise whether an exactly repeating orbit exists. This problem converts a promising numerical trajectory into a proof that a nearby periodic solution exists and stays away from collisions, then examines how small disturbances behave over one cycle.

### Key terms

- **Three-body problem** — The motion of three mutually gravitating bodies in the specified Newtonian model.

- **Restricted model** — A version in which one body has negligible mass and does not affect the other two.

- **Periodic orbit** — A trajectory that returns to the same state after a fixed positive time.

- **Shooting method** — Adjusting an initial state until its evolved state satisfies the return conditions.

- **Phase condition** — A rule fixing the arbitrary starting point along a periodic trajectory.

- **Validated integration** — Numerical evolution with guaranteed enclosures for errors.

- **Collision exclusion** — A proved positive lower bound on relevant body separations along the orbit.

- **Floquet multiplier** — An eigenvalue describing linear perturbation growth or rotation after one orbit.

- **Neutral direction** — A direction with constrained or trivial linear behavior arising from symmetries or conservation laws.

### Why this matters

Periodic orbits are organizing reference motions in gravitational dynamics. Certifying their existence gives subsequent studies of nearby motion a dependable starting point and separates exact solutions from visual patterns in a simulation.

### What progress would mean

Progress would enclose an orbit and its period, exclude collisions, and classify the reduced linear stability when the bounds permit. Linear stability would not alone prove nonlinear or indefinite stability, and a periodic-orbit proof would not establish connecting trajectories between different orbits.

## Revised core question

Can a numerically proposed periodic orbit be enclosed by a validated shooting proof and classified by its reduced Floquet spectrum?

## Precise scope, target and outputs

Choose one nondimensional Newtonian three-body model with positive rational masses, or a planar circular restricted model with fixed mass ratio. Declare collision exclusions and conserved quantities. Define a Poincaré section or phase condition to remove time translation, and reduce continuous symmetries. Solve the resulting shooting residual using a validated flow integrator and interval Newton/Krawczyk bounds for initial conditions and period. Integrate variational equations with enclosures for the monodromy matrix; remove the expected neutral directions before interpreting multipliers.

## Detailed critique of the synthesis framing

The original bundles orbit existence, complete family discovery, linear stability, invariant manifolds and heteroclinic connections into one certificate. A small shooting residual is not existence, and stable-looking Floquet multipliers are not nonlinear stability. Near-return trajectories do not establish a connection between invariant manifolds. The revised first target is a single orbit with a phase-fixed existence proof; continuation, manifold enclosure and connection proofs are explicit later stages. This also preserves the figure-eight idea incorrectly folded into PRD 27’s central configurations.

## Validation and milestones

Baseline: certify one supplied orbit and its period. Strong: validated continuation over a parameter interval and a reduced Floquet classification. A manifold connection requires additional covering/intersection proof, not a plot of approaching trajectories.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S11. Minimal reaction implementations of a specified computational behavior

**Source:** synthesis challenge 29; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

A reaction network can encode inputs as molecular amounts and produce an output through its reactions. But calling a network a computer is ambiguous unless we specify what counts as the input, the answer and an acceptable error. This problem asks for the smallest network that implements one precisely defined behavior within a bounded set of permitted reactions.

### Key terms

- **Reaction network** — A set of chemical species and rules for converting reactants into products.

- **Reaction grammar** — The restrictions defining which species and reaction forms are allowed.

- **Encoding** — The mapping from abstract inputs to initial molecular states or concentrations.

- **Boolean function** — A rule mapping a finite collection of binary inputs to a binary output.

- **Deterministic semantics** — A model in which concentrations follow fixed evolution equations.

- **Stochastic semantics** — A model in which individual reaction events occur randomly.

- **Readout** — The rule and time for translating the chemical state into an output.

- **Error bound** — A limit on deviation from the required output, or on the probability of an incorrect answer.

- **Cost model** — The explicit quantity being minimized, such as reaction or species count.

- **Universality** — The ability to implement arbitrary computations under a stated encoding and resource scheme.

### Why this matters

Chemical computation explores how local reactions can perform information processing, including ideas relevant to molecular sensing and control. Precise semantics make proposed constructions comparable and prevent hidden resources or flexible readout choices from creating misleading claims of simplicity.

### What progress would mean

Progress would provide a verified implementation and, if claimed, prove minimality within the bounded grammar and cost model. Implementing a finite truth table would establish that task only; computational universality requires a construction for arbitrary programs and an explicit account of resources.

## Revised core question

Within a bounded reaction grammar, what is the smallest network implementing a precisely defined input/output behavior with a proved error bound?

## Precise scope, target and outputs

Choose deterministic mass-action or stochastic reaction semantics, species/reaction bounds, stoichiometric arity, input encoding, output decoding and cost (species count first, then reaction count, for example). Begin with a finite Boolean input/output function and an explicit settling-time/error requirement. Enumerate networks modulo species relabeling; prove correctness over the complete allowed input set and exclude every lower-cost network within the grammar. For a universal-computation extension, supply a uniform compiler and a simulation theorem covering arbitrary program lengths, resource scaling and precision.

## Detailed critique of the synthesis framing

The synthesis joins minimal autocatalysis, self-replication and universal computation, but none implies the others. “Smallest chemistry” depends on kinetics, encoding, concentration precision and which resources count as free. A fixed finite truth table does not prove Turing universality, while an analog model with unbounded precision may hide computation in its inputs. The rewrite separates a bounded minimal implementation result from a universal simulation theorem and from the RAF structure in PRD 29.

## Validation and milestones

Baseline: one certified reaction implementation with explicit semantics. Strong: minimality within a finite grammar and input domain. Extension: a compiler/simulation theorem; avoid calling the baseline universal or biologically self-replicating.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

# S12. Capacity of a specified finite noisy genotype–phenotype channel

**Source:** synthesis challenge 30; [original synthesis](../NOTES/30-Most-Compelling-Pure-Thought-AI-Challenges.md). This topic is distinct from, or adds a distinct objective to, the canonical PRD catalog. [Crosswalk](../reviews/SYNTHESIS-CROSSWALK.md)

## Plain-language guide

### The problem in everyday terms

Suppose the same genetic sequence can produce different modeled outcomes because of noise. How reliably can an observer infer the chosen sequence from the outcome? This problem treats the relationship as an explicitly given communication channel and asks for certified bounds on the greatest information it can transmit under any stated input-cost limit.

### Key terms

- **Genotype** — An input genetic sequence or label in the finite model.

- **Phenotype** — An output trait or state as defined by the model.

- **Channel law** — The conditional probability of each output given each input.

- **Noise** — Variation in the output for a fixed input, as encoded by that channel law.

- **Input distribution** — The probabilities assigned to selecting the possible genotypes.

- **Mutual information** — A measure of how much observing the output reduces uncertainty about the input.

- **Shannon capacity** — The largest mutual information over the allowed input distributions for this channel.

- **Bit** — The information unit obtained when logarithms use base two.

- **Cost constraint** — A bound on average input cost, such as a prescribed sequence-dependent resource measure.

- **Relative entropy** — A measure of discrepancy between probability distributions, also called Kullback–Leibler divergence.

### Why this matters

An explicit channel separates the number of possible outputs from their reliability and distinguishability. Capacity bounds give a rigorous way to compare modeled noise mechanisms or resource constraints, and to check whether a numerical optimization has found the best input distribution.

### What progress would mean

Progress would bracket capacity with a feasible input distribution and a valid upper certificate. For a deterministic unconstrained channel, the number of reachable outputs supplies a simple benchmark. The answer describes the specified probabilities and costs; it is not an intrinsic information value of a genome or a measure of evolutionary fitness.

## Revised core question

For a finite conditional phenotype law P(y|g) and optional genotype cost constraint, what certified lower and upper bounds can be placed on Shannon capacity?

## Precise scope, target and outputs

Supply finite genotype and phenotype alphabets, a stochastic matrix W(y|g), and optional costs c(g) with E[c]≤C. Define capacity max_p I_p(G;Y), with logarithms base two. A chosen input distribution gives a lower bound. For the unconstrained channel, any output distribution q with adequate support gives upper bound max_g D(W(.|g)||q); constrained dual bounds must include the cost multiplier. Use interval logarithms and validated probabilities. Connect to PRD 30 by constructing W from a fully declared folding/noise rule.

## Detailed critique of the synthesis framing

Mutual information is a property of a joint distribution, not a map alone. The original omits the input distribution, noise, phenotype alphabet and optimization constraints. For a deterministic finite map with unrestricted input distribution, capacity is simply log₂ of the number of reachable phenotypes, so a purported hard capacity problem may collapse. Fano/Le Cam bounds answer specific inference questions and are not automatic capacity certificates. The rewrite specifies a nontrivial noisy channel, separates achievability from converse bounds, and keeps biophysical interpretation conditional on the chosen channel.

## Validation and milestones

Baseline: binary symmetric-channel capacity and deterministic-map capacity reproduced. Strong: a certified bracket for a finite GP channel and a cost-constrained extension with a valid dual. Check rows sum to one, handle zero probabilities explicitly, and report unresolved numerical gaps.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


---

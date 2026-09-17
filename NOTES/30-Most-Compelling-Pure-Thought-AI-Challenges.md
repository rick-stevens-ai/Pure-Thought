# Thirty synthesis challenges: revised problem summaries

Updated 2026-09-17. This synthesis retains the original synthesis numbering. It is a different portfolio from the canonical 30 PRDs. Every topic is mapped to its revised destination; the 12 distinct supplemental objectives are preserved. Claims and milestones follow the updated review and evidence standard.

[All 42 complete summaries](../COMPLETE-PROBLEM-SUMMARIES.md) · [Full crosswalk](../reviews/SYNTHESIS-CROSSWALK.md) · [Revised synthesis PDF](30-Pure-Thought-Challenges-Report.pdf) · [Preserved original](../archive/original-synthesis/30-Most-Compelling-Pure-Thought-AI-Challenges.md)


# Synthesis 01. AdS₃ "Pure Gravity" via the Modular Bootstrap

**Mapping:** [PRD 01](../PRDs/01-AdS3-Modular-Bootstrap.md). Covered by canonical revision.

## 01. Extremal holomorphic CFT partition functions: certified necessary tests

**Question.** For c=24k, which extremal holomorphic partition-function candidates pass exact genus-one spectral tests, and can any candidate be excluded by an additional, explicitly stated consistency condition?

**Why it matters.** Quantum gravity is hard partly because we do not know which mathematically plausible theories are internally consistent. These tests can eliminate candidates without experiments or a complete construction. They also connect gravity to the arithmetic of modular functions: counting quantum states becomes a problem about exact number sequences.

**Scope and model.** Start with bosonic holomorphic theories, a unique vacuum, c=24k, and k=1,…,4. This is a deliberately restricted version of the original gravity question. Use h for holomorphic weight; do not identify it with the full scaling dimension Δ=h+h̄. The extremal ansatz has no nonvacuum Virasoro primary with h≤k. Treat the nonchiral modular bootstrap as a separate extension with independent (h,h̄) data.

Keep the motivating existence question, but make the first deliverable a necessary-condition test. A modular function with integer coefficients is not a constructed CFT or a demonstrated gravity dual.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/01-AdS3-Modular-Bootstrap.md).

For the holomorphic sector with $c=24k$, $q=e^{2\pi i\tau}$ and $J=j-744$,

$$\chi_0=q^{-k}\prod_{n=2}^{\infty}(1-q^n)^{-1},\qquad
\chi_h=\frac{q^{h-k+1/24}}{\eta(\tau)}\quad(h>0).$$

$$Z_k=P_k(J)=\chi_0+\sum_{h\ge k+1}d_h\chi_h,\qquad
P_k(J)-\chi_0=O(q).$$

Determine $P_k$ exactly and test $d_h\in\mathbb Z_{\ge0}$ through a declared $H$. At $k=1$, $d_2=196883$. Passing this finite necessary test does not construct the CFT or establish OPE consistency.

**Deliverables.** Export the polynomial coefficients, exact q-series, primary multiplicities, conventions, cutoff, and a checker that independently expands η and J. An exclusion must name the violated necessary condition. An additional genus-two or OPE test must state its own assumptions and how its constraints follow from CFT consistency.

**Validation.** Recover J=q⁻¹+196884q+… at k=1 and d₂=196883 after subtracting the vacuum descendant. Test the checker with a deliberately altered coefficient. Check agreement of two series-construction routes. Do not report all-order coefficient positivity without a tail theorem.

**Milestones.** Baseline: exact k=1,…,4 tables with explicitly finite coverage. Strong: one additional sewing/OPE condition with independently checked consequences. Research extension: a new exclusion theorem or a full CFT construction; neither is a required outcome or a promised deadline.

[Detailed critique](../reviews/01-AdS3-Modular-Bootstrap.md).


# Synthesis 02. Gravitational Positivity & Causality Bounds (Flat Space)

**Mapping:** [PRD 02](../PRDs/02-Gravitational-Positivity-Bounds.md). Covered by canonical revision.

## 02. Conditional positivity bounds for a specified gravitational amplitude

**Question.** Under a fully stated infrared prescription and high-energy assumption, what certified inequality constrains one selected low-energy coefficient of scalar scattering coupled to gravity?

**Why it matters.** These bounds can tell us that an apparently sensible low-energy model cannot fit particular assumptions about its underlying physics. That is valuable when direct access to microscopic gravity is unavailable. A reliable bound also prevents researchers from interpreting an artifact of a singular calculation as a physical restriction.

**Scope and model.** Use four-dimensional scattering of one identical real massive scalar coupled to Einstein gravity. This replaces the redundant schematic pure-gravity R² coefficient list with an amplitude-defined observable. Set m=1 as the mass unit; specify κ, the EFT subtraction scale, perturbative order, and whether the target is a tree coefficient or a renormalized coefficient. Graviton helicity amplitudes are a later extension.

The first task is to establish a valid sum rule. Do not run an optimization until the pole subtraction, massless cuts, crossed channel, and large-contour contribution have been derived.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/02-Gravitational-Positivity-Bounds.md).

$$s+t+u=4m^2,\qquad \nu=s-2m^2+\frac t2,\qquad
b_2(t)=\frac12\partial_\nu^2 B(0,t).$$

$$B=A-A_{\mathrm{light\ poles}}-A_{\mathrm{chosen\ low\ cuts}},\qquad
b_2(t)=I_{\mathrm{UV}}(t)+R_\infty(t).$$

The last equality is the sum rule to derive, with its kernel, subtractions and contour assumptions. Only a justified smearing or regulated forward prescription can turn it into a positivity statement. If that prescription proves $I_{\mathrm{UV}}\ge0$ and $R_\infty\ge-R_*$, then $b_2\ge-R_*$. An uncontrolled contour term must remain unknown.

**Deliverables.** Return a derivation dossier, normalized coefficient definition, assumptions, and either a conditional inequality b₂≥L or an explanation of the missing hypothesis. If optimizing, export a dual functional with certified kernel positivity and all truncation errors. A surviving coefficient is “not excluded by these tests,” not “UV complete.”

**Validation.** Check s↔u symmetry, dimensional consistency, and pole residues. Reproduce the nongravitational forward positivity argument in the κ→0 limit. Use a fully specified example amplitude to test the subtraction algebra. Perturb a certificate beyond its margin and verify rejection.

**Milestones.** Baseline: one correct sum rule and its gravity-free limit. Strong: a numerically useful certified conditional bound with a finite positive margin. Research extension: improve the bound or relax a high-energy assumption; label any sharpness claim with both matching constructions and upper/lower evidence.

[Detailed critique](../reviews/02-Gravitational-Positivity-Bounds.md).


# Synthesis 03. Celestial CFT Bootstrap

**Mapping:** [PRD 03](../PRDs/03-Celestial-CFT-Bootstrap.md). Covered by canonical revision.

## 03. Celestial amplitudes: distributional crossing and soft-limit consistency

**Question.** Can a regulated celestial transform of a specified four-graviton tree amplitude reproduce Lorentz covariance, crossing, and the leading conformally soft residue in one consistent convention?

**Why it matters.** A successful reformulation could expose symmetries or constraints that are difficult to see in ordinary scattering variables. That would give theorists another way to investigate gravity. But a change of mathematical language is useful only if singularities, conservation laws and limiting cases survive the translation correctly.

**Scope and model.** Fix four-dimensional Einstein gravity at tree level, a nonzero four-graviton helicity amplitude, incoming/outgoing signs εᵢ, and pᵢ=εᵢωᵢq(zᵢ,z̄ᵢ). Declare metric, helicity, normalization, and analytic-continuation conventions. The target is an identity pipeline for amplitudes, not a classification of celestial CFTs.

Use Δᵢ=1+iλᵢ on the principal series where appropriate and hᵢ=(Δᵢ+Jᵢ)/2, h̄ᵢ=(Δᵢ−Jᵢ)/2. Identify explicitly which continuations away from that contour are used for soft residues.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/03-Celestial-CFT-Bootstrap.md).

$$p_i=\epsilon_i\omega_i q_i,\quad
\Delta_i=1+i\lambda_i,\quad
h_i=\frac{\Delta_i+J_i}{2},\quad
\bar h_i=\frac{\Delta_i-J_i}{2}.$$

$$\widetilde{\mathcal A}=
\left(\prod_{i=1}^4\int_0^\infty d\omega_i\,\omega_i^{\Delta_i-1}\right)
\delta^{(4)}\!\left(\sum_i\epsilon_i\omega_iq_i\right)\mathcal M_4.$$

The expression is a regulated distribution. A leading soft term proportional to $\omega_s^{-1}$ yields the local integral $\int_0^\Lambda\omega_s^{\Delta_s-2}d\omega_s=\Lambda^{\Delta_s-1}/(\Delta_s-1)$ before continuation, exposing the pole at $\Delta_s=1$. Support, helicity, crossing and regulator removal remain proof obligations; ordinary Euclidean OPE positivity is not assumed.

**Deliverables.** Produce exact momentum-space expressions, regulated transform formulas, support conditions, and symbolic covariance/crossing/soft identities. If a transform cannot be defined under the chosen assumptions, return a precise obstruction at that step. A positive bootstrap is an extension only after deriving an applicable inner product and positivity statement.

**Validation.** Check the original amplitude’s factorization first. Test Mellin formulas against analytic one-variable examples in their convergence strips. Verify soft residues before and after transformation under stated limit-interchange hypotheses. Use smeared distributions for numerical checks; never evaluate a delta distribution as an ordinary function.

**Milestones.** Baseline: one four-point transform with explicit support and checked leading soft residue. Strong: a second helicity/process example and a verified crossing relation. Research extension: derive a justified positive constraint in a restricted subsector before attempting an SDP island.

[Detailed critique](../reviews/03-Celestial-CFT-Bootstrap.md).


# Synthesis 04. Modular-Lightcone Bootstrap for Holographic CFTs

**Mapping:** [PRD 04](../PRDs/04-Modular-Lightcone-Bootstrap.md). Covered by canonical revision.

## 04. Lightcone bootstrap with controlled large-spin remainders

**Question.** For one scalar four-point function in a unitary CFT with d>2, what controlled bounds on large-spin double-twist data follow from specified low-twist exchanges?

**Why it matters.** Consistency can reveal features shared by many strongly interacting theories, even when their detailed dynamics cannot be solved. Large-spin patterns are also useful in theories proposed to describe gravity through a lower-dimensional boundary. Controlled errors turn suggestive asymptotic patterns into statements that can actually constrain a finite calculation.

**Scope and model.** Start in d=3 with an identical scalar φ of fixed dimension Δφ. Specify the normalization of C_T, the stress-tensor OPE coefficient fixed by its Ward identity, and an explicit twist gap above the low-twist exchanges retained. Use a large-C_T expansion only when a factorization assumption and its error order are stated.

This challenge studies analytic large-spin behavior. Challenge 07 instead studies the spinning stress-tensor correlator. Torus modular invariance is restricted to a separate d=2 variant.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/04-Modular-Lightcone-Bootstrap.md).

$$F_{\Delta,\ell}=v^{\Delta_\phi}g_{\Delta,\ell}(u,v)
-u^{\Delta_\phi}g_{\Delta,\ell}(v,u),\qquad
F_{\mathrm{id}}+\sum_{\mathcal O}\lambda_{\phi\phi\mathcal O}^{2}F_{\Delta,\ell}=0.$$

$$\Delta_{n,\ell}=2\Delta_\phi+2n+\ell+\gamma_{n,\ell},\qquad
|\gamma_{0,\ell}-\widehat\gamma_{0,\ell}|\le R(\ell)\quad(\ell\ge\ell_0).$$

The inequality is a target requiring explicit spectral and Regge assumptions and a proved remainder. An asymptotic series alone does not imply this finite-spin inequality. Higher-dimensional crossing is not torus modular invariance.

**Deliverables.** Export block normalizations, low-twist input, the derivation or inversion formula, coefficients, and a decomposition of R into omitted exchanges, expansion terms, and numerical errors. State clearly whether the result is an identity, a conditional bound, or an asymptotic estimate.

**Validation.** Use generalized-free-field crossing as an algebraic benchmark, while noting that it is not by itself the target local CFT with a finite stress tensor. Recover double-twist accumulation in a known analytic example. Verify every uniform estimate on the specified cross-ratio domain and all spins covered by the claim.

**Milestones.** Baseline: reproduce an established leading large-spin result with conventions checked. Strong: obtain one explicit remainder estimate, or identify the extra data needed to obtain it. Research extension: a tighter conditional bound on low-twist exchanges or bulk couplings with a stated holographic dictionary.

[Detailed critique](../reviews/04-Modular-Lightcone-Bootstrap.md).


# Synthesis 05. Positive Geometry for Gravity

**Mapping:** [PRD 05](../PRDs/05-Positive-Geometry-Gravity.md). Covered by canonical revision.

## 05. Positive-geometry tests for a fixed gravity integrand

**Question.** For a specified low-loop supergravity integrand, does a precisely bounded class of candidate positive geometries reproduce its differential form, including every boundary and pole at infinity?

**Why it matters.** A geometric organization could explain cancellations and suggest more efficient calculations or new identities in quantum gravity. Even a carefully limited impossibility theorem is informative: it tells us which geometric proposals are too restrictive and where a different idea is needed.

**Scope and model.** Start with the four-point one-loop N=8 supergravity integrand, in a declared four-dimensional representation and helicity sector. Record any regulator and the limits of four-dimensional cut reconstruction. Derive or reproduce the target before searching. Tree-level and scalar-box canonical forms are calibration examples.

Define the search class before claiming an obstruction: ambient coordinates, dimension, allowed rational maps, inequalities, permitted external prefactors, and bounds on degree/number of boundaries. “No geometry exists” is not a finite computational question without such restrictions.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/05-Positive-Geometry-Gravity.md).

$$\Omega_{\mathrm{target}}=I(\ell;k)\,d^4\ell,
\qquad \Omega_{\mathrm{target}}\stackrel{?}{=}f(k)\,\pi_*\Omega_G.$$

For a standard positive geometry, each oriented boundary must satisfy the canonical residue relation $\operatorname{Res}_{\partial G}\Omega_G=\Omega_{\partial G}$. Fix the geometry class, maps, prefactors, representation and equality convention first. Verify finite boundaries, intersections and infinity; an obstruction is limited to that class. An integrated symbol is not the integrand's canonical form.

**Deliverables.** Return either an explicit G, map, canonical form, and exact equality check, or a proof excluding the declared ansatz class. A failed search returns “unresolved within budget.” A dlog decomposition alone is a candidate diagnostic, not a geometry certificate.

**Validation.** Verify simplex/box normalization independently. Check factorization and generalized cuts of the target. Rational reconstruction must be followed by an exact polynomial identity after denominators are cleared, with singular loci accounted for. Check orientation signs and residues at infinity explicitly.

**Milestones.** Baseline: exact target and pole/residue audit. Strong: one verified representation or one restricted no-go theorem. Research extension: broaden the geometry class or multiplicity; explain how the result changes understanding of gravity amplitudes rather than treating a computational failure as a universal obstruction.

[Detailed critique](../reviews/05-Positive-Geometry-Gravity.md).


# Synthesis 06. Non-perturbative S-matrix Bootstrap with Gravity

**Mapping:** [PRD 06](../PRDs/06-Nonperturbative-S-matrix-Bootstrap.md). Covered by canonical revision.

## 06. S-matrix bootstrap with an explicit gravitational infrared prescription

**Question.** Can one certify an exclusion for a low-energy scalar-scattering parameter using a crossing-symmetric partial-wave relaxation whose infrared and truncation assumptions are explicit?

**Why it matters.** The method could identify restrictions shared by many possible microscopic theories. That is useful when no single theory is established. Its reliability depends on not mistaking a limitation of the search formula, grid or cutoff for a limitation imposed by physics.

**Scope and model.** Begin with identical massive scalars in four dimensions and no gravity to validate the machinery. The gravitational stage must first define an infrared-finite observable or a justified regulator/pole treatment, including massless radiation. Keep one scalar mass and one amplitude coefficient fixed; optimize a second coefficient in a declared normalization.

This is complementary to Challenge 02: the focus is amplitude feasibility and partial-wave constraints rather than deriving a low-energy positivity sum rule. A nongravitational baseline does not count as solving the gravitational stage.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/06-Nonperturbative-S-matrix-Bootstrap.md).

$$A=16\pi\sum_{\ell\ge0}(2\ell+1)a_\ell(s)P_\ell(\cos\theta),\quad
\rho=\sqrt{1-4m^2/s},\quad S_\ell=1+2i\rho a_\ell.$$

$$|S_\ell|\le1\quad\Longleftrightarrow\quad
\operatorname{Im}a_\ell\ge\rho|a_\ell|^2\qquad(s>4m^2).$$

Equality requires an elastic sector. A certified outer relaxation can exclude a physical amplitude only after the inclusion of every admissible amplitude and all tail bounds are proved. Failure of an inner ansatz excludes only that ansatz. The gravitational stage has a separate infrared prerequisite.

**Deliverables.** Export the amplitude conventions, IR prescription, finite optimization problem, mapping from the continuum problem, and dual certificate with its positivity/error margins. Feasible ansatz points are examples satisfying the verified constraints only. If the IR construction or tail control fails, state which conclusion remains unavailable.

**Validation.** Check free scattering and a specified analytic nongravitational benchmark. Verify the unitarity disk algebra independently. Test between grid points and beyond cutoffs using proved bounds. Demonstrate that known admissible examples are not falsely excluded. Increasing precision is a diagnostic; rigorous enclosures are needed for certification.

**Milestones.** Baseline: certified finite nongravitational relaxation. Strong: justify an IR treatment and certify one conditional gravitational exclusion. Research extension: improved continuum bounds or a constructive amplitude with a precisely delimited consistency claim.

[Detailed critique](../reviews/06-Nonperturbative-S-matrix-Bootstrap.md).


# Synthesis 07. Extremal Higher-Dimensional CFTs with Stress Tensor

**Mapping:** [PRD 07](../PRDs/07-Extremal-CFTs-Stress-Tensor.md). Covered by canonical revision.

## 07. Stress-tensor bootstrap: a sector-specific gap bound in three dimensions

**Question.** At fixed stress-tensor two- and three-point data, can crossing exclude a proposed gap in the parity-even scalar sector of T×T in a unitary three-dimensional CFT?

**Why it matters.** The stress tensor is present in a broad class of local theories, so restrictions derived from it can have wide reach. Such restrictions help map which strongly interacting quantum theories are possible. They can also constrain proposed boundary descriptions of gravity, where the stress tensor has a special role.

**Scope and model.** Fix d=3, parity invariance, a unique conserved stress tensor, a normalization for C_T, and the remaining independent TTT parameter. The initial observable is the first parity-even nonidentity scalar appearing with nonzero coupling in T×T. It is not a bound on every operator in the theory. State separate gaps for any other restricted spin/parity sectors.

Use the full conserved spinning correlator. A scalar block toy implementation is permitted for learning but is not the target stress-tensor bootstrap.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/07-Extremal-CFTs-Stress-Tensor.md).

$$V_{\mathrm{id}}+V_T(C_T,t_4)+
\sum_{\mathcal O}\lambda_{\mathcal O}^{\mathsf T}V_{\Delta,\ell,\pi}
\lambda_{\mathcal O}=0.$$

$$\alpha(V_{\mathrm{id}}+V_T)>0,\qquad
\alpha(V_{\Delta,\ell,\pi})\succeq0
\quad\text{for every allowed exchange}.$$

These inequalities contradict crossing and exclude the proposed sector-specific gap. Tensor structures, conserved spinning blocks, Ward identities and continuous exchange coverage are required. In three dimensions a conserved symmetric-traceless spin-$\ell$ primary has $\Delta=\ell+1$.

**Deliverables.** Export tensor/block conventions, conservation and Ward identities, fixed input data, derivative basis, spin treatment, and functional coefficients. Attach rigorous block-approximation and high-spin control. If only the finite approximant is checked, label the result accordingly.

**Validation.** Recover a selected published stress-tensor benchmark using exactly its conventions and assumptions. Check the allowed region rather than the excluded region. Verify matrix positivity, not just individual scalar components. Free scalar/fermion data provide normalization tests; numerical agreement alone is not a proof of the continuum bound.

**Milestones.** Baseline: reproduce one finite benchmark and publish the assumptions. Strong: turn one exclusion into a certified result with omitted-sector control. Research extension: improve a bound, add another sector, or obtain a conditional extremal spectrum. Saturation does not construct a CFT.

[Detailed critique](../reviews/07-Extremal-CFTs-Stress-Tensor.md).


# Synthesis 08. Swampland via Modularity & Higher-Form Symmetries

**Mapping:** [PRD 08](../PRDs/08-Swampland-Modularity-Symmetries.md). Covered by canonical revision.

## 08. Exact modular-data and symmetry-anomaly consistency tests

**Question.** For supplied rational-CFT modular data and a specified finite internal symmetry, which algebraic consistency tests can be certified, and which failures obstruct a proposed gauging?

**Why it matters.** Exact consistency checks can detect false candidate data before researchers invest in a full construction. They also clarify which symmetries can be gauged and which require additional structure, such as anomaly cancellation or inflow from another system. This is useful both for studying quantum field theories and for interpreting holographic gravity proposals.

**Scope and model.** Use a bosonic rational CFT with a declared chiral algebra and finitely many characters. Input exact algebraic S,T matrices, vacuum label, and a nonnegative integer multiplicity matrix M. Start with the Ising model. An optional finite 0-form group G requires separate defect/twisted-sector data and an anomaly cocycle; these cannot be inferred from the untwisted partition function alone.

The result concerns the supplied CFT data and gauging problem. A general bulk quantum-gravity compatibility verdict is outside the baseline.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/08-Swampland-Modularity-Symmetries.md).

$$S^\dagger S=I,\quad S^2=C,\quad(ST)^3=C,\quad
MS=SM,\quad MT=TM,\quad M_{00}=1,\quad M_{ij}\in\mathbb Z_{\ge0}.$$

$$N_{ij}^{\ k}=\sum_a\frac{S_{ia}S_{ja}S_{ka}^*}{S_{0a}}
\in\mathbb Z_{\ge0},\qquad \omega\in Z^3(G,U(1)),\quad\delta\beta=\omega.$$

Fix the $T$ convention and any assumed symmetry of $S$. An explicit cochain $\beta$ trivializes the specified anomaly; a complete cohomology calculation may prove no such cochain exists. This gauging obstruction does not make an anomalous boundary theory inconsistent, and untwisted modular data alone do not supply defect data.

**Deliverables.** Return pass/fail/unresolved for each named axiom, with exact arithmetic traces. Include the number field and embeddings. Even passing every modular-data test gives necessary algebraic consistency only; full CFT sewing and realizability are separate obligations.

**Validation.** Verify Ising S entries including 1/√2 in algebraic arithmetic. Reject altered modular relations, noninteger multiplicities, and a failed cocycle identity. Accept the existence of a boundary global symmetry rather than flagging it as automatically inconsistent. Cross-check a small cyclic-group cohomology example.

**Milestones.** Baseline: exact modular-data checker on Ising and one second supplied model. Strong: one finite-group twisted-sector/gauging analysis. Research extension: classify a explicitly finite family of data or derive a conditional bulk consequence using a stated holographic theorem.

[Detailed critique](../reviews/08-Swampland-Modularity-Symmetries.md).


# Synthesis 09. Topological Band Theory Without Materials Data

**Mapping:** [PRD 09](../PRDs/09-Topological-Band-Theory.md), [PRD 15](../PRDs/15-Topological-Quantum-Chemistry.md). Covered by canonical revision.

## 09. Certified Chern phases in a bounded tight-binding family

**Question.** Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?

**Why it matters.** Topological models explain why some properties depend on global structure rather than microscopic details. A certified phase map provides reliable examples for theory and simulation, and it helps identify which ingredients a proposed design actually needs. Proving a gap everywhere is important because a small missed gap closing can invalidate a computed topological label.

**Scope and model.** Begin with the two-band square-lattice model $H_m(k)=\sin(k_x)\sigma_x+\sin(k_y)\sigma_y+[m+\cos(k_x)+\cos(k_y)]\sigma_z$, k∈[−π,π]². Use one occupied band and a fixed orientation. Analyze rational parameter intervals separated from m=−2,0,2.

For subsequent searches fix dimension, orbitals, filling, hopping support, coefficient domain, and symmetry representation. A rational grid is finite; a continuous coefficient box requires a different completeness argument. “All Hamiltonians for a space group” is not an enumerable specification.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/09-Topological-Band-Theory.md).

$$H_m(k)=\sin k_x\,\sigma_x+\sin k_y\,\sigma_y+
(m+\cos k_x+\cos k_y)\sigma_z.$$

$$E_{r+1}(k)-E_r(k)\ge\delta>0,\qquad
C=\frac{1}{2\pi i}\int_{\mathrm{BZ}}
\operatorname{Tr}\!\left(P[\partial_{k_x}P,\partial_{k_y}P]\right)\,d^2k.$$

Prove the gap over the full claimed parameter and momentum domain. Fix orientation and the occupied projector. An analytic argument or validated integral enclosing one integer certifies $C$; a momentum mesh alone does not.

**Deliverables.** Return model coefficients, gap lower bounds, invariant and sign convention, certified parameter coverage, and uncovered boxes. A minimality claim must specify whether it minimizes orbitals, hopping range, or coefficients and must exclude every smaller object in that declared class.

**Validation.** Recover gap closings at −2,0,2 and trivial/nontrivial intervals on either side. Reverse orientation and verify C changes sign. A fine-grid lattice Chern calculation is a useful cross-check, but must not substitute for the continuous gap proof. Check a time-reversal-invariant occupied bundle has zero total Chern number.

**Milestones.** Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

[Detailed critique](../reviews/09-Topological-Band-Theory.md).

## 15. Symmetry-indicator algebra and the limits of band diagnosis

**Question.** For one fixed symmetry setting, can compatibility relations and atomic band representations be generated exactly, and what topology can their quotient actually detect?

**Why it matters.** Symmetry labels are often much simpler to work with than the full wavefunctions across momentum space. Knowing what those labels do and do not prove can make topological diagnosis more efficient and prevent false declarations that a system is ordinary. The distinction is essential for turning a classification table into a trustworthy scientific tool.

**Scope and model.** Choose one small symmetry setting, initially a spinless two-dimensional inversion-symmetric lattice without time reversal. Specify the space-group generators, orbital/site representations, and high-symmetry momenta. Derive all input representations from group operations, with external tables used only as checks.

The result is an algebraic classification of symmetry data. A zero indicator is not a certificate of a trivial occupied bundle.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/15-Topological-Quantum-Chemistry.md).

$$A\subset B\subset\mathbb Z^r,\qquad X=B/A,\qquad
UA_{\mathrm{mat}}V=D,\quad U,V\text{ unimodular}.$$

$$b=A_{\mathrm{mat}}n,\quad n\in\mathbb Z^s
\qquad\text{versus}\qquad b=A_{\mathrm{mat}}n,\quad n\in\mathbb Z_{\ge0}^s.$$

Use a declared basis for the compatibility lattice $B$ in the Smith decomposition. Integer membership and nonnegative atomic symmetry-data decomposition are distinct tests. Neither a zero indicator nor matching atomic labels alone constructs localized Wannier functions for a given Hamiltonian.

**Deliverables.** Export group conventions, irreps, compatibility matrix, atomic generators, Smith decomposition, indicator mapping, and witnesses for any integer/nonnegative decomposition. Report “indicator trivial; topology unresolved” when symmetry data are insufficient. Use Berry/Wilson-loop information when making a stronger claim about a specific H.

**Validation.** Verify all representation identities and dimension counts. Reconstruct the Smith identity U A_mat V=D in the chosen B basis, with U,V unimodular. Test known atomic data, incompatible data, and models sharing symmetry labels but differing in an additional invariant. Keep stable and fragile notions distinct.

**Milestones.** Baseline: exact quotient and decompositions for one setting. Strong: connect the diagnosis to explicit Hamiltonians and identify its kernel. Research extension: another symmetry group or a rigorous fragile obstruction with a definition of allowed added bands.

[Detailed critique](../reviews/15-Topological-Quantum-Chemistry.md).


# Synthesis 10. Flat Chern Bands with Provable Geometry

**Mapping:** [PRD 10](../PRDs/10-Flat-Chern-Bands.md). Covered by canonical revision.

## 10. Flatness–locality–geometry tradeoffs for Chern bands

**Question.** For fixed orbital count, hopping range, and gap normalization, how small can a Chern band’s bandwidth and geometric nonuniformity be, with certified bounds?

**Why it matters.** Flat topological bands are useful settings for studying collective quantum behavior driven by interactions. But there are tradeoffs: restricting the model’s spatial range changes which flatness properties are possible. Quantifying those tradeoffs can show whether a proposed improvement is attainable or requires a more complicated model.

**Scope and model.** Choose a two-dimensional isolated rank-one band, target C=1, and a compact coefficient family with range R and fixed orbital positions. Initially use a specified two-band family extending Challenge 09. Impose a positive gap floor and a norm bound to prevent trivial energy rescaling.

Maintain two distinct tracks: finite-range approximate flatness, and exact spectral flattening with generally non-finite-range hopping. Do not require exact nonzero-Chern flatness with strictly finite-range hopping under the no-go theorem’s hypotheses.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/10-Flat-Chern-Bands.md).

$$W=\max_k E_n(k)-\min_k E_n(k),\quad
\Delta=\min_k\min_{j\ne n}|E_j(k)-E_n(k)|,\quad\text{minimize }W/\Delta.$$

$$g_{ij}=\operatorname{Re}\langle\partial_i u|(1-P)|\partial_j u\rangle,
\quad \bar F=\frac{2\pi C}{A},\quad A=\operatorname{Area}(\mathrm{BZ}),$$

$$\operatorname{Var}(F)=\frac1A\int_{\mathrm{BZ}}(F-\bar F)^2d^2k,
\qquad D_g=\frac1A\int_{\mathrm{BZ}}(\operatorname{tr}g-|F|)d^2k.$$

Fix a gap floor, norm bound, orbital positions and coordinate metric. Use the Chern convention of 09. Strictly finite-range approximate flatness and generally long-range exact flattening are separate tasks; these diagnostics alone do not prove an interacting fractional phase.

**Deliverables.** Export coefficient/domain constraints, W upper and Δ lower bounds, certified C, geometric enclosures, and optimization gaps. An achieved objective gives an upper bound on the minimum; a global lower bound needs a covering/relaxation proof. For infinite-range flattening, give decay and truncation estimates.

**Validation.** Verify normalization ∫F=2πC and the pointwise metric inequality in the chosen convention. Check that truncating a spectrally flattened model generally reintroduces bandwidth. Include a trivial exactly flat band as a negative topology test. Compare candidate objectives using identical units, embedding, and norm constraints.

**Milestones.** Baseline: one certified near-flat Chern model and a correct locality audit. Strong: global objective bounds on one compact family or a rigorous truncation tradeoff. Research extension: optimal geometry within a specified class; many-body FCI stability requires an added interacting Hamiltonian and filling.

[Detailed critique](../reviews/10-Flat-Chern-Bands.md).


# Synthesis 11. RBCE (Relativistic Band & Crystal-Field Engineering)

**Mapping:** [S01](../supplemental/S01.md). Distinct objective preserved in supplement.

## S01. Crystal-field and spin–orbit anisotropy in a finite ion model

**Question.** For a fixed d-shell occupancy and point-group representation, which anisotropy values are possible in a bounded crystal-field/spin–orbit Hamiltonian family?

**Why it matters.** Directional magnetic response is a basic ingredient in understanding magnetic materials and controlling spin states. A finite ion model lets us separate the roles of local geometry and spin–orbit coupling before tackling interactions among many ions.

**Scope and model.** Fix one ion model (start with d¹), a chosen point group such as C₄v, and a finite orbital-spin basis. Supply exact matrices for L,S and the group action. Define H(B)=H_CF+λL·S+μ_B B·(L+2S), with crystal-field coefficients and λ in a compact normalized box. Specify the temperature or isolated low-energy multiplet before defining susceptibility or the effective g tensor. For a Kramers doublet, extract a g tensor from the projected magnetic moment and use its singular values, which remove doublet-basis ambiguity. Require a certified excitation gap above that doublet. Optimize a bounded ratio or difference with a denominator bounded away from zero. Export matrices, symmetry identities, interval eigenvalue bounds, and global objective brackets.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S01.md).

$$H(B)=H_{\mathrm{CF}}+\lambda L\cdot S+\mu_B B\cdot(L+2S),$$

$$P H(B)P=E_0I+\frac{\mu_B}{2}\sum_{a,b}B_a g_{ab}\sigma_b+O(|B|^2).$$

Use the isolated Kramers-doublet projector $P$ and exact orbital-spin conventions. The singular values of $g$ remove doublet-basis ambiguity. Certify the gap above the doublet and bound any denominator in an anisotropy ratio on the compact parameter box. This finite ion model does not establish bulk magnetism.

**Critique.** The synthesis’s “all d^n configurations across every point group” has no interaction model, coefficient bounds, or objective normalization. Symmetry permits tensor components but does not determine their magnitudes; arbitrary crystal-field or spin–orbit scaling can make an optimization meaningless. “Rare-earth-like” is a motivation, not an observable. Many-electron d^n ions also require Coulomb/Hund parameters and a specified approximation. The revision retains an anisotropy atlas but begins with a finite, normalized spectral problem and separates a single-ion response from bulk magnetocrystalline anisotropy.

**Validation and milestones.** Baseline: exact symmetry-allowed Hamiltonian basis and one doublet g tensor. Strong: certified parameter-box extrema with a doublet-gap condition. Extend to d² only after specifying interactions. Check the isotropic limit, group covariance, and basis invariance of g singular values.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 12. Photonic/Phononic Crystals: Rigorous Bandgap Optimization

**Mapping:** [PRD 11](../PRDs/11-Photonic-Topological-Crystals.md), [S02](../supplemental/S02.md). Distinct objective preserved in supplement.

## 11. Certified topological bands in an idealized Maxwell medium

**Question.** Can a specified lossless periodic electromagnetic model support a rigorously isolated band group with nonzero Chern number, and can its interface modes be validated?

**Why it matters.** Controlling where and in which direction waves travel is a basic aim of optical engineering. Topology offers one mechanism for robust interface behavior under appropriate conditions. A certified Maxwell calculation would also expose whether an apparent effect is caused by numerical artifacts or by assumptions a real material cannot satisfy.

**Scope and model.** Choose a two-dimensional periodic, lossless, frequency-independent constitutive model with Hermitian positive ε(r), μ(r), specified boundary conditions, and a mechanism breaking physical time reversal. Treat the constitutive tensors as mathematical inputs. Real-frequency dispersion, loss, and fabrication feasibility are extensions requiring additional physical data.

Start by validating a reciprocal scalar dielectric benchmark with zero total Chern number for a time-reversal-invariant band group; then introduce an explicitly nonreciprocal tensor.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/11-Photonic-Topological-Crystals.md).

$$(\nabla+ik)\times\mu^{-1}(r)(\nabla+ik)\times u
=\omega^2\epsilon(r)u,\qquad
\langle u,v\rangle_\epsilon=\int_{\mathrm{cell}}u^*\cdot\epsilon v\,dr.$$

Use positive Hermitian constitutive tensors, the transverse constraint, units $c=1$, and an isolated positive-frequency band group. Certify its gap and projector invariant in the weighted metric. Nonzero total Chern number requires breaking the relevant physical time-reversal symmetry; scalar reciprocal benchmarks provide rejection controls.

**Deliverables.** Return ideal geometry and constitutive tensors, solver discretization, eigenvalue enclosures, spurious-mode controls, gap and topology certificates. Interface calculations must state both media and termination. Report robustness only for an explicit perturbation class and bound that preserves the required gap/symmetry.

**Validation.** Check homogeneous-medium dispersion, mesh/basis convergence, Hermiticity, and transverse modes. Separate Galerkin error from k-space interpolation error. Verify reciprocal controls cannot acquire a net Chern number from numerical artifacts. Compare an interface spectrum with the bulk-index difference without claiming immunity to arbitrary disorder.

**Milestones.** Baseline: validated reciprocal eigenproblem. Strong: one certified ideal nonreciprocal topological design. Research extension: a realistic dispersive constitutive model or gap optimization within a bounded geometry family. Fabrication drawings alone do not certify material behavior.

[Detailed critique](../reviews/11-Photonic-Topological-Crystals.md).

## S02. Bandgap optimization in a bounded wave-operator family

**Question.** Within a fixed finite family of periodic media, what lower bound on a complete bandgap and what upper bound on the best possible gap can be certified?

**Why it matters.** Controlling which waves propagate supports ideas for sound isolation, vibration management and optical filtering. The same mathematical challenge arises across these applications: a promising numerical design needs reliable evidence that its gap exists everywhere and that any optimality claim covers the stated design space.

**Scope and model.** Choose either a scalar acoustic operator or a full Maxwell operator, never an unspecified mixture. Initially choose a periodic scalar elliptic operator −∇·a(r)∇ with 0<a_min≤a(r)≤a_max, a fixed unit cell and a finite geometry parameter box. For adjacent Bloch eigenvalues define the complete gap g=min_k λ_{n+1}(k)−max_k λ_n(k), with a declared normalized objective. A grid-local separation is not a complete gap. Use validated eigenvalue bounds plus k-space variation estimates to cover the Brillouin zone. Branch-and-bound over parameters yields a candidate lower bound and an optimization upper bound, keeping the inequality direction explicit.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S02.md).

$$\lambda_- =\max_k\lambda_n(k),\qquad
\lambda_+=\min_k\lambda_{n+1}(k),\qquad
g=\lambda_+-\lambda_-. $$

A possible normalized objective is $G=2g/(\lambda_++\lambda_-)$ when the denominator is positive; frequency-gap objectives must instead use $\omega_j=\sqrt{\lambda_j}$ consistently.

Use a fixed self-adjoint wave model, bounded coefficients and geometry class, with a positive denominator. A complete gap needs $\lambda_+>\lambda_-$ across the entire Brillouin zone and all required polarizations. A certified candidate lower bound does not establish global optimality; that needs coverage of the full design class.

**Critique.** The old description mixes photonic and phononic operators, index contrast and generic gap optimization without fixing constitutive laws, geometry regularity, or constraints. “Lowest possible contrast” and “largest possible gap” are two objectives needing an ordering or Pareto definition. A converged plane-wave plot does not establish a complete gap, and a locally optimized geometry does not prove global optimality. This is distinct from PRD 11’s topological invariant: a wide ordinary bandgap can be topologically trivial.

**Validation and milestones.** Baseline: one certified complete gap for a supplied geometry. Strong: bound the optimum to a declared tolerance over the entire compact parameter box, or report unresolved boxes. Validate a homogeneous medium and a simple one-dimensional layered case before multidimensional optimization.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 13. Universal Bounds for Effective Properties of Composites

**Mapping:** [S03](../supplemental/S03.md). Distinct objective preserved in supplement.

## S03. Effective conductivity bounds for two-phase composites

**Question.** For fixed positive phase conductivities and volume fraction, can rigorous effective-conductivity bounds and matching constructions be obtained in a specified microstructure class?

**Why it matters.** Composite design seeks useful bulk properties from available ingredients. Bounds reveal whether a proposed arrangement performs well, whether a computation is accurate, and how much improvement remains possible. Simple layered structures also provide exact reference cases for checking more complicated methods.

**Scope and model.** Choose scalar conductivities a₁,a₂>0, periodic unit cell Y, phase fraction f, and a unit direction e. Define eᵀA_eff e=min_{periodic u}〈a|e+∇u|²〉. Use an admissible potential trial field for an upper bound and a divergence-free flux variational principle for a lower bound. Distinguish bounds valid for every arrangement from optimization over a finite geometry family. Fix isotropy assumptions before comparing isotropic bounds. Cross-property elasticity/thermal results are separate extensions with both constitutive laws specified.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S03.md).

$$e\cdot A^*e=\min_{\phi\ \mathrm{periodic}}
\langle a(x)|e+\nabla\phi|^2\rangle,\qquad a(x)\in\{a_1,a_2\},$$

$$\left(\frac f{a_1}+\frac{1-f}{a_2}\right)^{-1}|e|^2
\le e\cdot A^*e\le(fa_1+(1-f)a_2)|e|^2.$$

Assume positive scalar phases, fixed volume fraction $f$, and a normalized cell average. Potential trials give upper bounds and the corresponding flux principle gives lower bounds. A finite microstructure list does not characterize an unrestricted G-closure.

**Critique.** The original “universal bounds” problem spans conductivity, permittivity and elasticity, including tensor and frequency effects with different variational principles. “Beyond Hashin–Shtrikman” is not a measurable novelty requirement until the symmetry class and available information are fixed. A finite set of candidate microstructures cannot characterize the unrestricted G-closure. The revision makes one scalar static problem precise and makes the bound direction and construction obligation explicit.

**Validation and milestones.** Baseline: verify arithmetic and harmonic directional bounds with layered composites. Strong: matching bounds for one declared geometry class or a certified gap to the unrestricted bound. Check volume fraction and ellipticity exactly; report discretization and geometry errors separately.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 14. Topological Mechanics: Maxwell Frames & Programmable Metamaterials

**Mapping:** [PRD 12](../PRDs/12-Topological-Mechanical-Metamaterials.md). Covered by canonical revision.

## 12. Rigidity and boundary modes of periodic Maxwell frames

**Question.** For a geometrically specified isostatic frame and boundary termination, can the zero-mode count and localization be proved from its compatibility matrix?

**Why it matters.** Geometry-controlled softness could help explain and design structures with directional compliance, boundary motion or localized mechanical response. The scientific value is understanding which behaviors follow from a robust index and which are accidental features of one geometry. That distinction matters before translating an ideal spring model into a physical structure.

**Scope and model.** Use unstressed central-force springs with exact rational or algebraic site coordinates, positive spring constants, and explicit periodic bond offsets. Fix the unit-cell gauge, dimension, masses, and boundary termination. Begin with a small frame and a known periodic Maxwell lattice.

Connectivity alone is insufficient: bond directions determine the compatibility matrix. Distinguish rigid-body motions, floppy modes, and states of self stress. Claims concern linearized ideal mechanics.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/12-Topological-Mechanical-Metamaterials.md).

$$Q=C^\dagger,\qquad n_0-n_{ss}=dN-N_b,\qquad
K=C^\dagger\operatorname{diag}(k_b)C,\quad D=M^{-1/2}KM^{-1/2}.$$

$$\nu_\gamma=\frac1{2\pi i}\oint_\gamma d\log\det C(k),
\qquad \det C(k)\ne0\text{ on }\gamma.$$

The first count includes rigid-body motions and self stress. Supply actual bond directions, unit-cell gauge and termination. Acoustic zeros and the local boundary count must be included before identifying edge modes from winding.

**Deliverables.** Export positions, bonds, C, exact rank/kernel calculations, self-stress basis, winding certificate, and a boundary-specific mode count. For localization, provide a transfer-matrix/root-modulus bound separated from the unit circle. Specify perturbations preserving the nonvanishing condition and positivity of spring constants.

**Validation.** Check one elementary underconstrained and one overconstrained frame against the index equation. Verify K is positive semidefinite and translations are zero modes where appropriate. Compare two terminations of the same bulk. Reject winding calculations crossing a zero of det C.

**Milestones.** Baseline: exact Maxwell–Calladine counts on finite frames. Strong: one periodic frame with certified winding and termination-dependent edge localization. Research extension: a bounded family with robust mode counts. Elastic beams, prestress, nonlinear stability, and manufacturing tolerance require separate models.

[Detailed critique](../reviews/12-Topological-Mechanical-Metamaterials.md).


# Synthesis 15. Real-Space Topological Invariants for Disordered Media

**Mapping:** [PRD 09](../PRDs/09-Topological-Band-Theory.md), [S04](../supplemental/S04.md). Related PRD; supplement preserves distinct objective.

## 09. Certified Chern phases in a bounded tight-binding family

**Question.** Within a declared finite-range Hamiltonian family, which parameter regions have a proven band gap and occupied-band Chern number, and what minimality statements follow within that family?

**Why it matters.** Topological models explain why some properties depend on global structure rather than microscopic details. A certified phase map provides reliable examples for theory and simulation, and it helps identify which ingredients a proposed design actually needs. Proving a gap everywhere is important because a small missed gap closing can invalidate a computed topological label.

**Scope and model.** Begin with the two-band square-lattice model $H_m(k)=\sin(k_x)\sigma_x+\sin(k_y)\sigma_y+[m+\cos(k_x)+\cos(k_y)]\sigma_z$, k∈[−π,π]². Use one occupied band and a fixed orientation. Analyze rational parameter intervals separated from m=−2,0,2.

For subsequent searches fix dimension, orbitals, filling, hopping support, coefficient domain, and symmetry representation. A rational grid is finite; a continuous coefficient box requires a different completeness argument. “All Hamiltonians for a space group” is not an enumerable specification.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/09-Topological-Band-Theory.md).

$$H_m(k)=\sin k_x\,\sigma_x+\sin k_y\,\sigma_y+
(m+\cos k_x+\cos k_y)\sigma_z.$$

$$E_{r+1}(k)-E_r(k)\ge\delta>0,\qquad
C=\frac{1}{2\pi i}\int_{\mathrm{BZ}}
\operatorname{Tr}\!\left(P[\partial_{k_x}P,\partial_{k_y}P]\right)\,d^2k.$$

Prove the gap over the full claimed parameter and momentum domain. Fix orientation and the occupied projector. An analytic argument or validated integral enclosing one integer certifies $C$; a momentum mesh alone does not.

**Deliverables.** Return model coefficients, gap lower bounds, invariant and sign convention, certified parameter coverage, and uncovered boxes. A minimality claim must specify whether it minimizes orbitals, hopping range, or coefficients and must exclude every smaller object in that declared class.

**Validation.** Recover gap closings at −2,0,2 and trivial/nontrivial intervals on either side. Reverse orientation and verify C changes sign. A fine-grid lattice Chern calculation is a useful cross-check, but must not substitute for the continuous gap proof. Check a time-reversal-invariant occupied bundle has zero total Chern number.

**Milestones.** Baseline: complete phase diagram for H_m away from transition values. Strong: a certified atlas for one bounded symmetry-compatible family. Research extension: new range/band tradeoffs or minimality theorems, explicitly relative to the permitted family.

[Detailed critique](../reviews/09-Topological-Band-Theory.md).

## S04. Finite-volume certification of a disordered topological index

**Question.** For a local disordered lattice Hamiltonian with a specified bulk spectral-gap hypothesis, when does a finite-volume index provably equal its infinite-volume counterpart?

**Why it matters.** Topological behavior is interesting partly because it can survive certain imperfections. Establishing that robustness requires methods that work without a perfect crystal and distinguish a finite-sample observation from a bulk conclusion. A rigorous bridge between the two makes numerical evidence more informative.

**Scope and model.** Fix a finite-range two-dimensional tight-binding Hamiltonian on Z², bounded disorder, Fermi energy E_F and a bulk spectral gap. Initially use bounded perturbations small enough to preserve a known clean gap. Choose one index construction, such as a spectral localizer, and one theorem relating finite and infinite volume. Define position operators, truncation radius and boundary convention; compute the finite Hermitian signature or index with interval spectral bounds. Verify every locality, commutator, radius and scale inequality in the selected theorem. Mobility-gap cases require a different theorem and are an extension.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S04.md).

$$\operatorname{dist}(E_F,\operatorname{spec}H_0)\ge g>0,
\qquad\|V\|<g\quad\Longrightarrow\quad
\operatorname{dist}(E_F,\operatorname{spec}(H_0+V))\ge g-\|V\|.$$

This bounded self-adjoint perturbation estimate supplies a spectral-gap hypothesis. Equality of a chosen finite-volume index with the bulk index still requires a specific localizer or other index theorem with all radius, locality, commutator and scale conditions checked. Mobility-gap claims need a different theorem.

**Critique.** The synthesis lists Bott indices, noncommutative Chern numbers and localizers as if they shared one universal finite-size error bound. They have different hypotheses and finite-volume behavior. An integer from a finite matrix is not automatically the bulk invariant, and one disorder realization does not establish robustness for all disorder. The revised task certifies the finite-to-infinite transfer under a stated gap/locality theorem; it is distinct from the periodic model classification in PRD 09.

**Validation and milestones.** Baseline: recover a clean model’s known index with a validated finite-to-bulk theorem. Strong: certify a full bounded disorder class using gap preservation. Include trivial controls and a gap-closing case that the checker declines to certify.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 16. N-Representability & 2-RDM Variational Chemistry

**Mapping:** [PRD 16](../PRDs/16-N-Representability-2RDM.md). Covered by canonical revision.

## 16. Certified energy brackets from reduced-density-matrix relaxations

**Question.** For a specified finite fermionic Hamiltonian, how tight an independently certified ground-state energy interval can P,Q,G and selected higher-order RDM constraints produce?

**Why it matters.** Reliable energy calculations underpin the theoretical study of bonding and interacting electrons. A bracket tells us how much uncertainty remains, rather than merely reporting a convincing decimal number. The approach may extract useful information without storing a complete many-electron wavefunction, although general representability remains computationally difficult.

**Scope and model.** Fix N≥2 fermions in M spin orbitals with an exact rational/algebraic, number-conserving one- and two-body Hamiltonian. Start with a small Hubbard-type system where exact diagonalization in the same N-particle sector is feasible. Molecular integrals are a later input with an explicit basis and numerical error enclosure.

Use ensemble N-representability. The task is to certify bounds within a finite orbital model, not solve general electronic structure in polynomial time.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/16-N-Representability-2RDM.md).

$$\gamma_{ij}=\langle a_i^\dagger a_j\rangle,\qquad
\Gamma_{ij,kl}=\langle a_i^\dagger a_j^\dagger a_l a_k\rangle,$$

$$\operatorname{Tr}\gamma=N,\quad
\sum_{ij}\Gamma_{ij,ij}=N(N-1),\quad
\sum_j\Gamma_{ij,kj}=(N-1)\gamma_{ik}.$$

$$L_{\mathrm{dual}}\le E_{\mathrm{relax}}\le E_0\le
\langle\Psi|H|\Psi\rangle=U,\qquad \langle\Psi|\Psi\rangle=1.$$

Derive two-particle, two-hole and particle-hole moment matrices from anticommutation in this convention. Necessary PSD constraints give an outer relaxation. Certify dual feasibility for $L$ and a physical $N$-fermion state for $U$; a relaxed RDM need not represent any state.

**Deliverables.** Export the Hamiltonian and indexing convention, moment constraints, certified dual objective L, state witness for U, and gap U−L. Separate solver error, relaxation error, and finite-basis/model error. Report any property bounds via separately optimized observables; a relaxed Γ need not be physical.

**Validation.** Compare with exact diagonalization in identical conventions. Check contractions using explicit Slater determinants and correlated states. Adding valid necessary constraints must not lower the exact relaxed minimum. Verify dual PSD matrices and equality residual corrections rigorously, rather than interpreting the raw solver objective as a bound.

**Milestones.** Baseline: valid L,U brackets for small fixed models. Strong: quantify improvement from G or T constraints across a specified instance set. Research extension: a new valid constraint with demonstrated tightening, or a theorem on a tractable family; novelty needs comparison with existing hierarchies.

[Detailed critique](../reviews/16-N-Representability-2RDM.md).


# Synthesis 17. Non-empirical Density Functional Theory via Convex Analysis

**Mapping:** [S05](../supplemental/S05.md). Distinct objective preserved in supplement.

## S05. Finite-basis density functionals and exact convex constraints

**Question.** For a fixed finite lattice Hamiltonian, can the ensemble constrained-search density functional be bracketed and its convexity/duality verified?

**Why it matters.** Density-based methods reduce a difficult many-electron description to a much smaller set of variables. Exact finite examples reveal which constraints an approximation should respect and provide benchmarks for evaluating proposed energy formulas.

**Scope and model.** Fix a finite N-particle Hilbert space, kinetic T, interaction W, and commuting site-density operators $\widehat n_i$. Define $F(n)=\min_{\rho\succeq0,\operatorname{Tr}\rho=1,\operatorname{Tr}(\rho\widehat n_i)=n_i}\operatorname{Tr}\rho(T+W)$ on ensemble-representable densities. Derive the dual $F(n)=\sup_v[E(v)-v\cdot n]$ with $E(v)=\min_{\rho\succeq0,\operatorname{Tr}\rho=1}\operatorname{Tr}\rho(T+W+\sum_i v_i\widehat n_i)$, under the finite-dimensional convex formulation. Use exact states for primal upper bounds and potentials with certified E(v) lower bounds for dual lower bounds. If defining an exchange-correlation remainder, specify T_s and Hartree conventions separately.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S05.md).

$$F(n)=\min_{\substack{\rho\succeq0,\ \operatorname{Tr}\rho=1\\
\operatorname{Tr}(\rho\widehat n_i)=n_i}}
\operatorname{Tr}\rho(T+W),$$

$$E(v)=\min_{\rho\succeq0,\ \operatorname{Tr}\rho=1}
\operatorname{Tr}\rho\!\left(T+W+\sum_i v_i\widehat n_i\right),
\qquad F(n)=\sup_v[E(v)-v\cdot n].$$

Work in the fixed finite $N$-particle Hilbert space and on ensemble-representable densities. The supremum need not be attained by a finite potential at boundary densities. Feasible density matrices give upper bounds; certified energy lower bounds at chosen potentials give lower bounds. Convexity of $F$ does not imply convexity of an exchange-correlation remainder.

**Critique.** “The cone of exchange-correlation functionals satisfying all exact constraints” is not justified: normalization and affine conditions need not define a cone, and convexity of the ensemble universal functional does not imply convexity of E_xc after subtracting other functionals. Constraint satisfaction alone does not identify the exact physical functional. The broad continuum task also leaves admissible density spaces and spin scaling unspecified. The revision first tests exact convex duality in a finite model, preserving nonempirical functional development as an extension.

**Validation and milestones.** Baseline: exact functional values and dual brackets on a two-site model. Strong: a certified piecewise characterization or a new valid inequality for a larger finite model. Verify particle-number bounds, convexity inequalities, and agreement with direct constrained search.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 18. Strictly-Correlated Electrons as Multi-Marginal Optimal Transport

**Mapping:** [PRD 18](../PRDs/18-Optimal-Transport-Chemistry.md). Covered by canonical revision.

## 18. Certified optimal-transport bounds for strictly correlated electrons

**Question.** For a supplied normalized electron density, can a discretized Coulomb multi-marginal transport problem yield verified bounds, with continuum error separated from optimization error?

**Why it matters.** Strongly correlated electrons are difficult to approximate because their motions are strongly linked. This limiting problem gives a mathematically structured reference for studying that linkage and for developing density-functional ideas. Verified bounds help distinguish a true property of the continuum model from an artifact of a spatial grid or optimization tolerance.

**Scope and model.** Take an electron density ρ≥0 with ∫ρ=N and marginal μ=ρ/N. Begin with N=2 and a compactly supported mathematical density or a finite rational discrete model. Specify treatment of the diagonal Coulomb singularity. A continuous-density result additionally needs domain-tail and discretization estimates.

The main target is the strictly correlated interaction functional. Wasserstein comparison of densities is a separate optional metric task; it is not a minimum-energy chemical reaction path.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/18-Optimal-Transport-Chemistry.md).

$$\int\rho=N,\qquad \mu=\rho/N,\qquad
c(r_1,\ldots,r_N)=\sum_{i<j}|r_i-r_j|^{-1},$$

$$V_{\mathrm{SCE}}[\rho]=\inf_{\Gamma\in\Pi(\mu,\ldots,\mu)}\int c\,d\Gamma,
\qquad \sum_i u(r_i)\le c(r_1,\ldots,r_N),$$

$$\underbrace{\int u\rho}_{L}\ \le\ V_{\mathrm{SCE}}[\rho]\ \le\
\underbrace{\int c\,d\Gamma_{\mathrm{feasible}}}_{U}.$$

The coupling is a probability measure; it may be symmetrized and is not antisymmetric. This is a Coulomb interaction functional, not exact exchange. For a discrete problem, the same inequalities apply only to the discrete optimum until continuum error is bounded.

**Deliverables.** Return marginal conventions, cost and singularity policy, primal plan, dual potentials, rigorously checked constraints, and L≤V≤U. State whether V is a finite discrete optimum or the continuum functional. Sinkhorn candidates must be converted to feasible bounds for the unregularized problem, with entropic bias controlled.

**Validation.** Verify a small rational-cost transport LP exactly and a one-dimensional quadratic-cost quantile example as a separate solver test. For Coulomb densities, use a case with a known co-motion construction under its actual hypotheses. Reject comparing normalized densities with unequal total mass unless normalization or an unbalanced-OT model is explicitly chosen.

**Milestones.** Baseline: certified two-marginal discrete bounds. Strong: one continuum enclosure with singularity/tail control. Research extension: N>2 with symmetry reduction and validated bounds, or a demonstrably improved relaxation. Do not promise deterministic Monge maps for arbitrary multimarginal inputs.

[Detailed critique](../reviews/18-Optimal-Transport-Chemistry.md).


# Synthesis 19. Complete Isomer Enumeration with Proofs

**Mapping:** [PRD 17](../PRDs/17-Isomer-Enumeration-Molecular-Graphs.md). Covered by canonical revision.

## 17. Complete enumeration of a precisely defined molecular-graph class

**Question.** For a fixed formula and explicit valence grammar, can every connected constitutional graph be generated exactly once, with a checkable completeness argument?

**Why it matters.** Complete enumeration turns an intuitive picture of “possible structures” into a checkable mathematical set. Such sets are useful for testing structure-generation methods and exploring how bonding constraints shape chemical possibilities. The guarantee is especially valuable when a search is used to support a claim that no alternative structure exists.

**Scope and model.** Initially allow neutral closed-shell C,H,O graphs with bond orders 1,2,3, no self-bonds, and valences 4,1,2. Hydrogens may be implicit if reconstructed uniquely from residual valence. Specify whether different Kekulé graphs are distinct; the baseline uses literal bond-order graphs and no aromatic equivalence.

Constitutional graph validity is not chemical stability or synthesizability. Stereoisomers, conformers, ions, radicals, isotopes, and 3D geometries require added definitions and are not baseline outputs.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/17-Isomer-Enumeration-Molecular-Graphs.md).

Let $\mathcal G$ contain exactly the graphs satisfying the fixed atom, bond, valence and connectivity grammar. Then

$$\text{catalog}=\mathcal G/\!\cong.$$

Isomorphisms preserve the declared atom and bond labels. Every output must satisfy the grammar, no two representatives may be equivalent, and every allowed input must map to a representative. Canonical labeling proves deduplication only; completeness needs a coverage proof for generation and pruning. Stereochemistry requires a separately declared equivalence model.

**Deliverables.** Export canonical graphs, reconstructed hydrogens, counts, labeling conventions, and generation/coverage proof. Provide a validator independent of the optimized generator. If a resource limit stops enumeration, output a partial catalog and coverage status rather than “all isomers.”

**Validation.** Recover alkane constitutional counts for C₄H₁₀ and C₅H₁₂, namely 2 and 3. Cross-check all tiny instances with brute force. Random relabelings must preserve canonical codes. Verify each graph’s formula and weighted valence; reject disconnected mixtures.

**Milestones.** Baseline: complete small-formula catalog with independent counts. Strong: complete C₆H₁₂O under the stated grammar. Research extension: stereochemical orbit enumeration with meso cases and explicit stereocenter/double-bond conventions; keep conformational continua separate.

[Detailed critique](../reviews/17-Isomer-Enumeration-Molecular-Graphs.md).


# Synthesis 20. Inverse Statistical Mechanics: Potentials with Provable Ground States

**Mapping:** [S06](../supplemental/S06.md). Distinct objective preserved in supplement.

## S06. Inverse pair-potential design over an explicit competitor class

**Question.** Can a parameterized pair potential make a target periodic structure an energy minimizer among a precisely specified set of competitors over a density interval?

**Why it matters.** Interaction design is a way to explore self-assembly: simple local rules may favor organized structures. Reliable comparisons expose whether apparent success comes from a useful mechanism or from overlooking a competing arrangement or a long-range contribution.

**Scope and model.** Fix dimension, density interval, and v_θ(r)=Σ_jθ_jφ_j(r) with bounded coefficients and convergent lattice sums. Define E_θ(Λ)=1/2 Σ_{x∈Λ∖{0}}v_θ(|x|). Begin with a finite list of normalized lattice competitors and use interval sums plus tail bounds to certify E_target≤E_competitor−δ uniformly on a density interval. Continuous competitors require a compact parameterization and global coverage. State whether particle displacements or variable cell size are allowed. General configuration ground states require additional universal lower bounds beyond lattice comparisons.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S06.md).

$$v_\theta(r)=\sum_j\theta_j\phi_j(r),\qquad
E_\theta(\Lambda)=\frac12\sum_{x\in\Lambda\setminus\{0\}}v_\theta(|x|),$$

$$E_\theta(\Lambda_{\mathrm{target}})\le
E_\theta(\Lambda_{\mathrm{competitor}})-\delta.$$

Require convergent sums and certified tails, uniformly on the declared density interval and competitor set. The conclusion is best among those competitors. Unrestricted configurations, infinitesimal deformations and finite-temperature free energies require separate arguments.

**Critique.** The original jumps from optimizing lattice energies to provable crystals over both densities and temperatures. Beating selected lattices does not exclude aperiodic competitors or infinitesimal deformations, and a zero-temperature energy gap is not a finite-temperature free-energy theorem. A strictly uniform energy gap against all arbitrarily close deformations may be impossible under continuity. The rewrite gives a restricted inverse-design result whose proof scope is honest; unrestricted crystallization and thermal stability remain separate research extensions.

**Validation and milestones.** Baseline: exact or interval-verified ordering for a finite competitor list. Strong: uniform density/parameter-region certificate. Validate a direct long-cutoff sum against the certified tail bound; label the result “best among these competitors” unless a universal proof is supplied.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 21. LDPC & Hypergraph-Product Quantum Error Correcting Codes

**Mapping:** [PRD 21](../PRDs/21-Quantum-LDPC-Codes.md). Covered by canonical revision.

## 21. Quantum LDPC instances with independently certified code parameters

**Question.** For a specified CSS construction, can code dimension and distance bounds be verified independently, and can decoder performance be measured under one explicit noise model?

**Why it matters.** Useful quantum computation requires reliable logical information despite noisy components. Sparse checks offer a route to distributing error correction efficiently, but code parameters and decoder behavior must be demonstrated rather than assumed. Exact small examples are valuable for detecting mistakes that large simulations can conceal.

**Scope and model.** Start with hypergraph-product codes from supplied binary parity-check matrices H₁ and H₂. Include the product of two full-rank 3×7 Hamming parity-check matrices as a small benchmark. Separate finite instances, asymptotic families, and noisy syndrome extraction.

Use independent X errors with perfect syndrome measurements for the first decoder experiment. A family is LDPC only when both row and column weights stay bounded as n grows.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/21-Quantum-LDPC-Codes.md).

$$H_X=[H_1\otimes I_{n_2}\mid I_{m_1}\otimes H_2^{\mathsf T}],
\qquad H_Z=[I_{n_1}\otimes H_2\mid H_1^{\mathsf T}\otimes I_{m_2}],$$

$$H_XH_Z^{\mathsf T}=0,\quad n=n_1n_2+m_1m_2,\quad
k=n-\operatorname{rank}_{\mathbb F_2}H_X-\operatorname{rank}_{\mathbb F_2}H_Z,$$

$$d_X=\min_{x\in\ker H_Z\setminus\operatorname{row}H_X}|x|,
\qquad d_Z=\min_{z\in\ker H_X\setminus\operatorname{row}H_Z}|z|,
\qquad d=\min(d_X,d_Z).$$

Two full-rank $3\times7$ Hamming checks give $n=58$, $k=16$; the included independent validator also checks $d=3$. A logical witness gives an upper distance bound; exclusion of all smaller logicals gives the lower bound. Finite code parameters do not establish an asymptotic LDPC family or decoding threshold.

**Deliverables.** Export sparse matrices, GF(2) row reductions, logical bases, explicit upper witnesses and lower-bound proofs. SAT certificates need a verified encoding of nontrivial homology and weight. Decoder experiments report physical noise, recovery rule, logical-failure test, sample counts, confidence intervals, and runtime.

**Validation.** For the two 3×7 inputs verify n=58 and k=16; certify the claimed distance independently rather than assuming it. Inject all small-weight errors on tiny codes. A matching syndrome is not success unless error plus recovery is a stabilizer. Compare optimized checks with an independent GF(2) implementation.

**Milestones.** Baseline: exact small product code and certified distance bracket. Strong: several instances plus honest finite-size decoder curves. Research extension: a new family tradeoff or proved decoder guarantee. Reproducing asymptotically good codes is not itself a new existence result.

[Detailed critique](../reviews/21-Quantum-LDPC-Codes.md).


# Synthesis 22. New Bell Inequalities & Tsirelson-type Bounds

**Mapping:** [PRD 22](../PRDs/22-Bell-Inequalities-Nonlocality.md). Covered by canonical revision.

## 22. Bell inequalities with certified local and quantum value brackets

**Question.** For a fixed rational Bell functional in a small bipartite scenario, can its exact local value and a certified quantum lower/upper bracket be produced?

**Why it matters.** Bell inequalities make the difference between classical local explanations and quantum correlations precise. They also underlie ways to certify quantum behavior from observed input-output relationships. Reliable quantum bounds are essential: a numerical optimization value without a valid upper certificate or achieving strategy may misstate what quantum theory permits.

**Scope and model.** Start with two parties, two binary settings each, and CHSH. Fix normalization and no-signaling affine coordinates. The target quantum value is the supremum over finite-dimensional tensor-product strategies; identify its closure when discussing limits. Finite NPA levels provide outer bounds through the commuting-operator framework.

Facet discovery and randomness certification are extensions with separate definitions. Do not equate a finite SDP solution with a realizable strategy.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/22-Bell-Inequalities-Nonlocality.md).

$$\beta(P)=\sum_{abxy}c_{abxy}P(ab|xy),\quad
\beta_L=\max_{P\text{ deterministic local}}\beta(P),\quad
L_Q\le\beta_Q\le U_Q.$$

A finite-dimensional state and measurements certify $L_Q$; a verified NPA dual or operator SOS certifies $U_Q$. For CHSH, $\beta_L=2$ and an achieving two-qubit strategy plus an upper-bound identity give $\beta_Q=2\sqrt2$. NPA's commuting-operator limit must not be identified with the closure of finite-dimensional tensor-product strategies. A facet in affine dimension $D$ needs saturating affine rank $D-1$.

**Deliverables.** Export coefficients, local vertices, state/measurements, moment-word list, algebraic relations, dual matrices/SOS, and exact or interval objective bounds. To certify a facet in affine dimension D, verify the inequality for every vertex and show saturating vertices have affine dimension D−1. Declare relabeling symmetries when counting classes.

**Validation.** Check all 16 CHSH deterministic strategies and β_L=2. Reproduce 2√2 with an explicit two-qubit strategy and an operator upper-bound identity. Reject an invalid dual PSD matrix or a wrong commutation relation. Distinguish eight CHSH variants from one class under relabeling; positivity facets must be handled separately.

**Milestones.** Baseline: exact CHSH lower and upper certificates. Strong: a preselected additional rational functional with a certified bracket, whether or not it closes. Research extension: a new inequality with normalized comparison, or device-independent entropy bounds with an explicit adversary and statistical model.

[Detailed critique](../reviews/22-Bell-Inequalities-Nonlocality.md).


# Synthesis 23. Certified Spectral Gaps for Parent Hamiltonians

**Mapping:** [S07](../supplemental/S07.md). Distinct objective preserved in supplement.

## S07. Finite-size criteria for parent-Hamiltonian spectral gaps

**Question.** For a specified frustration-free one-dimensional parent-Hamiltonian family, can a validated local spectral calculation imply a uniform many-body gap?

**Why it matters.** The gap helps distinguish low-energy behavior and enters rigorous statements about correlations and stability under suitable conditions. Proving a uniform gap turns a finite numerical observation into a statement about a many-body family, which is often the scientifically relevant object.

**Scope and model.** Fix exact MPS tensors, on-site dimension, interaction range, boundary convention, and local positive projectors h_i. Verify frustration freeness and ground-space assumptions. Select a precise Knabe-type or martingale theorem appropriate to that model. Compute interval enclosures for the required finite-segment nonzero gaps and verify the theorem’s threshold to derive a uniform lower bound for the stated sizes/boundaries. Degenerate ground spaces require identifying the first strictly positive eigenvalue rather than assuming a unique ground state.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S07.md).

$$H_N=\sum_i h_i,\qquad h_i\succeq0,\qquad
\Delta_N=\min(\operatorname{spec}H_N\setminus\{0\}),$$

$$\text{target: }\inf_{N\ \mathrm{in\ the\ stated\ family}}\Delta_N\ge\delta>0.$$

Verify frustration freeness and ground-space assumptions. A specified finite-size gap theorem must supply the implication from validated local gaps to the displayed uniform bound. Positive gaps on finitely many chains are insufficient; failure of a sufficient criterion is not a gaplessness proof.

**Critique.** The synthesis’s jump from MPS to arbitrary PEPS makes a tractable finite-size criterion look like a general spectral-gap decision method. Such general algorithms are obstructed by undecidability results. A positive finite-chain gap alone says nothing uniform as size grows; one must meet a theorem’s hypotheses and threshold. The revision targets certificate automation for an explicit restricted family and preserves unresolved outcomes when the chosen sufficient criterion fails.

**Validation and milestones.** Baseline: certify a standard parent model with a valid finite-size theorem. Strong: a uniform parameter interval with an explicit gap lower bound. Test a deliberately gapless/threshold-failing control without claiming that criterion failure proves gaplessness.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Cubitt, Pérez-García and Wolf, Undecidability of the Spectral Gap](https://arxiv.org/abs/1502.04573) motivates the restriction to a specified certifiable family.


# Synthesis 24. Topological Order & Anyon Condensation with Proofs

**Mapping:** [PRD 24](../PRDs/24-Topological-Quantum-Error-Correction.md), [S08](../supplemental/S08.md). Related PRD; supplement preserves distinct objective.

## 24. Topological code verification and noise-specific decoder evaluation

**Question.** Can a toric-code family’s algebraic parameters be proved, and can a fixed decoder’s logical failure probability be evaluated under a precisely specified error model?

**Why it matters.** Geometric codes connect an intuitive picture of local errors to precise algebraic guarantees. They are important models for protecting quantum information using structured measurements. Understanding logical failure, rather than only syndrome cancellation, is essential for assessing whether increasing the code size actually improves reliability.

**Scope and model.** Use an L×L square torus with qubits on edges and L≥3. Start with independent Z errors of probability p and perfect syndrome extraction. Specify the decoder, tie-breaking rule, and whether recovery weights are uniform or likelihood based.

Keep exact code properties separate from statistical decoder performance. Planar boundaries, color codes, measurement noise, and circuit-level noise are distinct extensions.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/24-Topological-Quantum-Error-Correction.md).

$$C_2\xrightarrow{\partial_2}C_1\xrightarrow{\partial_1}C_0,
\quad\partial_1\partial_2=0,\quad H_X=\partial_1,\quad H_Z=\partial_2^{\mathsf T},$$

$$n=2L^2,\quad k=2,\quad d=L\quad(L\ge3),\qquad
e+r\in\operatorname{im}\partial_2\quad\text{for successful }Z\text{-error recovery}.$$

A vanishing residual syndrome only gives $e+r\in\ker\partial_1$ and can hide a nontrivial logical cycle. Statistical decoder curves require the fixed noise model and decoder; finite-size crossings are threshold estimates, not limit theorems.

**Deliverables.** Export lattice incidence data, ranks and logical bases, distance argument, decoder specification, and failure counts with confidence intervals. Exhaustive small-size evaluation can give an exact failure polynomial in p. Monte Carlo output must retain trials and failure counts, including zero-failure upper intervals.

**Validation.** Check n,k,d on several small tori. Verify logical anticommutation via primal/dual intersection. Enumerate all low-weight errors on tiny instances and test the stabilizer-membership success criterion. Validate a matching solution’s optimality separately from logical decoding success.

**Milestones.** Baseline: exact construction and distance proof with a tested decoder. Strong: reproducible finite-size failure curves and uncertainty-aware threshold estimates. Research extension: a rigorous threshold bound or noisy-measurement/circuit model, with no transfer of numerical thresholds across noise conventions.

[Detailed critique](../reviews/24-Topological-Quantum-Error-Correction.md).

## S08. Gapped boundaries from exact anyon data

**Question.** For one fully specified two-dimensional topological order, can candidate condensates and their gapped boundaries be verified from exact categorical data?

**Why it matters.** Boundaries determine how a topological phase interfaces with its surroundings and which excitations can disappear or remain distinguishable there. Understanding these options supports the theory of topological phases and the organization of quantum information in such systems.

**Scope and model.** Start with the toric-code modular category, explicit simple objects, fusion, twists, F and R symbols and conventions. Enumerate candidate condensable algebra objects within a declared multiplicity bound. Check multiplication/unit, associativity, commutativity and separability conditions, and the Lagrangian dimension condition for a boundary to vacuum. For the abelian toric-code benchmark recover electric and magnetic condensates. If constructing a lattice boundary, supply its Hamiltonian and check commuting-projector and gap claims separately.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S08.md).

$$A=\bigoplus_a n_a a,\qquad n_a\in\mathbb Z_{\ge0},\qquad
(\dim A)^2=\operatorname{Dim}\mathcal C.$$

For a vacuum boundary this Lagrangian dimension condition supplements, rather than replaces, the unit, multiplication, associativity, commutativity and separability checks using exact categorical coherence data. In the toric-code category the reference condensates are $1\oplus e$ and $1\oplus m$. A microscopic boundary Hamiltonian needs its own gap proof.

**Critique.** Fusion rules and modular S,T tables alone do not in general specify or prove all categorical coherence data. The synthesis also mixes 2D anyons with 3D topological orders, which require different structures. A listed condensate is not automatically a microscopic boundary realization. The rewrite retains an exact categorical task, bounds the enumeration and separates algebraic classification from lattice realization and anomaly/inflow interpretation.

**Validation and milestones.** Baseline: verify the toric-code coherence data and two standard vacuum boundaries. Strong: a second fully specified category or a completeness proof within a bounded class. Invalid F/R symbols and a nonbosonic condensate are rejection controls.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Li, Yang and Dong, Gapped Boundaries of Kitaev’s Quantum Double Models](https://arxiv.org/abs/2504.19512) provides a primary example of linking Lagrangian algebras to explicit lattice boundary terms.


# Synthesis 25. Sign-Problem (Un)avoidability

**Mapping:** [S09](../supplemental/S09.md). Distinct objective preserved in supplement.

## S09. Stoquasticity under a bounded class of basis changes

**Question.** For a finite rational Pauli Hamiltonian, is there an allowed product Clifford transformation making all off-diagonal entries real and nonpositive?

**Why it matters.** Knowing when signs can be removed helps separate difficulties caused by representation from those that survive a proposed remedy. A bounded search also creates small, exact benchmarks for algorithms intended to find useful basis changes.

**Scope and model.** Fix n qubits, rational coefficients, and allowed U=⊗_iU_i with U_i from the finite single-qubit Clifford group modulo phase. Define stoquasticity in the computational basis for the full transformed H, not automatically term by term. Exhaustively enumerate or use an exactly equivalent constraint encoding. Export U and an exact matrix/sign check if successful; otherwise export complete enumeration or a checked UNSAT proof. Continuous local unitaries require a semialgebraic formulation and are a distinct extension.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S09.md).

$$U=\bigotimes_{i=1}^nU_i,\quad U_i\in\mathrm{Clifford}_1/\mathrm{phase},
\qquad H'=UHU^\dagger,$$

$$\operatorname{Im}H'_{xy}=0,\qquad\operatorname{Re}H'_{xy}\le0\quad(x\ne y).$$

Apply these conditions to the full transformed matrix, allowing cancellations between terms. Exhaustive coverage or an equivalent checked UNSAT proof excludes only this finite transformation class. Stoquasticity alone does not guarantee efficient mixing or sampling.

**Critique.** The original omits which local basis changes are allowed, so “unavoidable sign problem” has no definite quantifier. A SAT search over finitely many transformations cannot exclude all continuous local rotations. Requiring every term to be stoquastic can also be stronger than requiring their sum to be stoquastic because entries can cancel. Finally, stoquasticity does not guarantee efficient sampling or mixing. The rewrite makes a finite, falsifiable basis-cure problem and limits every no-go conclusion to its transformation class.

**Validation and milestones.** Baseline: exhaustive small-n decisions with explicit transformations or coverage. Strong: verified SAT encoding checked against brute force. Extend to a continuous rotation class only with exact equations, inequalities, and global proof machinery.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

[Marvian, Lidar and Hen, On the computational complexity of curing non-stoquastic Hamiltonians](https://pmc.ncbi.nlm.nih.gov/articles/PMC6450938/) supports the need to specify the allowed basis class.


# Synthesis 26. KAM/Nekhoroshev Stability Domains for Planetary Systems

**Mapping:** [PRD 26](../PRDs/26-KAM-Theory-Planetary-Stability.md), [PRD 28](../PRDs/28-Nekhoroshev-Stability-Theory.md). Covered by canonical revision.

## 26. A posteriori certification of an invariant Hamiltonian torus

**Question.** For an explicitly specified analytic near-integrable Hamiltonian and an approximate torus, can every hypothesis of a quantitative KAM theorem be verified?

**Why it matters.** Long-term stability is difficult to infer from a finite trajectory simulation. A theorem can instead prove that a particular organized motion exists despite perturbations. This offers a foundation for studying stability mechanisms in celestial and other Hamiltonian systems, provided the model and theorem assumptions are appropriate.

**Scope and model.** Begin with H(I,θ)=ω·I+|I|²/2+ε[cos θ₁+cos(θ₁−θ₂)] in two degrees of freedom, ω=(1,(1+√5)/2), on a stated complex neighborhood of a real action box. Choose a rational ε only after validating the unperturbed case. Seek an embedding with rotation vector ω.

This is a mathematical model, not a solar-system stability assertion. Planetary extensions require an explicit canonical reduction, handling Kepler degeneracy, collision exclusions, and input uncertainty.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/26-KAM-Theory-Planetary-Stability.md).

$$H(I,\theta)=\omega\cdot I+\frac{|I|^2}{2}
+\epsilon[\cos\theta_1+\cos(\theta_1-\theta_2)],
\qquad\omega=(1,(1+\sqrt5)/2),$$

$$E(\theta)=X_H(K(\theta))-DK(\theta)\omega,
\qquad |k\cdot\omega|\ge\frac\gamma{|k|_1^\tau}
\quad(k\in\mathbb Z^2\setminus\{0\}).$$

Prove the all-integer Diophantine bound and every analytic-width, twist, tail and smallness condition of a specified quantitative theorem. Its conclusion is a nearby invariant torus, not stability of arbitrary planetary initial conditions. Failure of a sufficient criterion is not instability.

**Deliverables.** Export H, ε, domain, exact frequency representation, Fourier coefficients, all theorem constants, interval inequalities, and a checker. State the conclusion: existence of a nearby quasiperiodic invariant torus with a quantified embedding error. A failed sufficient condition yields “not certified,” not “unstable.”

**Validation.** Verify E=0 for K₀(θ)=(I=0,θ) at ε=0. Derive an all-integer Diophantine bound for the golden-ratio frequency. Check Fourier tails and loss of analytic width. Test a resonant frequency where the stated theorem must refuse certification.

**Milestones.** Baseline: unperturbed certificate and complete arithmetic/theorem setup. Strong: a nonzero-ε torus certificate. Research extension: a larger perturbation range or a reduced planetary model. One invariant torus does not certify a neighborhood of arbitrary observed initial conditions or a phase-space measure bound.

[Detailed critique](../reviews/26-KAM-Theory-Planetary-Stability.md).

## 28. Explicit finite-time action confinement from a normal form

**Question.** For a specified analytic Hamiltonian and action domain, what rigorously computable action excursion and confinement time follow from a certified normal form or quantitative Nekhoroshev theorem?

**Why it matters.** Finite-time guarantees answer a question that simulations alone cannot settle beyond their run length. They also distinguish a valid stability argument from an impressive-looking exponential formula with missing constants or units. Understanding slow drift is relevant to the mathematical study of planetary and other nearly integrable systems.

**Scope and model.** Use H=h(I)+εf(I,θ), h=ω·I+|I|²/2, in two or three degrees of freedom on a compact action box with an explicit complex extension. Supply ω, a finite Fourier polynomial f, and rational ε. Bound the gradient away from zero where the selected theorem requires it.

The initial target is a valid dimensional time bound, even if modest. Planetary and post-Newtonian models are extensions requiring their own reductions and hypotheses.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/28-Nekhoroshev-Stability-Theory.md).

$$H\circ\Phi=Z(J)+R(J,\theta),\qquad
|I-J|\le\eta,\qquad\|\partial_\theta R\|\le r,$$

$$|I(t)-I(0)|\le2\eta+r|t|.$$

This estimate assumes the angle-independent normal form shown and a proved domain-containment interval. Include resonant drift if $Z$ depends on angles. A quantitative Nekhoroshev theorem instead requires all its constants, hypotheses and threshold in $T=C\exp(c\epsilon^{-a})$, with units supplied by $C$. The number $\exp(10^{0.3})\approx7.35413$ has no unit of time.

**Deliverables.** Export normal-form order, transformations, inverse/domain bounds, Fourier tails, r, η, and a domain-containment proof for the claimed interval. A Nekhoroshev result must include all constants and its smallness threshold in T=C exp(c ε^−a) with stated units; do not choose a from a desired time target.

**Validation.** Verify integrable ε=0 behavior and a low-order symbolic normal form. Check symplecticity with validated remainder control. Confirm the action bound stays inside the certified domain. Independently recompute exp(10^0.3)≈7.35: it is a dimensionless factor, not 10¹³ years.

**Milestones.** Baseline: one explicit finite-time confinement certificate. Strong: optimized truncation with a stronger bound or full quantitative quasi-convex theorem application. Research extension: exponential scaling across a parameter family; do not infer actual solar-system stability from ε alone.

[Detailed critique](../reviews/28-Nekhoroshev-Stability-Theory.md).


# Synthesis 27. Periodic Orbits & Invariant Manifolds in N-Body Problem

**Mapping:** [S10](../supplemental/S10.md). Distinct objective preserved in supplement.

## S10. Certified periodic orbits of a specified three-body model

**Question.** Can a numerically proposed periodic orbit be enclosed by a validated shooting proof and classified by its reduced Floquet spectrum?

**Why it matters.** Periodic orbits are organizing reference motions in gravitational dynamics. Certifying their existence gives subsequent studies of nearby motion a dependable starting point and separates exact solutions from visual patterns in a simulation.

**Scope and model.** Choose one nondimensional Newtonian three-body model with positive rational masses, or a planar circular restricted model with fixed mass ratio. Declare collision exclusions and conserved quantities. Define a Poincaré section or phase condition to remove time translation, and reduce continuous symmetries. Solve the resulting shooting residual using a validated flow integrator and interval Newton/Krawczyk bounds for initial conditions and period. Integrate variational equations with enclosures for the monodromy matrix; remove the expected neutral directions before interpreting multipliers.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S10.md).

$$F(z_0,T)=\varphi_T(z_0)-z_0=0,\qquad
M=D\varphi_T(z_0).$$

The shooting residual must first be augmented and reduced using a phase condition, conserved quantities and continuous symmetries so the validated root problem has the correct dimension. Interval flow and root bounds establish existence; the reduced Floquet multipliers classify linear behavior. Neither a near return nor stable-looking multipliers prove nonlinear stability or a manifold connection.

**Critique.** The original bundles orbit existence, complete family discovery, linear stability, invariant manifolds and heteroclinic connections into one certificate. A small shooting residual is not existence, and stable-looking Floquet multipliers are not nonlinear stability. Near-return trajectories do not establish a connection between invariant manifolds. The revised first target is a single orbit with a phase-fixed existence proof; continuation, manifold enclosure and connection proofs are explicit later stages. This also preserves the figure-eight idea incorrectly folded into PRD 27’s central configurations.

**Validation and milestones.** Baseline: certify one supplied orbit and its period. Strong: validated continuation over a parameter interval and a reduced Floquet classification. A manifold connection requires additional covering/intersection proof, not a plot of approaching trajectories.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 28. Central Configurations Classification

**Mapping:** [PRD 27](../PRDs/27-N-Body-Central-Configurations.md). Covered by canonical revision.

## 27. Central configurations in a bounded, symmetry-reduced N-body problem

**Question.** For fixed positive masses and a declared symmetry quotient, can all collision-free central configurations in the chosen family be certified and their relative-equilibrium linearization analyzed?

**Why it matters.** Central configurations are organizing structures in gravitational dynamics and provide exact reference cases amid otherwise complicated motion. A complete small catalog can test numerical methods and reveal how mass choices and symmetry affect possible motions. The problem also links gravity to exact algebraic root finding.

**Scope and model.** Start with three labeled positive rational masses in the plane. Fix G=1, center of mass zero, moment of inertia I=Σm_i|r_i|²=1, and quotient by O(2), so mirror images count together. Permute only equal masses when explicitly requested. Extend to a specified four-body symmetry family after validating the complete three-body case.

Central configurations generate homographic motions; general periodic choreographies are a separate problem.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/27-N-Body-Central-Configurations.md).

$$U=\sum_{i<j}\frac{m_im_j}{|r_i-r_j|},\quad
\sum_i m_i r_i=0,\quad I=\sum_i m_i|r_i|^2=1,$$

$$\nabla_{r_i}U=-\lambda m_i r_i,\qquad\lambda=U/I>0,
\qquad d_{ij}^2=|r_i-r_j|^2,\quad d_{ij}>0.$$

Preserve collision exclusions when clearing denominators. For three labeled positive masses modulo $O(2)$, recover three collinear orderings and one equilateral shape. Complete coverage requires more than isolated root boxes. Relative-equilibrium stability uses the reduced rotating-frame system including Coriolis terms; a figure-eight choreography is not a central configuration.

**Deliverables.** Return normalized configurations, distance enclosures, residual identities, existence/uniqueness proofs, equivalence representatives, and completeness scope. For planar relative equilibria analyze the reduced rotating-frame first-order system including Coriolis terms. Distinguish spectral stability from nonlinear stability.

**Validation.** For three labeled masses under the stated O(2) quotient, recover three collinear orderings (one per middle body) and one equilateral shape. Check λ=U at I=1. Test collision artifacts introduced by denominator clearing. For stability, reproduce a known three-body criterion using identical mass and reduction conventions.

**Milestones.** Baseline: complete certified three-body catalog. Strong: complete enumeration in one four-body symmetry family or mass instance, with all charts covered. Research extension: a five-body restricted family; no claim to solve the general finiteness problem without an appropriately global proof.

[Detailed critique](../reviews/27-N-Body-Central-Configurations.md).


# Synthesis 29. Minimal Autocatalytic Cores & Universality

**Mapping:** [PRD 29](../PRDs/29-Chemical-Reaction-Networks-Origins.md), [S11](../supplemental/S11.md). Distinct objective preserved in supplement.

## 29. Autocatalytic sets: maximality, irreducibility and dynamical viability

**Question.** For a finite catalytic reaction system, can its maximal RAF and smallest RAF subsets be certified, while keeping structural autocatalysis distinct from sustained growth?

**Why it matters.** Collectively supported chemistry is relevant to thinking about how organized reaction systems can arise and persist. Exact minimality results help distinguish essential reactions from dispensable ones in a model. Just as importantly, separating structural closure from energy supply and kinetics prevents a network diagram from being mistaken for proof of a viable replicating system.

**Scope and model.** Input species S, reactions R with integer stoichiometry, food set F, and catalysis relation C⊂S×R. Begin with at most 12 reactions for exhaustive cross-checks. Catalysis is supplied model data, not inferred from a stoichiometric cycle.

Define three separate targets: unique maximal RAF, an inclusion-minimal (irreducible) RAF, and a minimum-cardinality RAF. Thermodynamic and kinetic viability require extra inputs and are separate extensions.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/29-Chemical-Reaction-Networks-Origins.md).

$$R'\subseteq R,\quad R'\ne\varnothing,\quad
\operatorname{reactants}(r)\subseteq\operatorname{cl}_{R'}(F),$$

$$\forall r\in R'\ \exists s\in\operatorname{cl}_{R'}(F):\ (s,r)\in C.$$

Compute closure by reactant availability without imposing catalysis in the closure step. These conditions define a RAF. Inclusion-maximal, inclusion-minimal (irreducible), and minimum-cardinality RAFs are separate targets. Catalysis-respecting startup and sustained growth require additional conditions.

**Deliverables.** Export closure rounds, catalyst witnesses, maxRAF elimination trace, and minimality/cardinality proof with explicit scope. For thermodynamic extension specify chemical potentials, chemostats, and driving. For kinetic extension provide rates, dilution, initial conditions and a positive sustained-state/growth criterion; RAF existence alone is insufficient.

**Validation.** Cross-check all subsets for tiny networks. Include food-catalyzed, self-catalyzed, catalyst-free, and mutually dependent examples. Verify a maximum RAF can contain several irreducible RAFs of different sizes. Deletion tests for irreducibility must use a complete RAF detector on each remaining network, not merely test whether the entire remainder is itself RAF.

**Milestones.** Baseline: checked maxRAF/RAF witnesses. Strong: complete minimum-cardinality results for a bounded network family. Research extension: characterize which RAFs additionally support a declared driven mass-action regime. Biological self-replication and evolution require their own operational definitions.

[Detailed critique](../reviews/29-Chemical-Reaction-Networks-Origins.md).

## S11. Minimal reaction implementations of a specified computational behavior

**Question.** Within a bounded reaction grammar, what is the smallest network implementing a precisely defined input/output behavior with a proved error bound?

**Why it matters.** Chemical computation explores how local reactions can perform information processing, including ideas relevant to molecular sensing and control. Precise semantics make proposed constructions comparable and prevent hidden resources or flexible readout choices from creating misleading claims of simplicity.

**Scope and model.** Choose deterministic mass-action or stochastic reaction semantics, species/reaction bounds, stoichiometric arity, input encoding, output decoding and cost (species count first, then reaction count, for example). Begin with a finite Boolean input/output function and an explicit settling-time/error requirement. Enumerate networks modulo species relabeling; prove correctness over the complete allowed input set and exclude every lower-cost network within the grammar. For a universal-computation extension, supply a uniform compiler and a simulation theorem covering arbitrary program lengths, resource scaling and precision.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S11.md).

$$\min_{\mathcal R\in\mathfrak G}\operatorname{cost}(\mathcal R)
\quad\text{subject to}\quad
\forall x\in\mathcal X:\ \operatorname{error}(\mathcal R,x;T)\le\varepsilon.$$

Declare the finite reaction grammar $\mathfrak G$, semantics, encoding, decoding, time requirement, input set and cost ordering. Correctness of one network gives an upper cost bound; excluding every lower-cost network gives minimality within that grammar. A finite truth table does not prove universal computation or self-replication.

**Critique.** The synthesis joins minimal autocatalysis, self-replication and universal computation, but none implies the others. “Smallest chemistry” depends on kinetics, encoding, concentration precision and which resources count as free. A fixed finite truth table does not prove Turing universality, while an analog model with unbounded precision may hide computation in its inputs. The rewrite separates a bounded minimal implementation result from a universal simulation theorem and from the RAF structure in PRD 29.

**Validation and milestones.** Baseline: one certified reaction implementation with explicit semantics. Strong: minimality within a finite grammar and input domain. Extension: a compiler/simulation theorem; avoid calling the baseline universal or biologically self-replicating.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.


# Synthesis 30. Genotype→Phenotype Channel Capacity Bounds

**Mapping:** [PRD 30](../PRDs/30-Genotype-Phenotype-Mapping.md), [S12](../supplemental/S12.md). Related PRD; supplement preserves distinct objective.

## 30. An exact finite genotype–phenotype map and its mutation graph

**Question.** For a fully specified combinatorial RNA folding rule on short sequences, what are the exact neutral components, mutation robustness, and accessible phenotype counts?

**Why it matters.** The structure of a genotype–phenotype map helps explain how variation can accumulate without immediately changing a selected outcome, and how new outcomes become accessible. Exact small maps provide benchmarks for sampling methods and expose how conclusions depend on folding rules and tie choices.

**Scope and model.** Use alphabet {A,C,G,U}, sequence length L=4,…,8 initially, allowed pairs AU/UA, GC/CG and GU/UG, minimum hairpin length three unpaired bases, no pseudoknots, and one pair per base. Maximize pair count and break ties by lexicographically smallest dot-bracket string under a declared character order.

This defines a mathematical GP map, not a thermodynamic RNA predictor. A nearest-neighbor free-energy extension must name its parameter set; empirical parameters are then acknowledged model inputs.

### Mathematical formulation and claim boundary

[Canonical specification](../PRDs/30-Genotype-Phenotype-Mapping.md).

$$\Phi:\{A,C,G,U\}^L\longrightarrow\mathcal S_L,\qquad
r(s)=\frac{|\{t:d_H(s,t)=1,\ \Phi(t)=\Phi(s)\}|}{3L},$$

$$\sum_{\varphi}|\Phi^{-1}(\varphi)|=4^L,\qquad
\deg_{H(L,4)}s=3L.$$

Specify pair-count maximization, allowed pairs, hairpin length, pseudoknot exclusion and deterministic tie-breaking. Enumerate every connected component of each phenotype preimage. This combinatorial map is not a thermodynamic predictor. At $L=20$, $4^L=1{,}099{,}511{,}627{,}776$; sampling does not establish a complete catalog.

**Deliverables.** Export folding rules, tie-breaking, sequence-to-structure table, neutral components, exact counts and rational robustness. A search restricted to a component must not report the full phenotype frequency. Fitness is a separately supplied function; evolutionary dynamics require population size, mutation kernel, selection rule and initial state.

**Validation.** Cross-check dynamic programming with exhaustive legal-structure enumeration on short sequences, especially ties and prohibited pairs. Verify Σ_φ|Φ⁻¹(φ)|=4^L and every Hamming vertex degree is 3L. Count neutral edges independently and recover average robustness from twice the edge count. Treat any larger-L sample as statistical evidence.

**Milestones.** Baseline: complete exact maps through L=8. Strong: increase L within measured resources and characterize component-level robustness/evolvability. Research extension: a proved asymptotic property or a specified noisy GP-channel capacity problem. Do not hard-code a universal percolation threshold or empirical target correlation.

[Detailed critique](../reviews/30-Genotype-Phenotype-Mapping.md).

## S12. Capacity of a specified finite noisy genotype–phenotype channel

**Question.** For a finite conditional phenotype law P(y|g) and optional genotype cost constraint, what certified lower and upper bounds can be placed on Shannon capacity?

**Why it matters.** An explicit channel separates the number of possible outputs from their reliability and distinguishability. Capacity bounds give a rigorous way to compare modeled noise mechanisms or resource constraints, and to check whether a numerical optimization has found the best input distribution.

**Scope and model.** Supply finite genotype and phenotype alphabets, a stochastic matrix W(y|g), and optional costs c(g) with E[c]≤C. Define capacity max_p I_p(G;Y), with logarithms base two. A chosen input distribution gives a lower bound. For the unconstrained channel, any output distribution q with adequate support gives upper bound max_g D(W(.|g)||q); constrained dual bounds must include the cost multiplier. Use interval logarithms and validated probabilities. Connect to PRD 30 by constructing W from a fully declared folding/noise rule.

### Mathematical formulation and claim boundary

[Supplemental specification](../supplemental/S12.md).

$$C=\max_p I_p(G;Y),\qquad
I_p(G;Y)\le C\le\max_g D(W(\cdot|g)\|q).$$

For a cost limit $c_*$, any $\lambda\ge0$ and adequately supported $q$ give

$$C(c_*)\le\lambda c_*+
\max_g\{D(W(\cdot|g)\|q)-\lambda c(g)\}.$$

Use base-two logarithms and verified stochastic rows. A feasible input distribution supplies a lower bound, and interval logarithms certify numerical values. For a deterministic unrestricted finite map, capacity is $\log_2|\operatorname{im}\Phi|$; this differs from neutral-network connectivity.

**Critique.** Mutual information is a property of a joint distribution, not a map alone. The original omits the input distribution, noise, phenotype alphabet and optimization constraints. For a deterministic finite map with unrestricted input distribution, capacity is simply log₂ of the number of reachable phenotypes, so a purported hard capacity problem may collapse. Fano/Le Cam bounds answer specific inference questions and are not automatic capacity certificates. The rewrite specifies a nontrivial noisy channel, separates achievability from converse bounds, and keeps biophysical interpretation conditional on the chosen channel.

**Validation and milestones.** Baseline: binary symmetric-channel capacity and deterministic-map capacity reproduced. Strong: a certified bracket for a finite GP channel and a cost-constrained extension with a valid dual. Check rows sum to one, handle zero probabilities explicitly, and report unresolved numerical gaps.

Use the shared [evidence standard](../reviews/EVIDENCE-STANDARD.md). Unsuccessful search is unresolved unless a complete exclusion proof is supplied. These are proposed baseline specifications, not completed scientific results.

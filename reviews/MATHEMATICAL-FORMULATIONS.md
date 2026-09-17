# Report mathematical formulations

Updated 2026-09-17 from the canonical specifications and the detailed critiques. These displayed formulations collect definitions and claim directions for the regenerated reports. They are research targets, not claims that the proposed certificates have been computed. Each section inherits the scope, hypotheses and validation obligations of its linked specification; the [evidence standard](EVIDENCE-STANDARD.md) applies throughout.

## 01

[Canonical specification](../PRDs/01-AdS3-Modular-Bootstrap.md).

For the holomorphic sector with $c=24k$, $q=e^{2\pi i\tau}$ and $J=j-744$,

$$\chi_0=q^{-k}\prod_{n=2}^{\infty}(1-q^n)^{-1},\qquad
\chi_h=\frac{q^{h-k+1/24}}{\eta(\tau)}\quad(h>0).$$

$$Z_k=P_k(J)=\chi_0+\sum_{h\ge k+1}d_h\chi_h,\qquad
P_k(J)-\chi_0=O(q).$$

Determine $P_k$ exactly and test $d_h\in\mathbb Z_{\ge0}$ through a declared $H$. At $k=1$, $d_2=196883$. Passing this finite necessary test does not construct the CFT or establish OPE consistency.

## 02

[Canonical specification](../PRDs/02-Gravitational-Positivity-Bounds.md).

$$s+t+u=4m^2,\qquad \nu=s-2m^2+\frac t2,\qquad
b_2(t)=\frac12\partial_\nu^2 B(0,t).$$

$$B=A-A_{\mathrm{light\ poles}}-A_{\mathrm{chosen\ low\ cuts}},\qquad
b_2(t)=I_{\mathrm{UV}}(t)+R_\infty(t).$$

The last equality is the sum rule to derive, with its kernel, subtractions and contour assumptions. Only a justified smearing or regulated forward prescription can turn it into a positivity statement. If that prescription proves $I_{\mathrm{UV}}\ge0$ and $R_\infty\ge-R_*$, then $b_2\ge-R_*$. An uncontrolled contour term must remain unknown.

## 03

[Canonical specification](../PRDs/03-Celestial-CFT-Bootstrap.md).

$$p_i=\epsilon_i\omega_i q_i,\quad
\Delta_i=1+i\lambda_i,\quad
h_i=\frac{\Delta_i+J_i}{2},\quad
\bar h_i=\frac{\Delta_i-J_i}{2}.$$

$$\widetilde{\mathcal A}=
\left(\prod_{i=1}^4\int_0^\infty d\omega_i\,\omega_i^{\Delta_i-1}\right)
\delta^{(4)}\!\left(\sum_i\epsilon_i\omega_iq_i\right)\mathcal M_4.$$

The expression is a regulated distribution. A leading soft term proportional to $\omega_s^{-1}$ yields the local integral $\int_0^\Lambda\omega_s^{\Delta_s-2}d\omega_s=\Lambda^{\Delta_s-1}/(\Delta_s-1)$ before continuation, exposing the pole at $\Delta_s=1$. Support, helicity, crossing and regulator removal remain proof obligations; ordinary Euclidean OPE positivity is not assumed.

## 04

[Canonical specification](../PRDs/04-Modular-Lightcone-Bootstrap.md).

$$F_{\Delta,\ell}=v^{\Delta_\phi}g_{\Delta,\ell}(u,v)
-u^{\Delta_\phi}g_{\Delta,\ell}(v,u),\qquad
F_{\mathrm{id}}+\sum_{\mathcal O}\lambda_{\phi\phi\mathcal O}^{2}F_{\Delta,\ell}=0.$$

$$\Delta_{n,\ell}=2\Delta_\phi+2n+\ell+\gamma_{n,\ell},\qquad
|\gamma_{0,\ell}-\widehat\gamma_{0,\ell}|\le R(\ell)\quad(\ell\ge\ell_0).$$

The inequality is a target requiring explicit spectral and Regge assumptions and a proved remainder. An asymptotic series alone does not imply this finite-spin inequality. Higher-dimensional crossing is not torus modular invariance.

## 05

[Canonical specification](../PRDs/05-Positive-Geometry-Gravity.md).

$$\Omega_{\mathrm{target}}=I(\ell;k)\,d^4\ell,
\qquad \Omega_{\mathrm{target}}\stackrel{?}{=}f(k)\,\pi_*\Omega_G.$$

For a standard positive geometry, each oriented boundary must satisfy the canonical residue relation $\operatorname{Res}_{\partial G}\Omega_G=\Omega_{\partial G}$. Fix the geometry class, maps, prefactors, representation and equality convention first. Verify finite boundaries, intersections and infinity; an obstruction is limited to that class. An integrated symbol is not the integrand's canonical form.

## 06

[Canonical specification](../PRDs/06-Nonperturbative-S-matrix-Bootstrap.md).

$$A=16\pi\sum_{\ell\ge0}(2\ell+1)a_\ell(s)P_\ell(\cos\theta),\quad
\rho=\sqrt{1-4m^2/s},\quad S_\ell=1+2i\rho a_\ell.$$

$$|S_\ell|\le1\quad\Longleftrightarrow\quad
\operatorname{Im}a_\ell\ge\rho|a_\ell|^2\qquad(s>4m^2).$$

Equality requires an elastic sector. A certified outer relaxation can exclude a physical amplitude only after the inclusion of every admissible amplitude and all tail bounds are proved. Failure of an inner ansatz excludes only that ansatz. The gravitational stage has a separate infrared prerequisite.

## 07

[Canonical specification](../PRDs/07-Extremal-CFTs-Stress-Tensor.md).

$$V_{\mathrm{id}}+V_T(C_T,t_4)+
\sum_{\mathcal O}\lambda_{\mathcal O}^{\mathsf T}V_{\Delta,\ell,\pi}
\lambda_{\mathcal O}=0.$$

$$\alpha(V_{\mathrm{id}}+V_T)>0,\qquad
\alpha(V_{\Delta,\ell,\pi})\succeq0
\quad\text{for every allowed exchange}.$$

These inequalities contradict crossing and exclude the proposed sector-specific gap. Tensor structures, conserved spinning blocks, Ward identities and continuous exchange coverage are required. In three dimensions a conserved symmetric-traceless spin-$\ell$ primary has $\Delta=\ell+1$.

## 08

[Canonical specification](../PRDs/08-Swampland-Modularity-Symmetries.md).

$$S^\dagger S=I,\quad S^2=C,\quad(ST)^3=C,\quad
MS=SM,\quad MT=TM,\quad M_{00}=1,\quad M_{ij}\in\mathbb Z_{\ge0}.$$

$$N_{ij}^{\ k}=\sum_a\frac{S_{ia}S_{ja}S_{ka}^*}{S_{0a}}
\in\mathbb Z_{\ge0},\qquad \omega\in Z^3(G,U(1)),\quad\delta\beta=\omega.$$

Fix the $T$ convention and any assumed symmetry of $S$. An explicit cochain $\beta$ trivializes the specified anomaly; a complete cohomology calculation may prove no such cochain exists. This gauging obstruction does not make an anomalous boundary theory inconsistent, and untwisted modular data alone do not supply defect data.

## 09

[Canonical specification](../PRDs/09-Topological-Band-Theory.md).

$$H_m(k)=\sin k_x\,\sigma_x+\sin k_y\,\sigma_y+
(m+\cos k_x+\cos k_y)\sigma_z.$$

$$E_{r+1}(k)-E_r(k)\ge\delta>0,\qquad
C=\frac{1}{2\pi i}\int_{\mathrm{BZ}}
\operatorname{Tr}\!\left(P[\partial_{k_x}P,\partial_{k_y}P]\right)\,d^2k.$$

Prove the gap over the full claimed parameter and momentum domain. Fix orientation and the occupied projector. An analytic argument or validated integral enclosing one integer certifies $C$; a momentum mesh alone does not.

## 10

[Canonical specification](../PRDs/10-Flat-Chern-Bands.md).

$$W=\max_k E_n(k)-\min_k E_n(k),\quad
\Delta=\min_k\min_{j\ne n}|E_j(k)-E_n(k)|,\quad\text{minimize }W/\Delta.$$

$$g_{ij}=\operatorname{Re}\langle\partial_i u|(1-P)|\partial_j u\rangle,
\quad \bar F=\frac{2\pi C}{A},\quad A=\operatorname{Area}(\mathrm{BZ}),$$

$$\operatorname{Var}(F)=\frac1A\int_{\mathrm{BZ}}(F-\bar F)^2d^2k,
\qquad D_g=\frac1A\int_{\mathrm{BZ}}(\operatorname{tr}g-|F|)d^2k.$$

Fix a gap floor, norm bound, orbital positions and coordinate metric. Use the Chern convention of 09. Strictly finite-range approximate flatness and generally long-range exact flattening are separate tasks; these diagnostics alone do not prove an interacting fractional phase.

## 11

[Canonical specification](../PRDs/11-Photonic-Topological-Crystals.md).

$$(\nabla+ik)\times\mu^{-1}(r)(\nabla+ik)\times u
=\omega^2\epsilon(r)u,\qquad
\langle u,v\rangle_\epsilon=\int_{\mathrm{cell}}u^*\cdot\epsilon v\,dr.$$

Use positive Hermitian constitutive tensors, the transverse constraint, units $c=1$, and an isolated positive-frequency band group. Certify its gap and projector invariant in the weighted metric. Nonzero total Chern number requires breaking the relevant physical time-reversal symmetry; scalar reciprocal benchmarks provide rejection controls.

## 12

[Canonical specification](../PRDs/12-Topological-Mechanical-Metamaterials.md).

$$Q=C^\dagger,\qquad n_0-n_{ss}=dN-N_b,\qquad
K=C^\dagger\operatorname{diag}(k_b)C,\quad D=M^{-1/2}KM^{-1/2}.$$

$$\nu_\gamma=\frac1{2\pi i}\oint_\gamma d\log\det C(k),
\qquad \det C(k)\ne0\text{ on }\gamma.$$

The first count includes rigid-body motions and self stress. Supply actual bond directions, unit-cell gauge and termination. Acoustic zeros and the local boundary count must be included before identifying edge modes from winding.

## 13

[Canonical specification](../PRDs/13-Higher-Order-Topological-Insulators.md).

$$W_x(k_y)=\mathcal P\exp\!\left(i\oint\mathcal A_x(k_x,k_y)\,dk_x\right),
\qquad \mathcal A_{x,ab}=i\langle u_a|\partial_{k_x}u_b\rangle.$$

$$\widetilde W_y=\mathcal P\exp\!\left(i\oint\widetilde{\mathcal A}_y\,dk_y\right),
\qquad p_y^{\mathrm{sector}}=\frac1{2\pi}\arg\det\widetilde W_y\pmod1.$$

Use consistent holonomy orientation in an overlap implementation. The second connection belongs to a smoothly transported, spectrally isolated Wannier sector, not to the first loop's eigenphases alone. Prove both bulk and Wannier gaps and the relevant quantizing symmetry. Define corner charge with a reference background and termination; zero-energy corner states require additional assumptions.

## 14

[Canonical specification](../PRDs/14-Topological-Semimetals-Weyl-Dirac.md).

$$d(k)=(\sin k_x,\sin k_y,2-\cos k_x-\cos k_y-\cos k_z),
\quad d(k_*)=0,$$

$$k_*=(0,0,\pm\pi/2),\qquad
V_{ij}=\partial_{k_j}d_i(k_*),\qquad \chi=\operatorname{sgn}\det V.$$

The two determinants are $+1$ and $-1$ in the stated coordinate order. Relate this convention explicitly to the occupied-band flux. Certify absence of other nodes on the complement and the topology of gapped slices; surface arcs additionally depend on the chosen boundary.

## 15

[Canonical specification](../PRDs/15-Topological-Quantum-Chemistry.md).

$$A\subset B\subset\mathbb Z^r,\qquad X=B/A,\qquad
UA_{\mathrm{mat}}V=D,\quad U,V\text{ unimodular}.$$

$$b=A_{\mathrm{mat}}n,\quad n\in\mathbb Z^s
\qquad\text{versus}\qquad b=A_{\mathrm{mat}}n,\quad n\in\mathbb Z_{\ge0}^s.$$

Use a declared basis for the compatibility lattice $B$ in the Smith decomposition. Integer membership and nonnegative atomic symmetry-data decomposition are distinct tests. Neither a zero indicator nor matching atomic labels alone constructs localized Wannier functions for a given Hamiltonian.

## 16

[Canonical specification](../PRDs/16-N-Representability-2RDM.md).

$$\gamma_{ij}=\langle a_i^\dagger a_j\rangle,\qquad
\Gamma_{ij,kl}=\langle a_i^\dagger a_j^\dagger a_l a_k\rangle,$$

$$\operatorname{Tr}\gamma=N,\quad
\sum_{ij}\Gamma_{ij,ij}=N(N-1),\quad
\sum_j\Gamma_{ij,kj}=(N-1)\gamma_{ik}.$$

$$L_{\mathrm{dual}}\le E_{\mathrm{relax}}\le E_0\le
\langle\Psi|H|\Psi\rangle=U,\qquad \langle\Psi|\Psi\rangle=1.$$

Derive two-particle, two-hole and particle-hole moment matrices from anticommutation in this convention. Necessary PSD constraints give an outer relaxation. Certify dual feasibility for $L$ and a physical $N$-fermion state for $U$; a relaxed RDM need not represent any state.

## 17

[Canonical specification](../PRDs/17-Isomer-Enumeration-Molecular-Graphs.md).

Let $\mathcal G$ contain exactly the graphs satisfying the fixed atom, bond, valence and connectivity grammar. Then

$$\text{catalog}=\mathcal G/\!\cong.$$

Isomorphisms preserve the declared atom and bond labels. Every output must satisfy the grammar, no two representatives may be equivalent, and every allowed input must map to a representative. Canonical labeling proves deduplication only; completeness needs a coverage proof for generation and pruning. Stereochemistry requires a separately declared equivalence model.

## 18

[Canonical specification](../PRDs/18-Optimal-Transport-Chemistry.md).

$$\int\rho=N,\qquad \mu=\rho/N,\qquad
c(r_1,\ldots,r_N)=\sum_{i<j}|r_i-r_j|^{-1},$$

$$V_{\mathrm{SCE}}[\rho]=\inf_{\Gamma\in\Pi(\mu,\ldots,\mu)}\int c\,d\Gamma,
\qquad \sum_i u(r_i)\le c(r_1,\ldots,r_N),$$

$$\underbrace{\int u\rho}_{L}\ \le\ V_{\mathrm{SCE}}[\rho]\ \le\
\underbrace{\int c\,d\Gamma_{\mathrm{feasible}}}_{U}.$$

The coupling is a probability measure; it may be symmetrized and is not antisymmetric. This is a Coulomb interaction functional, not exact exchange. For a discrete problem, the same inequalities apply only to the discrete optimum until continuum error is bounded.

## 19

[Canonical specification](../PRDs/19-Chemical-Reaction-Networks.md).

$$\dot x=S v(x),\qquad v_j(x)=k_j\prod_i x_i^{\alpha_{ij}},\qquad
x\in(x_0+\operatorname{im}S)\cap\mathbb R_{\ge0}^n.$$

$$f(x)=6-11x+6x^2-x^3=-(x-1)(x-2)(x-3),
\qquad (f'(1),f'(2),f'(3))=(-2,1,-2).$$

The scalar benchmark has two locally stable positive equilibria and one unstable equilibrium. General equilibrium enumeration needs complete coverage within the compatibility class, boundary treatment, and a reduced Jacobian. Positivity invariance, persistence and multistability are different claims.

## 20

[Canonical specification](../PRDs/20-Ab-Initio-Path-Integrals.md).

$$\beta_P=\beta/P,\quad\omega_P=P/(\beta\hbar),\quad q_{P+1}=q_1,$$

$$\pi_P(q)\propto\exp\!\left[-\beta_P\sum_{s=1}^P
\left\{\frac{m\omega_P^2}{2}(q_s-q_{s+1})^2+V(q_s)\right\}\right],$$

$$E_{\mathrm{prim}}=\frac{P}{2\beta}
-\frac{mP}{2\beta^2\hbar^2}\sum_s(q_s-q_{s+1})^2
+\frac1P\sum_sV(q_s).$$

Coordinate observables use bead averages. Derive the estimator from the same partition-function convention. Separate sampling, integration, finite-$P$ and model errors; convergence fits and empirical autocorrelation diagnostics do not by themselves give rigorous enclosures.

## 21

[Canonical specification](../PRDs/21-Quantum-LDPC-Codes.md).

$$H_X=[H_1\otimes I_{n_2}\mid I_{m_1}\otimes H_2^{\mathsf T}],
\qquad H_Z=[I_{n_1}\otimes H_2\mid H_1^{\mathsf T}\otimes I_{m_2}],$$

$$H_XH_Z^{\mathsf T}=0,\quad n=n_1n_2+m_1m_2,\quad
k=n-\operatorname{rank}_{\mathbb F_2}H_X-\operatorname{rank}_{\mathbb F_2}H_Z,$$

$$d_X=\min_{x\in\ker H_Z\setminus\operatorname{row}H_X}|x|,
\qquad d_Z=\min_{z\in\ker H_X\setminus\operatorname{row}H_Z}|z|,
\qquad d=\min(d_X,d_Z).$$

Two full-rank $3\times7$ Hamming checks give $n=58$, $k=16$; the included independent validator also checks $d=3$. A logical witness gives an upper distance bound; exclusion of all smaller logicals gives the lower bound. Finite code parameters do not establish an asymptotic LDPC family or decoding threshold.

## 22

[Canonical specification](../PRDs/22-Bell-Inequalities-Nonlocality.md).

$$\beta(P)=\sum_{abxy}c_{abxy}P(ab|xy),\quad
\beta_L=\max_{P\text{ deterministic local}}\beta(P),\quad
L_Q\le\beta_Q\le U_Q.$$

A finite-dimensional state and measurements certify $L_Q$; a verified NPA dual or operator SOS certifies $U_Q$. For CHSH, $\beta_L=2$ and an achieving two-qubit strategy plus an upper-bound identity give $\beta_Q=2\sqrt2$. NPA's commuting-operator limit must not be identified with the closure of finite-dimensional tensor-product strategies. A facet in affine dimension $D$ needs saturating affine rank $D-1$.

## 23

[Canonical specification](../PRDs/23-Entanglement-Measures-Witnesses.md).

$$\mathcal N(\rho)=\frac{\|\rho^{T_B}\|_1-1}{2},\qquad
\langle a\otimes b|W|a\otimes b\rangle\ge0,\quad \operatorname{Tr}(W\rho)<0.$$

$$\rho=\sum_jp_j|a_jb_j\rangle\langle a_jb_j|,
\qquad p_j\ge0,\quad\sum_jp_j=1.$$

The witness conditions certify entanglement; the normalized product-state decomposition certifies separability. In general, passing finite PPT/DPS tests remains unresolved. For the two-qubit family $\rho_p=p|\psi^-\rangle\langle\psi^-|+(1-p)I/4$, $0\le p\le1$, separability holds exactly for $p\le1/3$.

## 24

[Canonical specification](../PRDs/24-Topological-Quantum-Error-Correction.md).

$$C_2\xrightarrow{\partial_2}C_1\xrightarrow{\partial_1}C_0,
\quad\partial_1\partial_2=0,\quad H_X=\partial_1,\quad H_Z=\partial_2^{\mathsf T},$$

$$n=2L^2,\quad k=2,\quad d=L\quad(L\ge3),\qquad
e+r\in\operatorname{im}\partial_2\quad\text{for successful }Z\text{-error recovery}.$$

A vanishing residual syndrome only gives $e+r\in\ker\partial_1$ and can hide a nontrivial logical cycle. Statistical decoder curves require the fixed noise model and decoder; finite-size crossings are threshold estimates, not limit theorems.

## 25

[Canonical specification](../PRDs/25-Quantum-Algorithms-Complexity.md).

$$\theta=\arcsin(N^{-1/2}),\qquad
p_r=\sin^2((2r+1)\theta),\qquad
r\approx\frac{\pi}{4\theta}-\frac12.$$

Choose a nonnegative integer by checking neighboring counts and explicitly handle small instances under the declared oracle model. The target is a matched $O(\sqrt N)$ algorithm and $\Omega(\sqrt N)$ lower bound with identical promise and error tolerance. An oracle separation has the form $\mathrm{BQP}^O\ne\mathrm{BPP}^O$ and does not prove $\mathrm{BQP}\ne\mathrm{BPP}$.

## 26

[Canonical specification](../PRDs/26-KAM-Theory-Planetary-Stability.md).

$$H(I,\theta)=\omega\cdot I+\frac{|I|^2}{2}
+\epsilon[\cos\theta_1+\cos(\theta_1-\theta_2)],
\qquad\omega=(1,(1+\sqrt5)/2),$$

$$E(\theta)=X_H(K(\theta))-DK(\theta)\omega,
\qquad |k\cdot\omega|\ge\frac\gamma{|k|_1^\tau}
\quad(k\in\mathbb Z^2\setminus\{0\}).$$

Prove the all-integer Diophantine bound and every analytic-width, twist, tail and smallness condition of a specified quantitative theorem. Its conclusion is a nearby invariant torus, not stability of arbitrary planetary initial conditions. Failure of a sufficient criterion is not instability.

## 27

[Canonical specification](../PRDs/27-N-Body-Central-Configurations.md).

$$U=\sum_{i<j}\frac{m_im_j}{|r_i-r_j|},\quad
\sum_i m_i r_i=0,\quad I=\sum_i m_i|r_i|^2=1,$$

$$\nabla_{r_i}U=-\lambda m_i r_i,\qquad\lambda=U/I>0,
\qquad d_{ij}^2=|r_i-r_j|^2,\quad d_{ij}>0.$$

Preserve collision exclusions when clearing denominators. For three labeled positive masses modulo $O(2)$, recover three collinear orderings and one equilateral shape. Complete coverage requires more than isolated root boxes. Relative-equilibrium stability uses the reduced rotating-frame system including Coriolis terms; a figure-eight choreography is not a central configuration.

## 28

[Canonical specification](../PRDs/28-Nekhoroshev-Stability-Theory.md).

$$H\circ\Phi=Z(J)+R(J,\theta),\qquad
|I-J|\le\eta,\qquad\|\partial_\theta R\|\le r,$$

$$|I(t)-I(0)|\le2\eta+r|t|.$$

This estimate assumes the angle-independent normal form shown and a proved domain-containment interval. Include resonant drift if $Z$ depends on angles. A quantitative Nekhoroshev theorem instead requires all its constants, hypotheses and threshold in $T=C\exp(c\epsilon^{-a})$, with units supplied by $C$. The number $\exp(10^{0.3})\approx7.35413$ has no unit of time.

## 29

[Canonical specification](../PRDs/29-Chemical-Reaction-Networks-Origins.md).

$$R'\subseteq R,\quad R'\ne\varnothing,\quad
\operatorname{reactants}(r)\subseteq\operatorname{cl}_{R'}(F),$$

$$\forall r\in R'\ \exists s\in\operatorname{cl}_{R'}(F):\ (s,r)\in C.$$

Compute closure by reactant availability without imposing catalysis in the closure step. These conditions define a RAF. Inclusion-maximal, inclusion-minimal (irreducible), and minimum-cardinality RAFs are separate targets. Catalysis-respecting startup and sustained growth require additional conditions.

## 30

[Canonical specification](../PRDs/30-Genotype-Phenotype-Mapping.md).

$$\Phi:\{A,C,G,U\}^L\longrightarrow\mathcal S_L,\qquad
r(s)=\frac{|\{t:d_H(s,t)=1,\ \Phi(t)=\Phi(s)\}|}{3L},$$

$$\sum_{\varphi}|\Phi^{-1}(\varphi)|=4^L,\qquad
\deg_{H(L,4)}s=3L.$$

Specify pair-count maximization, allowed pairs, hairpin length, pseudoknot exclusion and deterministic tie-breaking. Enumerate every connected component of each phenotype preimage. This combinatorial map is not a thermodynamic predictor. At $L=20$, $4^L=1{,}099{,}511{,}627{,}776$; sampling does not establish a complete catalog.

## S01

[Supplemental specification](../supplemental/S01.md).

$$H(B)=H_{\mathrm{CF}}+\lambda L\cdot S+\mu_B B\cdot(L+2S),$$

$$P H(B)P=E_0I+\frac{\mu_B}{2}\sum_{a,b}B_a g_{ab}\sigma_b+O(|B|^2).$$

Use the isolated Kramers-doublet projector $P$ and exact orbital-spin conventions. The singular values of $g$ remove doublet-basis ambiguity. Certify the gap above the doublet and bound any denominator in an anisotropy ratio on the compact parameter box. This finite ion model does not establish bulk magnetism.

## S02

[Supplemental specification](../supplemental/S02.md).

$$\lambda_- =\max_k\lambda_n(k),\qquad
\lambda_+=\min_k\lambda_{n+1}(k),\qquad
g=\lambda_+-\lambda_-. $$

A possible normalized objective is $G=2g/(\lambda_++\lambda_-)$ when the denominator is positive; frequency-gap objectives must instead use $\omega_j=\sqrt{\lambda_j}$ consistently.

Use a fixed self-adjoint wave model, bounded coefficients and geometry class, with a positive denominator. A complete gap needs $\lambda_+>\lambda_-$ across the entire Brillouin zone and all required polarizations. A certified candidate lower bound does not establish global optimality; that needs coverage of the full design class.

## S03

[Supplemental specification](../supplemental/S03.md).

$$e\cdot A^*e=\min_{\phi\ \mathrm{periodic}}
\langle a(x)|e+\nabla\phi|^2\rangle,\qquad a(x)\in\{a_1,a_2\},$$

$$\left(\frac f{a_1}+\frac{1-f}{a_2}\right)^{-1}|e|^2
\le e\cdot A^*e\le(fa_1+(1-f)a_2)|e|^2.$$

Assume positive scalar phases, fixed volume fraction $f$, and a normalized cell average. Potential trials give upper bounds and the corresponding flux principle gives lower bounds. A finite microstructure list does not characterize an unrestricted G-closure.

## S04

[Supplemental specification](../supplemental/S04.md).

$$\operatorname{dist}(E_F,\operatorname{spec}H_0)\ge g>0,
\qquad\|V\|<g\quad\Longrightarrow\quad
\operatorname{dist}(E_F,\operatorname{spec}(H_0+V))\ge g-\|V\|.$$

This bounded self-adjoint perturbation estimate supplies a spectral-gap hypothesis. Equality of a chosen finite-volume index with the bulk index still requires a specific localizer or other index theorem with all radius, locality, commutator and scale conditions checked. Mobility-gap claims need a different theorem.

## S05

[Supplemental specification](../supplemental/S05.md).

$$F(n)=\min_{\substack{\rho\succeq0,\ \operatorname{Tr}\rho=1\\
\operatorname{Tr}(\rho\widehat n_i)=n_i}}
\operatorname{Tr}\rho(T+W),$$

$$E(v)=\min_{\rho\succeq0,\ \operatorname{Tr}\rho=1}
\operatorname{Tr}\rho\!\left(T+W+\sum_i v_i\widehat n_i\right),
\qquad F(n)=\sup_v[E(v)-v\cdot n].$$

Work in the fixed finite $N$-particle Hilbert space and on ensemble-representable densities. The supremum need not be attained by a finite potential at boundary densities. Feasible density matrices give upper bounds; certified energy lower bounds at chosen potentials give lower bounds. Convexity of $F$ does not imply convexity of an exchange-correlation remainder.

## S06

[Supplemental specification](../supplemental/S06.md).

$$v_\theta(r)=\sum_j\theta_j\phi_j(r),\qquad
E_\theta(\Lambda)=\frac12\sum_{x\in\Lambda\setminus\{0\}}v_\theta(|x|),$$

$$E_\theta(\Lambda_{\mathrm{target}})\le
E_\theta(\Lambda_{\mathrm{competitor}})-\delta.$$

Require convergent sums and certified tails, uniformly on the declared density interval and competitor set. The conclusion is best among those competitors. Unrestricted configurations, infinitesimal deformations and finite-temperature free energies require separate arguments.

## S07

[Supplemental specification](../supplemental/S07.md).

$$H_N=\sum_i h_i,\qquad h_i\succeq0,\qquad
\Delta_N=\min(\operatorname{spec}H_N\setminus\{0\}),$$

$$\text{target: }\inf_{N\ \mathrm{in\ the\ stated\ family}}\Delta_N\ge\delta>0.$$

Verify frustration freeness and ground-space assumptions. A specified finite-size gap theorem must supply the implication from validated local gaps to the displayed uniform bound. Positive gaps on finitely many chains are insufficient; failure of a sufficient criterion is not a gaplessness proof.

## S08

[Supplemental specification](../supplemental/S08.md).

$$A=\bigoplus_a n_a a,\qquad n_a\in\mathbb Z_{\ge0},\qquad
(\dim A)^2=\operatorname{Dim}\mathcal C.$$

For a vacuum boundary this Lagrangian dimension condition supplements, rather than replaces, the unit, multiplication, associativity, commutativity and separability checks using exact categorical coherence data. In the toric-code category the reference condensates are $1\oplus e$ and $1\oplus m$. A microscopic boundary Hamiltonian needs its own gap proof.

## S09

[Supplemental specification](../supplemental/S09.md).

$$U=\bigotimes_{i=1}^nU_i,\quad U_i\in\mathrm{Clifford}_1/\mathrm{phase},
\qquad H'=UHU^\dagger,$$

$$\operatorname{Im}H'_{xy}=0,\qquad\operatorname{Re}H'_{xy}\le0\quad(x\ne y).$$

Apply these conditions to the full transformed matrix, allowing cancellations between terms. Exhaustive coverage or an equivalent checked UNSAT proof excludes only this finite transformation class. Stoquasticity alone does not guarantee efficient mixing or sampling.

## S10

[Supplemental specification](../supplemental/S10.md).

$$F(z_0,T)=\varphi_T(z_0)-z_0=0,\qquad
M=D\varphi_T(z_0).$$

The shooting residual must first be augmented and reduced using a phase condition, conserved quantities and continuous symmetries so the validated root problem has the correct dimension. Interval flow and root bounds establish existence; the reduced Floquet multipliers classify linear behavior. Neither a near return nor stable-looking multipliers prove nonlinear stability or a manifold connection.

## S11

[Supplemental specification](../supplemental/S11.md).

$$\min_{\mathcal R\in\mathfrak G}\operatorname{cost}(\mathcal R)
\quad\text{subject to}\quad
\forall x\in\mathcal X:\ \operatorname{error}(\mathcal R,x;T)\le\varepsilon.$$

Declare the finite reaction grammar $\mathfrak G$, semantics, encoding, decoding, time requirement, input set and cost ordering. Correctness of one network gives an upper cost bound; excluding every lower-cost network gives minimality within that grammar. A finite truth table does not prove universal computation or self-replication.

## S12

[Supplemental specification](../supplemental/S12.md).

$$C=\max_p I_p(G;Y),\qquad
I_p(G;Y)\le C\le\max_g D(W(\cdot|g)\|q).$$

For a cost limit $c_*$, any $\lambda\ge0$ and adequately supported $q$ give

$$C(c_*)\le\lambda c_*+
\max_g\{D(W(\cdot|g)\|q)-\lambda c(g)\}.$$

Use base-two logarithms and verified stochastic rows. A feasible input distribution supplies a lower bound, and interval logarithms certify numerical values. For a deterministic unrestricted finite map, capacity is $\log_2|\operatorname{im}\Phi|$; this differs from neutral-network connectivity.

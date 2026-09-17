# The 30 Most Compelling Pure Thought AI Challenges
## A Synthesis Report

This report synthesizes 15 detailed domain analyses to identify the most compelling scientific challenges that can be tackled using **pure thought + fresh code only**—no external datasets, no experiments, no legacy software. Each challenge is grounded in axioms, symmetries, variational principles, and produces **verifiable artifacts**: proofs, certificates, impossibility theorems, or constructive models.

---

## I. QUANTUM GRAVITY & PARTICLE PHYSICS (8 challenges)

### 1. AdS₃ "Pure Gravity" via the Modular Bootstrap

**The Challenge:** Determine whether extremal 2D CFTs with only Virasoro descendants below the gap exist for large central charge c=24k, corresponding to pure AdS₃ gravity duals.

**Why Compelling:** This is a foundational question in quantum gravity. The answer—existence or rigorous impossibility—would be a genuine landmark result.

**Pure Thought Approach:** Entirely first-principles: modular invariance, unitarity, integrality constraints. Build high-precision modular bootstrap solvers with exact rational certificates and integer-degeneracy constraints. When infeasible, return dual certificates proving impossibility.

**Verifiable Artifacts:** SDP dual functionals, infeasibility certificates, or explicit extremal spectra; publishable within 6-12 months.

---

### 2. Gravitational Positivity & Causality Bounds (Flat Space)

**The Challenge:** Derive rigorous bounds on higher-derivative operators (R², R³ terms) in graviton effective field theories using analyticity, crossing, unitarity, and causality constraints.

**Why Compelling:** Determines what effective field theories are consistent with quantum gravity—a central question in the "swampland" program.

**Pure Thought Approach:** Dispersion relations with rigorous error control + convex optimization (SoS/SDP). Certificates are dual polynomials proving regions are forbidden/allowed.

**Verifiable Artifacts:** Multi-coupling bounds with machine-checkable proofs tightening known results; 3-9 month timeline.

---

### 3. Celestial CFT Bootstrap

**The Challenge:** Carve out the consistent space of celestial CFTs by enforcing SL(2,ℂ) covariance, crossing, unitarity, and soft-theorem constraints on graviton scattering amplitudes.

**Why Compelling:** Connects 4D quantum gravity to 2D celestial CFT structure—a new frontier in holography.

**Pure Thought Approach:** Mellin transform pipeline for amplitudes derived from first principles; impose celestial crossing/Regge/soft limits as SDP inequalities.

**Verifiable Artifacts:** First rigorous "islands" or no-go regions for subsectors; 6-12 month timeline.

---

### 4. Modular-Lightcone Bootstrap for Holographic CFTs

**The Challenge:** For large-c, sparse-spectrum CFTs, derive sharp bounds on higher-spin gaps and OPE coefficients implied by causality/chaos constraints.

**Why Compelling:** Provides quantitative "gravitational bootstrap" constraints—rules out bulk higher-derivative terms that would violate causality.

**Pure Thought Approach:** Lorentzian inversion + lightcone expansions certified via SDP/SoS; formal certificates proving operators must be small to avoid superluminality.

**Verifiable Artifacts:** Tighter exclusion plots for higher-spin exchanges; 6-9 months.

---

### 5. Positive Geometry for Gravity

**The Challenge:** Identify or rule out amplituhedron-like positive-geometry structures for (super)gravity sectors beyond planar N=4 SYM.

**Why Compelling:** Would reveal hidden mathematical structures in quantum gravity amplitudes or prove fundamental obstructions.

**Pure Thought Approach:** Canonical-form solvers for polytopes/Grassmannians; finite-field sampling + rational reconstruction; symbol integrability checks.

**Verifiable Artifacts:** New positive-geometry proposals or no-go theorems for specific helicity/loop sectors; 9-12 months.

---

### 6. Non-perturbative S-matrix Bootstrap with Gravity

**The Challenge:** Carve out the space of unitary, crossing-symmetric, analytic 2→2 amplitudes obeying gravitational soft theorems and proper high-energy behavior.

**Why Compelling:** First rigorous non-perturbative constraints on quantum gravity + matter systems.

**Pure Thought Approach:** Partial-wave + Roy-like equations adapted to massless exchange; impose Weinberg soft behavior as hard constraints.

**Verifiable Artifacts:** Dual certificates excluding inconsistent Wilson-coefficient tuples; islands/exclusion regions within 6-12 months.

---

### 7. Extremal Higher-Dimensional CFTs with Stress Tensor

**The Challenge:** Determine whether "nearly extremal" unitary CFTs exist in d=3,4 with large gaps to higher-spin currents (pure-gravity-like holographic duals).

**Why Compelling:** Connects CFT constraints to the existence/uniqueness of pure Einstein gravity in AdS.

**Pure Thought Approach:** Mixed-correlator bootstrap including stress tensor; certify via dual functionals.

**Verifiable Artifacts:** Stronger universal lower bounds on higher-spin gaps; 6-12 months.

---

### 8. Swampland via Modularity & Higher-Form Symmetries

**The Challenge:** Use modularity, integrality, and higher-form symmetry constraints to produce theorem-level obstructions to quantum-gravity-inconsistent spectra.

**Why Compelling:** Rigorous "no-go" theorems for theories in the swampland—mathematically definitive results.

**Pure Thought Approach:** Couple modular bootstrap to discrete/higher-form anomaly constraints (cobordism-style).

**Verifiable Artifacts:** "No such CFT" results under explicit symmetry assumptions; 9-12 months.

---

## II. MATERIALS SCIENCE (7 challenges)

### 9. Topological Band Theory Without Materials Data

**The Challenge:** Complete constructive classification of which band topologies are possible for minimal orbital contents and symmetries across all (magnetic) space groups, plus minimal tight-binding models realizing them.

**Why Compelling:** Foundational atlas mapping symmetry → achievable topologies with hard theorems and no-go results.

**Pure Thought Approach:** Compute elementary band representations; enumerate minimal tight-binding graphs; certify Chern/ℤ₂ invariants via K-theory/Wilson loops.

**Verifiable Artifacts:** Atlas of minimal models with proofs; no-go theorems where impossible; 3-6 months for non-magnetic catalog.

---

### 10. Flat Chern Bands with Provable Geometry

**The Challenge:** Design compact tight-binding lattices realizing flat bands with target Chern numbers and provably good quantum geometry (uniform Berry curvature, optimal Fubini-Study metric bounds).

**Why Compelling:** Critical for fractional Chern insulator realization—provides blueprints with mathematical guarantees.

**Pure Thought Approach:** Enumerate finite-range hopping graphs; certify Chern numbers exactly; prove flatness bounds using SoS relaxations.

**Verifiable Artifacts:** Hamiltonians + certificates of Chern number and flatness; Wannier obstruction certificates.

---

### 11. RBCE (Relativistic Band & Crystal-Field Engineering)

**The Challenge:** Proof-level classification of single-ion anisotropy and g-tensor anisotropy for all d^n configurations across every point group, identifying maximizers of magnetocrystalline anisotropy.

**Why Compelling:** Directly enables "rare-earth-like behavior without rare earths"—high-impact for permanent magnets and quantum materials.

**Pure Thought Approach:** Enumerate point-group irreps; build CF+SOC Hamiltonians symbolically; derive closed-form anisotropy bounds; prove necessary/sufficient symmetry conditions for Ising-like behavior.

**Verifiable Artifacts:** Theorem-backed atlas with max-anisotropy certificates and explicit model Hamiltonians.

---

### 12. Photonic/Phononic Crystals: Rigorous Bandgap Optimization

**The Challenge:** Design microstructures with certified complete bandgaps at lowest possible index contrast, with provable gap-to-midgap maxima under symmetry constraints.

**Why Compelling:** Yields fabrication-ready unit cells with airtight mathematical guarantees—high practical value.

**Pure Thought Approach:** Verified FEM/plane-wave solvers with interval arithmetic outputting rigorous eigenvalue bounds; topology optimization with SoS/shape-derivative certificates.

**Verifiable Artifacts:** Microstructure blueprints + interval-verified band diagrams + dual certificates for optimality.

---

### 13. Universal Bounds for Effective Properties of Composites

**The Challenge:** Derive sharp bounds on effective conductivity, permittivity, elasticity for given phase properties and symmetries; cross-property bounds linking elastic and thermal responses.

**Why Compelling:** Goes beyond classical Hashin-Shtrikman bounds—provides fundamental limits no microstructure can beat.

**Pure Thought Approach:** G-closure problems as SDP moment relaxations; construct microstructures attaining bounds via inverse homogenization.

**Verifiable Artifacts:** New analytical bounds + constructive microstructures + proof certificates.

---

### 14. Topological Mechanics: Maxwell Frames & Programmable Metamaterials

**The Challenge:** Complete classification and constructive design of Maxwell frames with topological polarization (robust boundary modes) and programmable mechanical response.

**Why Compelling:** Bridges rigidity theory and topological physics—enables design of mechanical metamaterials with guaranteed properties.

**Pure Thought Approach:** Enumerate periodic frameworks under space-group constraints; compute topological invariants; use SAT/MILP for isostaticity with DRAT proofs.

**Verifiable Artifacts:** Lattice designs with certified zero-mode counts and topological indices; printable unit cells.

---

### 15. Real-Space Topological Invariants for Disordered Media

**The Challenge:** Certified algorithms for Chern/ℤ₂ indices in aperiodic/disordered systems with finite-volume error bounds.

**Why Compelling:** Extends topological classification beyond perfect crystals—mathematically rigorous treatment of realistic disorder.

**Pure Thought Approach:** Implement non-commutative Chern numbers, Bott indices, spectral localizers; prove a priori error bounds vs. system size.

**Verifiable Artifacts:** Algorithms + proofs; reference models showing certified quantization in finite domains.

---

## III. CHEMISTRY (5 challenges)

### 16. N-Representability & 2-RDM Variational Chemistry

**The Challenge:** Close the gap between necessary/sufficient constraints for 2-electron reduced density matrix feasibility; deliver provably tighter constraint families.

**Why Compelling:** Direct path to ground-state energies without wavefunction—foundational for quantum chemistry.

**Pure Thought Approach:** Pure convex geometry; energies are linear in 2-RDM. Implement SDP solver with exact rational logging; add hierarchies beyond P,Q,G,T via SoS.

**Verifiable Artifacts:** Ground-state lower bounds with SDP dual certificates; infeasibility certificates; Lean-formalized lemmas.

---

### 17. Non-empirical Density Functional Theory via Convex Analysis

**The Challenge:** Characterize the cone of exchange-correlation functionals satisfying all exact constraints (scaling, spin-bounds, Lieb-Oxford, convexity); construct extremal functionals.

**Why Compelling:** DFT from first principles—no empirical fitting, only mathematical constraints.

**Pure Thought Approach:** Encode constraints as convex sets; use support-function computations (SDP/SoS) for guaranteed energy bounds; build self-interaction-free subclass.

**Verifiable Artifacts:** Certified XC bounds with proofs of constraint satisfaction; analytic forms or numeric oracles.

---

### 18. Strictly-Correlated Electrons as Multi-Marginal Optimal Transport

**The Challenge:** Compute and bound the SCE functional W_SCE[ρ] by solving multi-marginal optimal transport with Coulomb cost; derive co-motion function constructions.

**Why Compelling:** Pure optimal transport formulation of strong-correlation limit—mathematically elegant and rigorous.

**Pure Thought Approach:** Sparse/discretized OT with exact certificates; symmetry-adapted decompositions for spherical & crystalline densities.

**Verifiable Artifacts:** Certified upper/lower bounds on W_SCE with dual potentials as proof objects.

---

### 19. Complete Isomer Enumeration with Proofs

**The Challenge:** Exhaustively enumerate constitutional and stereoisomers up to size N under valence rules, with isomorphism certificates proving completeness.

**Why Compelling:** Resolves fundamental combinatorial questions in chemical space—every possible molecule within constraints.

**Pure Thought Approach:** Pólya-Redfield + canonical labeling; SAT to enforce stereochemical constraints; emit DRAT proofs of completeness/minimality.

**Verifiable Artifacts:** Exact counts, canonical SMILES/graphs, and proof logs.

---

### 20. Inverse Statistical Mechanics: Potentials with Provable Ground States

**The Challenge:** Design pair or few-body potentials that provably stabilize target crystals (diamond, wurtzite, bcc) over ranges of densities/temperatures.

**Why Compelling:** "Potentials with proofs"—enables rational design of self-assembling materials with mathematical guarantees.

**Pure Thought Approach:** Extend Cohn-Kumar bounds to 3D molecular settings; compute rigorous energy gaps with interval-verified lattice sums.

**Verifiable Artifacts:** Closed-form potentials + proofs of stability; parameter maps with certified gaps.

---

## IV. QUANTUM INFORMATION & MANY-BODY THEORY (5 challenges)

### 21. LDPC & Hypergraph-Product Quantum Error Correcting Codes

**The Challenge:** Construct new QECC families with better rate-distance-degree tradeoffs and provable decoding guarantees in adversarial/noise-model-free regimes.

**Why Compelling:** Critical for scalable quantum computing—approaching capacity with practical constraints.

**Pure Thought Approach:** Pure combinatorics & algebraic topology; LLM-led search over Tanner graphs; SAT for distance certificates; design BP/OSD/ML decoders with provable bounds.

**Verifiable Artifacts:** Code constructions with distance/minimum-weight certificates; impossibility theorems for certain parameter triples; 90 days to novel instances.

---

### 22. New Bell Inequalities & Tsirelson-type Bounds

**The Challenge:** Discover Bell inequalities with maximal quantum-classical gaps; certify with NPA (Navascués-Pironio-Acín) duals.

**Why Compelling:** Fundamental tests of quantum nonlocality with tight bounds—deepens understanding of quantum-classical boundary.

**Pure Thought Approach:** Noncommutative SDP/SoS for operator inequalities; auto-search inequality families; produce dual certificates.

**Verifiable Artifacts:** Inequality families with optimal quantum values (SDP duals) and explicit achieving measurements/states.

---

### 23. Certified Spectral Gaps for Parent Hamiltonians

**The Challenge:** Generalize Knabe/martingale techniques to automate discovery of local gap witnesses for MPS/PEPS parent Hamiltonians.

**Why Compelling:** Rigorous many-body theory—provable lower bounds on gaps guarantee phases of matter.

**Pure Thought Approach:** Tensor-network engine with interval arithmetic; automated gap-witness search producing machine-checked inequalities.

**Verifiable Artifacts:** Provable lower bounds on gaps for new families; Lean/Isabelle lemmas.

---

### 24. Topological Order & Anyon Condensation with Proofs

**The Challenge:** Classify gapped boundaries/defects; derive anomaly constraints on 2D/3D topological orders from categorical data.

**Why Compelling:** Complete mathematical understanding of topological phases—fusion rules, braiding, anomalies.

**Pure Thought Approach:** Fusion categories, modular data, anomaly tests with machine-checked coherence; construct exactly-solvable commuting-projector models.

**Verifiable Artifacts:** Fusion/braiding tables with obstruction cocycle certificates; explicit Hamiltonians.

---

### 25. Sign-Problem (Un)avoidability

**The Challenge:** Decide stoquasticity under local basis changes; characterize classes where sign-free forms are impossible.

**Why Compelling:** Resolves fundamental question about quantum Monte Carlo applicability—sharp dividing line between tractable/intractable.

**Pure Thought Approach:** SAT/ILP certificates of (non)existence; explicit unitaries achieving sign-free form when possible.

**Verifiable Artifacts:** Certificates of stoquasticity or impossibility proofs; explicit basis transformations.

---

## V. PLANETARY SYSTEMS & CELESTIAL MECHANICS (3 challenges)

### 26. KAM/Nekhoroshev Stability Domains for Planetary Systems

**The Challenge:** Compute quantitative KAM existence and Nekhoroshev stability domains for near-integrable multi-planet Hamiltonians, including post-Newtonian corrections.

**Why Compelling:** Rigorous long-time stability guarantees for planetary systems—pure dynamical systems theory with explicit time-scale bounds.

**Pure Thought Approach:** High-order averaging + validated numerics with rigorous remainder bounds; compute Diophantine constants; certify invariant tori.

**Verifiable Artifacts:** Formal normal forms + interval certificates for tori existence and explicit time-scale bounds.

---

### 27. Periodic Orbits & Invariant Manifolds in N-Body Problem

**The Challenge:** Discover new families of periodic orbits in N-body and restricted three-body problems; classify stability via validated Floquet analysis.

**Why Compelling:** Fundamental objects in celestial mechanics—"choreographies" and transport structures with machine-verified existence.

**Pure Thought Approach:** Variational solvers with symmetry constraints; validated continuation (Krawczyk/radii polynomials) to enclose true orbits; Conley index/covering relations for heteroclinic connections.

**Verifiable Artifacts:** Libraries of orbits with machine-verified existence/stability; connection proofs with interval error bounds.

---

### 28. Central Configurations Classification

**The Challenge:** Certified classification of central configurations for N=5-8 under symmetry/mass patterns; prove upper/lower bounds on their counts.

**Why Compelling:** Classic Smale problem—polynomial systems amenable to complete algebraic certification.

**Pure Thought Approach:** Gröbner/resultant elimination + α-theory to certify isolated solutions; SAT proofs for nonexistence in constrained subcases.

**Verifiable Artifacts:** Exhaustive catalogs with proof certificates.

---

## VI. BIOLOGY & ORIGIN OF LIFE (2 challenges)

### 29. Minimal Autocatalytic Cores & Universality

**The Challenge:** Find provably minimal chemical reaction networks that are autocatalytic or universal (can implement arbitrary computation/self-replication) under mass-action kinetics.

**Why Compelling:** Foundational for origin-of-life theory—what is the simplest chemistry capable of self-replication?

**Pure Thought Approach:** CRNs are finite algebraic objects; properties reduce to graph/semigroup and dynamical criteria. Branch-and-bound with isomorphism-free enumeration; DRAT/SMT certificates of minimality.

**Verifiable Artifacts:** Catalog of minimal autocatalytic sets with machine-checkable witnesses; Lean proofs for key lemmas.

---

### 30. Genotype→Phenotype Channel Capacity Bounds

**The Challenge:** Derive upper/lower bounds on mutual information between genome of length L and coarse-grained phenotype under biophysically constrained maps.

**Why Compelling:** Fundamental limits on biological design—what information can evolution actually encode?

**Pure Thought Approach:** Model as constrained circuits or graph dynamics; derive rate-distortion and Fano/Le Cam bounds; construct achievability codes to close gaps.

**Verifiable Artifacts:** Theorems with constructive encoders/decoders and proofs of capacity bounds.

---

## CORE INFRASTRUCTURE: Day-0 Toolchain

All 30 challenges share common infrastructure that should be built once and reused:

### 1. Autoprover Farm
Multi-agent proof workshop attempting multiple proof styles (induction, contradiction, SoS, compactness); round-trips through Lean/Isabelle.

### 2. Symbolic-Numeric Engine
From-scratch CAS: exact rationals, algebraic numbers, Gröbner bases, resultants, tensor calculus, interval arithmetic, automatic differentiation; self-coded SDP/QP/LP solvers with proof logging.

### 3. Search with Certificates
SAT/SMT (CDCL, DPLL(T)), branch-and-bound, A* over combinatorial spaces with certified proof traces (DRAT/FRAT).

### 4. Bootstrap & Amplitude Toolbox
Crossing/positivity SDPs, symbol alphabets, finite-field reconstructions, IBP reductions, sector decomposition—all verified.

### 5. Tensor Network & Coding Theory Lab
TN contractions, MPO/MPS/PEPS variational solvers with rigorous error bars; constructive code search with distance certificates.

### 6. Proof Back-Ends
Lean/Isabelle for theorems; DRAT/FRAT logs for SAT; SoS/SDP dual certificates; interval-arithmetic bounds—all artifacts auto-generated and re-checked.

---

## SELECTION PRINCIPLES

These 30 challenges were selected based on:

1. **Proofs > Predictions**: Each resolves to theorems, bounds, constructive counterexamples, or certificates (SAT/SMT proofs, SDP duals, cut-generating proofs, Gröbner bases).

2. **Self-Consistency or First Principles Only**: Relies on unitarity, causality, crossing, locality, symmetry, convexity (physics); quantum mechanics and variational principles (chemistry); formal axioms (math/CS).

3. **Search with Cheap Verification**: Uses massive LLM-guided search but demands fast verifiers (proof checkers, symbolic algebra identities).

4. **Formalized by Default**: Everything emits human-readable proofs AND machine-checkable artifacts (Lean/Isabelle/Coq, DRAT/FRAT, SoS certificates, interval bounds).

5. **Scientific Impact**: Each challenge addresses foundational questions with potential for publishable, citable advances.

6. **Tractability**: Milestones achievable within 3-12 months with focused effort.

---

## EXECUTION STRATEGY

### Per-Track Pattern
- Self-contained toolchain emitted by AI (solver + checker + proof exporter)
- Many-shot strategy generation (varying gauges, bases, relaxations)
- Parallel search with early proof certificates
- Human-quality explanations synthesized after machine certificate exists

### Token Budget Allocation
- 80% to search/proof attempts
- 20% to verification and proof polishing

### Prioritization for Initial Focus
**Highest-likelihood short-term wins:**
- AdS₃ modular bootstrap (Track 1)
- LDPC quantum codes (Track 21)
- Minimal autocatalytic networks (Track 29)
- RBCE symmetry atlas (Track 11)
- Gravitational positivity bounds (Track 2)

**12-Month Macro-Plan:**
- **Months 1-2**: Stand up core infrastructure (Autoprover, Symbolic-Numeric Engine, SAT/SMT with proof logging, minimal Lean kernel, SDP/SoS solver)
- **Months 3-4**: Replicate cornerstone results to validate pipelines
- **Months 5-8**: Push new bounds in multiple tracks
- **Months 9-12**: Deliver at least 3 "best-known" world results with machine-checkable proofs

---

## WHY THIS PORTFOLIO IS RIGHT FOR PURE THOUGHT + CODE

1. **Data-Free & Experiment-Free**: Every track is grounded in axioms, symmetry, or first principles.

2. **Objectively Verifiable**: Each yields artifacts (proofs/certificates) verifiable without trust.

3. **Scientifically Nontrivial**: Tighter bounds in CFT/EFT, new QECC families, stronger extremal combinatorics, constructive symmetry classifications—all publishable and foundational.

4. **Diverse Impact**: Spans quantum gravity, materials design, quantum computing, chemistry, planetary dynamics, and origin of life.

5. **Certificate-Driven**: Progress is measured by theorem-level results, not numerical approximations.

This portfolio represents the frontier of what pure mathematical reasoning, amplified by AI, can achieve in fundamental science—no experiments, no data, just axioms, algorithms, and proofs.

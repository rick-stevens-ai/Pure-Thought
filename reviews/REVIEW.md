# Scientific and editorial review of Pure Thought

**Date:** 2026-09-11
**Source commit:** `f854d4e12b55c1f5b5c2a964b75f2d23496d981e`
**Deliverables:** 30 rewritten canonical problems, 30 individual detailed critiques, 12 supplemental synthesis-topic revisions, a complete synthesis crosswalk, and a shared evidence standard.

The portfolio has a useful central idea: choose scientific questions whose results can be checked independently of the search that found them. The original descriptions often undermine that idea by leaving the model or quantifier unspecified, treating numerical evidence as proof, or requiring a result that the stated mathematics cannot establish. Some equations and concrete benchmarks are also incorrect. The revisions address the scientific framing, not just wording.

Start with the [revised catalog](../PRDs/README.md). Each problem links to its own critique and preserved original. Use the [crosswalk](SYNTHESIS-CROSSWALK.md) for the different topics in the earlier synthesis, and the [evidence standard](EVIDENCE-STANDARD.md) when implementing any challenge.

## What was reviewed and changed

The primary review covers the problem statements, mathematical formulations, certificate claims, and success criteria across all 32 original PRD files representing 30 IDs. Both variants of 07 and 08 were considered and consolidated into one canonical description each. Representative implementation sketches were examined where they directly expose a specification error; they were not executed or repaired.

All 30 numbered synthesis entries were mapped. Twelve distinct objectives receive supplementary rewrites and critiques so that the different synthesis portfolio is preserved. The LaTeX sources were inventoried and spot-checked for shared claims. Draft transcripts and their document copies were inventoried as historical ideation material. This work is not an exhaustive technical audit of every report equation, bibliography entry, draft idea, or embedded program, nor a page-by-page review of the PDFs.

The revised PRDs replace tutorial-length pseudocode with concise research contracts: a core question, explicit assumptions, mathematical target, outputs, verification controls, and achievable baseline. More ambitious results remain visible as research extensions. The archived originals preserve the explanatory and implementation material for comparison. The original PDFs and TeX are explicitly labeled historical and remain unchanged.

## Highest-priority findings

| Problem | Original issue | Revised treatment |
|---|---|---|
| 01 | Holomorphic weight, full scaling dimension and extremal gap mixed; character normalization wrong | One holomorphic sector, correct characters, exact finite necessary tests |
| 02 | Redundant curvature basis and naive gravitational positivity | One amplitude coefficient and a derived conditional sum rule |
| 03 | Ordinary positive CFT bootstrap assumed for distributional celestial amplitudes | Regulated transforms, support, crossing and derived soft residues |
| 04 | Generic higher-dimensional thermal modular invariance asserted | Lightcone crossing in d>2; separate d=2 modular extension |
| 05 | Integrands, canonical forms and integrated symbols conflated | Fixed differential form and bounded geometry ansatz |
| 06 | Gravitational IR issues and finite-ansatz infeasibility treated as universal | IR prerequisite and explicit inner/outer distinction |
| 07 | Wrong conserved-current dimensions and positivity on the wrong domain | Full spinning crossing and allowed-sector PSD conditions |
| 08 | Boundary global symmetry flagged as inconsistent with bulk gravity | Modular-data and gauging tests, with bulk/boundary distinction |
| 09 | Unbounded enumeration and false higher-Chern band-count requirements | Bounded Hamiltonian family and scoped minimality |
| 10 | Exact flatness, Chern topology and strict locality jointly required | Separate finite-range approximation and long-range flattening |
| 11 | Scalar dielectric equations used for nonreciprocal Chern phases | Explicit constitutive tensors and validated Maxwell spectrum |
| 12 | Incorrect Maxwell count; geometry replaced by connectivity | Compatibility matrix, self stress, and boundary index |
| 13 | Chern-like expression labeled quadrupole; nested loops constructed incorrectly | Wannier-sector holonomy and boundary-charge convention |
| 14 | Node finding treated as exhaustive; surface arcs overdetermined by bulk | Root coverage, chirality, slice topology and specified surface |
| 15 | Symmetry indicators presented as a complete topology classification | Quotient algebra, nonnegative decompositions and unresolved kernel |
| 16 | Upper/lower energy bounds mixed; Q condition misidentified | Dual lower bound plus physical-state upper witness |
| 17 | “All molecules” undefined; deduplication treated as completeness | Explicit graph grammar and coverage proof |
| 18 | SCE Coulomb transport labeled exact exchange | Correct positive interaction functional and primal/dual bounds |
| 19 | Persistence, positivity and multistability confused | Compatibility-class roots and theorem-scoped dynamical claims |
| 20 | Ring-polymer temperature and energy estimators inconsistent | One density convention and a separated error budget |
| 21 | Wrong hypergraph-product size; instance evidence treated as asymptotic | Exact GF(2) construction, distance witnesses and scoped noise model |
| 22 | NPA convergence target misstated; facet rank/count wrong | Quantum value bracket and commuting-operator distinction |
| 23 | Finite relaxation feasibility treated as separability | Witness, decomposition, or unresolved |
| 24 | Matching syndrome treated as decoding success; thresholds mixed | Homology test and noise-specific performance |
| 25 | Oracle separation presented as an unrelativized separation | Explicit oracle, output and resource model |
| 26 | Finite small-divisor scan presented as a KAM certificate | Exact frequencies and a quantitative all-hypothesis theorem check |
| 27 | Figure-eight classified as central; rotating-frame stability reduced to Hessian signs | Symmetry-reduced configurations; full dynamical linearization |
| 28 | Incorrect steepness criterion and dimensionless exponential labeled trillions of years | Explicit normal-form remainder and dimensional time bound |
| 29 | maxRAF defined as minimal; structural catalysis equated with viability | Maximal/irreducible/minimum distinctions and separate dynamics |
| 30 | Pair maximization equated with MFE; neighborhood and scale errors | Deterministic finite map and exact mutation graph |

Each individual critique gives source locations, explains why the issue changes the result, and states replacement acceptance criteria. “Blocking” describes an invalid target or inference in the original document, not proof that the scientific field or broader ambition is unworkable.

## Corrections with especially large consequences

A passing necessary-condition test is not an existence proof. This affects modular spectra, bootstrap amplitudes, RDMs, symmetry indicators, entanglement relaxations and candidate geometries. Conversely, failure of a finite inner ansatz cannot exclude a larger physical class. The revised problems specify which implication is justified and what extra witness would close the gap.

Some assumptions directly conflict with established limitations. Exact nontrivial Chern flat bands cannot simultaneously satisfy strict finite-range locality under the applicable no-go theorem; the revised optimization changes its feasible class accordingly. [Chen et al., flat-band locality obstruction](https://arxiv.org/abs/1311.4956).

The original Bell specification equates NPA convergence with the finite-dimensional tensor-product quantum set. The hierarchy’s commuting-measurement interpretation and the distinction demonstrated by MIP*=RE require separate upper bounds and constructive strategies. [Navascués, Pironio and Acín](https://arxiv.org/abs/0803.4290), [Ji et al.](https://arxiv.org/abs/2001.04383).

The chemistry corrections change actual optimization targets. RDM necessary-condition relaxations minimize over an enlarged set, while a physical trial state supplies the other side of the energy bracket. Coulomb optimal transport describes the strongly correlated interaction limit; it is not the exact exchange functional claimed in PRD 18. [Cotar, Friesecke and Klüppelberg](https://arxiv.org/abs/1104.0603).

Boundary global symmetry is not an automatic violation of a bulk no-global-symmetry principle: it can be the boundary manifestation of bulk gauge symmetry. The replacement for PRD 08 therefore tests supplied modular and anomaly data without inventing a general gravity-compatibility classifier. [Harlow and Ooguri](https://arxiv.org/abs/1810.05338).

Several corrections can be checked without specialized literature: two 3×7 Hamming parity-check inputs give 7²+3²=58 hypergraph-product qubits; the Schlögl cubic supplied in the revision has three positive roots and two locally stable ones; exp(10^0.3) is approximately 7.35 and has no unit of years; and 4²⁰ exceeds one trillion. These are useful regression cases because they catch misleading milestones before a long computation begins.

## Portfolio structure and research design

The main editorial change is to distinguish a research program from its first executable problem. A broad ambition such as “classify all topological materials” is replaced by a complete result in a declared Hamiltonian family, while the larger ambition remains an extension. This is a proposed design choice, not a claim that the selected small model is the only important scientific target.

The shared evidence standard replaces the blanket demand for exact arithmetic with a more useful requirement: every claimed result must state its evidence level and error control. Numerical search and heuristics can discover candidates; exact algebra, validated numerics, or a formal proof can verify an appropriate claim. Rewriting an SDP solver or CAS from scratch is optional and should not consume the scientific project unless independent solver development is itself the goal.

Milestones no longer promise novel theorems, comprehensive classifications or publication within a fixed number of months. Baselines reproduce a reference result. Strong milestones extend coverage or certification. Research extensions name the open contribution without guaranteeing it. Concrete tolerances and budgets are fixed before an implementation run, rather than presented here as arbitrary universal scientific constants.

The topics also have useful boundaries: 02 derives positivity sum rules while 06 studies amplitude feasibility; 04 treats scalar large-spin analysis while 07 treats spinning crossing; 09 proves model topology while 15 diagnoses symmetry data; 19 studies reaction dynamics while 29 studies catalytic closure; 21 studies general sparse codes while 24 studies geometric codes; 26 concerns invariant tori while 28 concerns finite-time action confinement.

For a first implementation pilot, the revised small-instance tasks in 17, 19, 21, 23, 24 and 30 offer explicit reference cases and relatively direct failure controls. This is an editorial assessment of verifier tractability, not a forecast of publication value. The gravitational IR/celestial tasks and quantitative KAM/Nekhoroshev tasks have theorem-derivation prerequisites that should be completed before large searches are funded.

## Repository issues resolved or recorded

- The active catalog now has 30 canonical descriptions and two explicit redirect files, rather than two ambiguous duplicate IDs.
- A crosswalk records that the original synthesis is a different portfolio and preserves its distinct subjects.
- The landing page no longer claims “production-ready” science, completed proofs, or universal 6–12 month delivery.
- The original README advertises MIT licensing and points to a LICENSE file, but no such file is present in the reviewed commit. No license text has been invented or added; the repository owner should resolve that separately.
- Historical generation scripts can recreate old content and should not be used to regenerate the new specifications.
- Original source hashes and a validation report make coverage and preservation auditable.

## Limits and next use

This delivery is a scientific-specification revision, not a solution of the problems or a validation of the archived executable sketches. Primary sources were checked selectively for material corrections; no comprehensive current novelty search is claimed. Before pursuing a research extension, compare its exact proposed theorem and assumptions with the current literature and choose a domain expert review appropriate to the subject.

Use the rewritten PRD as the implementation contract and its critique as the rationale. Use the archive to trace what changed, not as a second conflicting specification.

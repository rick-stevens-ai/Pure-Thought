# Evidence and execution standard

This standard applies to the revised specifications. It is a proposed contract for future research runs; this editorial revision does not supply the scientific certificates themselves.

## What “pure thought” means here

Use explicitly stated mathematical models and deductions without fitting conclusions to experimental outcomes. Synthetic examples, exact model inputs, analytic references, numerical search, and established software are allowed. Physical constants, electronic-structure approximations, constitutive laws, and empirical parameter sets must be identified when used. A model-derived result is conditional on that model.

The original ban on legacy software conflicts with its proposed reliance on NumPy, SciPy, CVXPY, nauty, and other libraries. Reimplementing every optimizer and algebra system is not necessary for scientific independence. Prefer a small independently checkable certificate and a documented trusted base. A fresh implementation can be a project objective, but must not be confused with a mathematical proof or novelty claim.

## Four evidence levels

| Label | What it establishes | What it does not establish |
|---|---|---|
| Numerical evidence | A reproducible computation at stated precision, sample size, and discretization | Exact truth, continuum coverage, or a universal theorem |
| Finite-model certificate | An exact/validated statement about the specified finite model or finite relaxation | An omitted tail, thermodynamic limit, or physical realization |
| Conditional mathematical theorem | A proof over the stated domain with every hypothesis and error controlled | Truth outside those assumptions or a demonstrated physical system |
| Formal proof | A checked theorem in a named proof assistant, with imported axioms/dependencies listed | Correct physical modeling merely by virtue of formalization |

Formalization is an optional extra deliverable unless the specific project makes it mandatory. High precision, JSON output, solver status and small residuals do not by themselves confer any of these proof levels.

## Claim directions

For minimization over a larger feasible set, the optimum is a lower bound on the original minimum. A physical feasible point supplies an upper bound. For maximization the directions reverse. Track feasibility of the primal/dual witnesses separately from their objective values.

Infeasibility of an inner ansatz excludes only that ansatz. Infeasibility of an outer relaxation can exclude the original problem if the map into the relaxation is proved and all approximations preserve necessity. Passing necessary tests means “not excluded”; existence needs a constructive sufficient argument.

A smallest observed example is not a minimality theorem. A complete catalog must define the universe, equivalence relation, boundaries, and proof that pruning misses nothing. A finite-size numerical trend does not prove an asymptotic limit.

## Minimum run manifest

Before calculation, record the problem ID and revision hash, exact inputs and conventions, quantified domain, equivalence relation, target observable, accepted evidence level, tolerances, and resource limits. Record dependencies and precision/rounding mode. Where randomization is used, preserve seeds, trial counts and raw sufficient statistics. Resource limits are planning decisions, not evidence that a mathematical solution cannot exist.

An output bundle should include:

- The statement being claimed, including all assumptions and quantifiers.
- Construction or proof object, plus an independently runnable verifier and its trust assumptions.
- Numerical/error ledger separating roundoff, discretization, truncation, optimization, sampling and modeling error where relevant.
- Positive reference cases and negative controls that the verifier must reject.
- Coverage: included domain, omitted domain, unresolved regions and termination reason.

These are requirements for future implementations, not a promise of polynomial-size certificates for every problem.

## Checking specific certificate types

**LP/SDP/SOS:** verify equality constraints, objective direction, and cone feasibility using exact arithmetic or outward-rounded intervals. Rationalizing a numerical solution can break PSD constraints. Correct residuals with justified norm bounds; do not silently project and reuse the old objective. Distinguish a proof of a polynomial approximation from a proof for the original function.

**Interval numerics:** state function enclosures, domains and outward rounding. A small root box needs existence and, if claimed, uniqueness. Cover the complement for a global enumeration. Include singularities and tails rather than discarding them numerically.

**SAT/SMT:** a solver’s UNSAT label is not a certificate. Preserve a proof supported by a checker and the proof that the encoding, symmetry reduction and blocked solutions match the mathematical universe. DRAT applies to an appropriate Boolean CNF proof, not automatically to every real-arithmetic or SDP claim.

**Statistics:** specify the estimand, dependence assumptions, confidence procedure and bias. Zero observed failures does not prove zero failure probability. MCMC error bars based on estimated autocorrelation are not generally rigorous mixing guarantees.

## Milestones and novelty

Baseline means a working, independently validated result on a concrete reference problem. Strong means a broader or sharper result with its claimed evidence level. Research extension means a possible novel result whose feasibility and significance need review. A negative theorem can be valuable; an unsuccessful search can also be useful if its scope is documented, but the two are not interchangeable.

Do not promise “publication-quality” in a fixed number of months. Before claiming novelty, conduct a focused current literature comparison against the exact theorem/model/metric. The sources in this revision check key definitions and limitations; they are not an exhaustive novelty audit across all 30 fields.

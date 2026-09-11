# Challenge 02: Gravitational Positivity & Causality Bounds

**Domain:** Quantum Gravity & Particle Physics
**Difficulty:** High
**Timeline:** 3-9 months
**Prerequisites:** Scattering amplitudes, dispersion relations, convex optimization

---

## Problem Statement

### Scientific Context
Effective field theories (EFTs) with gravity cannot be arbitrary. Quantum consistency, unitarity, causality, and analyticity impose stringent constraints on the coefficients of higher-derivative terms like R², R³ in the gravitational action. These constraints distinguish theories that can be UV-completed into consistent quantum gravity from those in the "swampland."

### The Core Question
**What are the sharp bounds on Wilson coefficients of higher-derivative gravitational operators imposed purely by consistency conditions?**

For example, for corrections to Einstein gravity:
```
S = ∫ d⁴x √g [M_pl² R + a₁ R² + a₂ R_μν R^μν + a₃ R_μνρσ R^μνρσ + ...]
```

What ranges of {a₁, a₂, a₃, ...} are consistent with:
- Unitarity (positive-norm states)
- Causality (no superluminal propagation)
- Analyticity (dispersion relations)
- Crossing symmetry

### Why This Matters
- **Swampland program:** Determines which low-energy theories can couple to gravity
- **Phenomenology:** Constrains quantum gravity corrections to GR
- **Rigorous:** Produces mathematical

 no-go theorems, not heuristics

---

## Mathematical Formulation

### Problem Definition

Consider graviton-graviton scattering amplitude M(s,t) where s,t are Mandelstam variables.

**Constraints from physics:**

1. **Unitarity:** Im M(s,t) ≥ 0 in physical region
2. **Analyticity:** M(s,t) is analytic except on physical cuts
3. **Crossing:** M(s,t) = M(t,s) = M(u,t) where u = -s-t
4. **Causality (shockwave):** Time delay δt ≥ 0 for shockwave scattering
   - Translates to: certain combinations of Wilson coefficients must be positive

5. **Dispersion relation:** For fixed t < 0,
   ```
   M(s,t) = M(0,t) + s²/π ∫_{s_th}^∞ ds' Im M(s',t)/(s'²(s'-s))
   ```

**Optimization formulation:**

Given a set of Wilson coefficients {a₁, a₂, ..., aₙ}, determine feasibility:

```
Minimize/Maximize: aᵢ
Subject to:
  - Dispersion relation holds
  - Im M(s,t) ≥ 0 (unitarity)
  - Crossing symmetry  satisfied
  - Causality bounds satisfied
  - Regge bound: M(s,t) ~ s² at large s
```

This is a **Sum-of-Squares (SoS) or Semidefinite Program (SDP)** that can be solved with certificates.

### Certificate of Correctness

**Allowed region certificate:**
- Explicit Wilson coefficients {a₁*, a₂*, ...}
- Explicit scattering amplitude M(s,t) satisfying all constraints
- Verification: compute amplitude, check unitarity numerically

**Forbidden region certificate:**
- Dual SoS certificate: polynomial p(s,t) such that:
  - p(s,t) is positive on physical region
  - ∫ p(s,t) [violated constraint] ds dt < 0
  - This proves the region is impossible

---

## Implementation Approach

### Phase 1: 2→2 Graviton Scattering at Tree Level (Month 1-2)

**Build amplitude calculator:**

1. **Einstein gravity amplitude**
   ```python
   def einstein_amplitude(s, t, M_pl):
       """
       Pure GR amplitude for graviton-graviton → graviton-graviton
       """
       u = -s - t
       return (s*t*u) / M_pl**2  # Schematic
   ```

2. **Higher-derivative corrections**
   ```python
   def corrected_amplitude(s, t, M_pl, wilson_coeffs):
       """
       Amplitude including R², R³, ... corrections
       """
       M_0 = einstein_amplitude(s, t, M_pl)
       M_R2 = wilson_coeffs['a1'] * (s**2 + t**2 + u**2)
       M_R3 = wilson_coeffs['a2'] * (s**3 + t**3 + u**3)
       return M_0 + M_R2 / M_pl**4 + M_R3 / M_pl**6 + ...
   ```

3. **Verify crossing symmetry**
   ```python
   assert abs(M(s,t) - M(t,s)) < 1e-10
   assert abs(M(s,t) - M(u,t)) < 1e-10
   ```

### Phase 2: Dispersion Relations (Month 2-3)

**Implement forward dispersion relation:**

```python
def dispersion_relation(M, s, t, s_min, s_max):
    """
    Check if M(s,t) satisfies dispersion relation

    M(s,t) = poly(s,t) + s^N/π ∫ ds' Im M(s',t)/(s'^N(s'-s))
    """
    # Subtracted dispersion relation
    lhs = M(s, t)

    # Polynomial subtraction
    poly_part = sum(c_n * s**n for n, c_n in subtractions)

    # Dispersive integral
    def integrand(s_prime):
        return imag(M(s_prime + 1j*epsilon, t)) / (s_prime**N * (s_prime - s))

    dispersive_part = s**N / np.pi * quad(integrand, s_min, s_max)[0]

    rhs = poly_part + dispersive_part

    return abs(lhs - rhs)  # Should be ~ 0
```

### Phase 3: Causality from Shockwave Scattering (Month 3-4)

**Eikonal phase shift:**

The shockwave time delay is related to the eikonal phase:
```
δ(b) = (1/2s) ∫ d²q/(2π)² e^{iq·b} M(s, t=-q²)
```

**Causality:** δ(b) ≥ 0 for all impact parameters b

This translates to positivity constraints on M(s,t) at fixed t.

### Phase 4: SDP Formulation & Solver (Month 4-6)

**Set up SDP:**

```python
import cvxpy as cp

# Variables: Wilson coefficients
a = cp.Variable(n_coeffs)

# Constraints
constraints = []

# 1. Unitarity: Im M ≥ 0
for s_i, t_i in grid_points:
    Im_M = imaginary_part_amplitude(s_i, t_i, a)
    constraints.append(Im_M >= 0)

# 2. Causality: shockwave positivity
for b_i in impact_parameters:
    delta = eikonal_phase(b_i, a)
    constraints.append(delta >= 0)

# 3. Dispersion relation (approximate as polynomial constraints)
# ... implement as sum-of-squares

# Objective: maximize/minimize specific coefficient
objective = cp.Maximize(a[0])  # e.g., bound on a₁

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK, verbose=True)

# Extract dual certificate
dual_cert = constraints[i].dual_value
```

### Phase 5: Exact Certificates (Month 6-9)

**Generate SoS certificates:**

Use Sum-of-Squares decomposition to prove bounds:
```python
from sympy import symbols, expand
from sympy.polys.polytools import Poly

s, t = symbols('s t', real=True)

# Claim: a₁ ≥ a₁_min
# Proof: Show that (a₁ - a₁_min) can be written as SoS

def find_sos_certificate(constraint_poly, variables):
    """
    Find polynomials {p_i} such that:
    constraint_poly = Σ p_i² + (physical constraints)
    """
    # Use SDP to find {p_i}
    # Return explicit polynomials
    pass
```

---

## Example Starting Prompt

```
I need you to derive and implement bounds on gravitational Wilson coefficients
from causality and unitarity.

GOAL: Bound the coefficient a₁ of the R² term in the gravitational action.

PHASE 1 - Set up the amplitude:
1. Write down the tree-level graviton-graviton scattering amplitude
   in Einstein gravity (just GR).

2. Add the R² correction term and write the corrected amplitude M(s,t; a₁).

3. Verify crossing symmetry: M(s,t) = M(t,s) = M(u,t).

PHASE 2 - Implement dispersion relation:
4. Write the forward dispersion relation (t=0) for graviton scattering.

5. Express the dispersive integral in terms of Im M(s, 0).

6. Use unitarity: Im M(s,0) ≥ 0 for s > threshold.

PHASE 3 - Causality constraint:
7. Implement the eikonal phase shift δ(b) from the amplitude.

8. Impose δ(b) ≥ 0 for all impact parameters b.

9. Translate this into a constraint on a₁.

PHASE 4 - Solve the optimization:
10. Formulate as SDP: maximize a₁ subject to all constraints.

11. Also minimize a₁ to get the allowed range.

12. Extract dual certificate from the SDP solver.

PHASE 5 - Verify:
13. Check that the certificate is a valid SoS decomposition.

14. Export certificate in machine-readable format.

Please implement this step-by-step, verifying each constraint independently
before combining them.
```

---

## Success Criteria

### Minimum Viable Result (3 months)

✅ **Single coefficient bound:**
- Rigorous bounds on a₁ (R² coefficient)
- Both upper and lower bounds
- Dual certificate extracted and verified

✅ **Validation:**
- Reproduce known bounds from literature (if any)
- Independent verification of certificate

### Strong Result (6 months)

✅ **Multi-parameter bounds:**
- Simultaneous bounds on {a₁, a₂, a₃}
- Allowed region in 3D parameter space
- Boundary of region certified with SoS

✅ **Novel bounds:**
- Tighter than previous literature, or
- New multi-coupling constraints
- Machine-checkable certificates

### Publication-Quality Result (9 months)

✅ **Complete EFT space:**
- All Wilson coefficients up to dimension 8 operators
- Full allowed region characterized
- Phase diagram of allowed vs. forbidden regions

✅ **Formal proofs:**
- Certificates formalized in Lean/Isabelle
- Automated theorem proving for no-go regions
- Published with proof repository

---

## Verification Protocol

### Automated Checks

```python
def verify_bound_certificate(coeff_bounds, dual_certificate):
    """
    Verify that the dual certificate proves the bound.
    """
    # 1. Check SoS decomposition
    sos_polynomials = extract_sos_from_certificate(dual_certificate)
    reconstructed = sum(p**2 for p in sos_polynomials)
    assert is_equivalent(reconstructed, constraint_polynomial)

    # 2. Verify positivity on physical region
    test_points = generate_physical_region_samples(n=1000)
    for s, t in test_points:
        assert eval_certificate(dual_certificate, s, t) >= -1e-10

    # 3. Check bound is saturated correctly
    critical_amplitude = construct_amplitude_from_certificate(dual_certificate)
    verify_unitarity(critical_amplitude)
    verify_causality(critical_amplitude)
    verify_crossing(critical_amplitude)

    return "VERIFIED"
```

### Exported Artifacts

1. **Bound certificate:** `bound_a1.sos`
   - Sum-of-Squares decomposition
   - Exact rational coefficients
   - Verifiable by independent SoS checkers

2. **Allowed region:** `allowed_region.json`
   ```json
   {
     "coefficients": ["a1", "a2", "a3"],
     "bounds": {
       "a1": {"min": -0.5, "max": 0.5},
       "a2": {"min": 0.0, "max": 1.0}
     },
     "constraints": ["unitarity", "causality", "crossing"]
   }
   ```

3. **Proof script:** `gravitational_bounds.lean`

---

## Milestone Checklist

- [ ] Tree-level amplitude calculator implemented
- [ ] Crossing symmetry verified numerically
- [ ] Forward dispersion relation implemented
- [ ] Unitarity bounds imposed
- [ ] Eikonal/shockwave causality constraints derived
- [ ] SDP solver setup with certificate extraction
- [ ] First single-coefficient bound obtained
- [ ] Dual certificate verified independently
- [ ] Multi-parameter bounds computed
- [ ] Formal verification initiated
- [ ] Publication draft ready

---

**Next Steps:** Start with implementing the tree-level amplitude and verify crossing symmetry before adding any constraints.

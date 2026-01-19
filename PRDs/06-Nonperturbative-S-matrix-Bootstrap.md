# Challenge 06: Non-perturbative S-matrix Bootstrap with Gravity

**Domain:** Quantum Gravity & Particle Physics
**Difficulty:** High
**Timeline:** 6-12 months
**Prerequisites:** S-matrix theory, dispersion relations, partial-wave unitarity

---

## Problem Statement

### Scientific Context
The S-matrix bootstrap program uses only fundamental axioms—unitarity, crossing symmetry, and analyticity—to constrain scattering amplitudes non-perturbatively. Unlike perturbative calculations, this approach makes no assumption about weak coupling and can produce rigorous bounds on effective field theory (EFT) parameters.

When gravity is included (massless graviton exchange), additional constraints emerge:
- Weinberg's soft graviton theorem
- Regge boundedness (polynomial growth at high energies)
- Causality from graviton propagation

### The Core Question
**What are the allowed regions for EFT Wilson coefficients when gravity is coupled to matter, derived purely from S-matrix axioms?**

For example, consider scalar-graviton scattering with effective Lagrangian:
```
L = L_GR + (1/2)(∂φ)² - (1/2)m²φ² + Σᵢ cᵢ Oᵢ/Λⁱ
```

What ranges of {c₁, c₂, c₃, ...} are consistent with:
- Unitarity
- Crossing symmetry
- Analyticity (dispersion relations)
- Weinberg soft graviton theorem
- Regge bounds

### Why This Matters
- **Non-perturbative rigor:** Valid beyond weak coupling
- **Model-independent:** Applies to any QFT + gravity
- **Swampland program:** Identifies which EFTs can UV-complete with gravity
- **Complementary to positivity:** Different systematic from EFT positivity bounds

---

## Mathematical Formulation

### Problem Definition

Consider 2→2 scattering: φ(p₁) + φ(p₂) → φ(p₃) + φ(p₄) mediated by graviton exchange.

**Mandelstam variables:**
```
s = (p₁ + p₂)²
t = (p₁ - p₃)²
u = (p₁ - p₄)²
s + t + u = 4m²
```

**Amplitude:** A(s,t)

**Partial-wave expansion:**
```
A(s,t) = 16π Σ_{J=0}^∞ (2J+1) a_J(s) P_J(cos θ)
```

where cos θ = 1 + 2t/(s-4m²)

### Constraints from Physics

**1. Unitarity:**

For elastic scattering (s < first inelastic threshold):
```
Im a_J(s) = ρ(s) |a_J(s)|²
```

where ρ(s) = √(1 - 4m²/s) is phase space.

This implies:
```
|a_J(s)| ≤ 1/ρ(s)   (unitarity bound)
```

**2. Crossing Symmetry:**
```
A(s,t) = A(t,s) = A(u,t)
```

**3. Analyticity & Dispersion Relations:**

Roy-Steiner equations (fixed-t dispersion relation):
```
Re a_J(s) = polynomial(s) + (s-s₀)/π ∫_{4m²}^∞ ds' Im a_J(s')/(s'-s₀) / ((s'-s)(s'-s))
```

**4. Weinberg Soft Graviton Theorem:**

As graviton momentum q → 0:
```
M(..., q^μ, ε^{μν}) → Σᵢ (εμν p^μ_i p^ν_i)/(p_i · q) M_n(...without graviton)
```

This constrains residues at t=0, u=0 poles.

**5. Regge Bound:**

For fixed t, s → ∞:
```
|A(s,t)| ≤ C s²   (gravity Regge bound)
```

This bounds the number of subtractions in dispersion relations.

### Optimization Formulation

**Feasibility problem:**

Given Wilson coefficients {c₁, ..., c_n}, determine if there exist partial waves {a_J(s)} satisfying all constraints.

**Bounding problem:**

```
Maximize/Minimize: c_i
Subject to:
  - Unitarity: |a_J(s)| ≤ 1/ρ(s)
  - Crossing: Σ_J (2J+1) a_J(s) P_J(z) = Σ_J (2J+1) a_J(t) P_J(z')
  - Dispersion relations hold
  - Soft theorem residues correct
  - Regge bound satisfied
```

This is a **semi-infinite linear program** (infinite-dimensional due to continuum s, finite J).

### Certificate of Correctness

**If coefficients are allowed:**
- Explicit partial waves {a_J(s)} satisfying all constraints
- Verification: check unitarity bound pointwise
- Verification: evaluate crossing equation on grid
- Verification: verify dispersion integral converges

**If coefficients are forbidden:**
- **Dual certificate:** A functional α(s, J) such that:
  ```
  α applied to (constraints) gives contradiction
  α ≥ 0 on physical region
  ```
- This proves mathematically that no consistent S-matrix exists

---

## Implementation Approach

### Phase 1: Single-Channel Scalar-Graviton (Months 1-3)

**Build partial-wave infrastructure:**

```python
import numpy as np
import scipy.special as sp
from mpmath import mp

mp.dps = 100  # High precision

def legendre_polynomial(J, z):
    """Compute P_J(z)"""
    return sp.eval_legendre(J, z)

def partial_wave_projection(amplitude_func, J, s_val):
    """
    Project amplitude onto J-th partial wave

    a_J(s) = (1/2) ∫₋₁¹ dz P_J(z) A(s, t(z))
    """
    def integrand(z):
        t = compute_t_from_z(s_val, z)
        return legendre_polynomial(J, z) * amplitude_func(s_val, t)

    result, error = mp.quad(integrand, [-1, 1])
    return result / 2

def compute_t_from_z(s, z):
    """
    t = (s - 4m²)(z-1)/2
    """
    return (s - 4*m**2) * (z - 1) / 2
```

**Tree-level amplitude (Einstein gravity + scalar):**

```python
def tree_amplitude_scalar_graviton(s, t, m, M_pl):
    """
    Leading order: gravitational attraction between scalars

    A ~ G_N s t u / M_pl²
    """
    u = 4*m**2 - s - t
    return (8 * np.pi / M_pl**2) * s * t * u / (s * t * u)  # Simplified

def tree_amplitude_with_corrections(s, t, m, M_pl, wilson_coeffs):
    """
    Include higher-derivative corrections

    A = A_tree + c₁ s²/Λ² + c₂ t²/Λ² + ...
    """
    A_tree = tree_amplitude_scalar_graviton(s, t, m, M_pl)

    # Higher-derivative corrections
    corrections = 0
    corrections += wilson_coeffs['c1'] * s**2 / wilson_coeffs['Lambda']**2
    corrections += wilson_coeffs['c2'] * t**2 / wilson_coeffs['Lambda']**2
    corrections += wilson_coeffs['c3'] * s * t / wilson_coeffs['Lambda']**2

    return A_tree + corrections
```

### Phase 2: Dispersion Relations (Months 3-5)

**Implement Roy equations:**

```python
def dispersion_kernel(s, s_prime, s0, J):
    """
    Kernel for partial-wave dispersion relation

    K(s, s') such that:
    Re a_J(s) = poly + ∫ K(s,s') Im a_J(s') ds'
    """
    # Fixed-t dispersion relation kernel
    return (s - s0) / ((s_prime - s0) * (s_prime - s))

def roy_equation(a_J_real, a_J_imag, s_grid, J):
    """
    Self-consistency equation for partial waves

    Re a_J(s) must match dispersive integral of Im a_J(s')
    """
    s0 = 4 * m**2  # Threshold

    for s in s_grid:
        # Left-hand side: input real part
        lhs = a_J_real(s)

        # Right-hand side: dispersive integral
        def integrand(s_prime):
            K = dispersion_kernel(s, s_prime, s0, J)
            return K * a_J_imag(s_prime)

        rhs_integral = np.trapz([integrand(sp) for sp in s_grid], s_grid) / np.pi
        rhs_poly = polynomial_subtraction(s, J)  # Subtraction polynomial
        rhs = rhs_poly + rhs_integral

        # Verify self-consistency
        if abs(lhs - rhs) > 1e-6:
            return False, s, lhs, rhs

    return True, None, None, None
```

**Unitarity relation:**

```python
def unitarity_constraint(a_J, s, m):
    """
    Below inelastic threshold:
    Im a_J(s) = ρ(s) |a_J(s)|²

    where ρ(s) = √(1 - 4m²/s)
    """
    rho = np.sqrt(1 - 4*m**2 / s)

    # Elastic unitarity
    Im_aJ_expected = rho * abs(a_J)**2
    Im_aJ_actual = np.imag(a_J)

    return np.isclose(Im_aJ_actual, Im_aJ_expected, rtol=1e-8)
```

### Phase 3: Crossing Symmetry (Months 5-7)

**Implement crossing equations:**

```python
def crossing_equation(partial_waves, s, t, J_max):
    """
    Crossing: A(s,t) = A(t,s) = A(u,t)

    Σ_J (2J+1) a_J(s) P_J(z_s) = Σ_J (2J+1) a_J(t) P_J(z_t)
    """
    # Compute scattering angle for s-channel
    z_s = compute_scattering_angle(s, t)
    # Compute scattering angle for t-channel
    z_t = compute_scattering_angle(t, s)

    # s-channel sum
    A_s = sum((2*J+1) * partial_waves['s'][J](s) * legendre_polynomial(J, z_s)
              for J in range(J_max))

    # t-channel sum
    A_t = sum((2*J+1) * partial_waves['t'][J](t) * legendre_polynomial(J, z_t)
              for J in range(J_max))

    return abs(A_s - A_t) < 1e-8
```

### Phase 4: Soft Theorem Constraints (Months 7-8)

**Weinberg soft graviton:**

```python
def soft_graviton_residue(amplitude, m):
    """
    Extract residue at t=0 (soft graviton exchange)

    A(s,t) ~ R_soft/t as t → 0

    Weinberg: R_soft = specific function of s, m
    """
    # Compute expected soft factor
    def weinberg_soft_factor(s, m):
        # Universal gravitational coupling
        return 8 * np.pi / M_pl**2 * (s - 2*m**2)

    expected_residue = weinberg_soft_factor(s, m)

    # Extract actual residue from amplitude
    t_small = 1e-6
    actual_residue = amplitude(s, t_small) * t_small

    # Verify match
    assert np.isclose(actual_residue, expected_residue, rtol=1e-6), \
        f"Soft theorem violated: {actual_residue} vs {expected_residue}"
```

### Phase 5: Optimization via Linear/Semidefinite Programming (Months 8-11)

**Formulate as optimization:**

```python
import cvxpy as cp

def setup_smatrix_bootstrap(s_grid, J_max, wilson_bounds=None):
    """
    Set up optimization problem to bound Wilson coefficients
    """
    # Discretize: partial waves at grid points
    # Variables: a_J[s_i] for J=0,...,J_max and s_i in s_grid

    num_s_points = len(s_grid)
    a_real = {}
    a_imag = {}

    for J in range(J_max):
        a_real[J] = cp.Variable(num_s_points)
        a_imag[J] = cp.Variable(num_s_points)

    # Variables: Wilson coefficients
    c = cp.Variable(n_wilson_coeffs)

    constraints = []

    # 1. UNITARITY
    for J in range(J_max):
        for i, s in enumerate(s_grid):
            if s < inelastic_threshold:
                rho_s = np.sqrt(1 - 4*m**2/s)
                # |a_J|² ≤ Im a_J / rho
                constraints.append(
                    a_real[J][i]**2 + a_imag[J][i]**2 <= a_imag[J][i] / rho_s
                )

    # 2. CROSSING SYMMETRY
    # Discretize crossing equation at test points
    for s_test, t_test in crossing_test_points:
        s_channel_sum = compute_partial_wave_sum(a_real, a_imag, s_test, t_test, 's')
        t_channel_sum = compute_partial_wave_sum(a_real, a_imag, t_test, s_test, 't')
        constraints.append(s_channel_sum == t_channel_sum)

    # 3. DISPERSION RELATIONS
    for J in range(J_max):
        for i, s in enumerate(s_grid):
            dispersive_integral = compute_dispersive_integral(
                a_imag[J], s, s_grid
            )
            poly_part = subtraction_polynomial(s, J, c)  # Depends on Wilson coeffs
            constraints.append(a_real[J][i] == poly_part + dispersive_integral)

    # 4. SOFT THEOREM
    # Residue at t=0 must match Weinberg
    soft_constraint = extract_t_zero_residue(a_real, a_imag, c)
    weinberg_value = compute_weinberg_residue(s_grid[0], m)
    constraints.append(soft_constraint == weinberg_value)

    # 5. REGGE BOUND
    # At large s: A(s,t) ~ s^2
    # Constrains high partial waves
    for J in range(J_max):
        constraints.append(a_real[J][-1] <= regge_bound(s_grid[-1], J))

    # OBJECTIVE: Maximize c[0] (for example)
    objective = cp.Maximize(c[0])

    problem = cp.Problem(objective, constraints)
    return problem, c, a_real, a_imag
```

**Solve and extract certificate:**

```python
def solve_and_extract_certificate(problem):
    """
    Solve optimization and extract dual certificate if infeasible
    """
    problem.solve(solver=cp.MOSEK, verbose=True)

    if problem.status == 'optimal':
        return {
            'status': 'feasible',
            'wilson_coeffs': c.value,
            'partial_waves': {J: a.value for J, a in a_real.items()}
        }
    elif problem.status == 'infeasible':
        # Extract dual variables (certificate of infeasibility)
        dual_cert = {}
        for i, constraint in enumerate(problem.constraints):
            dual_cert[f'constraint_{i}'] = constraint.dual_value

        return {
            'status': 'infeasible',
            'certificate': dual_cert
        }
```

### Phase 6: Verification & Formal Proofs (Months 11-12)

```python
def verify_smatrix_solution(partial_waves, wilson_coeffs, s_grid):
    """
    Comprehensive verification of solution
    """
    print("Verifying S-matrix bootstrap solution...")

    # 1. Unitarity
    print("  Checking unitarity...")
    for J in range(J_max):
        for s in s_grid:
            assert check_unitarity(partial_waves[J](s), s)

    # 2. Crossing
    print("  Checking crossing symmetry...")
    for s, t in test_points:
        assert check_crossing(partial_waves, s, t)

    # 3. Dispersion relations
    print("  Checking dispersion relations...")
    for J in range(J_max):
        assert check_dispersion_relation(partial_waves[J], s_grid)

    # 4. Soft theorem
    print("  Checking soft graviton theorem...")
    assert check_soft_theorem(partial_waves, m, M_pl)

    # 5. Regge bound
    print("  Checking Regge bound...")
    assert check_regge_bound(partial_waves, s_grid[-1])

    print("All checks passed! Solution verified.")
    return True
```

---

## Example Starting Prompt

```
I need you to implement the S-matrix bootstrap for scalar-graviton scattering
to derive rigorous bounds on EFT Wilson coefficients.

GOAL: Find the allowed range for the Wilson coefficient c₁ in the effective
Lagrangian L = L_EH + (∂φ)² + c₁ φ² R/Λ² using only S-matrix axioms.

PHASE 1 - Build partial-wave machinery:
1. Implement partial-wave expansion: project amplitude A(s,t) onto Legendre
   polynomials to get a_J(s).

2. Write the tree-level amplitude for φφ → φφ with graviton exchange.

3. Add the c₁ correction term and verify dimensional consistency.

PHASE 2 - Implement constraints:
4. Write the unitarity bound: |a_J(s)| ≤ 1/ρ(s) for s < threshold.

5. Implement the Roy dispersion relation:
   Re a_J(s) = polynomial + ∫ K(s,s') Im a_J(s') ds'

6. Verify crossing symmetry numerically for the tree amplitude.

PHASE 3 - Soft theorem:
7. Extract the residue of A(s,t) at t=0 (soft graviton limit).

8. Compute Weinberg's universal soft factor and verify they match.

PHASE 4 - Optimization:
9. Formulate as LP/SDP: find c₁ maximizing/minimizing subject to:
   - Unitarity constraints
   - Dispersion relations
   - Crossing symmetry
   - Soft theorem
   - Regge bound

10. Solve using cvxpy + MOSEK.

PHASE 5 - Extract certificate:
11. If feasible: extract explicit partial waves and verify all constraints.

12. If infeasible: extract dual functional proving impossibility.

Please implement this step-by-step with exact arithmetic where possible
and cross-check against known results in the literature.
```

---

## Success Criteria

### Minimum Viable Result (6 months)

✅ **Infrastructure complete:**
- Partial-wave projection working
- Dispersion relations implemented
- Crossing symmetry verified for tree-level

✅ **First bound obtained:**
- Rigorous bound on one Wilson coefficient
- Dual certificate extracted (if infeasible)
- Independent verification confirms result

### Strong Result (9 months)

✅ **Multi-parameter bounds:**
- Simultaneous constraints on {c₁, c₂, c₃}
- Allowed region in parameter space mapped
- Comparison with EFT positivity bounds

✅ **Multiple channels:**
- Scalar-scalar + graviton
- Scalar-graviton → scalar-graviton
- Consistency across channels verified

### Publication-Quality Result (12 months)

✅ **Comprehensive EFT space:**
- All Wilson coefficients to dimension-8 bounded
- Systematic comparison with swampland criteria
- Identification of universal bounds

✅ **Formal verification:**
- Certificates formalized in Lean/Isabelle
- All proofs machine-checkable
- Publication with proof repository

---

## Verification Protocol

```python
def verify_wilson_bound(c_value, certificate_type, certificate_data):
    """
    Verify claimed Wilson coefficient bound
    """
    if certificate_type == 'feasible':
        # Verify explicit partial waves satisfy all constraints
        partial_waves = certificate_data['partial_waves']

        assert all(check_unitarity(a_J, s) for a_J in partial_waves for s in s_grid)
        assert all(check_crossing(partial_waves, s, t) for s, t in test_points)
        assert check_dispersion_relations(partial_waves, s_grid)
        assert check_soft_theorem(partial_waves)

        return "FEASIBLE VERIFIED"

    elif certificate_type == 'infeasible':
        # Verify dual certificate proves impossibility
        dual_functional = certificate_data['dual']

        # Dual must be positive on allowed region
        assert verify_dual_positivity(dual_functional)

        # Dual applied to constraints gives contradiction
        gap = evaluate_dual_on_constraints(dual_functional)
        assert gap < -1e-10  # Negative gap proves infeasibility

        return "IMPOSSIBILITY PROVEN"
```

---

## Milestone Checklist

- [ ] Partial-wave projection implemented and tested
- [ ] Tree-level amplitude verified against known results
- [ ] Unitarity bounds imposed and checked
- [ ] Roy dispersion relations implemented
- [ ] Crossing symmetry verified numerically
- [ ] Soft graviton theorem constraint added
- [ ] Regge bound implemented
- [ ] LP/SDP solver infrastructure working
- [ ] First Wilson coefficient bound obtained
- [ ] Dual certificate extracted and verified
- [ ] Multi-parameter optimization completed
- [ ] Formal verification initiated
- [ ] Publication draft with certificates

---

**Next Steps:** Start with implementing partial-wave projection for scalar scattering. Verify against known amplitudes before adding gravity. Build robust dispersion relation infrastructure with high-precision arithmetic.

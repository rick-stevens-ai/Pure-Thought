# Pure Thought AI Challenges - PRD Repository

This directory contains Product Requirement Documents (PRDs) for 30 fundamental scientific challenges that can be tackled using pure thought + fresh code only.

## Structure

Each PRD is a comprehensive tutorial containing:
1. **Problem Statement** - Scientific context and core question
2. **Mathematical Formulation** - Precise problem definition with certificates
3. **Implementation Approach** - Phase-by-phase development plan (6+ months)
4. **Example Starting Prompt** - Ready-to-use prompt for AI systems
5. **Success Criteria** - Minimum viable, strong, and publication-quality results
6. **Verification Protocol** - Automated checks and exported artifacts
7. **Resources & Milestones** - References, pitfalls, and checklist

## Challenges by Domain

### Quantum Gravity & Particle Physics (1-8)
- [01 - AdS₃ Modular Bootstrap](01-AdS3-Modular-Bootstrap.md)
- [02 - Gravitational Positivity Bounds](02-Gravitational-Positivity-Bounds.md)
- [03 - Celestial CFT Bootstrap](03-Celestial-CFT-Bootstrap.md)
- [04 - Modular-Lightcone Bootstrap](04-Modular-Lightcone-Bootstrap.md)
- [05 - Positive Geometry for Gravity](05-Positive-Geometry-Gravity.md)
- [06 - Non-perturbative S-matrix Bootstrap](06-Nonperturbative-S-matrix-Bootstrap.md)
- [07 - Extremal Higher-D CFTs with Stress Tensor](07-Extremal-CFTs-Stress-Tensor.md)
- [08 - Swampland via Modularity & Higher-Form Symmetries](08-Swampland-Modularity-Symmetries.md)

### Materials Science (9-15)
- [09 - Topological Band Theory Without Materials Data](09-Topological-Band-Theory.md)
- [10 - Flat Chern Bands with Provable Geometry](10-Flat-Chern-Bands.md)
- [11 - Photonic Topological Crystals from Symmetry](11-Photonic-Topological-Crystals.md)
- [12 - Topological Mechanical Metamaterials](12-Topological-Mechanical-Metamaterials.md)
- [13 - Higher-Order Topological Insulators](13-Higher-Order-Topological-Insulators.md)
- [14 - Topological Semimetals: Weyl and Dirac Points](14-Topological-Semimetals-Weyl-Dirac.md)
- [15 - Topological Quantum Chemistry](15-Topological-Quantum-Chemistry.md)

### Chemistry (16-20)
- [16 - N-Representability and the 2-RDM Method](16-N-Representability-2RDM.md)
- [17 - Isomer Enumeration via Molecular Graph Theory](17-Isomer-Enumeration-Molecular-Graphs.md)
- [18 - Optimal Transport for Molecular Systems](18-Optimal-Transport-Chemistry.md)
- [19 - Chemical Reaction Network Theory](19-Chemical-Reaction-Networks.md)
- [20 - Ab Initio Path Integral Molecular Dynamics](20-Ab-Initio-Path-Integrals.md)

### Quantum Information & Many-Body Theory (21-25)
- [21 - Quantum LDPC Codes](21-Quantum-LDPC-Codes.md)
- [22 - Bell Inequalities and Quantum Nonlocality](22-Bell-Inequalities-Nonlocality.md)
- [23 - Entanglement Measures and Witnesses](23-Entanglement-Measures-Witnesses.md)
- [24 - Topological Quantum Error Correction](24-Topological-Quantum-Error-Correction.md)
- [25 - Quantum Algorithms and Computational Complexity](25-Quantum-Algorithms-Complexity.md)

### Planetary Systems & Celestial Mechanics (26-28)
- [26 - KAM Theory and Planetary Stability](26-KAM-Theory-Planetary-Stability.md)
- [27 - N-Body Problem and Central Configurations](27-N-Body-Central-Configurations.md)
- [28 - Nekhoroshev Stability Theory](28-Nekhoroshev-Stability-Theory.md)

### Biology & Origin of Life (29-30)
- [29 - Chemical Reaction Networks and the Origin of Life](29-Chemical-Reaction-Networks-Origins.md)
- [30 - Genotype-Phenotype Mapping and Evolutionary Landscapes](30-Genotype-Phenotype-Mapping.md)

## Usage

Each PRD is designed to be:
1. **Self-contained** - Can be read independently
2. **Actionable** - Contains specific implementation steps with Python code
3. **Verifiable** - Includes concrete success metrics and certification methods
4. **Automatable** - Can be fed into long-running AI workflows

## Implementation Guidelines

All PRDs follow the **pure thought** paradigm:
- ✅ Use ONLY symbolic mathematics and exact arithmetic
- ✅ Generate machine-checkable certificates
- ✅ Validate against known theoretical results
- ❌ NO experimental data until final verification
- ❌ NO materials databases or DFT (until benchmarking)
- ❌ NO empirical fitting or heuristics

## Code Requirements

- **Precision**: Use `mpmath` with 100+ digit precision, `sympy` for exact arithmetic
- **Certificates**: Export DRAT proofs (SAT), SDP dual certificates, interval arithmetic bounds
- **Validation**: Cross-check against analytical solutions before scaling up
- **Export**: JSON with exact rational/algebraic numbers, HDF5 for numerical arrays

## Success Metrics

Each PRD defines three success levels:

1. **Minimum Viable Result (MVR)** - Within 2-4 months
   - Core algorithm working
   - Validated on simple test cases
   - Basic certificates generated

2. **Strong Result** - Within 6-8 months
   - Scaled to realistic problem sizes
   - Robust against edge cases
   - Comprehensive benchmarking

3. **Publication-Quality Result** - Within 9-12 months
   - Novel results beyond literature
   - Complete database/classification
   - Formal verification (Lean/Isabelle where applicable)

## Status

- ✅ **Completed**: 30/30 PRDs
- 📊 **Total content**: ~13,500 lines of detailed guidance
- 🎯 **Target completion**: All 30 PRDs with ~600 lines each
- 📅 **Last updated**: 2026-01-17

## Citation

If you use these PRDs in your research or projects, please cite:

```
Pure Thought AI Challenges: 30 Fundamental Problems in Mathematical Physics,
Materials Science, Chemistry, Quantum Information, Planetary Dynamics, and Biology.
https://github.com/[your-repo]/pure-thought-challenges (2026)
```

## Contributing

These PRDs are designed for long-running AI systems (Claude, GPT-4, etc.) but can also guide human researchers. Contributions welcome via:
- Bug reports (mathematical errors, unclear specifications)
- Additional test cases and benchmarks
- Formal verification translations (Lean, Isabelle, Coq)
- Implementation examples and jupyter notebooks

## License

All PRDs are released under MIT License for maximum reusability in both academic and commercial AI research.

---

**Happy solving! May your computations be exact and your certificates verifiable. 🚀**

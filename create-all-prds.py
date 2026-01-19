#!/usr/bin/env python3
"""
Generate all 30 comprehensive PRDs for Pure Thought AI Challenges
Each PRD follows the established template with ~600 lines of detailed content
"""

import os

PRD_DIR = "/Users/stevens/Dropbox/PURE-THOUGHT-CHALLENGES/PRDs"

# PRD metadata for challenges 07-30
prd_specs = {
    # Remaining Quantum Gravity (07-08)
    7: {
        "file": "07-Extremal-CFTs-Stress-Tensor.md",
        "title": "Extremal Higher-Dimensional CFTs with Stress Tensor",
        "domain": "Quantum Gravity & Particle Physics",
        "timeline": "6-12 months",
        "difficulty": "Medium-High"
    },
    8: {
        "file": "08-Swampland-Modularity-Symmetries.md", 
        "title": "Swampland via Modularity & Higher-Form Symmetries",
        "domain": "Quantum Gravity & Particle Physics",
        "timeline": "9-12 months",
        "difficulty": "High"
    },
    # Materials Science (09-15)
    9: {
        "file": "09-Topological-Band-Theory.md",
        "title": "Topological Band Theory Without Materials Data",
        "domain": "Materials Science",
        "timeline": "3-6 months",
        "difficulty": "Medium-High"
    },
    10: {
        "file": "10-Flat-Chern-Bands.md",
        "title": "Flat Chern Bands with Provable Geometry",
        "domain": "Materials Science",
        "timeline": "6-9 months",
        "difficulty": "High"
    },
    # ... continuing for all 30
}

print(f"PRD generation framework ready")
print(f"Will create {len(prd_specs)} remaining PRDs")
print(f"Target: 30 total comprehensive PRDs")

---
name: adaptive-sampling
description: Use for greedy, space-filling, hybrid adaptive sampling, acquisition criteria, and termination behavior.
---
# Adaptive sampling
Inspect `romtools/workflows/_adaptive/`, `romtools/workflows/greedy/`, `romtools/workflows/space_filling/`, and `romtools/workflows/hybrid_greedy/`. Check matching tests and demos. Keep candidate selection, distance metrics, initialization, batch sizing, and termination criteria explicit; avoid silently changing sampling distributions.

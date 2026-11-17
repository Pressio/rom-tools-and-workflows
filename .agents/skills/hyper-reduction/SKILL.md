---
name: hyper-reduction
description: Use for DEIM, QDEIM, ECSW, sampling meshes, interpolation, and weighted residual approximations.
---
# Hyper-reduction

## When to use
Use for DEIM, QDEIM, ECSW, sampling meshes, interpolation, and weighted residual approximations.

## Read first
`romtools/hyper_reduction/`; `examples/h2_air_flame_deim/example.py`; `examples/h2_air_flame_qdeim/example.py`; `tests/romtools/hyper_reduction/`.

## Procedure
1. Identify whether the method approximates a nonlinear vector, operator, or weighted residual/Jacobian; choose the correct snapshots.
2. Verify basis orientation, selected row/element indices, rank, conditioning, and reconstruction dimensions.
3. For DEIM/QDEIM check interpolation solvability and deterministic selection; for ECSW check nonnegative weights, tolerances, and mesh/element consistency.
4. Do not assume vector-entry sampling and finite-element integration-point sampling are equivalent.

## Verification
Use exact low-rank synthetic operators, reconstruction error, residual norms, positivity where required, and rank-deficient edge cases. Compare with a non-hyper-reduced reference.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record snapshot shapes, selected indices/elements, basis rank, condition metrics, weight tolerances, and reconstruction error against a baseline.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

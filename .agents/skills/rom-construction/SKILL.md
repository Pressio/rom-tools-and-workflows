---
name: rom-construction
description: Use for POD/SVD, projection, reduced operators, basis/vector-space construction, and ROM correctness.
---
# ROM construction

## When to use
Use for POD/SVD, projection, reduced operators, basis/vector-space construction, and ROM correctness.

## Read first
`romtools/rom/`; `romtools/vector_space/`; `romtools/linalg/`; `examples/distributed_svd/`; `docs/source/demos/basis_construction.rst`.

## Procedure
1. Identify snapshot shape, state dimension, sample axis, basis column convention, and inner product.
2. Distinguish Galerkin, Petrov-Galerkin, and data-driven ROM assumptions; do not interchange residual and state projection.
3. Check orthogonality with the correct weighting and treatment of offsets/centering.
4. For distributed operations, inspect communication and partition semantics before assuming serial NumPy behavior.

## Verification
Test basis rank/orthogonality, projection reconstruction on known subspaces, reduced operator dimensions, and deterministic small-matrix references; test MPI-specific paths only in authorized environments.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record snapshot/basis shapes, centering/weighting, rank criteria, singular values or truncation metric, and reconstruction/projection error.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

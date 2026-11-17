---
name: multifidelity-methods
description: Use for MF-EKI, MF-VI, surrogate coupling, control variates, and high/low-fidelity allocation.
---
# Multifidelity methods

## When to use
Use for MF-EKI, MF-VI, surrogate coupling, control variates, and high/low-fidelity allocation.

## Read first
`romtools/workflows/inverse/mf_eki_drivers.py`; `romtools/workflows/inverse/vi/`; `tests/romtools/workflows/inverse/test_mf_eki_auto_rom.py`; `tests/romtools/workflows/inverse/test_mf_vi_acv.py`; `examples/eki_mf_eki_demo/`.

## Procedure
1. Identify the high-fidelity and low-fidelity models, shared parameter ordering, QoI shape, and paired versus independent sample sets.
2. Inspect the exact estimator: covariance centering, ensemble-size denominator, gain regularization, observation covariance, and control-variate coefficients.
3. Distinguish model evaluations from surrogate training/update costs; include both in efficiency claims.
4. Check when ROM/surrogate updates occur and whether samples and optimizer state are reused. Do not silently alter a reference formulation.

## Verification
Compare with a direct full-fidelity baseline on a small analytic problem; test unequal ensemble sizes, zero/weak correlation, covariance symmetry and finite gains, and regression gold data before numerical changes.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

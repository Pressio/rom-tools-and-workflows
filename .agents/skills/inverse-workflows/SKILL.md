---
name: inverse-workflows
description: Use for EKI, EGO, variational inference, observation operators, and inverse workflow configuration.
---
# Inverse workflows

## When to use
Use for EKI, EGO, variational inference, observation operators, and inverse workflow configuration.

## Read first
`romtools/workflows/inverse/eki_drivers.py`; `romtools/workflows/inverse/ego_drivers.py`; `romtools/workflows/inverse/vi/`; `examples/eki_mf_eki_demo/example.py`; `examples/vi_mf_vi_demo/`; `tests/romtools/workflows/inverse/`.

## Procedure
1. Establish whether the objective is optimization, ensemble-based inversion, or approximate posterior inference; choose a corresponding public driver.
2. Inspect observation vector, noise covariance, parameter transforms/bounds, and QoI ordering and dimensions.
3. EKI updates ensembles via sample cross-covariances; VI optimizes a distributional objective; EGO uses a surrogate acquisition strategy. Do not treat their outputs as interchangeable.
4. For multifidelity inverse work, load `multifidelity-methods` as well. Check dispatch and blocking behavior against the chosen driver, not against sampling defaults.

## Verification
Verify a small analytic inverse problem and noise scaling, covariance shapes, finite updates, and convergence behavior. Numerical algorithm changes require the regression suite under `tests/romtools/workflows/regression/inverse/`; EKI spread is not automatically a calibrated posterior.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

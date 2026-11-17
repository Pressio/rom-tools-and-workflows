---
name: inverse-workflows
description: Use for EKI, EGO, variational inference, and multifidelity inverse problems or algorithm selection.
---
# Inverse workflows
Read `romtools/workflows/inverse/` public exports, the selected driver, and tests under `tests/romtools/workflows/inverse/`. Match the method to the goal: calibration versus posterior approximation, available observations and covariance, model cost, and surrogate availability. Check `examples/eki_mf_eki_demo/` and `examples/vi_mf_vi_demo/` where relevant. Never equate EKI ensemble spread with a validated Bayesian posterior. Numerical changes require regression review under `tests/romtools/workflows/regression/inverse/`.

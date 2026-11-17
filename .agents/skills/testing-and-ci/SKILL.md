---
name: testing-and-ci
description: Use when adding tests, diagnosing CI, changing numerical algorithms, or updating regression baselines.
---
# Testing and CI
Find the nearest tests under `tests/romtools/` and inspect `.github/workflows/` for actual CI commands. Run focused tests first, then affected regression and broader suites. Distinguish unexecuted, running, failed, and passed checks. When numerical changes invalidate gold data, verify the new result independently before updating it; do not relax tolerances merely to pass CI.

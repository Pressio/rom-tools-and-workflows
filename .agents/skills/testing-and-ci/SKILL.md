---
name: testing-and-ci
description: Use for adding tests, debugging GitHub Actions, numerical regression updates, or test coverage.
---
# Testing and CI

## When to use
Use for adding tests, debugging GitHub Actions, numerical regression updates, or test coverage.

## Read first
`tests/romtools/`; `tests/romtools/workflows/regression/inverse/`; `.github/workflows/`; `helper_scripts/validate_agent_skills.py`.

## Procedure
1. Identify nearest unit tests and public behavior contract before changing implementation.
2. Run the smallest relevant pytest target, then affected regressions and the broader suite when feasible.
3. Use deterministic seeds and numerical reference oracles; distinguish API compatibility, numerical accuracy, and workflow integration.
4. When gold data changes, independently justify the new result; never regenerate gold data or loosen tolerances solely to make CI green.
5. Separate local tests from CI and report exact commands and outcomes.

## Verification
Report passed/failed/skipped/not-run accurately, including missing dependencies. For skills changes, run the validator and inspect symlink adapters; actual Codex/Claude discovery remains a separate integration check.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record exact commands, environment/dependency versions when relevant, pass/fail/skip counts, regression comparisons, and unresolved failures.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

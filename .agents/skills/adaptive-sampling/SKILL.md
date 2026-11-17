---
name: adaptive-sampling
description: Use for greedy, space-filling, hybrid adaptive algorithms, acquisition rules, and stopping criteria.
---
# Adaptive sampling

## When to use
Use for greedy, space-filling, hybrid adaptive algorithms, acquisition rules, and stopping criteria.

## Read first
`romtools/workflows/_adaptive/`; `romtools/workflows/greedy/run_greedy.py`; `romtools/workflows/space_filling/`; `romtools/workflows/hybrid_greedy/run_hybrid_greedy.py`; `examples/hybrid_greedy_demo.py`; `tests/romtools/workflows/greedy/test_greedy.py`.

## Procedure
1. Identify whether candidates are discrete or continuously sampled; document the parameter-space metric and normalization.
2. Separate initialization, candidate scoring, selection, evaluation, and stopping logic; inspect shared `_adaptive` utilities before adding duplicated code.
3. Greedy scores model error indicators; space filling prioritizes geometric coverage. Hybrid batches combine policies; do not conflate their termination criteria.
4. Check duplicate prevention, deterministic ties, batch sizes, concurrency, and checkpoint/resume semantics.

## Verification
Use small known candidate sets to verify selected indices and distances; test fixed seeds, overlapping regions, stopping thresholds, and serial-versus-concurrent behavior where supported.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record candidate pool/seed, metric, acquisition values or selected indices, batch decisions, termination reason, and checkpoint paths.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

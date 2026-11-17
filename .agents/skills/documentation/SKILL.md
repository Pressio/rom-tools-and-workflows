---
name: documentation
description: Use for API documentation, Sphinx navigation, demo descriptions, examples, and cross-references.
---
# Documentation

## When to use
Use for API documentation, Sphinx navigation, demo descriptions, examples, and cross-references.

## Read first
`docs/source/`; `docs/source/demos/index.rst`; `docs/source/demos/models.rst`; `examples/`; `.github/workflows/`.

## Procedure
1. Identify the public export and the actual implementation before documenting signatures.
2. Keep API references task-focused and avoid duplicating entire implementation docstrings in navigation pages.
3. Demo introductions should briefly state the method, what the example demonstrates, and where to find runnable code; link rather than repeat derivations.
4. Use valid Sphinx references and check whether generated pages depend on autodoc or notebooks.

## Verification
Build docs when dependencies are available; check warnings, broken references, duplicated toctree entries, and representative rendered demo/API pages. Do not claim links work based only on source syntax.

## Reviewable execution trace
Follow the common protocol in `.agents/TRACE_PROTOCOL.md`. For this task specifically: Record public symbols/pages inspected, cross-reference changes, documentation build command, warnings, and links checked.

## Handoff
State which implementation and tests were inspected, what changed, commands actually run, numerical comparisons, and remaining uncertainty. Prefer loading only this skill plus directly relevant neighboring skills.

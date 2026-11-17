# Reviewable agent execution protocol

Purpose: make agent-assisted ROM Tools work reproducible, auditable, and easy to review without requiring verbose hidden reasoning or expensive logging.

## Default stages (adapt to task size)

1. **Scope and plan (before editing):** record objective, relevant skill(s), public contract, expected files, numerical acceptance criteria, risk, and whether external actions require approval. For multi-file work, make a short numbered checklist. A one-line plan is sufficient for trivial tasks.
2. **Inspect:** identify exact source, tests, examples, and API signatures consulted. Do not assume stale skill descriptions override current code.
3. **Implement:** record significant files changed and concise decision rationales/tradeoffs, especially deviations from the plan. Preserve existing public behavior unless explicitly authorized.
4. **Verify:** execute focused tests and numerical oracles, then broader affected tests as feasible. Record commands, exit status, and numerical tolerances/results. Never mark unrun tests as passing.
5. **Review and handoff:** inspect the diff for compatibility, unexpected side effects, sensitive information, and regression changes. Summarize completion, evidence, limitations, and next actions.

## Trace format

For substantial tasks, maintain a concise Markdown record (e.g. `agent_trace.md` in an explicitly designated output directory, or a PR/issue comment). Do not create an unsolicited permanent trace file in the repository root. The trace should include:

- Task ID / commit / environment and timestamp if available
- Skills requested and evidence of actual discovery/read (not just presence on disk)
- Plan checklist with completed, skipped, and blocked stages
- Source paths inspected and decisions made (brief justifications, not private chain-of-thought)
- Files changed and public API compatibility notes
- Commands executed, exit codes, test counts, and numerical comparisons
- External actions, approvals, job IDs, and outcomes when applicable
- Deviations, failures, unresolved uncertainty, and reviewer checklist

Use tool-native logs as evidence when available, with pointers to their locations. Avoid claiming that a model used a skill solely because the skill was installed. Keep traces concise, factual, and structured; no hidden reasoning or full internal conversation dumps.

## Safety and reproducibility

Never store tokens, credentials, private datasets, or unnecessary raw prompts in traces. Redact machine-specific paths and user identifiers where needed. Never execute remote/HPC jobs without explicit authorization. Record seeds, versions, and configuration hashes when numerical reproducibility depends on them.

## Review gate

Before reporting completion, answer: Does the diff match the plan? Did the API remain compatible? Are numerical results independently checked? Which tests ran? Which steps were not verified? What would a reviewer need to reproduce the result?

# Work environment pilot diagnostic — 2026-11-17

## Plan and scope
Validate the CDR reference, inspect benchmark isolation and CLI compatibility, then execute one full-skills pilot before attempting the model matrix. Preserve public/numerical behavior; CPU only. No HPC jobs submitted.

## Inspection
Read root AGENTS.md, .agents/TRACE_PROTOCOL.md, testing-and-ci skill, benchmark README, run.py, CDR task/prompt, reference tests, independent grading tests, and pyproject.toml.

## Verified results
- Python 3.12.14; Codex CLI 0.167.0-alpha.4.
- `codex login status`: logged in using ChatGPT; no separate API key configured for this experiment.
- `python -m pytest -q benchmarks/agent_skills/reference/tests`: 3 passed. Numerical solution and QoI oracle checks passed.
- `python helper_scripts/validate_agent_skills.py`: 11 canonical skills and Claude adapters validated. This does not demonstrate discovery/use by an executing agent.
- `codex debug models`: refreshed catalog includes gpt-6.2-sol, gpt-6.2-astra and other models. Catalog visibility does not establish successful inference.

## Pilot outcome: infrastructure blocked
Attempted gpt-6.2-sol/full/trial 0. No agent task executed, no generated solution, and no token usage was returned. Do not count the subsequent missing-solution grader failures as model failures.

The original runner exports a workspace without Git history and removes reference/grader files, but does not itself hide the original checkout. An external bubblewrap wrapper was used to hide the original /workspace and /tmp, bind only the exported workspace into /tmp/eval, and provide temporary Codex state with read-only authentication/configuration. Codex workspace-write sandbox remained enabled.

Initial wrapper mount/state errors were corrected. Codex's filesystem-sandbox helper then failed on network namespace setup. Reproduction independent of benchmark/model:

```bash
bwrap --ro-bind / / --proc /proc --dev /dev --unshare-net -- /bin/true
```

Exit 1: `bwrap: loopback: Failed to create NETLINK_ROUTE socket: Operation not permitted`.

Enabling workspace-write network access did not resolve startup. The installed CLI's legacy Landlock alternative also failed: `filesystem-restricted execution requires bubblewrap to isolate app-server sockets`. Neither alternative reached inference. No sandbox bypass was used.

## Remaining work
Run on an execution host supporting this CLI's network namespace and sandbox requirements, or use a supported CLI/runtime combination. Then validate agent execution, actual skill-read evidence, token instrumentation, independent grading, and retained solution artifacts before running comparisons. Isolation of all credentials and connected services has not been established; do not describe this setup as fully credential-isolated.

Runner follow-up: distinguish infrastructure errors from task failures and avoid interpreting missing-solution grader errors after agent startup failure as model quality results. Preserve trial workspaces/diffs for review. No benchmark runner or numerical implementation changes were made in this diagnostic.

## Follow-up implementation
The runner now explicitly requests workspace-write sandboxing, marks unsuccessful agent exits/timeouts as ungraded, skips numerical grading in those cases, supports a single guidance condition, and optionally retains the pre-grading submission. The summary handles null token usage and reports ungraded counts. These changes do not resolve the host restriction or enforce complete read isolation.

`python -m pytest -q benchmarks/agent_skills/tests benchmarks/agent_skills/reference/tests`: **8 passed**, one third-party pytest deprecation warning. Five runner tests cover terminal/missing usage, agent error/timeout/success gating, retained submissions without grader files, and null-usage summaries. Agent execution is mocked in runner tests; the three reference tests perform real numerical checks. No successful model trial or model comparison is claimed.

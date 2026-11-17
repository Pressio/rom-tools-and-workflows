# Repository agent skills

Canonical skills live in `.agents/skills/<name>/SKILL.md`. Root `AGENTS.md` routes tasks to skills. `.agents/TRACE_PROTOCOL.md` defines the common plan → inspect → implement → verify → review evidence trail.

## Design rules
- Progressive disclosure: load only relevant skills; do not preload all eleven.
- Source first: inspect actual signatures and tests.
- Reviewable behavior: follow explicit stages, leave concise evidence of decisions, tests, deviations, and unresolved work.
- Distinguish installed skills from confirmed skill discovery.
- Do not record hidden reasoning, credentials, or full tool transcripts.

## Harness adapters
Codex uses root `AGENTS.md` as a navigation entry point; native skill discovery must be checked in the installed version. Claude uses root `CLAUDE.md` and symlinks in `.claude/skills/`. Keep canonical skills harness-neutral.

## Validation
Run `python helper_scripts/validate_agent_skills.py`; separately check actual harness discovery and trace generation in a pilot agent run.

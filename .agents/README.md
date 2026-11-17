# Repository agent skills

The canonical content is `.agents/skills/<name>/SKILL.md`. Root `AGENTS.md` routes tasks to skills; each skill contains a trigger, source-of-truth paths, a procedure, verification criteria, and a handoff checklist.

## Design rules
- **Progressive disclosure:** keep `AGENTS.md` short and load only relevant skills.
- **Source first:** never duplicate full signatures that may drift; inspect the current code.
- **Actionable guidance:** state decisions, failure modes, numerical checks, and expected artifacts rather than generic advice.
- **Honest evidence:** report tests actually executed and distinguish structural skill presence from runtime discovery.
- **Harness neutrality:** no vendor-specific CLI commands in canonical skills; harness adapters only point to them.

## Adapters
Codex uses root `AGENTS.md` as a navigation entry point; do not assume it automatically discovers `.agents/skills` without explicit verification. Claude Code uses root `CLAUDE.md` and symlinks in `.claude/skills/`. Other harnesses should link, not duplicate, canonical skill content.

## Maintenance
Run `python helper_scripts/validate_agent_skills.py`. When changing an API, update the relevant skill's entry points and numerical checks. Confirm paths and actual harness discovery in a real agent run; static validation alone cannot prove skill use.

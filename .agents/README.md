# Harness-independent agent knowledge

`.agents/skills/<name>/SKILL.md` is the **canonical** skill source. Skills describe tasks and point to authoritative code, tests, and examples. They must not assume a particular agent runtime or vendor-specific commands.

## Adapters
- **Codex:** root `AGENTS.md` and `.agents/skills/` are native discovery locations.
- **Claude Code:** root `CLAUDE.md` refers to `AGENTS.md`; `.claude/skills/<name>` links to the canonical skill directory.
- **Other harnesses:** add a small adapter (instruction entry point and skill-directory links) rather than copying skill content.

Run `python helper_scripts/validate_agent_skills.py` to check the metadata and adapters. Symlink discovery should additionally be tested in the actual harness; structural validation alone cannot establish runtime discovery.

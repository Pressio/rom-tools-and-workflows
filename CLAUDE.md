# Claude Code repository guidance

Read [AGENTS.md](AGENTS.md) for repository conventions and the subsystem map. Treat the current source code and tests as authoritative.

Task-specific skills are exposed under `.claude/skills/` as links to the canonical `.agents/skills/` directories. Load only the relevant skill for the current task; do not load every skill by default.

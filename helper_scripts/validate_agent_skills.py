"""Validate harness-independent skills and Claude adapter without dependencies."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
CLAUDE_SKILLS = ROOT / ".claude" / "skills"


def main():
    errors = []
    if not (ROOT / "AGENTS.md").is_file() or not (ROOT / "CLAUDE.md").is_file():
        errors.append("Missing AGENTS.md or CLAUDE.md")
    names = []
    for skill in sorted(SKILLS.iterdir()):
        if not skill.is_dir():
            continue
        names.append(skill.name)
        doc = skill / "SKILL.md"
        if not doc.is_file():
            errors.append(f"{skill}: missing SKILL.md")
            continue
        content = doc.read_text()
        match = re.match(r"\A---\n(.*?)\n---\n", content, re.S)
        if not match:
            errors.append(f"{doc}: missing YAML frontmatter")
            continue
        header = match.group(1)
        if not re.search(rf"(?m)^name:\s*{re.escape(skill.name)}\s*$", header):
            errors.append(f"{doc}: name must match directory")
        if not re.search(r"(?m)^description:\s*\S+", header):
            errors.append(f"{doc}: missing description")
        adapter = CLAUDE_SKILLS / skill.name
        if not adapter.is_symlink():
            errors.append(f"{adapter}: expected directory symlink")
        elif adapter.resolve() != skill.resolve():
            errors.append(f"{adapter}: wrong target")
    if not names:
        errors.append("No skills found")
    for entry in CLAUDE_SKILLS.iterdir():
        if entry.name not in names:
            errors.append(f"{entry}: unexpected adapter")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Validated {len(names)} canonical skills and Claude adapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Print success and token summaries from a Codex benchmark JSONL."""
import json
from collections import defaultdict
from pathlib import Path
import sys

groups = defaultdict(list)
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        groups[(r["task"], r["model"], r["condition"])].append(r)
for key, rows in sorted(groups.items()):
    successes = [r for r in rows if r["passed"]]
    total_tokens = [r["usage"].get("input_tokens", 0) + r["usage"].get("output_tokens", 0)
                    for r in rows if "input_tokens" in r["usage"] and "output_tokens" in r["usage"]]
    print(*key, f"success={len(successes)}/{len(rows)}",
          f"tokens_per_success={sum(total_tokens)/len(successes):.0f}" if successes and len(total_tokens) == len(rows)
          else "tokens_per_success=NA")

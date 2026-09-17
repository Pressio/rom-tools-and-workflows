#!/usr/bin/env python3
"""Summarize comparable pairs, preserving failures and missing usage."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(paths):
    records = {}
    for path in paths:
        for r in json.loads(path.read_text()):
            if r.get("status") != "prepared":
                key = (r["run_id"], r["trial_id"])
                if key in records and records[key] != r:
                    raise ValueError(f"Conflicting duplicate trial: {key}")
                records[key] = r
    groups = defaultdict(list)
    for r in records.values():
        key = tuple(r[k] for k in ("provider", "model", "effort", "cli_version", "task", "source_sha256", "skill_sha256", "harness_sha256", "delivery", "authorization"))
        key += (json.dumps(r.get("dependencies", {}), sort_keys=True), r.get("platform", "unknown"))
        groups[key].append(r)
    lines = ["# Sampling skill study", "", "Positive savings means fewer total tokens with the skill.", "",
             "| Provider/model/task | Baseline passes | Skill passes | Measured pairs | Median paired savings (range) | Both-pass pairs: median savings |", 
             "|---|---:|---:|---:|---:|---:|"]
    for key, rows in sorted(groups.items()):
        conditions = {c: [r for r in rows if r["condition"] == c] for c in ("baseline", "skill")}
        pairs = defaultdict(dict)
        for r in rows:
            pairs[(r["run_id"], r["repeat"])][r["condition"]] = r
        savings = []
        passing_savings = []
        for pair in pairs.values():
            if all(c in pair and pair[c].get("usage") for c in conditions):
                b, s = (pair[c]["usage"]["total_tokens"] for c in conditions)
                if b:
                    savings.append(100 * (b - s) / b)
                    if all(pair[c].get("successful") for c in conditions):
                        passing_savings.append(savings[-1])
        passes = [f"{sum(bool(r.get('successful')) for r in conditions[c])}/{len(conditions[c])}" for c in conditions]
        delta = f"{statistics.median(savings):.1f}% ({min(savings):.1f}% to {max(savings):.1f}%)" if savings else "unavailable"
        matched = f"{len(passing_savings)}: {statistics.median(passing_savings):.1f}%" if passing_savings else "0: unavailable"
        lines.append(f"| {key[0]} / {key[1]} / {key[4]} | {passes[0]} | {passes[1]} | {len(savings)} | {delta} | {matched} |")
    lines += ["", "Savings include measured failed trials; interpret together with pass rates. Missing usage is never zero.",
              "Groups separate CLI versions, effort, task, source, skill, harness and protocol. Same labels can identify distinct groups.",
              "Audit content, distribution implementation and final-report accuracy require human review; automated passes are partial quality evidence.",
              "Small samples are exploratory. Do not infer equivalent quality from a token reduction alone."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    print(summarize(sorted(args.directory.rglob("results.json"))))

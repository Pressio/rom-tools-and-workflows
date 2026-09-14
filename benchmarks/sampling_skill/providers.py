"""CLI adapters. No shell interpolation; prompts arrive on stdin."""
import json


def command(provider, executable, model, effort):
    if provider == "codex":
        return [executable, "exec", "--json", "--ephemeral", "--ignore-user-config",
                "--ignore-rules", "--skip-git-repo-check", "--sandbox", "workspace-write",
                "--disable", "apps", "--disable", "plugins", "--disable", "hooks",
                "--disable", "multi_agent", "--disable", "memories",
                "--disable", "skill_search", "--enable", "skip_host_skill_discovery",
                "-c", 'web_search="disabled"',
                "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-"]
    if provider == "claude":
        return [executable, "--bare", "-p", "--output-format", "json",
                "--no-session-persistence", "--model", model, "--effort", effort,
                "--allowedTools", "Read,Write,Edit,Glob,Grep,Bash"]
    raise ValueError(provider)


def parse_usage(provider, output):
    """Normalize totals without double counting cached input. Missing stays null."""
    events = []
    for line in output.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    if provider == "claude":
        try:
            events = [json.loads(output)]
        except ValueError:
            pass
    if provider == "codex":
        usages = [e["usage"] for e in events if e.get("type") == "turn.completed" and e.get("usage")]
        if not usages:
            return None
        raw_input = sum(u["input_tokens"] for u in usages)
        output_tokens = sum(u["output_tokens"] for u in usages)
        read = sum(u.get("cached_input_tokens", 0) for u in usages)
        write = sum(u["cache_write_input_tokens"] for u in usages) if all("cache_write_input_tokens" in u for u in usages) else None
        input_tokens = raw_input
    else:
        results = [e for e in events if e.get("type") == "result" and e.get("usage")]
        if not results:
            return None
        u = results[-1]["usage"]
        raw_input, output_tokens = u["input_tokens"], u["output_tokens"]
        read, write = u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0)
        input_tokens = raw_input + read + write
    return dict(input_tokens=input_tokens, output_tokens=output_tokens,
                cache_read_tokens=read, cache_write_tokens=write,
                total_tokens=input_tokens + output_tokens)


def provider_succeeded(provider, output):
    try:
        if provider == "claude":
            result = json.loads(output)
            return result.get("type") == "result" and not result.get("is_error", True)
        events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        return any(e.get("type") == "turn.completed" for e in events) and not any(
            e.get("type") in ("error", "turn.failed") for e in events)
    except ValueError:
        return False

"""Run isolated Codex CLI benchmark cells; no external Python dependencies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path(__file__).resolve().parent / "tasks"


def run_cmd(args, *, cwd, timeout, output=None):
    return subprocess.run(args, cwd=cwd, text=True, stdout=output or subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=False)


def parse_usage(log):
    usage = {}
    for line in Path(log).read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        info = event.get("usage") or event.get("token_usage") or {}
        if event.get("type") == "turn.completed":
            info = event.get("usage", info)
        if isinstance(info, dict):
            for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
                if isinstance(info.get(key), int):
                    usage[key] = info[key]
    return usage


def prepare(work, condition):
    if condition == "none":
        for name in ("AGENTS.md", "CLAUDE.md"):
            (work / name).unlink(missing_ok=True)
        shutil.rmtree(work / ".agents", ignore_errors=True)
        shutil.rmtree(work / ".claude", ignore_errors=True)
    elif condition == "guide":
        shutil.rmtree(work / ".agents", ignore_errors=True)
        shutil.rmtree(work / ".claude", ignore_errors=True)
        (work / "CLAUDE.md").unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="Actual Codex CLI model IDs")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="*", default=["parameter_space"])
    a = p.parse_args()
    if a.trials < 1:
        p.error("--trials must be positive")
    if not shutil.which("codex"):
        p.error("Codex CLI is not installed or on PATH")
    a.output.mkdir(parents=True, exist_ok=True)
    results = a.output / "results.jsonl"
    # Worktrees use the current checkout HEAD. A clean source checkout is required.
    check = run_cmd(["git", "status", "--porcelain"], cwd=ROOT, timeout=30)
    if check.returncode or check.stdout.strip():
        p.error("source checkout must be clean")
    for task in a.tasks:
        task_dir = TASKS / task
        spec = json.loads((task_dir / "task.json").read_text())
        for model in a.models:
            for condition in ("none", "guide", "full"):
                for trial in range(a.trials):
                    with tempfile.TemporaryDirectory(prefix="romtools-skill-eval-") as td:
                        work = Path(td) / "repo"
                        subprocess.run(["git", "worktree", "add", "--detach", str(work), "HEAD"],
                                       cwd=ROOT, check=True, capture_output=True)
                        try:
                            prepare(work, condition)
                            log = a.output / f"{task}-{model.replace('/', '_')}-{condition}-{trial}.jsonl"
                            prompt = (task_dir / "prompt.md").read_text()
                            start = time.monotonic()
                            timed_out = False
                            with log.open("w") as out:
                                try:
                                    cmd = ["codex", "exec", "--json", "--model", model, "--cd", str(work), prompt]
                                    result = run_cmd(cmd, cwd=work, timeout=spec["timeout_seconds"], output=out)
                                    rc = result.returncode
                                    stderr = result.stderr[-2000:]
                                except subprocess.TimeoutExpired:
                                    timed_out, rc, stderr = True, None, "timeout"
                            elapsed = time.monotonic() - start
                            # Verify independently, even if the agent reports success.
                            try:
                                verification = run_cmd(spec["verify"], cwd=work, timeout=120)
                                passed = verification.returncode == 0
                                verify_output = (verification.stdout + verification.stderr)[-2000:]
                            except subprocess.TimeoutExpired:
                                passed, verify_output = False, "verification timeout"
                            record = dict(task=task, model=model, condition=condition,
                                          trial=trial, passed=passed, agent_returncode=rc,
                                          timed_out=timed_out, seconds=elapsed,
                                          usage=parse_usage(log), stderr=stderr,
                                          verifier_output=verify_output)
                            with results.open("a") as out:
                                out.write(json.dumps(record) + "\n")
                            print(task, model, condition, trial, "PASS" if passed else "FAIL")
                        finally:
                            subprocess.run(["git", "worktree", "remove", "--force", str(work)],
                                           cwd=ROOT, check=False, capture_output=True)


if __name__ == "__main__":
    main()

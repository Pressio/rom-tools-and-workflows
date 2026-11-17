"""Codex CLI pilot runner. Run only in an isolated environment without secrets."""
import argparse
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path(__file__).resolve().parent / "tasks"


def parse_usage(path):
    """Read terminal turn usage, not summed intermediate/cumulative events."""
    usage = None
    for line in Path(path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "turn.completed":
            value = event.get("usage")
            if isinstance(value, dict):
                usage = {k: value[k] for k in ("input_tokens", "output_tokens",
                         "cached_input_tokens") if isinstance(value.get(k), int)}
    return usage


def prepare(work, condition):
    if condition in ("none", "guide"):
        shutil.rmtree(work / ".agents", ignore_errors=True)
        shutil.rmtree(work / ".claude", ignore_errors=True)
        (work / "CLAUDE.md").unlink(missing_ok=True)
    if condition == "none":
        (work / "AGENTS.md").unlink(missing_ok=True)


def run_one(model, condition, trial, task, output):
    task_dir = TASKS / task
    spec = json.loads((task_dir / "task.json").read_text())
    with tempfile.TemporaryDirectory(prefix="romtools-eval-") as temp:
        work = Path(temp) / "workspace"
        work.mkdir()
        # Export without .git so the agent cannot inspect withheld tests via history.
        archive = subprocess.Popen(["git", "archive", "--format=tar", "HEAD"],
                                   cwd=ROOT, stdout=subprocess.PIPE)
        extract = subprocess.run(["tar", "-xf", "-", "-C", str(work)],
                                 stdin=archive.stdout, capture_output=True)
        archive.stdout.close()
        archive_rc = archive.wait()
        if archive_rc or extract.returncode:
            raise RuntimeError("Failed to export repository snapshot")
        # The grader is kept in the trusted checkout, never exported to the agent.
        for testdir in (work / "benchmarks/agent_skills/tasks").glob("*/tests"):
            shutil.rmtree(testdir)
        prepare(work, condition)
        prompt = (task_dir / "prompt.md").read_text()
        log = output / f"{task}-{model.replace('/', '_')}-{condition}-{trial}.jsonl"
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "--model", model,
               "--cd", str(work), prompt]
        start = time.monotonic()
        timeout = False
        with log.open("w") as stream:
            try:
                result = subprocess.run(cmd, cwd=work, stdout=stream,
                                        stderr=subprocess.PIPE, text=True,
                                        timeout=spec["timeout_seconds"])
                agent_rc, stderr = result.returncode, result.stderr[-2000:]
            except subprocess.TimeoutExpired:
                timeout, agent_rc, stderr = True, None, "Agent timeout"
        elapsed = time.monotonic() - start
        # Restore trusted tests from the checkout after the agent exits.
        grader = task_dir / "tests"
        if grader.is_dir():
            dest = work / "benchmarks/agent_skills/tasks" / task / "tests"
            shutil.copytree(grader, dest)
        try:
            verification = subprocess.run(spec["verify"], cwd=work, text=True,
                                          capture_output=True, timeout=120)
            passed = verification.returncode == 0
            verify_log = (verification.stdout + verification.stderr)[-3000:]
        except subprocess.TimeoutExpired:
            passed, verify_log = False, "Verifier timeout"
        return dict(task=task, model=model, condition=condition, trial=trial,
                    passed=passed, agent_returncode=agent_rc, timed_out=timeout,
                    seconds=elapsed, usage=parse_usage(log), stderr=stderr,
                    verifier_output=verify_log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", default=["cdr_model_adapter"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if not shutil.which("codex"):
        parser.error("Codex CLI not found")
    if not shutil.which("tar"):
        parser.error("tar not found")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(task, model, condition, trial)
            for task in args.tasks for model in args.models
            for condition in ("none", "guide", "full")
            for trial in range(args.trials)]
    random.Random(args.seed).shuffle(jobs)
    with (args.output / "results.jsonl").open("a") as results:
        for task, model, condition, trial in jobs:
            record = run_one(model, condition, trial, task, args.output)
            results.write(json.dumps(record) + "\n")
            results.flush()
            print(task, model, condition, trial,
                  "PASS" if record["passed"] else "FAIL")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare, run, and export paired skill trials. Python 3.9+, POSIX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from fixtures import COMMON, MODEL, TASKS, WORKFLOW
from providers import command, parse_usage, provider_succeeded

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def execute(argv, cwd, env, prompt="", timeout=600):
    start = time.monotonic()
    with subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as proc:
        try:
            out, err = proc.communicate(prompt, timeout=timeout)
            status = "finished"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            status = "timeout"
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            status = "interrupted"
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
    return dict(stdout=out, stderr=err, exit_code=proc.returncode,
                status=status, seconds=time.monotonic() - start)


def snapshot(root):
    """Allowlist avoids prior generated answers, instructions, and credentials."""
    shutil.copytree(REPO / "romtools", root / "romtools",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "version.txt", "LICENSE"):
        shutil.copy2(REPO / name, root / name)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return digest(b"".join(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() for p in files))


def prepare(root, task, condition, skill):
    root.mkdir()
    source_hash = snapshot(root)
    if task in ("existing", "reuse"):
        (root / "model.py").write_text(MODEL)
    if task == "existing":
        (root / "workflow.py").write_text(WORKFLOW)
    if condition == "skill":
        (root / "workflow_skill").mkdir()
        (root / "workflow_skill/SKILL.md").write_bytes(skill)
    return source_hash


def environment(home, workspace, python, provider):
    # Do not inherit editor/session/MCP settings, PYTHONPATH, or nested agent IDs.
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR",
           "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY") if k in os.environ}
    env.update(HOME=str(home), CODEX_HOME=str(home / ".codex"),
               CLAUDE_CONFIG_DIR=str(home / ".claude"), PYTHONPATH=str(workspace),
               PYTHONNOUSERSITE="1", MPLCONFIGDIR=str(home / "matplotlib"))
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    for key in (("OPENAI_API_KEY",) if provider == "codex" else ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def trial(args, run_dir, task, condition, repeat, skill, executable, version):
    trial_id = f"{task}-{repeat}-{condition}"
    dest = run_dir / trial_id
    dest.mkdir()
    with tempfile.TemporaryDirectory(prefix="sampling-trial-") as tmp:
        base = Path(tmp)
        workspace, home = base / "workspace", base / "home"
        home.mkdir(mode=0o700)
        source_hash = prepare(workspace, task, condition, skill)
        env = environment(home, workspace, args.python, args.provider)
        # Only auth is copied, never config, history, skills, rules, or plugins.
        auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
        if args.provider == "codex" and auth.is_file() and not args.dry_run:
            (home / ".codex").mkdir(mode=0o700)
            shutil.copyfile(auth, home / ".codex/auth.json")
            (home / ".codex/auth.json").chmod(0o600)
        prompt = TASKS[task] + COMMON + f"\nPython executable: {args.python}\n"
        if condition == "skill":
            prompt += "Read workflow_skill/SKILL.md and follow this sampling workflow skill for the task.\n"
        (dest / "prompt.txt").write_text(prompt)
        argv = command(args.provider, executable, args.model, args.effort)
        record = dict(schema_version=1, run_id=run_dir.name, trial_id=trial_id,
                      provider=args.provider, model=args.model, effort=args.effort,
                      cli_version=version, task=task, condition=condition, repeat=repeat,
                      skill_sha256=digest(skill), source_sha256=source_hash,
                      prompt_sha256=digest(prompt.encode()), delivery="explicit-file",
                      authorization="preauthorized-local", python=args.python,
                      platform=platform.platform(), command=argv, timeout=args.timeout,
                      repository_commit=args.repository_commit, dependencies=args.dependencies,
                      harness_sha256=args.harness_hash)
        if args.dry_run:
            record.update(status="prepared", usage=None, quality=None)
        else:
            print(f"Running {trial_id} ...", flush=True)
            result = execute(argv, workspace, env, prompt, args.timeout)
            (dest / "events.jsonl").write_text(result.pop("stdout"))
            (dest / "stderr.txt").write_text(result.pop("stderr"))
            output = (dest / "events.jsonl").read_text()
            record.update(result)
            record["usage"] = parse_usage(args.provider, output)
            record["provider_completed"] = provider_succeeded(args.provider, output)
            validation = execute([args.python, str(HERE / "check_outputs.py"), str(workspace)],
                                 workspace, env, timeout=60)
            try:
                quality = json.loads(validation["stdout"])
            except ValueError:
                quality = {"passed": False, "checks": {}, "validator_error": validation["stderr"]}
            # Evaluation happens after the model exits, outside measured token usage.
            replay_dir = base / "replay"
            shutil.copytree(workspace, replay_dir, ignore=shutil.ignore_patterns("sampling_output", "__pycache__"))
            replay_env = environment(home, replay_dir, args.python, args.provider)
            observed = base / "observed.json"
            replay = execute([args.python, str(HERE / "replay.py"), str(observed)],
                             replay_dir, replay_env, timeout=60)
            (dest / "replay_stdout.txt").write_text(replay["stdout"])
            (dest / "replay_stderr.txt").write_text(replay["stderr"])
            quality["checks"]["replay_exit_zero"] = replay["exit_code"] == 0
            calls = json.loads(observed.read_text()) if observed.is_file() else []
            quality["observed_sampling_calls"] = calls
            quality["checks"]["sampling_configuration"] = calls == [dict(
                number_of_samples=8, evaluation_concurrency=1, random_seed=42,
                dry_run=False, names=["alpha", "beta"], dispatcher="LocalDispatcher")]
            samples = workspace / "sampling_output/sample_parameters.txt"
            repeated = replay_dir / "sampling_output/sample_parameters.txt"
            quality["checks"]["reproducible_samples"] = samples.is_file() and repeated.is_file() and samples.read_bytes() == repeated.read_bytes()
            for name, expected in (("model.py", MODEL), ("workflow.py", WORKFLOW)):
                if name == "model.py" and task in ("existing", "reuse") or name == "workflow.py" and task == "existing":
                    quality["checks"][name + "_unchanged"] = (workspace / name).is_file() and (workspace / name).read_text() == expected
            quality["checks"]["library_unchanged"] = all(
                (workspace / "romtools" / p.relative_to(REPO / "romtools")).is_file() and
                (workspace / "romtools" / p.relative_to(REPO / "romtools")).read_bytes() == p.read_bytes()
                for p in (REPO / "romtools").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
            quality["passed"] = quality["passed"] and all(v is True for k, v in quality["checks"].items() if k != "error")
            record["quality"] = quality
            record["successful"] = result["exit_code"] == 0 and record["provider_completed"] and quality["passed"]
            shutil.copytree(workspace, dest / "artifacts", ignore=shutil.ignore_patterns("romtools", "__pycache__", "*.pyc"))
        write_json(dest / "result.json", record)
        print(f"{trial_id}: {record['status']}; usage={record['usage']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["codex", "claude"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per session")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--executable", help="Provider CLI path")
    parser.add_argument("--skill", type=Path, default=REPO / ".agents/skills/workflow/SKILL.md")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts and manifest; never call a model")
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    args.python = str(Path(args.python).absolute())
    args.harness_hash = digest(b"".join(p.name.encode() + b"\0" + p.read_bytes() for p in sorted(HERE.glob("*.py"))))
    executable = shutil.which(args.executable or args.provider)
    if not args.dry_run and not executable:
        parser.error(f"{args.provider} executable not found")
    if args.provider == "claude" and not args.dry_run and not any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")):
        parser.error("Clean Claude trials require ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN; see README")
    skill = args.skill.read_bytes()
    version = "not checked (dry run)"
    args.repository_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    args.dependencies = {}
    if not args.dry_run:
        version = subprocess.check_output([executable, "--version"], text=True).strip()
        subprocess.run([args.python, "-c", "import numpy, scipy, yaml, romtools"], cwd=REPO, check=True)
        args.dependencies = json.loads(subprocess.check_output([args.python, "-c",
            "import importlib.metadata as m,json,platform; print(json.dumps(dict(python=platform.python_version(), **{p:m.version(p) for p in ['numpy','scipy','PyYAML','matplotlib']})))"], text=True))
    run_dir = args.output.resolve() / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True)
    (run_dir / "skill.txt").write_bytes(skill)
    records = []
    for repeat in range(args.repeats):
        for task_index, task in enumerate(args.tasks):
            order = ["baseline", "skill"] if (repeat + task_index) % 2 == 0 else ["skill", "baseline"]
            for condition in order:
                record = trial(args, run_dir, task, condition, repeat, skill, executable or args.provider, version)
                records.append(record)
                write_json(run_dir / "results.json", records)
                if not args.dry_run and (record.get("usage") is None or record.get("status") == "interrupted"):
                    print(f"Stopped: no token usage returned. Inspect {run_dir / record['trial_id'] / 'stderr.txt'}", file=sys.stderr)
                    return 2
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

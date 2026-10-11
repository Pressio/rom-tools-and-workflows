import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fixtures import MODEL, WORKFLOW
from providers import parse_usage
from run import prepare, REPO, HERE


class BenchmarkTests(unittest.TestCase):
    def test_codex_cache_not_double_counted(self):
        output = json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 20}})
        self.assertEqual(parse_usage("codex", output)["total_tokens"], 120)

    def test_claude_cache_added_once(self):
        output = json.dumps({"type": "result", "usage": {"input_tokens": 10,
            "cache_read_input_tokens": 60, "cache_creation_input_tokens": 30, "output_tokens": 20},
            "modelUsage": {"ignored": {"inputTokens": 100}}})
        self.assertEqual(parse_usage("claude", output)["total_tokens"], 120)

    def test_codex_optional_cache_write(self):
        output = json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 100, "cached_input_tokens": 60,
            "cache_write_input_tokens": 10, "output_tokens": 20}})
        usage = parse_usage("codex", output)
        self.assertEqual(usage["cache_write_tokens"], 10)
        self.assertEqual(usage["total_tokens"], 120)

    def test_missing_usage_is_unknown(self):
        self.assertIsNone(parse_usage("codex", '{"type":"turn.failed"}'))
        self.assertIsNone(parse_usage("claude", "authentication error"))

    def test_baseline_has_no_skill_or_generated_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            prepare(root, "generate", "baseline", b"UNIQUE_SKILL_CONTENT")
            self.assertFalse(list(root.rglob("SKILL.md")))
            self.assertFalse((root / "model.py").exists())
            self.assertFalse((root / "workflow.py").exists())
            self.assertFalse((root / ".claude").exists())

    def test_fixture_executes_and_bad_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.py").write_text(MODEL)
            (root / "workflow.py").write_text(WORKFLOW)
            env = dict(os.environ, PYTHONPATH=str(REPO))
            observed = root / "observed.json"
            run = subprocess.run([sys.executable, str(HERE / "replay.py"), str(observed)],
                                 cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(observed.read_text())[0]["random_seed"], 42)
            (root / "sampling_output/AUDIT.md").write_text("Fixture audit")
            from check_outputs import check
            self.assertTrue(check(root)["passed"])
            (root / "sampling_output/run_0/result.json").write_text('{"value": 999}')
            self.assertFalse(check(root)["passed"])

    def test_runner_pair_with_fake_cli(self):
        # Exercises isolation, collection and replay without any model requests.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / "fake-codex"
            fake.write_text('''#!/usr/bin/env python3
import json, pathlib, subprocess, sys
if "--version" in sys.argv:
    print("fake-test-only")
    sys.exit(0)
prompt = sys.stdin.read()
skill_exists = pathlib.Path("workflow_skill/SKILL.md").exists()
assert skill_exists == ("Read workflow_skill/SKILL.md" in prompt)
subprocess.run([sys.executable, "workflow.py"], check=True, stdout=subprocess.DEVNULL)
pathlib.Path("sampling_output/AUDIT.md").write_text("Test fixture only")
print(json.dumps({"type":"turn.completed", "usage":{"input_tokens":100,"cached_input_tokens":60,"output_tokens":20}}))
''')
            fake.chmod(0o755)
            env = dict(os.environ, CODEX_HOME=str(root / "no-auth"))
            result = subprocess.run([sys.executable, str(HERE / "run.py"),
                "--provider", "codex", "--model", "fake-test-only", "--executable", str(fake),
                "--tasks", "existing", "--repeats", "1", "--output", str(root / "results")],
                cwd=REPO, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(next((root / "results").glob("*/results.json")).read_text())
            self.assertEqual(len(data), 2)
            self.assertTrue(all(r["successful"] for r in data), data)
            self.assertEqual(data[0]["source_sha256"], data[1]["source_sha256"])
            self.assertTrue(all(r["usage"]["total_tokens"] == 120 for r in data))


if __name__ == "__main__":
    unittest.main()

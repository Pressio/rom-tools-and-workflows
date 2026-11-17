"""Runner contract checks without model calls or remote jobs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_runner', ROOT / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_usage_terminal_event_and_missing_instrumentation(tmp_path):
    log = tmp_path / 'events.jsonl'
    log.write_text('not json\n' + json.dumps({'type': 'turn.completed', 'usage':
        {'input_tokens': 12, 'output_tokens': 3, 'cached_input_tokens': 4}}) + '\n')
    assert runner.parse_usage(log) == dict(input_tokens=12, output_tokens=3, cached_input_tokens=4)
    log.write_text('{"type":"error"}\n')
    assert runner.parse_usage(log) is None


@pytest.mark.parametrize('outcome', ['error', 'timeout', 'success'])
def test_agent_exit_controls_grading_and_preserves_submission(tmp_path, monkeypatch, outcome):
    original_run = subprocess.run
    calls = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == 'tar':
            return original_run(cmd, **kwargs)
        work = Path(kwargs['cwd'])
        calls.append(cmd[0])
        if cmd[0] == 'codex':
            assert '--sandbox' in cmd and 'workspace-write' in cmd
            assert kwargs['stdin'] == subprocess.DEVNULL
            assert not (work / 'benchmarks/agent_skills/reference').exists()
            assert not (work / 'benchmarks/agent_skills/tasks/cdr_model_adapter/tests').exists()
            (work / 'submission_evidence.txt').write_text('retained')
            if outcome == 'timeout':
                raise subprocess.TimeoutExpired(cmd, 600)
            return subprocess.CompletedProcess(cmd, 0 if outcome == 'success' else 1,
                                               stderr='sandbox initialization failed')
        assert outcome == 'success', 'Grader must not run after an agent error'
        assert (work / 'benchmarks/agent_skills/tasks/cdr_model_adapter/tests').is_dir()
        return subprocess.CompletedProcess(cmd, 0, stdout='2 passed', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    result = runner.run_one('mock-model', 'full', 0, 'cdr_model_adapter', tmp_path,
                            keep_workspaces=True)
    assert result['status'] == {'error': 'agent_error', 'timeout': 'timeout', 'success': 'passed'}[outcome]
    assert result['graded'] == (outcome == 'success')
    assert result['passed'] == (outcome == 'success')
    assert len(calls) == (2 if outcome == 'success' else 1)
    retained = Path(result['workspace'])
    assert (retained / 'submission_evidence.txt').read_text() == 'retained'
    assert not (retained / 'benchmarks/agent_skills/tasks/cdr_model_adapter/tests').exists()


def test_summary_handles_null_usage_and_reports_ungraded(tmp_path):
    results = tmp_path / 'results.jsonl'
    results.write_text(json.dumps(dict(task='cdr', model='mock', condition='full',
        passed=False, usage=None, graded=False, status='agent_error')) + '\n')
    result = subprocess.run(['python', str(ROOT / 'summarize.py'), str(results)],
                            text=True, capture_output=True, check=True)
    assert 'ungraded=1' in result.stdout
    assert 'tokens_per_success=NA' in result.stdout

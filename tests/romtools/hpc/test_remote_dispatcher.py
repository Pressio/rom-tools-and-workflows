from unittest.mock import MagicMock

import numpy as np
import pytest

import romtools.hpc.dispatcher_base as dispatcher_base_module
from romtools.hpc.connection import Result
from romtools.hpc.remote_dispatcher import RemoteDispatcher

from conftest import ArchiveFakeConnection, FakeConnection


def _make_dispatcher(monkeypatch, config, connection, sampling_directory="hpctools"):
    """
    Build a RemoteDispatcher against an injected fake Connection, bypassing
    Configuration's real argv/YAML parsing by stubbing it out entirely.
    """
    stub_config = MagicMock()
    stub_config.to_dict.return_value = dict(config)
    monkeypatch.setattr(dispatcher_base_module, "Configuration", MagicMock(return_value=stub_config))
    return RemoteDispatcher(sampling_directory=sampling_directory, connection=connection)


def test_missing_remote_raises(monkeypatch, make_config):
    config = make_config(remote=None)

    with pytest.raises(ValueError, match="Remote host and user"):
        _make_dispatcher(monkeypatch, config, FakeConnection())


def test_missing_user_raises(monkeypatch, make_config):
    config = make_config(user=None)

    with pytest.raises(ValueError, match="Remote host and user"):
        _make_dispatcher(monkeypatch, config, FakeConnection())


def test_injected_connection_skips_real_ssh_handshake(monkeypatch, make_config):
    conn = FakeConnection(host="injected-host")
    dispatcher = _make_dispatcher(monkeypatch, make_config(), conn)

    assert dispatcher.conn is conn


def test_dispatch_without_slurm_runs_command_directly(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    result = dispatcher.dispatch("./my_app", with_slurm=False)

    assert result == "No SLURM job submitted."
    assert conn.calls == ["cd campaigns && ./my_app"]


def test_dispatch_with_slurm_submits_polls_and_collects(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 123\n", "", 0)),
        ("squeue -j 123 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
        ("sacct -j", Result("123|COMPLETED|0:0|0:0", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=["all"])
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    result = dispatcher.dispatch("./my_app")

    assert result.ok
    assert result.exit_code == 0
    assert any(c.startswith("cd campaigns/hpctools && sbatch") for c in conn.calls)
    assert any(c == "squeue -j 123 -h" for c in conn.calls)
    assert any(c.startswith("rm -f") for c in conn.calls)


def test_dispatch_uses_run_directory_when_given(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 5\n", "", 0)),
        ("squeue -j 5 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=None)
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.dispatch("./my_app", run_directory="run_00")

    assert any(c.startswith("cd campaigns/run_00 && sbatch") for c in conn.calls)

def test_dispatch_with_default_relative_remote_root_submits_resolvable_script_path(monkeypatch, make_config, tmp_path):
    """
    Regression test: remote_root defaults to a relative path ("hpctools_campaigns").
    __submit_slurm_job cd's into remote_root/run_directory and must not then hand
    sbatch a script path that is *also* prefixed with remote_root/run_directory,
    since that duplicated path can't resolve from the new working directory.
    """
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 5\n", "", 0)),
        ("squeue -j 5 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(poll_interval=0, collect=None)
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.dispatch("./my_app", run_directory="run_00")

    assert (
        "cd hpctools_campaigns/run_00 && sbatch --output=slurm.out "
        "--error=slurm.err hpctools_job_slurm.sh"
    ) in conn.calls


def test_dispatch_with_custom_script_uploads_to_run_directory(monkeypatch, make_config, tmp_path):
    """
    Regression test: __generate_slurm_script's custom-script branch uploaded to
    remote_root/sampling_directory even when a run_directory was given, while
    __submit_slurm_job cd's into remote_root/run_directory - a directory mismatch
    that left the uploaded script outside the directory sbatch is run from.
    """
    monkeypatch.chdir(tmp_path)
    local_script = tmp_path / "custom_job.sh"
    local_script.write_text("#!/bin/bash\n#SBATCH --job-name=custom\nsrun ./my_app\n")

    responses = [
        ("sbatch", Result("Submitted batch job 7\n", "", 0)),
        ("squeue -j 7 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(script=str(local_script), poll_interval=0, collect=None)
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.dispatch(run_directory="run_00")

    assert conn.put_calls == [(str(local_script), "hpctools_campaigns/run_00/custom_job.sh")]
    assert (
        "cd hpctools_campaigns/run_00 && sbatch --output=slurm.out "
        "--error=slurm.err custom_job.sh"
    ) in conn.calls


def test_dispatch_raises_when_sbatch_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("sbatch", Result("", "out of quota", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    with pytest.raises(RuntimeError, match="out of quota"):
        dispatcher.dispatch("./my_app")


def test_dispatch_raises_when_sbatch_output_unparseable(monkeypatch, make_config):
    conn = FakeConnection(responses=[("sbatch", Result("nonsense output", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    with pytest.raises(RuntimeError, match="Could not parse job ID"):
        dispatcher.dispatch("./my_app")


def test_keyboard_interrupt_during_poll_cancels_job(monkeypatch, make_config):
    responses = [
        ("sbatch", Result("Submitted batch job 42\n", "", 0)),
        ("squeue -j 42 -h", Result("42 R\n", "", 0)),
        ("scancel 42", Result("", "", 0)),
    ]
    conn = FakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", poll_interval=1)
    dispatcher = _make_dispatcher(monkeypatch, config, conn)
    monkeypatch.setattr(
        "romtools.hpc.remote_dispatcher.time.sleep",
        MagicMock(side_effect=KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        dispatcher.dispatch("./my_app")

    assert any(c.startswith("scancel") for c in conn.calls)


def test_put_resolves_relative_path_under_remote_root(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.put("local.txt", "results/out.txt")

    assert conn.put_calls == [("local.txt", "campaigns/results/out.txt")]


def test_put_preserves_absolute_remote_path(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.put("local.txt", "/abs/out.txt")

    assert conn.put_calls == [("local.txt", "/abs/out.txt")]


def test_get_resolves_relative_path_under_remote_root(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.get("results/out.txt", "local.txt")

    assert conn.get_calls == [("campaigns/results/out.txt", "local.txt")]


def test_path_exists_true_and_false(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    assert dispatcher.path_exists("some/file.txt") is True
    assert conn.calls == ["test -e campaigns/some/file.txt"]


def test_path_exists_false_when_command_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    assert dispatcher.path_exists("missing.txt") is False


def test_create_empty_dir_issues_mkdir(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.create_empty_dir("newdir")

    assert conn.calls == ["mkdir -p campaigns/newdir"]


def test_np_savetxt_writes_via_remote_heredoc(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.np_savetxt("results/array.txt", np.array([1, 2, 3]), fmt="%d")

    assert len(conn.calls) == 1
    written_cmd = conn.calls[0]
    assert "campaigns/results/array.txt" in written_cmd
    assert "1\n2\n3" in written_cmd


def test_np_savez_uploads_via_put(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(monkeypatch, config, conn)

    dispatcher.np_savez("results/data", a=np.array([1, 2]))

    assert len(conn.put_calls) == 1
    local_path, remote_path = conn.put_calls[0]
    assert remote_path == "campaigns/results/data.npz"
    assert local_path.endswith("data.npz")

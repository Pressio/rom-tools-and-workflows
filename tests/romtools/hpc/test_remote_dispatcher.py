from unittest.mock import MagicMock

import os

import numpy as np
import shlex
import pytest

from romtools.hpc.connection import Result
from romtools.hpc.dispatchers import RemoteDispatcher

from conftest import ArchiveFakeConnection, FakeConnection


def _make_dispatcher(config, connection, campaign_directory="hpctools"):
    """Build a RemoteDispatcher against an injected fake Connection."""
    return RemoteDispatcher(campaign_directory=campaign_directory,
                            connection=connection, config=config)


def test_missing_remote_raises(monkeypatch, make_config):
    config = make_config(remote=None)

    with pytest.raises(ValueError, match="Remote host and user"):
        _make_dispatcher(config, FakeConnection())


def test_missing_user_raises(monkeypatch, make_config):
    config = make_config(user=None)

    with pytest.raises(ValueError, match="Remote host and user"):
        _make_dispatcher(config, FakeConnection())


def test_injected_connection_skips_real_ssh_handshake(monkeypatch, make_config):
    conn = FakeConnection(host="injected-host")
    dispatcher = _make_dispatcher(make_config(), conn)

    assert dispatcher.conn is conn


def test_run_executes_command_directly_without_slurm(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.run("./my_app")

    assert result.ok
    assert conn.calls == ["cd campaigns && ./my_app"]


def test_run_reports_a_failed_command_without_raising(monkeypatch, make_config):
    """
    Both dispatchers report a failing command through the Result, so a model
    written against one behaves the same when handed the other. This used to
    raise RuntimeError while LocalDispatcher.run returned the failing Result.
    """
    conn = FakeConnection(responses=[("./my_app", Result("", "segfault", 139))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.run("./my_app")

    assert not result.ok
    assert result.exit_code == 139
    assert result.stderr == "segfault"


def test_submit_job_submits_polls_and_collects(monkeypatch, make_config, tmp_path):
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
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert result.exit_code == 0
    assert any(c.startswith("cd campaigns/hpctools && sbatch") for c in conn.calls)
    assert any(c == "squeue -j 123 -h" for c in conn.calls)
    assert any(c.startswith("rm -f") for c in conn.calls)


def test_submit_job_keeps_polling_when_squeue_fails(monkeypatch, make_config, tmp_path):
    """
    Regression test: wait() read empty stdout as a finished job without checking
    the exit code, so a transient slurm_load_jobs error ended the poll loop
    while the job was still running and the sample was reported as failed.
    """
    monkeypatch.chdir(tmp_path)
    squeue_results = iter([
        Result("", "slurm_load_jobs error: Connection timed out", 1),
        Result("123 R\n", "", 0),
        Result("", "", 0),
    ])
    responses = [
        ("sbatch", Result("Submitted batch job 123\n", "", 0)),
        ("squeue -j 123 -h", lambda: next(squeue_results)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
        ("sacct -j", Result("123|COMPLETED|0:0|0:0", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0,
                         timeout=30, collect=["all"])
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert len([c for c in conn.calls if c == "squeue -j 123 -h"]) == 3


def test_submit_job_stops_polling_when_the_job_id_is_unknown(monkeypatch, make_config, tmp_path):
    """A purged job is genuinely gone, so squeue failing that way ends the poll."""
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 123\n", "", 0)),
        ("squeue -j 123 -h", Result("", "slurm_load_jobs error: Invalid job id specified", 1)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
        ("sacct -j", Result("123|COMPLETED|0:0|0:0", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0,
                         timeout=30, collect=["all"])
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert len([c for c in conn.calls if c == "squeue -j 123 -h"]) == 1


def test_submit_job_falls_through_to_sacct_when_squeue_never_answers(monkeypatch, make_config, tmp_path):
    """A persistently broken squeue must not spin: sacct is the authority."""
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 123\n", "", 0)),
        ("squeue -j 123 -h", Result("", "slurm_load_jobs error: Connection timed out", 1)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
        ("sacct -j", Result("123|COMPLETED|0:0|0:0", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=["all"])
    dispatcher = _make_dispatcher(config, conn)

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert any(c.startswith("sacct -j 123") for c in conn.calls)


def test_submit_job_collects_results_and_extracts_them_locally(monkeypatch, make_config, tmp_path):
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
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job("./my_app")

    archive_name = "dispatcher-transfer-myjob.tar.gz"
    remote_archive_path = "campaigns/dispatcher-transfer-myjob.tar.gz"
    assert conn.get_calls == [(remote_archive_path, archive_name)]
    assert f"rm -f {remote_archive_path}" in conn.calls
    assert not (tmp_path / archive_name).exists()
    assert (tmp_path / "hpctools" / "result.txt").read_text() == "payload"


def test_results_are_extracted_even_when_the_remote_cleanup_fails(monkeypatch, make_config, tmp_path):
    """
    Regression test: removing the remote archive raises, which aborted
    collection after the download but before extraction, so the results were
    never unpacked. A stale remote archive is untidy, not fatal.
    """
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 123\n", "", 0)),
        ("squeue -j 123 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "read-only file system", 1)),
        ("sacct -j", Result("123|COMPLETED|0:0|0:0", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=["all"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job("./my_app")

    assert (tmp_path / "hpctools" / "result.txt").read_text() == "payload"


def test_submit_job_skips_local_extraction_when_no_collect_patterns(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 9\n", "", 0)),
        ("squeue -j 9 -h", Result("", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=None)
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job("./my_app")

    assert conn.get_calls == []
    assert not (tmp_path / "hpctools").exists()


def test_submit_job_uses_run_directory_when_given(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    responses = [
        ("sbatch", Result("Submitted batch job 5\n", "", 0)),
        ("squeue -j 5 -h", Result("", "", 0)),
        ("tar -czf", Result("", "", 0)),
        ("rm -f", Result("", "", 0)),
    ]
    conn = ArchiveFakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", job_name="myjob", poll_interval=0, collect=None)
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job("./my_app", run_directory="run_00")

    assert any(c.startswith("cd campaigns/run_00 && sbatch") for c in conn.calls)

def test_submit_job_with_default_relative_remote_root_submits_resolvable_script_path(monkeypatch, make_config, tmp_path):
    """
    Regression test: remote_root defaults to a relative path ("hpctools_campaigns").
    SlurmJobManager.submit cd's into remote_root/run_directory and must not then hand
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
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job("./my_app", run_directory="run_00")

    assert (
        "cd hpctools_campaigns/run_00 && sbatch --output=slurm.out "
        "--error=slurm.err hpctools_job_slurm.sh"
    ) in conn.calls


def test_submit_job_with_custom_script_uploads_to_run_directory(monkeypatch, make_config, tmp_path):
    """
    Regression test: _generate_slurm_script's custom-script branch uploaded to
    remote_root/campaign_directory even when a run_directory was given, while
    SlurmJobManager.submit cd's into remote_root/run_directory - a directory mismatch
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
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.submit_job(run_directory="run_00")

    assert conn.put_calls == [(str(local_script), "hpctools_campaigns/run_00/custom_job.sh")]
    assert (
        "cd hpctools_campaigns/run_00 && sbatch --output=slurm.out "
        "--error=slurm.err custom_job.sh"
    ) in conn.calls


def test_submit_job_raises_when_sbatch_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("sbatch", Result("", "out of quota", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    with pytest.raises(RuntimeError, match="out of quota"):
        dispatcher.submit_job("./my_app")


def test_submit_job_raises_when_sbatch_output_unparseable(monkeypatch, make_config):
    conn = FakeConnection(responses=[("sbatch", Result("nonsense output", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    with pytest.raises(RuntimeError, match="Could not parse job ID"):
        dispatcher.submit_job("./my_app")


def test_keyboard_interrupt_during_poll_cancels_job(monkeypatch, make_config):
    responses = [
        ("sbatch", Result("Submitted batch job 42\n", "", 0)),
        ("squeue -j 42 -h", Result("42 R\n", "", 0)),
        ("scancel 42", Result("", "", 0)),
    ]
    conn = FakeConnection(responses=responses)
    config = make_config(remote_root="campaigns", poll_interval=1)
    dispatcher = _make_dispatcher(config, conn)
    monkeypatch.setattr(
        "romtools.hpc.components.slurm_job_manager.time.sleep",
        MagicMock(side_effect=KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        dispatcher.submit_job("./my_app")

    assert any(c.startswith("scancel") for c in conn.calls)


def test_put_resolves_relative_path_under_remote_root(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.put("local.txt", "results/out.txt")

    assert conn.put_calls == [("local.txt", "campaigns/results/out.txt")]


def test_put_preserves_absolute_remote_path(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.put("local.txt", "/abs/out.txt")

    assert conn.put_calls == [("local.txt", "/abs/out.txt")]


def test_get_resolves_relative_path_under_remote_root(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.get("results/out.txt", "local.txt")

    assert conn.get_calls == [("campaigns/results/out.txt", "local.txt")]


def test_upload_skips_when_no_upload_patterns(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns", upload=None)
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    assert conn.calls == []
    assert conn.put_calls == []


def test_upload_warns_but_does_not_raise_when_a_pattern_matches_nothing(monkeypatch, make_config, tmp_path):
    """
    Regression test: an unmatched pattern is a warning on LocalDispatcher but
    an uncaught FileNotFoundError here, so the same model behaved differently
    depending on which dispatcher it was handed.
    """
    monkeypatch.chdir(tmp_path)

    conn = FakeConnection()
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["missing.yaml"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    assert conn.put_calls == []
    assert not any("mkdir -p" in call for call in conn.calls)


def test_upload_packs_local_files_and_transfers_them(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("results")

    conn = FakeConnection(responses=[("archive_path=", Result("", "", 0))])
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["data.txt"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    tar_name = "dispatcher-transfer-myjob.tar.gz"
    (staged_archive, remote_path), = conn.put_calls
    assert os.path.basename(staged_archive) == tar_name
    assert not os.path.exists(staged_archive)  # staging area cleaned up after put
    assert remote_path == f"campaigns/run_00/{tar_name}"


def test_upload_packs_the_archive_outside_the_directory_it_packs(monkeypatch, make_config, tmp_path):
    """
    Regression test: an upload of everything wrote the archive into the working
    directory it was packing, so tar read the file it was still writing and
    failed with "file changed as we read it" on most runs.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("results")

    conn = FakeConnection(responses=[("archive_path=", Result("", "", 0))])
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["all"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    staged_archive, _ = conn.put_calls[0]
    assert os.path.dirname(staged_archive) != str(tmp_path)
    assert list(tmp_path.glob("*.tar.gz")) == []


def test_upload_without_a_run_directory_targets_the_campaign_directory(monkeypatch, make_config, tmp_path):
    """
    Regression test: upload() addressed the remote root while submit_job() runs
    in the campaign directory, staging inputs where the job could not see them.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("results")

    conn = FakeConnection(responses=[("archive_path=", Result("", "", 0))])
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["data.txt"])
    dispatcher = _make_dispatcher(config, conn, campaign_directory="hpctools")

    dispatcher.upload()

    tar_name = "dispatcher-transfer-myjob.tar.gz"
    _, remote_path = conn.put_calls[0]
    assert remote_path == f"campaigns/hpctools/{tar_name}"


def test_upload_creates_the_run_directory_it_transfers_into(monkeypatch, make_config, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("results")

    conn = FakeConnection(responses=[("archive_path=", Result("", "", 0))])
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["data.txt"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    assert any("mkdir -p" in call and "campaigns/run_00" in call for call in conn.calls)


def test_upload_extracts_the_same_archive_it_uploaded(monkeypatch, make_config, tmp_path):
    """
    Regression test: upload() previously built a fully-qualified remote tar path and
    then handed it to self.put(), which re-resolves relative paths against
    remote_root - prefixing it a second time and leaving the archive PUT at one
    remote path but extracted from another.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("results")

    conn = FakeConnection(responses=[("archive_path=", Result("", "", 0))])
    config = make_config(remote_root="campaigns", job_name="myjob", upload=["data.txt"])
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.upload("run_00")

    _, uploaded_path = conn.put_calls[0]
    extract_cmd = conn.calls[-1]

    parts = shlex.split(extract_cmd)

    assert uploaded_path in parts


def test_path_exists_true_and_false(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    assert dispatcher.path_exists("some/file.txt") is True
    assert conn.calls == ["test -e campaigns/some/file.txt"]


def test_path_exists_false_when_command_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    assert dispatcher.path_exists("missing.txt") is False


def test_create_empty_dir_issues_mkdir(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.create_empty_dir("newdir")

    assert conn.calls == ["mkdir -p campaigns/newdir"]


def test_list_dir_lists_remote_entries(monkeypatch, make_config):
    conn = FakeConnection(responses=[("ls -1", Result("iteration_0\niteration_1\n", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    assert dispatcher.list_dir("work") == ["iteration_0", "iteration_1"]
    assert conn.calls == ["ls -1 campaigns/work"]


def test_list_dir_is_empty_when_command_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("ls -1", Result("", "No such file", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    assert dispatcher.list_dir("missing") == []


def test_remove_deletes_remote_file(monkeypatch, make_config):
    conn = FakeConnection()
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.remove("work/restart.npz")

    assert conn.calls == ["rm -f campaigns/work/restart.npz"]


def test_remove_raises_when_command_fails(monkeypatch, make_config):
    conn = FakeConnection(responses=[("rm -f", Result("", "permission denied", 1))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    with pytest.raises(RuntimeError, match="permission denied"):
        dispatcher.remove("work/restart.npz")


def test_write_text_creates_the_parent_directory(make_config):
    """
    Regression test: the local manager created parents and the remote one did
    not, so the same model succeeded locally and raised RuntimeError remotely.
    """
    conn = FakeConnection()
    dispatcher = _make_dispatcher(make_config(remote_root="campaigns"), conn)

    dispatcher.write_text("work/stats.txt", "elbo: 1.0\n")

    assert len(conn.calls) == 1
    assert "mkdir -p campaigns/work" in conn.calls[0]
    assert "campaigns/work/stats.txt" in conn.calls[0]


def test_np_savetxt_writes_through_write_text(make_config):
    conn = FakeConnection()
    dispatcher = _make_dispatcher(make_config(remote_root="campaigns"), conn)

    dispatcher.np_savetxt("results/array.txt", np.array([1, 2, 3]), fmt="%d")

    assert len(conn.calls) == 1
    assert "campaigns/results/array.txt" in conn.calls[0]


def test_np_savez_uploads_via_put(monkeypatch, make_config):
    conn = FakeConnection(responses=[("test -e", Result("", "", 0))])
    config = make_config(remote_root="campaigns")
    dispatcher = _make_dispatcher(config, conn)

    dispatcher.np_savez("results/data", a=np.array([1, 2]))

    assert len(conn.put_calls) == 1
    local_path, remote_path = conn.put_calls[0]
    assert remote_path == "campaigns/results/data.npz"
    assert local_path.endswith("data.npz")


def test_np_savez_creates_the_parent_directory(make_config):
    """
    Regression test: a missing directory raised instead of being created, so
    np_savez disagreed with write_text about whose job the parent was.
    """
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])
    dispatcher = _make_dispatcher(make_config(remote_root="campaigns"), conn)

    dispatcher.np_savez("results/data", a=np.array([1, 2]))

    assert any("mkdir -p campaigns/results" in call for call in conn.calls)
    assert conn.put_calls[0][1] == "campaigns/results/data.npz"


def test_campaign_directory_ignores_a_trailing_slash(make_config):
    """
    Regression test: os.path.basename("sample_00/") is "", so the campaign was
    staged at the remote root, tarred from the root on collection, and then
    aborted in os.makedirs("") with a FileNotFoundError.
    """
    dispatcher = _make_dispatcher(make_config(), FakeConnection(),
                                  campaign_directory="sample_00/")

    assert dispatcher.campaign_directory == "sample_00"


def test_campaign_directory_keeps_a_nested_path(make_config):
    """
    Regression test: basename() collapsed "runs/sample_00" to "sample_00", so
    slurm and the transfers acted on a different directory than the workflow.
    """
    dispatcher = _make_dispatcher(make_config(), FakeConnection(),
                                  campaign_directory="runs/sample_00")

    assert dispatcher.campaign_directory == "runs/sample_00"


@pytest.mark.parametrize("campaign_directory", ["/scratch/me/campaign", "../escape", "", "."])
def test_campaign_directory_must_be_a_relative_path(make_config, campaign_directory):
    """
    Regression test: basename() silently reinterpreted an absolute path as one
    under the remote root rather than saying the path was unusable.
    """
    with pytest.raises(ValueError):
        _make_dispatcher(make_config(), FakeConnection(),
                         campaign_directory=campaign_directory)


def test_require_relative_path_rejects_an_absolute_path(monkeypatch, make_config):
    dispatcher = _make_dispatcher(make_config(), FakeConnection())

    with pytest.raises(ValueError, match="relative to the remote root"):
        dispatcher.require_relative_path("/scratch/work")


def test_require_relative_path_accepts_a_relative_path(monkeypatch, make_config):
    dispatcher = _make_dispatcher(make_config(), FakeConnection())

    dispatcher.require_relative_path("campaigns/work")


def test_require_supported_concurrency_rejects_concurrent_evaluation(monkeypatch, make_config):
    dispatcher = _make_dispatcher(make_config(), FakeConnection())

    with pytest.raises(ValueError, match="Concurrency > 1 is not supported"):
        dispatcher.require_supported_concurrency(2)


def test_require_supported_concurrency_accepts_serial_evaluation(monkeypatch, make_config):
    dispatcher = _make_dispatcher(make_config(), FakeConnection())

    dispatcher.require_supported_concurrency(1)

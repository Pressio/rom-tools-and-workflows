import os
import shlex
import sys

import numpy as np
import pytest

from romtools.hpc.dispatchers import LocalDispatcher


@pytest.fixture
def dispatcher():
    return LocalDispatcher()


@pytest.fixture
def fake_scheduler(tmp_path, monkeypatch):
    """
    Put stand-in sbatch, squeue, and sacct commands on PATH.

    A LocalDispatcher on a cluster node shells out to the real ones, so this
    exercises that path end to end without needing a cluster. Returns the file
    the stand-in sbatch records its invocations in.
    """
    def install(job_id="123", state="COMPLETED", exit_code="0:0"):
        bin_dir = tmp_path / "slurm-bin"
        bin_dir.mkdir(exist_ok=True)
        sbatch_log = tmp_path / "sbatch.log"

        bodies = {
            "sbatch": f'echo "cwd=$PWD args=$*" >> {shlex.quote(str(sbatch_log))}\n'
                      f'echo "Submitted batch job {job_id}"\n',
            "squeue": "",  # a finished job is no longer in the queue
            "sacct": f'echo "{job_id}|{state}|{exit_code}|{exit_code}"\n',
        }
        for name, body in bodies.items():
            command = bin_dir / name
            command.write_text(f"#!/bin/bash\n{body}")
            command.chmod(0o755)

        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        return sbatch_log

    return install


def test_configuration_comes_from_the_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-j", "myjob", "-n", "4", "-w", "02:00:00"])

    dispatcher = LocalDispatcher()

    assert dispatcher.get_config("job_name") == "myjob"
    assert dispatcher.get_config("num_nodes") == 4
    assert dispatcher.get_config("wall_time") == "02:00:00"


def test_configuration_comes_from_a_yaml_file(tmp_path, monkeypatch):
    config = tmp_path / "cluster.yaml"
    config.write_text("slurm:\n  job_name: from_yaml\n  num_nodes: 2\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-i", str(config)])

    dispatcher = LocalDispatcher()

    assert dispatcher.get_config("job_name") == "from_yaml"
    assert dispatcher.get_config("num_nodes") == 2


def test_put_copies_a_file(tmp_path, dispatcher):
    src = tmp_path / "src.txt"
    src.write_text("payload")
    dst = tmp_path / "nested" / "dst.txt"

    dispatcher.put(str(src), str(dst))

    assert dst.read_text() == "payload"


def test_get_copies_a_file(tmp_path, dispatcher):
    src = tmp_path / "src.txt"
    src.write_text("payload")
    dst = tmp_path / "dst.txt"

    dispatcher.get(str(src), str(dst))

    assert dst.read_text() == "payload"


def test_put_copies_a_directory(tmp_path, dispatcher):
    src_dir = tmp_path / "src_dir"
    src_dir.mkdir()
    (src_dir / "file.txt").write_text("payload")
    dst_dir = tmp_path / "dst_dir"

    dispatcher.put(str(src_dir), str(dst_dir))

    assert (dst_dir / "file.txt").read_text() == "payload"


def test_path_exists(tmp_path, dispatcher):
    existing = tmp_path / "exists.txt"
    existing.write_text("x")

    assert dispatcher.path_exists(str(existing))
    assert not dispatcher.path_exists(str(tmp_path / "missing.txt"))


def test_create_empty_dir_is_idempotent(tmp_path, dispatcher):
    target = tmp_path / "newdir"

    dispatcher.create_empty_dir(str(target))
    dispatcher.create_empty_dir(str(target))

    assert target.is_dir()


def test_list_dir_returns_entries(tmp_path, dispatcher):
    (tmp_path / "iteration_0").mkdir()
    (tmp_path / "file.txt").write_text("x")

    assert sorted(dispatcher.list_dir(str(tmp_path))) == ["file.txt", "iteration_0"]


def test_list_dir_of_missing_path_is_empty(tmp_path, dispatcher):
    assert dispatcher.list_dir(str(tmp_path / "missing")) == []


def test_remove_deletes_a_file(tmp_path, dispatcher):
    target = tmp_path / "restart.npz"
    target.write_text("x")

    dispatcher.remove(str(target))

    assert not target.exists()


def test_remove_of_missing_file_does_not_raise(tmp_path, dispatcher):
    dispatcher.remove(str(tmp_path / "missing.npz"))


def test_write_text_creates_parent_directories(tmp_path, dispatcher):
    target = tmp_path / "iteration_0" / "stats.txt"

    dispatcher.write_text(str(target), "elbo: 1.0\n")

    assert target.read_text() == "elbo: 1.0\n"


def test_np_savetxt_round_trip(tmp_path, dispatcher):
    path = tmp_path / "array.txt"
    arr = np.array([1, 2, 3])

    dispatcher.np_savetxt(str(path), arr, fmt="%d")

    assert np.array_equal(np.loadtxt(path, dtype=int), arr)


def test_np_savez_round_trip(tmp_path, dispatcher):
    path = tmp_path / "arrays"

    dispatcher.np_savez(str(path), a=np.array([1, 2]), b=np.array([3, 4]))

    loaded = np.load(str(path) + ".npz")
    assert np.array_equal(loaded["a"], [1, 2])
    assert np.array_equal(loaded["b"], [3, 4])


def test_run_runs_command_and_captures_output(tmp_path, dispatcher):
    result = dispatcher.run("echo hello", run_directory=str(tmp_path))

    assert result.ok
    assert result.stdout.strip() == "hello"


def test_run_without_run_directory_runs_in_cwd(dispatcher):
    result = dispatcher.run("echo hello")

    assert result.ok
    assert result.stdout.strip() == "hello"


def test_run_reports_failure_without_raising(dispatcher):
    result = dispatcher.run("exit 1")

    assert not result.ok
    assert result.exit_code == 1


def test_run_runs_relative_to_run_directory(tmp_path, dispatcher):
    marker = tmp_path / "marker.txt"
    marker.write_text("present")

    result = dispatcher.run("cat marker.txt", run_directory=str(tmp_path))

    assert result.ok
    assert result.stdout.strip() == "present"


def test_submit_job_submits_polls_and_returns_the_job_output(tmp_path, fake_scheduler):
    """
    A LocalDispatcher on a cluster node reaches SLURM directly: the job is
    submitted where the dispatcher runs, and its output is read back in place.
    """
    sbatch_log = fake_scheduler()
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "slurm.out").write_text("job stdout\n")
    (campaign / "slurm.err").write_text("job stderr\n")
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert result.exit_code == 0
    assert result.stdout == "job stdout\n"
    assert result.stderr == "job stderr\n"
    assert f"cwd={campaign}" in sbatch_log.read_text()


def test_submit_job_writes_the_generated_script_into_the_run_directory(tmp_path, fake_scheduler):
    fake_scheduler()
    run_dir = tmp_path / "run"
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path))

    dispatcher.submit_job("./my_app", run_directory=str(run_dir))

    script = (run_dir / "hpctools_job_slurm.sh").read_text()
    assert script.startswith("#!/bin/bash")
    assert "./my_app" in script


def test_submit_job_uses_a_configured_slurm_script_unmodified(tmp_path, monkeypatch, fake_scheduler):
    script = tmp_path / "job.sh"
    script.write_text(
        "#!/bin/bash\n"
        "#SBATCH --output=custom-%j.out\n"
        "#SBATCH --error=custom-%j.err\n"
        "srun ./my_app\n"
    )
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "custom-123.out").write_text("from the custom outfile\n")
    (campaign / "custom-123.err").write_text("")
    sbatch_log = fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "-s", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    result = dispatcher.submit_job()

    assert (campaign / "job.sh").read_text() == script.read_text()
    assert result.stdout == "from the custom outfile\n"
    # The script names its own output files, so sbatch is not told where to write
    assert "--output=" not in sbatch_log.read_text()


def test_submit_job_leaves_results_in_place_without_archiving_them(tmp_path, monkeypatch, fake_scheduler):
    """
    Results of a local job are already on this filesystem, so collect_results()
    has nothing to do, whatever collect patterns are configured.
    """
    fake_scheduler()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "-o", "all"])
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    dispatcher.submit_job("./my_app")

    assert list(tmp_path.glob("*.tar.gz")) == []


def test_submit_job_reports_a_failed_job(tmp_path, fake_scheduler):
    fake_scheduler(state="FAILED", exit_code="1:0")
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    result = dispatcher.submit_job("./my_app")

    assert not result.ok
    assert result.exit_code == 1


def test_submit_job_tolerates_missing_output_files(tmp_path, fake_scheduler):
    """
    Regression test: a job that wrote no output leaves no file behind. Reading
    one back raises FileNotFoundError locally, where the remote file manager
    raises RuntimeError, and neither should escape submit_job().
    """
    fake_scheduler()
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    result = dispatcher.submit_job("./my_app")

    assert result.ok
    assert result.stdout == ""
    assert result.stderr == ""


def test_submit_job_raises_when_given_neither_a_command_nor_a_script(dispatcher):
    with pytest.raises(ValueError, match="base command or a SLURM script"):
        dispatcher.submit_job()


def test_submit_job_describes_the_job_from_the_configuration(tmp_path, monkeypatch, fake_scheduler):
    fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "-j", "myjob", "-n", "4", "-w", "02:00:00"])
    campaign = tmp_path / "campaign"
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    dispatcher.submit_job("./my_app")

    script = (campaign / "myjob_slurm.sh").read_text()
    assert "#SBATCH --job-name=myjob" in script
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --time=02:00:00" in script


def test_require_absolute_path_rejects_a_relative_path(dispatcher):
    with pytest.raises(AssertionError, match="must provide an absolute path"):
        dispatcher.require_absolute_path("work")


def test_require_relative_path_is_a_no_op(tmp_path, dispatcher):
    """Local runs create directories on this machine, so absolute paths are fine."""
    dispatcher.require_relative_path(str(tmp_path))


def test_require_supported_concurrency_allows_concurrent_evaluation(dispatcher):
    dispatcher.require_supported_concurrency(4)

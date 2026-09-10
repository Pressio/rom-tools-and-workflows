import concurrent.futures
import multiprocessing
import os
import pickle
import shlex
import sys

import numpy as np
import pytest

from romtools.hpc.dispatchers import LocalDispatcher


@pytest.fixture
def dispatcher():
    return LocalDispatcher()


def _submit_in_worker(campaign_directory, run_directory):
    """Submit from a fresh dispatcher, the way a pool worker process does."""
    return LocalDispatcher(campaign_directory=campaign_directory).submit_job(
        "./my_app", run_directory=run_directory
    )


@pytest.fixture
def fake_scheduler(tmp_path, monkeypatch):
    """
    Put stand-in sbatch, squeue, and sacct commands on PATH.

    A LocalDispatcher on a cluster node shells out to the real ones, so this
    exercises that path end to end without needing a cluster. Returns the file
    the stand-in sbatch records its invocations in.
    """
    def install(job_id="123", state="COMPLETED", exit_code="0:0", write_output=False):
        bin_dir = tmp_path / "slurm-bin"
        bin_dir.mkdir(exist_ok=True)
        sbatch_log = tmp_path / "sbatch.log"

        # Write stdout where sbatch was told to, or to SLURM's own default if it was not
        write_body = (
            f'out="slurm-{job_id}.out"\n'
            'for arg in "$@"; do\n'
            '  case "$arg" in --output=*) out="${arg#--output=}";; esac\n'
            'done\n'
            f'echo "stdout in $PWD" > "${{out//%j/{job_id}}}"\n'
        ) if write_output else ""

        bodies = {
            "sbatch": f'echo "cwd=$PWD args=$*" >> {shlex.quote(str(sbatch_log))}\n'
                      f'{write_body}'
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
    monkeypatch.setattr(sys, "argv", ["prog", "--job_name", "myjob", "--num_nodes", "4", "--wall_time", "02:00:00"])

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
    monkeypatch.setattr(sys, "argv", ["prog", "--script", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    result = dispatcher.submit_job()

    assert (campaign / "job.sh").read_text() == script.read_text()
    assert result.stdout == "from the custom outfile\n"
    # The script names its own output files, so sbatch is not told where to write
    assert "--output=" not in sbatch_log.read_text()


def test_submit_job_accepts_a_slurm_script_already_in_the_job_directory(tmp_path, monkeypatch, fake_scheduler):
    """
    Regression test: staging a script that already sits in the job directory
    copied it onto itself, and the resulting SameFileError aborted the
    submission before sbatch. That is the normal case on a cluster node.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    script = campaign / "job.sh"
    script.write_text("#!/bin/bash\nsrun ./my_app\n")
    sbatch_log = fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "--script", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    dispatcher.submit_job()

    assert script.read_text() == "#!/bin/bash\nsrun ./my_app\n"
    submission = sbatch_log.read_text().rstrip()
    assert f"cwd={campaign}" in submission
    assert submission.endswith("job.sh")


def test_construction_is_immune_to_host_process_argv(tmp_path, monkeypatch):
    """
    A dispatcher given an explicit argv ignores the surrounding program's
    command line, so a workflow keeps its own switches whatever they mean here.
    """
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--script", "host.sh", "--job_name", "host-job", "--port", "not-a-port"],
    )

    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path), argv=[])

    assert dispatcher.config["script"] is None
    assert dispatcher.config["job_name"] == "hpctools_job"


def test_submit_job_leaves_results_in_place_without_archiving_them(tmp_path, monkeypatch, fake_scheduler):
    """
    Results of a local job are already on this filesystem, so collect_results()
    has nothing to do, whatever collect patterns are configured.
    """
    fake_scheduler()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--collect", "all"])
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    dispatcher.submit_job("./my_app")

    assert list(tmp_path.glob("*.tar.gz")) == []


def test_submit_job_reports_a_failed_job(tmp_path, fake_scheduler):
    fake_scheduler(state="FAILED", exit_code="1:0")
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    result = dispatcher.submit_job("./my_app")

    assert not result.ok
    assert result.exit_code == 1


@pytest.mark.parametrize("state", ["CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"])
def test_submit_job_reports_a_terminal_state_that_is_not_completed(tmp_path, fake_scheduler, state):
    """
    Regression test: sacct reports ExitCode 0:0 for a job that ended without
    running, such as one cancelled while pending, so the sample was recorded as
    passing and passed.txt was written for a job that never produced anything.
    """
    fake_scheduler(state=state, exit_code="0:0")
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    result = dispatcher.submit_job("./my_app")

    assert not result.ok
    assert result.exit_code != 0


def test_submit_job_keeps_the_signal_from_a_killed_job(tmp_path, fake_scheduler):
    """A job killed by a signal reports that signal, not the generic failure code."""
    fake_scheduler(state="CANCELLED", exit_code="0:15")
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    assert dispatcher.submit_job("./my_app").exit_code == -15


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


def test_submit_job_with_a_script_naming_only_an_output_file(tmp_path, monkeypatch, fake_scheduler):
    """
    Regression test: the --error default was nested inside the --output one, so
    a script that set only --output left the error file name as None and
    get_output() raised AttributeError. SLURM merges the streams in that case.
    """
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --output=custom-%j.out\nsrun ./my_app\n")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "custom-123.out").write_text("both streams\n")
    sbatch_log = fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "--script", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    result = dispatcher.submit_job()

    assert result.stdout == "both streams\n"
    # The merged file is read once, not reported twice
    assert result.stderr == ""
    assert "--error=" not in sbatch_log.read_text()


def test_submit_job_with_a_script_naming_only_an_error_file(tmp_path, monkeypatch, fake_scheduler):
    """The complementary case: sbatch is told where to put stdout, and only that."""
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --error=custom-%j.err\nsrun ./my_app\n")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "slurm.out").write_text("job stdout\n")
    (campaign / "custom-123.err").write_text("job stderr\n")
    sbatch_log = fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "--script", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    result = dispatcher.submit_job()

    assert result.stdout == "job stdout\n"
    assert result.stderr == "job stderr\n"
    assert "--output=slurm.out" in sbatch_log.read_text()
    assert "--error=" not in sbatch_log.read_text()


def test_submit_job_names_the_output_files_on_every_submission(tmp_path, fake_scheduler):
    """
    Regression test: the output file names were remembered on the manager, so
    the second sbatch was issued with no --output/--error and the job wrote to
    SLURM's own slurm-<jobid>.out while get_output() still read slurm.out.
    """
    sbatch_log = fake_scheduler()
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path / "campaign"))

    dispatcher.submit_job("./my_app")
    dispatcher.submit_job("./my_app")

    assert sbatch_log.read_text().count("--output=slurm.out") == 2
    assert sbatch_log.read_text().count("--error=slurm.err") == 2


def test_submit_job_reads_output_from_a_reused_dispatcher(tmp_path, fake_scheduler):
    """
    Regression test, end to end: a workflow reuses one dispatcher and gives each
    sample its own run directory. The second job used to be submitted without
    --output, so it wrote to SLURM's slurm-<jobid>.out while get_output() read
    the slurm.out that was never created, and the sample came back with no output.
    """
    campaign = tmp_path / "campaign"
    fake_scheduler(write_output=True)
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    dispatcher.submit_job("./my_app", run_directory=str(campaign / "run_0"))
    second = dispatcher.submit_job("./my_app", run_directory=str(campaign / "run_1"))

    assert "run_1" in second.stdout


def test_submit_job_with_a_script_keeps_its_own_output_files_on_every_submission(
    tmp_path, monkeypatch, fake_scheduler
):
    """The configured-script path stays merged-stream correct when submitted more than once."""
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n#SBATCH --output=custom-%j.out\nsrun ./my_app\n")
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "custom-123.out").write_text("both streams\n")
    sbatch_log = fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "--script", str(script)])
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    dispatcher.submit_job()
    second = dispatcher.submit_job()

    assert second.stdout == "both streams\n"
    assert second.stderr == ""
    assert "--output=" not in sbatch_log.read_text()
    assert "--error=" not in sbatch_log.read_text()


def test_submit_job_raises_when_given_neither_a_command_nor_a_script(dispatcher):
    with pytest.raises(ValueError, match="base command or a SLURM script"):
        dispatcher.submit_job()


def test_submit_job_describes_the_job_from_the_configuration(tmp_path, monkeypatch, fake_scheduler):
    fake_scheduler()
    monkeypatch.setattr(sys, "argv", ["prog", "--job_name", "myjob", "--num_nodes", "4", "--wall_time", "02:00:00"])
    campaign = tmp_path / "campaign"
    dispatcher = LocalDispatcher(campaign_directory=str(campaign))

    dispatcher.submit_job("./my_app")

    script = (campaign / "myjob_slurm.sh").read_text()
    assert "#SBATCH --job-name=myjob" in script
    assert "#SBATCH --nodes=4" in script
    assert "#SBATCH --time=02:00:00" in script


def test_remove_dir_deletes_a_directory_tree(tmp_path, dispatcher):
    target = tmp_path / "iteration_0"
    (target / "run_0").mkdir(parents=True)
    (target / "run_0" / "out.txt").write_text("x")

    dispatcher.remove_dir(str(target))

    assert not target.exists()


def test_remove_dir_of_a_missing_directory_does_not_raise(tmp_path, dispatcher):
    dispatcher.remove_dir(str(tmp_path / "missing"))


def test_remove_dir_reports_a_failure(tmp_path, dispatcher):
    """
    Regression test: this used shutil.rmtree(ignore_errors=True) and logged
    success regardless, so the local and remote managers disagreed about
    whether a failed removal is an error.
    """
    target = tmp_path / "locked"
    (target / "child").mkdir(parents=True)
    tmp_path.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="Failed to remove directory"):
            dispatcher.remove_dir(str(target))
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.mpi_skip
def test_concurrent_submit_jobs_do_not_interfere(tmp_path, fake_scheduler):
    """
    The docs say evaluation_concurrency > 1 is fine with a LocalDispatcher.
    Concurrent evaluations run in separate worker processes with their own run
    directory, so each submits, polls, and reads back its own job.
    """
    sbatch_log = fake_scheduler()
    campaign = tmp_path / "campaign"
    run_dirs = []
    for name in ("run_0", "run_1"):
        run_dir = campaign / name
        run_dir.mkdir(parents=True)
        (run_dir / "slurm.out").write_text(f"{name} stdout\n")
        run_dirs.append(run_dir)

    context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        results = list(executor.map(
            _submit_in_worker,
            [str(campaign)] * len(run_dirs),
            [str(d) for d in run_dirs],
        ))

    assert [r.stdout for r in results] == ["run_0 stdout\n", "run_1 stdout\n"]
    for run_dir in run_dirs:
        assert (run_dir / "hpctools_job_slurm.sh").is_file()
        assert f"cwd={run_dir}" in sbatch_log.read_text()


def test_dispatcher_survives_a_pickle_round_trip(tmp_path):
    """Spawned workers, which the VI workflows use, receive the dispatcher by pickle."""
    dispatcher = LocalDispatcher(campaign_directory=str(tmp_path))

    revived = pickle.loads(pickle.dumps(dispatcher))

    assert revived.campaign_directory == str(tmp_path)
    assert revived.slurm is not None


def test_require_absolute_path_rejects_a_relative_path(dispatcher):
    """
    Regression test: this was a bare `assert`, which `python -O` strips, so an
    optimized run silently accepted a relative working directory.
    """
    with pytest.raises(ValueError, match="must provide an absolute path"):
        dispatcher.require_absolute_path("work")


def test_require_relative_path_is_a_no_op(tmp_path, dispatcher):
    """Local runs create directories on this machine, so absolute paths are fine."""
    dispatcher.require_relative_path(str(tmp_path))


def test_require_supported_concurrency_allows_concurrent_evaluation(dispatcher):
    dispatcher.require_supported_concurrency(4)

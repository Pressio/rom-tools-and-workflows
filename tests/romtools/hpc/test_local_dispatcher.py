import os
import sys

import numpy as np
import pytest

from romtools.hpc.dispatchers import LocalDispatcher


@pytest.fixture
def dispatcher():
    return LocalDispatcher()


def test_construction_is_immune_to_host_process_argv(monkeypatch):
    """
    Regression test: LocalDispatcher must not read the real process argv when
    building its Configuration, since it has no use for remote/SLURM CLI
    flags. Previously, embedding LocalDispatcher() inside another process
    (e.g. pytest, whose own flags like -q/-p collide with this schema) would
    raise SystemExit/argparse errors instead of constructing successfully.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "-q", "no:warnings", "-p", "not-a-port"])

    dispatcher = LocalDispatcher()

    assert dispatcher.get_config("job_name") == "hpctools_job"
    assert dispatcher.get_config("port") == 22


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


def test_dispatch_runs_command_and_captures_output(tmp_path, dispatcher):
    result = dispatcher.dispatch("echo hello", run_directory=str(tmp_path))

    assert result.ok
    assert result.stdout.strip() == "hello"


def test_dispatch_without_run_directory_runs_in_cwd(dispatcher):
    result = dispatcher.dispatch("echo hello")

    assert result.ok
    assert result.stdout.strip() == "hello"


def test_dispatch_reports_failure_without_raising(dispatcher):
    result = dispatcher.dispatch("exit 1")

    assert not result.ok
    assert result.exit_code == 1


def test_dispatch_runs_relative_to_run_directory(tmp_path, dispatcher):
    marker = tmp_path / "marker.txt"
    marker.write_text("present")

    result = dispatcher.dispatch("cat marker.txt", run_directory=str(tmp_path))

    assert result.ok
    assert result.stdout.strip() == "present"


def test_require_absolute_path_rejects_a_relative_path(dispatcher):
    with pytest.raises(AssertionError, match="must provide an absolute path"):
        dispatcher.require_absolute_path("work")


def test_require_relative_path_is_a_no_op(tmp_path, dispatcher):
    """Local runs create directories on this machine, so absolute paths are fine."""
    dispatcher.require_relative_path(str(tmp_path))


def test_require_supported_concurrency_allows_concurrent_evaluation(dispatcher):
    dispatcher.require_supported_concurrency(4)

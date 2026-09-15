import sys

from romtools.hpc.dispatchers import (
    LocalDispatcher,
    RemoteDispatcher,
    resolve_dispatcher,
    resolve_local_dispatcher,
)

from hpc_fakes import FakeConnection


def _make_remote_dispatcher(make_config):
    return RemoteDispatcher(connection=FakeConnection(), config=make_config())


def test_resolve_dispatcher_defaults_to_local():
    assert isinstance(resolve_dispatcher(None), LocalDispatcher)


def test_resolve_dispatcher_passes_through_supplied_dispatcher(make_config):
    remote = _make_remote_dispatcher(make_config)

    assert resolve_dispatcher(remote) is remote


def test_resolve_local_dispatcher_reuses_a_local_dispatcher():
    local = LocalDispatcher()

    assert resolve_local_dispatcher(local) is local


def test_resolve_local_dispatcher_replaces_a_remote_dispatcher(make_config):
    remote = _make_remote_dispatcher(make_config)

    resolved = resolve_local_dispatcher(remote)

    assert isinstance(resolved, LocalDispatcher)
    assert resolved is not remote


def test_resolve_local_dispatcher_defaults_to_local():
    assert isinstance(resolve_local_dispatcher(None), LocalDispatcher)


def test_resolve_dispatcher_ignores_the_host_process_command_line(monkeypatch):
    """
    Regression test: the fallback dispatcher parsed sys.argv, so a workflow run
    as "driver.py -c my_deck.yaml" loaded its own deck as HPC configuration, and
    "pytest --timeout=300" set the dispatcher's sacct timeout.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "-c", "/nonexistent/my_deck.yaml", "--hpc-job-name", "host-job"])

    assert resolve_dispatcher(None).get_config("job_name") == "hpctools_job"


def test_resolve_local_dispatcher_ignores_the_host_process_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-job-name", "host-job"])

    assert resolve_local_dispatcher(None).get_config("job_name") == "hpctools_job"

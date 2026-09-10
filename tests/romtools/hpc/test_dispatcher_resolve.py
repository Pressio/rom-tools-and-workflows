import sys
from unittest.mock import MagicMock

import romtools.hpc.dispatchers.base_dispatcher as base_dispatcher_module
from romtools.hpc.dispatchers import (
    LocalDispatcher,
    RemoteDispatcher,
    resolve_dispatcher,
    resolve_local_dispatcher,
)

from conftest import FakeConnection


def _make_remote_dispatcher(monkeypatch, make_config):
    stub_config = MagicMock()
    stub_config.to_dict.return_value = make_config()
    monkeypatch.setattr(base_dispatcher_module, "Configuration", MagicMock(return_value=stub_config))
    return RemoteDispatcher(connection=FakeConnection())


def test_resolve_dispatcher_defaults_to_local():
    assert isinstance(resolve_dispatcher(None), LocalDispatcher)


def test_resolve_dispatcher_passes_through_supplied_dispatcher(monkeypatch, make_config):
    remote = _make_remote_dispatcher(monkeypatch, make_config)

    assert resolve_dispatcher(remote) is remote


def test_resolve_local_dispatcher_reuses_a_local_dispatcher():
    local = LocalDispatcher()

    assert resolve_local_dispatcher(local) is local


def test_resolve_local_dispatcher_replaces_a_remote_dispatcher(monkeypatch, make_config):
    remote = _make_remote_dispatcher(monkeypatch, make_config)

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
    monkeypatch.setattr(sys, "argv", ["prog", "-c", "/nonexistent/my_deck.yaml", "--job_name", "host-job"])

    assert resolve_dispatcher(None).get_config("job_name") == "hpctools_job"


def test_resolve_local_dispatcher_ignores_the_host_process_command_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--job_name", "host-job"])

    assert resolve_local_dispatcher(None).get_config("job_name") == "hpctools_job"

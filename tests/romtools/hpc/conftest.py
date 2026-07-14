import io
import sys
import tarfile

import pytest

from romtools.hpc.connection import Result


class FakeConnection:
    """
    Test double for romtools.hpc.connection.Connection.

    Scripts canned Result responses per substring of the command issued,
    so RemoteDispatcher/Collector orchestration logic can be exercised
    without ever shelling out to ssh/scp.
    """

    def __init__(self, host="test-host", responses=None):
        self.host = host
        self.calls = []
        self.put_calls = []
        self.get_calls = []
        self.closed = False
        self._responses = responses or []

    def run(self, command):
        self.calls.append(command)
        for matcher, result in self._responses:
            if matcher in command:
                return result() if callable(result) else result
        return Result(stdout="", stderr="", exited=0)

    def put(self, local, remote):
        self.put_calls.append((local, remote))

    def get(self, remote, local):
        self.get_calls.append((remote, local))

    def close(self):
        self.closed = True


class ArchiveFakeConnection(FakeConnection):
    """
    FakeConnection whose get() writes a real (small, valid) tar.gz archive to
    the requested local path, so Collector's local extraction logic can run
    against a real file without ever touching a real remote host.
    """

    def get(self, remote, local):
        super().get(remote, local)
        with tarfile.open(local, "w:gz") as tar:
            data = b"payload"
            info = tarfile.TarInfo(name="result.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def fake_connection():
    return FakeConnection()


@pytest.fixture
def make_config():
    def _make(**overrides):
        config = {
            "remote": "test-host",
            "user": "test-user",
            "port": 22,
            "script": None,
            "job_name": "test_job",
            "num_nodes": 1,
            "tasks_per_node": 1,
            "wall_time": "00:01:00",
            "partition": "short",
            "account": "test-account",
            "poll_interval": 0,
            "remote_root": "hpctools_campaigns",
            "debug": False,
            "collect": None,
            "user_defined": {},
        }
        config.update(overrides)
        return config

    return _make


@pytest.fixture(autouse=True)
def _isolate_argv(request, monkeypatch):
    """
    Configuration() unconditionally parses real sys.argv, and its schema
    reuses flags (-p, -n) that collide with common pytest/xdist flags.
    Pin argv to a minimal value for every test in this directory except
    test_configuration.py, which manages argv explicitly.
    """
    if "test_configuration" in request.node.nodeid:
        return
    monkeypatch.setattr(sys, "argv", ["prog"])

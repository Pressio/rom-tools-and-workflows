import io
import os
import shutil
import subprocess
import sys
import tarfile
import time

import pytest

from romtools.hpc.connection import Result
from romtools.hpc.configuration import Configuration


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

        # fallbacks
        if "date -d" in command:
            Result(stdout=str(time.time()), stderr="", exit_code=0)

        return Result(stdout="", stderr="", exit_code=0)

    def put(self, local, remote):
        self.put_calls.append((local, remote))

    def get(self, remote, local):
        self.get_calls.append((remote, local))

    def close(self):
        self.closed = True


class LocalShellConnection(FakeConnection):
    """
    FakeConnection that runs commands and transfers files against a local
    directory standing in for the remote host's login directory, so the whole
    call() round trip can be exercised without a real remote host.
    """

    def __init__(self, root, **kwargs):
        super().__init__(**kwargs)
        self.root = str(root)

    def run(self, command):
        self.calls.append(command)
        res = subprocess.run(
            ["bash", "-c", command],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return Result(res.stdout, res.stderr, res.returncode)

    def put(self, local, remote):
        self.put_calls.append((local, remote))
        target = self.__resolve(remote)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(local, target)

    def get(self, remote, local):
        self.get_calls.append((remote, local))
        shutil.copy2(self.__resolve(remote), local)

    def __resolve(self, remote):
        return remote if os.path.isabs(remote) else os.path.join(self.root, remote)


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
        config = Configuration.defaults()
        config.remote = "test-host"
        config.user = "test-user"
        config.timeout = 0
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    return _make


@pytest.fixture(autouse=True)
def _isolate_argv(request, monkeypatch):
    """
    Configuration() reads real sys.argv unless given an explicit list, so
    pytest's own command line would otherwise leak into these tests. Pin argv
    to a minimal value for every test in this directory except
    test_configuration.py, which manages argv explicitly.
    """
    if "test_configuration" in request.node.nodeid:
        return
    monkeypatch.setattr(sys, "argv", ["prog"])

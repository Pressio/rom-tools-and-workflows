import os
import re
import shutil
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

import romtools.hpc.dispatchers.base_dispatcher as base_dispatcher_module
from romtools.hpc.connection import Result
from romtools.hpc.dispatchers import LocalDispatcher, RemoteDispatcher
from romtools.hpc.dispatchers.call_runner import (
    build_call_runner,
    pack,
    resolve_target,
    unpack,
)
from romtools.hpc.dispatchers.caller import BaseCaller, build_call_command

from conftest import FakeConnection

MODEL_SOURCE = '''
import numpy as np

def scale(values, factor=1.0):
    return np.asarray(values) * factor

def summarize(values):
    return {"mean": float(np.mean(values)), "shape": np.asarray(values).shape}

def explode():
    raise ValueError("model blew up")

class Model:
    @staticmethod
    def double(value):
        return 2 * value
'''


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


def _make_remote_dispatcher(monkeypatch, config, connection):
    stub_config = MagicMock()
    stub_config.to_dict.return_value = dict(config)
    monkeypatch.setattr(base_dispatcher_module, "Configuration", MagicMock(return_value=stub_config))
    return RemoteDispatcher(connection=connection)


@pytest.fixture
def staged_model(tmp_path, request):
    """Write the sample model into tmp_path under a name unique to this test."""
    module_name = "call_model_" + re.sub(r"\W", "_", request.node.name)
    (tmp_path / f"{module_name}.py").write_text(MODEL_SOURCE)
    yield module_name
    sys.modules.pop(module_name, None)


@pytest.fixture
def remote_host(tmp_path, staged_model):
    """A fake remote login directory with the sample model in its run directory."""
    run_directory = tmp_path / "campaigns" / "run_00"
    run_directory.mkdir(parents=True)
    shutil.copy2(tmp_path / f"{staged_model}.py", run_directory)
    return tmp_path


# ----------------------------------------------------------------------
# Payload helpers
# ----------------------------------------------------------------------

def test_pack_and_unpack_round_trip_nested_structures():
    arrays = {}
    original = {
        "matrix": np.arange(6).reshape(2, 3),
        "pair": (np.array([1.0]), 2),
        "listed": [np.array([3, 4]), "text"],
    }

    restored = unpack(pack(original, arrays), arrays)

    assert np.array_equal(restored["matrix"], original["matrix"])
    assert isinstance(restored["pair"], tuple)
    assert np.array_equal(restored["pair"][0], original["pair"][0])
    assert restored["pair"][1] == 2
    assert np.array_equal(restored["listed"][0], original["listed"][0])
    assert restored["listed"][1] == "text"


def test_pack_converts_numpy_scalars_to_python_scalars():
    packed = pack({"value": np.float64(1.5)}, {})

    assert packed == {"value": 1.5}
    assert isinstance(packed["value"], float)


def test_build_call_runner_is_valid_python_and_shares_the_wire_format():
    runner = build_call_runner()

    compile(runner, "runner.py", "exec")
    assert runner.count("def pack(") == 1
    assert runner.count("def unpack(") == 1
    assert runner.count("def resolve_target(") == 1


# ----------------------------------------------------------------------
# Target resolution
# ----------------------------------------------------------------------

def test_resolve_target_rejects_a_target_without_a_qualname():
    with pytest.raises(ValueError, match="module:qualname"):
        resolve_target("mymodel.evaluate")


def test_resolve_target_reports_the_underlying_import_error():
    with pytest.raises(ImportError, match="no_such_module_anywhere"):
        resolve_target("no_such_module_anywhere:evaluate")


def test_resolve_target_finds_a_nested_attribute(tmp_path, staged_model, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))

    assert resolve_target(f"{staged_model}:Model.double")(3) == 6


# ----------------------------------------------------------------------
# LocalDispatcher.call
# ----------------------------------------------------------------------

def test_local_call_runs_a_target_staged_in_the_run_directory(tmp_path, staged_model):
    dispatcher = LocalDispatcher()

    result = dispatcher.call(
        f"{staged_model}:scale",
        np.array([1.0, 2.0]),
        factor=3.0,
        run_directory=str(tmp_path),
    )

    assert np.array_equal(result, np.array([3.0, 6.0]))


def test_local_call_restores_the_working_directory_and_sys_path(tmp_path, staged_model):
    dispatcher = LocalDispatcher()
    original_directory = os.getcwd()
    original_path = list(sys.path)

    dispatcher.call(f"{staged_model}:scale", np.array([1.0]), run_directory=str(tmp_path))

    assert os.getcwd() == original_directory
    assert sys.path == original_path


def test_local_call_restores_the_working_directory_when_the_target_raises(tmp_path, staged_model):
    dispatcher = LocalDispatcher()
    original_directory = os.getcwd()

    with pytest.raises(ValueError, match="model blew up"):
        dispatcher.call(f"{staged_model}:explode", run_directory=str(tmp_path))

    assert os.getcwd() == original_directory


def test_local_call_without_a_run_directory_uses_the_current_directory(tmp_path, staged_model, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    dispatcher = LocalDispatcher()

    assert dispatcher.call(f"{staged_model}:Model.double", 4) == 8


def test_local_call_rejects_a_malformed_target(tmp_path):
    dispatcher = LocalDispatcher()

    with pytest.raises(ValueError, match="module:qualname"):
        dispatcher.call("mymodel.evaluate", run_directory=str(tmp_path))


# ----------------------------------------------------------------------
# Remote call command
# ----------------------------------------------------------------------

def test_call_command_paths_are_relative_to_the_run_directory():
    cmd = build_call_command(None, "python3", ".call_id", "mymodel:evaluate")

    assert "python3 .call_id/runner.py mymodel:evaluate" in cmd
    assert ".call_id/input.json" in cmd
    assert "hpctools_campaigns" not in cmd


def test_call_command_includes_the_python_setup():
    cmd = build_call_command("module load python", "python3", ".call_id", "mymodel:evaluate")

    assert "module load python" in cmd


def test_call_command_is_skipped_entirely_when_the_cd_fails(tmp_path):
    """
    Regression test: errexit used to be reachable only through the caller's
    `cd`, so a failed `cd` skipped `set -e` but still ran the python setup and
    the runner in the login directory.
    """
    cmd = build_call_command("touch setup_ran", "true", ".call_id", "mymodel:evaluate")

    res = subprocess.run(
        ["bash", "-c", f"cd missing_directory && {cmd}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert not (tmp_path / "setup_ran").exists()


# ----------------------------------------------------------------------
# RemoteDispatcher.call
# ----------------------------------------------------------------------

def test_remote_call_round_trips_arrays_through_a_relative_remote_root(monkeypatch, make_config, remote_host, staged_model):
    """
    Regression test: remote_root defaults to a relative path, and call() used to
    hand already-resolved paths back to put/get, which resolve again - staging
    files under a doubled prefix while the command used the single one.
    """
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    result = dispatcher.call(
        f"{staged_model}:scale",
        np.array([1.0, 2.0]),
        factor=3.0,
        run_directory="run_00",
    )

    assert np.array_equal(result, np.array([3.0, 6.0]))


def test_remote_call_round_trips_tuples_and_scalars(monkeypatch, make_config, remote_host, staged_model):
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    result = dispatcher.call(f"{staged_model}:summarize", np.array([1.0, 3.0]), run_directory="run_00")

    assert result == {"mean": 2.0, "shape": (2,)}


def test_remote_call_imports_modules_staged_in_the_run_directory(monkeypatch, make_config, remote_host, staged_model):
    """
    Regression test: the runner is executed by path, so its own staging
    directory - not the run directory - led sys.path, leaving user modules
    staged in the run directory unimportable.
    """
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    assert dispatcher.call(f"{staged_model}:Model.double", 21, run_directory="run_00") == 42


def test_remote_call_without_a_run_directory_uses_the_remote_root(monkeypatch, make_config, remote_host, staged_model):
    shutil.copy2(remote_host / f"{staged_model}.py", remote_host / "campaigns")
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    assert dispatcher.call(f"{staged_model}:Model.double", 5) == 10


def test_remote_call_cleans_up_its_staging_directory(monkeypatch, make_config, remote_host, staged_model):
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    dispatcher.call(f"{staged_model}:Model.double", 1, run_directory="run_00")

    run_directory = remote_host / "campaigns" / "run_00"
    assert not [entry for entry in os.listdir(run_directory) if entry.startswith(".dispatcher_call_")]


def test_remote_call_cleans_up_when_the_target_fails(monkeypatch, make_config, remote_host, staged_model):
    """
    Regression test: cleanup ran only on the success path, so every failed call
    leaked a hidden staging directory in the remote run directory.
    """
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    with pytest.raises(RuntimeError):
        dispatcher.call(f"{staged_model}:explode", run_directory="run_00")

    run_directory = remote_host / "campaigns" / "run_00"
    assert not [entry for entry in os.listdir(run_directory) if entry.startswith(".dispatcher_call_")]


def test_remote_call_failure_surfaces_the_remote_traceback(monkeypatch, make_config, remote_host, staged_model):
    """
    Regression test: the diagnostic branch was unreachable, because the command
    was issued through a helper that raised on a non-zero exit code first.
    """
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command=sys.executable)
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    with pytest.raises(RuntimeError, match="model blew up"):
        dispatcher.call(f"{staged_model}:explode", run_directory="run_00")


def test_remote_call_reports_a_missing_python_command(monkeypatch, make_config, remote_host, staged_model):
    conn = LocalShellConnection(remote_host)
    config = make_config(remote_root="campaigns", python_command="no_such_python")
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    with pytest.raises(RuntimeError, match="no_such_python"):
        dispatcher.call(f"{staged_model}:scale", np.array([1.0]), run_directory="run_00")


def test_remote_call_runs_the_python_setup_first(monkeypatch, make_config, remote_host, staged_model):
    conn = LocalShellConnection(remote_host)
    config = make_config(
        remote_root="campaigns",
        python_command="$CALL_PYTHON",
        python_setup=f"export CALL_PYTHON={sys.executable}",
    )
    dispatcher = _make_remote_dispatcher(monkeypatch, config, conn)

    assert dispatcher.call(f"{staged_model}:Model.double", 3, run_directory="run_00") == 6


# ----------------------------------------------------------------------
# Dispatcher wiring
# ----------------------------------------------------------------------

def test_base_caller_has_no_execution_strategy():
    with pytest.raises(NotImplementedError):
        BaseCaller().call("mymodel:evaluate")


def test_dispatchers_delegate_to_their_own_caller(monkeypatch, make_config):
    local = LocalDispatcher()
    remote = _make_remote_dispatcher(monkeypatch, make_config(), FakeConnection())

    local.caller = MagicMock()
    remote.caller = MagicMock()

    local.call("mymodel:evaluate", 1, run_directory="here", keyword=2)
    remote.call("mymodel:evaluate", 1, run_directory="there", keyword=2)

    local.caller.call.assert_called_once_with("mymodel:evaluate", 1, run_directory="here", keyword=2)
    remote.caller.call.assert_called_once_with("mymodel:evaluate", 1, run_directory="there", keyword=2)

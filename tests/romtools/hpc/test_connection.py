from unittest.mock import MagicMock, patch

import pytest

from romtools.hpc.connection import Connection, Result


def _make_connection(**kwargs):
    """Construct a Connection with the ControlMaster handshake mocked out."""
    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        conn = Connection(**kwargs)
    return conn


def test_result_ok_reflects_exit_code():
    assert Result("out", "err", 0).ok is True
    assert Result("out", "err", 1).ok is False


def test_init_issues_control_master_handshake():
    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        Connection(host="myhost", user="alice", port=2222, persist_seconds=60)

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "ssh"
    assert "ControlMaster=yes" in cmd
    assert "ControlPersist=60" in cmd
    assert "-p" in cmd
    assert cmd[cmd.index("-p") + 1] == "2222"
    assert "alice@myhost" in cmd
    assert cmd[-1] == "-N"


def test_target_omits_user_when_not_provided():
    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        conn = Connection(host="myhost")

    assert conn.target == "myhost"
    assert conn.port == 22


def test_run_builds_ssh_command_and_wraps_result():
    conn = _make_connection(host="myhost", user="alice", port=22)

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="output", stderr="", returncode=0)
        result = conn.run("echo hi")

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "ssh"
    assert "alice@myhost" in cmd
    assert cmd[-1] == "echo hi"
    assert result.ok
    assert result.stdout == "output"


def test_get_builds_scp_command_and_creates_local_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    conn = _make_connection(host="myhost", user="alice", port=2222)

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        conn.get("/remote/path/file.txt", "local/dir/file.txt")

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "scp"
    assert "-P" in cmd
    assert cmd[cmd.index("-P") + 1] == "2222"
    assert "alice@myhost:/remote/path/file.txt" in cmd
    assert "local/dir/file.txt" in cmd
    assert (tmp_path / "local" / "dir").is_dir()


def test_get_defaults_local_path_to_remote_basename(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    conn = _make_connection(host="myhost")

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        conn.get("/remote/path/file.txt")

    cmd = mock_run.call_args.args[0]
    assert cmd[-1] == "file.txt"


def test_get_raises_on_scp_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    conn = _make_connection(host="myhost")

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="no such file")
        with pytest.raises(RuntimeError, match="SCP failed"):
            conn.get("/remote/missing.txt", "local.txt")


def test_put_builds_scp_command_with_default_remote_basename():
    conn = _make_connection(host="myhost", user="alice", port=22)

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        conn.put("/local/script.sh")

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "scp"
    assert "/local/script.sh" in cmd
    assert "alice@myhost:script.sh" in cmd


def test_put_raises_on_scp_failure():
    conn = _make_connection(host="myhost")

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="permission denied")
        with pytest.raises(RuntimeError, match="SCP failed"):
            conn.put("/local/script.sh", "/remote/script.sh")


def test_local_runs_command_via_shell_not_ssh():
    conn = _make_connection(host="myhost")

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="ok", stderr="", returncode=0)
        result = conn.local("echo hi")

    args, kwargs = mock_run.call_args
    assert args[0] == "echo hi"
    assert kwargs.get("shell") is True
    assert result.ok
    assert result.stdout == "ok"


def test_close_issues_ssh_exit():
    conn = _make_connection(host="myhost", user="alice")

    with patch("romtools.hpc.connection.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        conn.close()

    cmd = mock_run.call_args.args[0]
    assert cmd[0] == "ssh"
    assert "-O" in cmd
    assert cmd[cmd.index("-O") + 1] == "exit"
    assert "alice@myhost" in cmd

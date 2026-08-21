import io
import tarfile
import subprocess

import pytest

from romtools.hpc.connection import Result
from romtools.hpc.util.file_transfer import create_tarball, safe_extract_tar, validate_file_patterns

from conftest import ArchiveFakeConnection, FakeConnection


def run_local_bash(cmd: str) -> Result:
    res = subprocess.run(
        ["bash", "-c", cmd],
        cwd=".",
        capture_output=True,
        text=True
    )
    return Result(res.stdout, res.stderr, res.returncode)


# ----------------------------------------------------------------------------
# Pattern validation (Collector.__init__ -> __validate)
# ----------------------------------------------------------------------------


def test_none_yields_no_patterns():
    patterns = validate_file_patterns(None)

    assert patterns is None


def test_valid_patterns_are_kept():
    patterns = validate_file_patterns(["*.log", "results/"])

    assert patterns == ["*.log", "results/"]


def test_pattern_starting_with_dash_is_rejected():
    with pytest.raises(ValueError, match="may not begin with '-'"):
        validate_file_patterns(["-bad"])


def test_pattern_with_newline_is_rejected():
    with pytest.raises(ValueError, match="forbidden characters"):
        validate_file_patterns(["bad\npattern"])


def test_pattern_with_unsupported_characters_is_rejected():
    with pytest.raises(ValueError, match="unsupported characters"):
        validate_file_patterns(["bad; rm -rf /"])


def test_all_blank_patterns_raises():
    with pytest.raises(ValueError, match="no valid patterns"):
        validate_file_patterns(["   ", ""])


# ----------------------------------------------------------------------------
# create_tarball
# ----------------------------------------------------------------------------


def test_create_tarball_skips_collection_when_collect_is_none(tmp_path, monkeypatch):
    """
    Note: despite the collect_results/__pack_remote_results docstrings claiming
    collect=None retrieves the entire run directory, the implementation does
    the opposite -- collect=None means "nothing was requested" and collection
    is skipped entirely. This test documents the actual behavior; use the
    "all"/"*"/"any"/"everything" keyword to opt into whole-directory
    collection (see test_collect_results_with_all_keyword_packs_entire_directory).
    """
    monkeypatch.chdir(tmp_path)
    conn = ArchiveFakeConnection()
    with pytest.raises(ValueError):
        create_tarball(lambda _: None, lambda cmd: conn.run(cmd), "hpctools", "myjob.tar.gz", None)

    assert conn.calls == []
    assert conn.get_calls == []
    assert not (tmp_path / "hpctools").exists()


def test_create_tarball_with_all_keyword_packs_entire_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = ArchiveFakeConnection(
        responses=[
            ("tar -czf", Result("", "", 0)),
            ("rm -f", Result("", "", 0)),
        ]
    )
    create_tarball(lambda _: None, lambda cmd: conn.run(cmd), "campaigns", "myjob.tar.gz", ["all"])

    pack_cmd = next(c for c in conn.calls if c.startswith("tar -czf"))
    assert pack_cmd.endswith(" .")


def test_create_tarball_with_specific_patterns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    conn = ArchiveFakeConnection(
        responses=[
            ("for f in", Result("matched1.log\n", "", 0)),
            ("test -e", Result("", "", 0)),
            ("tar -czf", Result("", "", 0)),
            ("rm -f", Result("", "", 0)),
        ]
    )
    create_tarball(lambda _: None, lambda cmd: conn.run(cmd), "campaigns/hpctools", "myjob.tar.gz", ["*.log", "results.txt"])

    pack_cmd = next(c for c in conn.calls if c.startswith("tar -czf"))
    assert "matched1.log" in pack_cmd
    assert "results.txt" in pack_cmd


def test_create_tarball_raises_if_no_patterns_matched(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])

    with pytest.raises(FileNotFoundError):
        create_tarball(lambda _: None, lambda cmd: conn.run(cmd), "campaigns", "myjob.tar.gz", ["results.txt"])


def test_create_tarball_raises_when_pack_command_fails(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("tar -czf", Result("", "disk full", 1))])

    with pytest.raises(RuntimeError, match="disk full"):
        create_tarball(lambda _: None, lambda cmd: conn.run(cmd), "campaigns/hpctools", "myjob.tar.gz", ["all"])


# ----------------------------------------------------------------------------
# Safe tar extraction
# ----------------------------------------------------------------------------


def test_safe_extract_tar_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.tar.gz"
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    res = safe_extract_tar(run_local_bash, str(archive_path), str(target_dir))
    assert not res.ok

def test_safe_extract_tar_extracts_files_and_removes_archive(tmp_path):
    archive_path = tmp_path / "archive.tar.gz"
    target_dir = tmp_path / "target"

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            data = b"hello world"

            dir_info = tarfile.TarInfo(name="subdir/")
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            tar.addfile(dir_info)

            file_info = tarfile.TarInfo(name="subdir/hello.txt")
            file_info.size = len(data)
            file_info.mode = 0o644
            tar.addfile(file_info, io.BytesIO(data))

        res = safe_extract_tar(run_local_bash, str(archive_path), str(target_dir))

        assert res.ok
        assert (target_dir / "subdir" / "hello.txt").read_bytes() == data
        assert not archive_path.exists()

    finally:
        extracted_subdir = target_dir / "subdir"
        if extracted_subdir.exists():
            extracted_subdir.chmod(0o755)

def test_safe_extract_tar_rejects_nested_path_traversal(tmp_path):
    archive_path = tmp_path / "evil.tar.gz"
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    with tarfile.open(archive_path, "w:gz") as tar:
        data = b"pwned"
        info = tarfile.TarInfo(name="safe/../evil.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    res = safe_extract_tar(run_local_bash, str(archive_path), str(target_dir))

    assert not res.ok
    assert not (tmp_path / "evil.txt").exists()
    assert archive_path.exists()

import io
import tarfile

import pytest

from romtools.hpc.connection import Result
from romtools.hpc.util.file_transfer import pack_results, safe_extract_tar, local_cmd, validate
from subprocess import CalledProcessError

from conftest import ArchiveFakeConnection, FakeConnection


# ----------------------------------------------------------------------------
# Pattern validation (Collector.__init__ -> __validate)
# ----------------------------------------------------------------------------


def test_collect_none_yields_no_patterns():
    patterns = validate(None)

    assert patterns is None


def test_collect_valid_patterns_are_kept():
    patterns = validate(["*.log", "results/"])

    assert patterns == ["*.log", "results/"]


def test_collect_pattern_starting_with_dash_is_rejected():
    with pytest.raises(ValueError, match="may not begin with '-'"):
        validate(["-bad"])


def test_collect_pattern_with_newline_is_rejected():
    with pytest.raises(ValueError, match="forbidden characters"):
        validate(["bad\npattern"])


def test_collect_pattern_with_unsupported_characters_is_rejected():
    with pytest.raises(ValueError, match="unsupported characters"):
        validate(["bad; rm -rf /"])


def test_collect_all_blank_patterns_raises():
    with pytest.raises(ValueError, match="no valid patterns"):
        validate(["   ", ""])


# ----------------------------------------------------------------------------
# collect_results orchestration
# ----------------------------------------------------------------------------


def test_collect_results_skips_collection_when_collect_is_none(tmp_path, monkeypatch, make_config):
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
    pack_results(lambda _: None, lambda cmd: conn.run(cmd), "hpctools", "myjob.tar.gz", None)

    assert conn.calls == []
    assert conn.get_calls == []
    assert not (tmp_path / "hpctools").exists()


def test_collect_results_with_all_keyword_packs_entire_directory(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = ArchiveFakeConnection(
        responses=[
            ("tar -czf", Result("", "", 0)),
            ("rm -f", Result("", "", 0)),
        ]
    )
    pack_results(lambda _: None, lambda cmd: conn.run(cmd), "campaigns", "myjob.tar.gz", ["all"])

    pack_cmd = next(c for c in conn.calls if c.startswith("tar -czf"))
    assert pack_cmd.endswith(" .")


def test_collect_results_with_specific_patterns(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = ArchiveFakeConnection(
        responses=[
            ("for f in", Result("matched1.log\n", "", 0)),
            ("test -e", Result("", "", 0)),
            ("tar -czf", Result("", "", 0)),
            ("rm -f", Result("", "", 0)),
        ]
    )
    pack_results(lambda _: None, lambda cmd: conn.run(cmd), "campaigns/hpctools", "myjob.tar.gz", ["*.log", "results.txt"])

    pack_cmd = next(c for c in conn.calls if c.startswith("tar -czf"))
    assert "matched1.log" in pack_cmd
    assert "results.txt" in pack_cmd


def test_collect_results_raises_if_no_patterns_matched(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])

    with pytest.raises(RuntimeError, match="No files matched"):
        pack_results(lambda _: None, lambda cmd: conn.run(cmd), "campaigns", "myjob.tar.gz", ["results.txt"])


def test_collect_results_raises_when_pack_command_fails(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("tar -czf", Result("", "disk full", 1))])

    with pytest.raises(RuntimeError, match="disk full"):
        pack_results(lambda _: None, lambda cmd: conn.run(cmd), "campaigns/hpctools", "myjob.tar.gz", ["all"])


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

    with pytest.raises(CalledProcessError):
        safe_extract_tar(local_cmd, str(archive_path), str(target_dir))

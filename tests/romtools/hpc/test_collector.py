import io
import tarfile

import pytest

from romtools.hpc.collector import Collector
from romtools.hpc.connection import Result

from conftest import ArchiveFakeConnection, FakeConnection


# ----------------------------------------------------------------------------
# Pattern validation (Collector.__init__ -> __validate)
# ----------------------------------------------------------------------------


def test_collect_none_yields_no_patterns(fake_connection, make_config):
    config = make_config(collect=None)

    collector = Collector(fake_connection, config, sampling_directory="hpctools")

    assert collector.patterns is None


def test_collect_valid_patterns_are_kept(fake_connection, make_config):
    config = make_config(collect=["*.log", "results/"])

    collector = Collector(fake_connection, config, sampling_directory="hpctools")

    assert collector.patterns == ["*.log", "results/"]


def test_collect_pattern_starting_with_dash_is_rejected(fake_connection, make_config):
    config = make_config(collect=["-bad"])

    with pytest.raises(ValueError, match="may not begin with '-'"):
        Collector(fake_connection, config, sampling_directory="hpctools")


def test_collect_pattern_with_newline_is_rejected(fake_connection, make_config):
    config = make_config(collect=["bad\npattern"])

    with pytest.raises(ValueError, match="forbidden characters"):
        Collector(fake_connection, config, sampling_directory="hpctools")


def test_collect_pattern_with_unsupported_characters_is_rejected(fake_connection, make_config):
    config = make_config(collect=["bad; rm -rf /"])

    with pytest.raises(ValueError, match="unsupported characters"):
        Collector(fake_connection, config, sampling_directory="hpctools")


def test_collect_all_blank_patterns_raises(fake_connection, make_config):
    config = make_config(collect=["   ", ""])

    with pytest.raises(ValueError, match="no valid patterns"):
        Collector(fake_connection, config, sampling_directory="hpctools")


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
    config = make_config(remote_root="campaigns", job_name="myjob", collect=None)
    collector = Collector(conn, config, sampling_directory="hpctools")

    collector.collect_results()

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
    config = make_config(remote_root="campaigns", job_name="myjob", collect=["all"])
    collector = Collector(conn, config, sampling_directory="hpctools")

    collector.collect_results()

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
    config = make_config(remote_root="campaigns", job_name="myjob", collect=["*.log", "results.txt"])
    collector = Collector(conn, config, sampling_directory="hpctools")

    collector.collect_results()

    pack_cmd = next(c for c in conn.calls if c.startswith("tar -czf"))
    assert "matched1.log" in pack_cmd
    assert "results.txt" in pack_cmd
    assert (tmp_path / "hpctools" / "result.txt").read_text() == "payload"


def test_collect_results_raises_if_no_patterns_matched(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("test -e", Result("", "", 1))])
    config = make_config(remote_root="campaigns", job_name="myjob", collect=["results.txt"])
    collector = Collector(conn, config, sampling_directory="hpctools")

    with pytest.raises(RuntimeError, match="No files matched"):
        collector.collect_results()


def test_collect_results_raises_when_pack_command_fails(tmp_path, monkeypatch, make_config):
    monkeypatch.chdir(tmp_path)
    conn = FakeConnection(responses=[("tar -czf", Result("", "disk full", 1))])
    config = make_config(remote_root="campaigns", job_name="myjob", collect=["all"])
    collector = Collector(conn, config, sampling_directory="hpctools")

    with pytest.raises(RuntimeError, match="disk full"):
        collector.collect_results()


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

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        Collector._Collector__safe_extract_tar(str(archive_path), str(target_dir))

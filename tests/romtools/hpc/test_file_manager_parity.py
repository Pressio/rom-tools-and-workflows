"""
Local and remote file managers must answer file writes the same way.

A model is written once and moved between dispatchers unchanged, so a write
that creates its parent directory on one host has to create it on the other,
and content has to arrive byte for byte either way.
"""

import numpy as np
import pytest

from romtools.hpc.components.file_manager import LocalFileManager, RemoteFileManager
from romtools.hpc.logger import Logger

from conftest import LocalShellConnection


@pytest.fixture(params=["local", "remote"])
def file_manager(request, tmp_path, monkeypatch):
    """The same contract, once against the filesystem and once through bash."""
    logger = Logger(False)
    if request.param == "local":
        monkeypatch.chdir(tmp_path)
        return LocalFileManager(config={}, logger=logger)

    root = tmp_path / "remote_root"
    root.mkdir()
    return RemoteFileManager(LocalShellConnection(tmp_path),
                             config={"remote_root": "remote_root"}, logger=logger)


def test_write_text_creates_missing_parent_directories(file_manager):
    """
    Regression test: only the local manager created parents, so a model that
    worked locally raised RuntimeError the moment it ran through SSH.
    """
    file_manager.write_text("work/nested/stats.txt", "elbo: 1.0\n")

    assert file_manager.read_text("work/nested/stats.txt") == "elbo: 1.0\n"


@pytest.mark.parametrize("content", [
    "trailing newline\n",
    "no trailing newline",
    "blank line between\n\nparagraphs\n",
    "quotes ' \" and $VAR and `cmd`\n",
    "a line reading __HPCTOOLS_FILE_EOF__\nand one after it\n",
    pytest.param("x" * 200_000 + "\n", id="larger-than-one-ssh-command"),
])
def test_write_text_round_trips_content_exactly(file_manager, content):
    """
    Regression test: the remote heredoc appended a newline the local writer did
    not, and truncated any content containing the heredoc delimiter. Content
    beyond one argv element then failed outright with E2BIG.
    """
    file_manager.write_text("out.txt", content)

    assert file_manager.read_text("out.txt") == content


def test_np_savetxt_creates_missing_parent_directories(file_manager):
    """Regression test: np_savetxt disagreed with write_text about the parent."""
    file_manager.np_savetxt("results/nested/array.txt", np.array([1, 2, 3]), fmt="%d")

    assert file_manager.read_text("results/nested/array.txt").split() == ["1", "2", "3"]


def test_np_savez_creates_missing_parent_directories(file_manager):
    """Regression test: the remote manager raised here instead of creating it."""
    file_manager.np_savez("arrays/nested/data", a=np.array([1, 2]))

    assert file_manager.path_exists("arrays/nested/data.npz")

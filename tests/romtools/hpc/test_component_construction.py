"""
Regression tests: the components that reach the filesystem used to default
`files` to None, so construction succeeded and failed much later with an
AttributeError deep inside a job. All three are only ever built with keywords,
so the argument is required.
"""

# Every call here deliberately omits the argument under test
# pylint: disable=missing-kwoa

import pytest

from conftest import FakeConnection
from romtools.hpc.components.caller import BaseCaller, RemoteCaller
from romtools.hpc.components.file_manager import BaseFileManager, LocalFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.components.transfer_manager import TransferManager
from romtools.hpc.connection import run_local_bash
from romtools.hpc.logger import Logger


def test_slurm_job_manager_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        SlurmJobManager(run_local_bash)


def test_transfer_manager_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        TransferManager(FakeConnection())


def test_remote_caller_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        RemoteCaller(FakeConnection())


@pytest.mark.parametrize("build", [
    lambda: BaseFileManager(),
    lambda: BaseCaller(),
    lambda: SlurmJobManager(run_local_bash, files=LocalFileManager()),
    lambda: TransferManager(FakeConnection(), files=LocalFileManager()),
])
def test_components_log_without_being_handed_a_logger(build):
    """
    Regression test: logger defaulted to None but was dereferenced on every
    path, so a component built without one raised AttributeError on its first
    message rather than at construction.
    """
    component = build()

    assert isinstance(component.logger, Logger)
    component.logger.debug("no logger was supplied")

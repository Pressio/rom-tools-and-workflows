import pytest

from conftest import FakeConnection
from romtools.hpc.components.caller import BaseCaller
from romtools.hpc.components.file_manager import BaseFileManager, LocalFileManager
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.components.transfer_manager import (
    BaseTransferManager,
    LocalTransferManager,
    RemoteTransferManager,
)
from romtools.hpc.connection import run_local_bash
from romtools.hpc.logger import Logger


@pytest.mark.parametrize("build", [
    lambda: BaseFileManager(),
    lambda: BaseCaller(),
    lambda: SlurmJobManager(run_local_bash, files=LocalFileManager()),
    lambda: BaseTransferManager(files=LocalFileManager()),
    lambda: LocalTransferManager(files=LocalFileManager()),
    lambda: RemoteTransferManager(FakeConnection(), files=LocalFileManager()),
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

import pytest

from hpc_fakes import FakeConnection
from romtools.hpc.components.caller import BaseCaller, LocalCaller
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
    lambda: LocalFileManager(),
    lambda: LocalCaller(files=LocalFileManager()),
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


@pytest.mark.parametrize("build", [
    lambda campaign: SlurmJobManager(run_local_bash, files=LocalFileManager(), campaign_directory=campaign),
    lambda campaign: BaseTransferManager(files=LocalFileManager(), campaign_directory=campaign),
])
def test_campaign_components_agree_on_the_default_directory(build):
    """
    Submitting a job and staging its inputs have to mean the same directory, or
    the job runs where its inputs are not.
    """
    component = build("campaign")

    assert component.job_directory("run_0") == "run_0"
    assert component.job_directory() == "campaign"


@pytest.mark.parametrize("base, missing", [
    (BaseFileManager, "resolve_path"),
    (BaseCaller, "call"),
])
def test_an_incomplete_component_is_rejected_at_construction(base, missing):
    """
    A helper that forgets one of its base's methods has to fail when the
    dispatcher is built, not part-way through a job that has already queued.
    """
    incomplete = type("Incomplete", (base,), {})

    with pytest.raises(TypeError, match=missing):
        incomplete()

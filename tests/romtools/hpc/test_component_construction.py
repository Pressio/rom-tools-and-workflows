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
from romtools.hpc.components.caller import RemoteCaller
from romtools.hpc.components.slurm_job_manager import SlurmJobManager
from romtools.hpc.components.transfer_manager import TransferManager
from romtools.hpc.connection import run_local_bash


def test_slurm_job_manager_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        SlurmJobManager(run_local_bash)


def test_transfer_manager_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        TransferManager(FakeConnection())


def test_remote_caller_requires_a_file_manager():
    with pytest.raises(TypeError, match="files"):
        RemoteCaller(FakeConnection())

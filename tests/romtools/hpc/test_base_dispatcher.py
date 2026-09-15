import pytest

from romtools.hpc.connection import Result
from romtools.hpc.dispatchers import BaseDispatcher


class SchedulerlessDispatcher(BaseDispatcher):
    """The smallest complete dispatcher: it runs commands but reaches no scheduler."""

    def run(self, cmd: str, run_directory: str = None) -> Result:
        return Result(cmd, "", 0)


def test_the_base_dispatcher_cannot_be_instantiated():
    """
    The base is a facade over helpers that only subclasses install, so
    constructing it directly would produce a dispatcher that cannot do anything.
    """
    with pytest.raises(TypeError, match="abstract"):
        BaseDispatcher()


def test_a_subclass_that_does_not_implement_run_cannot_be_instantiated():
    class RunlessDispatcher(BaseDispatcher):
        pass

    with pytest.raises(TypeError, match="run"):
        RunlessDispatcher()


def test_submit_job_without_a_scheduler_is_not_implemented():
    """
    Regression test: the base submit_job() used to have an empty body, so a
    dispatcher with no scheduler installed silently returned None instead of
    running anything.
    """
    with pytest.raises(NotImplementedError, match="no batch scheduler"):
        SchedulerlessDispatcher().submit_job("./my_app")

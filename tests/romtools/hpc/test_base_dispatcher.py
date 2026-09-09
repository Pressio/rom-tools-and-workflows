import pytest

from romtools.hpc.dispatchers import BaseDispatcher


def test_run_is_not_implemented():
    with pytest.raises(NotImplementedError):
        BaseDispatcher().run("echo hello")


def test_submit_job_without_a_scheduler_is_not_implemented():
    """
    Regression test: the base submit_job() used to have an empty body, so a
    dispatcher with no scheduler installed silently returned None instead of
    running anything.
    """
    with pytest.raises(NotImplementedError, match="no batch scheduler"):
        BaseDispatcher().submit_job("./my_app")


def test_collect_results_is_a_no_op_by_default():
    """Work that ran on this machine leaves its results in place."""
    BaseDispatcher().collect_results()

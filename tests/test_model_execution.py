"""Tests for the shared model executor."""
import pytest

from romtools.workflows.model_execution import ModelEvaluation, execute_models


class BlockingModel:
    def run_model(self, directory, parameters):
        return parameters.get("exit_code", 0)


class Handle:
    def __init__(self, ticks, status=0):
        self.ticks = ticks
        self.status = status

    def done(self):
        self.ticks -= 1
        return self.ticks <= 0

    def result(self):
        return self.status


class NonBlockingModel:
    blocking = False

    def __init__(self):
        self.handles = []
        self.max_active = 0

    def run_model(self, directory, parameters):
        handle = Handle(parameters["ticks"], parameters.get("exit_code", 0))
        self.handles.append(handle)
        return handle


def evaluations(*parameters):
    return [
        ModelEvaluation(f"run_{index}", parameter)
        for index, parameter in enumerate(parameters)
    ]


def test_blocking_default_and_order():
    samples = evaluations({}, {}, {})
    assert execute_models(BlockingModel(), samples) == samples


def test_blocking_failure():
    with pytest.raises(RuntimeError, match="run_0"):
        execute_models(BlockingModel(), evaluations({"exit_code": 2}))


def test_nonblocking_out_of_order_and_bounded_submission():
    model = NonBlockingModel()
    samples = evaluations({"ticks": 5}, {"ticks": 1}, {"ticks": 2})
    assert execute_models(model, samples, max_concurrent=2, poll_interval=0) == samples
    assert len(model.handles) == 3


def test_nonblocking_failure():
    with pytest.raises(RuntimeError, match="exit status 7"):
        execute_models(
            NonBlockingModel(),
            evaluations({"ticks": 1, "exit_code": 7}),
            poll_interval=0,
        )


def test_invalid_concurrency():
    with pytest.raises(ValueError):
        execute_models(BlockingModel(), evaluations({}), max_concurrent=0)

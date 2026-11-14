"""Tests for shared blocking and non-blocking model execution."""

import os

import pytest

from romtools.workflows.model_execution import ModelEvaluation, execute_models


class BlockingModel:
    def run_model(self, directory, parameters):
        self.value = parameters.get("x", 0)
        self.pid = os.getpid()
        return parameters.get("exit_code", 0)


def _blocking_output(model, evaluation):
    return model.pid, model.value


class Handle:
    def __init__(self, model, directory, parameters):
        self.model = model
        self.directory = directory
        self.ticks = parameters["ticks"]
        self.status = parameters.get("exit_code", 0)
        self.failure = parameters.get("failure")

    def done(self):
        if self.failure == "poll":
            raise LookupError("poll failed")
        self.ticks -= 1
        return self.ticks <= 0

    def result(self):
        self.model.active -= 1
        self.model.finished.append(self.directory)
        if self.failure == "result":
            raise LookupError("result failed")
        return self.status


class NonBlockingModel:
    blocking = False

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.finished = []
        self.pid = os.getpid()

    def run_model(self, directory, parameters):
        assert self.pid == os.getpid()
        if parameters.get("failure") == "submit":
            raise LookupError("submit failed")
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return Handle(self, directory, parameters)


def evaluations(*parameters):
    return [
        ModelEvaluation(f"run_{index}", parameter)
        for index, parameter in enumerate(parameters)
    ]


def test_blocking_default_and_order():
    samples = evaluations({}, {}, {})
    assert execute_models(BlockingModel(), samples) == samples


def test_blocking_worker_preserves_model_state_for_postprocessing():
    samples = evaluations({"x": 1}, {"x": 2})
    completed = execute_models(BlockingModel(), samples, 2, postprocess=_blocking_output)
    pids, values = zip(*(evaluation.output for evaluation in completed))
    assert values == (1, 2)
    assert all(pid != os.getpid() for pid in pids)


def test_blocking_failure():
    with pytest.raises(RuntimeError, match="run_0"):
        execute_models(BlockingModel(), evaluations({"exit_code": 2}))


def test_nonblocking_out_of_order_and_bounded_submission():
    model = NonBlockingModel()
    samples = evaluations({"ticks": 5}, {"ticks": 1}, {"ticks": 2})

    def postprocess(model, evaluation):
        assert evaluation.run_directory in model.finished
        return evaluation.run_directory

    completed = execute_models(
        model, samples, max_concurrent=2, poll_interval=0, postprocess=postprocess,
    )
    assert [sample.output for sample in completed] == ["run_0", "run_1", "run_2"]
    assert model.finished == ["run_1", "run_2", "run_0"]
    assert model.max_active == 2
    assert model.active == 0


@pytest.mark.parametrize(("failure", "error_type", "message"), [
    ("submit", LookupError, "submit failed"),
    ("poll", LookupError, "poll failed"),
    ("result", LookupError, "result failed"),
    ("status", RuntimeError, "exit status 7"),
])
def test_nonblocking_failure_drains_submitted_jobs(failure, error_type, message):
    model = NonBlockingModel()
    samples = evaluations(
        {"ticks": 10}, {"ticks": 1, "failure": failure, "exit_code": 7}, {"ticks": 1},
    )

    def postprocess(model, evaluation):
        pytest.fail("failed or drained jobs must not be postprocessed")

    with pytest.raises(error_type, match=message):
        execute_models(model, samples, 2, poll_interval=0, postprocess=postprocess)

    assert model.active == 0
    assert "run_0" in model.finished
    assert "run_2" not in model.finished


def test_nonblocking_can_finish_other_evaluations_after_failure():
    model = NonBlockingModel()
    samples = evaluations({"ticks": 3, "failure": "poll"}, {"ticks": 1}, {"ticks": 1})

    results = execute_models(model, samples, 2, poll_interval=0, return_exceptions=True)

    assert isinstance(results[0], LookupError)
    assert results[1:] == samples[1:]
    assert set(model.finished) == {"run_0", "run_1", "run_2"}
    assert model.active == 0


def test_invalid_concurrency():
    with pytest.raises(ValueError):
        execute_models(BlockingModel(), evaluations({}), max_concurrent=0)

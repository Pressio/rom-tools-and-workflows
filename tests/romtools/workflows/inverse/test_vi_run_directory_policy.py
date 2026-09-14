import inspect
import warnings

import numpy as np
import pytest

import romtools.workflows as workflows
from romtools.workflows.inverse import _inverse_utils
from romtools.workflows.inverse import vi_run_directory_policy as directory_policy


class _InMemoryQoiModel:
    def __init__(self):
        self.run_directories = []

    def populate_run_directory(self, run_directory, parameter_sample):
        del parameter_sample
        self.run_directories.append(run_directory)

    def run_model(self, run_directory, parameter_sample):
        del run_directory, parameter_sample
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        del run_directory
        return np.array([parameter_sample["theta"]])


def test_run_vi_iteration_can_skip_sample_directories(tmp_path):
    model = _InMemoryQoiModel()
    run_directory_base = f"{tmp_path}/run_"

    results = _inverse_utils.run_vi_iteration(
        model=model,
        observations=np.array([0.0]),
        run_directory_base=run_directory_base,
        parameter_names=["theta"],
        parameter_samples=np.array([[1.0], [2.0]]),
        evaluation_concurrency=1,
        create_run_directories=False,
    )

    assert not (tmp_path / "run_0").exists()
    assert not (tmp_path / "run_1").exists()
    assert model.run_directories == [
        f"{run_directory_base}0",
        f"{run_directory_base}1",
    ]
    np.testing.assert_allclose(results["qois"], np.array([[1.0, 2.0]]))


def test_run_vi_iteration_creates_sample_directories_by_default(tmp_path):
    model = _InMemoryQoiModel()
    run_directory_base = f"{tmp_path}/run_"

    _inverse_utils.run_vi_iteration(
        model=model,
        observations=np.array([0.0]),
        run_directory_base=run_directory_base,
        parameter_names=["theta"],
        parameter_samples=np.array([[1.0], [2.0]]),
        evaluation_concurrency=1,
    )

    assert (tmp_path / "run_0").is_dir()
    assert (tmp_path / "run_1").is_dir()


def test_default_directory_policy_warns(monkeypatch):
    monkeypatch.setattr(directory_policy, "_BASE_RUN_VI", lambda *args, **kwargs: "done")

    with pytest.warns(UserWarning, match="create_run_directories=True"):
        assert directory_policy.run_vi() == "done"


def test_disabled_directory_policy_does_not_warn(monkeypatch):
    observed_policy = []

    def _fake_run_vi(*args, **kwargs):
        del args, kwargs
        observed_policy.append(
            directory_policy._ACTIVE_CREATE_RUN_DIRECTORIES.get()
        )
        return "done"

    monkeypatch.setattr(directory_policy, "_BASE_RUN_VI", _fake_run_vi)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert directory_policy.run_vi(create_run_directories=False) == "done"

    assert caught == []
    assert observed_policy == [False]


def test_public_vi_signatures_expose_create_run_directories():
    for function in (
        workflows.run_vi,
        workflows.run_mf_vi,
        workflows.mf_vi_with_auto_rom,
    ):
        parameter = inspect.signature(function).parameters["create_run_directories"]
        assert parameter.default is True
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

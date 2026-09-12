import os

import numpy as np
import pytest

import romtools.workflows.inverse.eki_drivers as eki_module
import romtools.workflows.inverse.mf_eki_drivers as mf_module


class DummyParameterSpace:
    def get_names(self):
        return ["x"]

    def get_dimensionality(self):
        return 1

    def generate_samples(self, number_of_samples, seed=None):
        return np.linspace(0.0, 1.0, number_of_samples)[:, None]


def test_eki_caps_accepted_step_growth(monkeypatch, tmp_path):
    parameter_space = DummyParameterSpace()

    def fake_iteration(model, observations, run_directory_base, parameter_names,
                       parameter_samples, evaluation_concurrency, dispatcher):
        os.makedirs(os.path.dirname(run_directory_base), exist_ok=True)
        mean_parameter = float(np.mean(parameter_samples))
        qois = np.full((1, parameter_samples.shape[0]), mean_parameter)
        error = max(1.0e-12, 10.0 - mean_parameter)
        return {
            "qois": qois,
            "mean-qoi": np.array([mean_parameter]),
            "errors": np.full((1, parameter_samples.shape[0]), error),
        }

    monkeypatch.setattr(eki_module, "run_eki_iteration", fake_iteration)
    monkeypatch.setattr(
        eki_module,
        "compute_eki_update",
        lambda parameter_samples, *args, **kwargs: np.ones_like(parameter_samples),
    )

    workdir = tmp_path / "eki"
    eki_module.run_eki(
        model=object(),
        parameter_space=parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_eki_directory=str(workdir),
        ensemble_size=3,
        initial_step_size=0.6,
        max_step_size=1.0,
        step_size_growth_factor=2.0,
        relaxation_parameter=1.0,
        error_norm_tolerance=0.0,
        delta_params_tolerance=0.0,
        max_iterations=3,
    )

    with np.load(workdir / "iteration_1" / "restart.npz") as restart:
        assert float(restart["step_size"]) == pytest.approx(1.0)
    with np.load(workdir / "iteration_2" / "restart.npz") as restart:
        assert float(restart["step_size"]) == pytest.approx(1.0)


def test_eki_rejects_initial_step_above_max(tmp_path):
    with pytest.raises(AssertionError, match="initial_step_size must not exceed max_step_size"):
        eki_module.run_eki(
            model=object(),
            parameter_space=DummyParameterSpace(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_eki_directory=str(tmp_path / "eki"),
            initial_step_size=1.1,
            max_step_size=1.0,
        )


def test_mf_eki_rejects_initial_step_above_max(tmp_path):
    with pytest.raises(AssertionError, match="initial_step_size must not exceed max_step_size"):
        mf_module.run_mf_eki(
            model=object(),
            rom_model_builder=object(),
            parameter_space=DummyParameterSpace(),
            observations=np.array([0.0]),
            observations_covariance=np.eye(1),
            absolute_eki_directory=str(tmp_path / "mf_eki"),
            initial_step_size=1.1,
            max_step_size=1.0,
        )


def test_auto_mf_eki_forwards_max_step_size(monkeypatch):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured.update(kwargs)
        return "ok", None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)
    result = mf_module.mf_eki_with_auto_rom(
        model=object(),
        parameter_space=DummyParameterSpace(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        max_step_size=0.4,
    )

    assert result == ("ok", None)
    assert captured["max_step_size"] == pytest.approx(0.4)

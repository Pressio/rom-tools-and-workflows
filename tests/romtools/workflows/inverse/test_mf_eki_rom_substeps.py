import numpy as np
import pytest

import romtools.workflows.inverse.mf_eki_drivers as mf_module
from romtools.workflows.parameter_spaces import MonteCarloSampler, UniformParameterSpace


class LinearQoiModel:
    def populate_run_directory(self, run_directory, parameter_sample):
        return None

    def run_model(self, run_directory, parameter_sample):
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        return np.array([parameter_sample["u"] + 0.5 * parameter_sample["v"]])


class LinearQoiModelBuilder:
    def build_from_training_dirs(self, offline_data_dir, training_data_dirs,
                                 training_parameters, training_qois):
        return LinearQoiModel()


def _run_kwargs(directory):
    return {
        "model": LinearQoiModel(),
        "rom_model_builder": LinearQoiModelBuilder(),
        "parameter_space": UniformParameterSpace(
            ["u", "v"],
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            sampler=MonteCarloSampler,
        ),
        "observations": np.array([0.8]),
        "observations_covariance": np.array([[1.0e-2]]),
        "parameter_mins": np.array([0.0, 0.0]),
        "parameter_maxes": np.array([1.0, 1.0]),
        "absolute_eki_directory": str(directory),
        "fom_ensemble_size": 4,
        "rom_extra_ensemble_size": 4,
        "rom_tolerance": np.inf,
        "initial_step_size": 0.1,
        "regularization_parameter": 1.0e-6,
        "step_size_growth_factor": 1.1,
        "step_size_decay_factor": 2.0,
        "max_step_size_decrease_trys": 5,
        "relaxation_parameter": 100.0,
        "error_norm_tolerance": 0.0,
        "delta_params_tolerance": 0.0,
        "max_rom_training_history": 2,
        "max_iterations": 4,
        "random_seed": 3,
    }


@pytest.mark.mpi_skip
def test_rom_substep_window_predicate():
    assert not mf_module._rom_substeps_are_enabled(0, 1, 3, 2)
    assert mf_module._rom_substeps_are_enabled(1, 1, 3, 2)
    assert mf_module._rom_substeps_are_enabled(2, 1, 3, 2)
    assert not mf_module._rom_substeps_are_enabled(3, 1, 3, 2)
    assert mf_module._rom_substeps_are_enabled(100, 1, None, 2)
    assert not mf_module._rom_substeps_are_enabled(2, 0, None, 0)


@pytest.mark.mpi_skip
def test_rom_substeps_respect_bounds(monkeypatch):
    calls = []

    def fake_iteration(model, observations, run_directory_base, parameter_names,
                       parameter_samples, evaluation_concurrency, dispatcher):
        calls.append((run_directory_base, parameter_samples.copy()))
        n = parameter_samples.shape[0]
        return {
            "qois": np.zeros((1, n)),
            "mean-qoi": np.zeros(1),
            "errors": np.ones((1, n)),
        }

    def fake_update(parameter_samples, qois, mean_qoi, errors,
                    observations_covariance, regularization_parameter):
        return np.ones_like(parameter_samples) * 10.0

    monkeypatch.setattr(mf_module, "run_eki_iteration", fake_iteration)
    monkeypatch.setattr(mf_module, "compute_eki_update", fake_update)

    result = mf_module._apply_rom_only_substeps(
        rom_model=object(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        parameter_sample_sets=[
            np.array([[0.2, 0.4], [0.5, 0.5]]),
            np.array([[0.9, 0.1]]),
        ],
        parameter_names=["u", "v"],
        step_size=1.0,
        regularization_parameter=0.0,
        parameter_mins=np.zeros(2),
        parameter_maxes=np.ones(2),
        absolute_eki_directory="/tmp/test-mf-eki-substeps",
        outer_iteration=2,
        num_rom_substeps=2,
        rom_evaluation_concurrency=1,
        rom_dispatcher=object(),
    )

    assert len(calls) == 2
    assert "/iteration_2/rom_substep_0/" in calls[0][0]
    assert "/iteration_2/rom_substep_1/" in calls[1][0]
    assert result[0].shape == (2, 2)
    assert result[1].shape == (1, 2)
    for sample_set in result:
        assert np.all(sample_set >= 0.0)
        assert np.all(sample_set <= 1.0)


@pytest.mark.mpi_skip
def test_num_rom_substeps_zero_reproduces_existing_behavior(tmp_path):
    baseline = mf_module.run_mf_eki(**_run_kwargs(tmp_path / "baseline"))
    explicit_zero = mf_module.run_mf_eki(
        **_run_kwargs(tmp_path / "explicit_zero"),
        rom_substep_start_iteration=1,
        rom_substep_end_iteration=3,
        num_rom_substeps=0,
    )
    assert np.array_equal(baseline[0], explicit_zero[0])
    assert np.array_equal(baseline[1], explicit_zero[1])


@pytest.mark.mpi_skip
def test_rom_substeps_only_run_in_requested_outer_window(monkeypatch, tmp_path):
    outer_iterations = []

    def fake_substeps(**kwargs):
        outer_iterations.append(kwargs["outer_iteration"])
        return [samples.copy() for samples in kwargs["parameter_sample_sets"]]

    monkeypatch.setattr(mf_module, "_apply_rom_only_substeps", fake_substeps)
    mf_module.run_mf_eki(
        **_run_kwargs(tmp_path / "window"),
        rom_substep_start_iteration=1,
        rom_substep_end_iteration=3,
        num_rom_substeps=2,
    )
    assert outer_iterations == [1, 2]


@pytest.mark.mpi_skip
def test_restart_restores_rom_substep_schedule(monkeypatch, tmp_path):
    outer_iterations = []

    def fake_substeps(**kwargs):
        outer_iterations.append(kwargs["outer_iteration"])
        return [samples.copy() for samples in kwargs["parameter_sample_sets"]]

    monkeypatch.setattr(mf_module, "_apply_rom_only_substeps", fake_substeps)
    first_kwargs = _run_kwargs(tmp_path / "restart")
    first_kwargs["max_iterations"] = 3
    mf_module.run_mf_eki(
        **first_kwargs,
        rom_substep_start_iteration=1,
        rom_substep_end_iteration=4,
        num_rom_substeps=1,
    )
    assert outer_iterations == [1]

    outer_iterations.clear()
    restart_path = tmp_path / "restart" / "iteration_2" / "restart.npz"
    restart_kwargs = _run_kwargs(tmp_path / "restart")
    restart_kwargs["max_iterations"] = 5
    mf_module.run_mf_eki(
        **restart_kwargs,
        restart_file=str(restart_path),
    )
    assert outer_iterations == [2, 3]


@pytest.mark.mpi_skip
def test_auto_rom_forwards_rom_substep_settings(monkeypatch):
    captured = {}

    def fake_run_mf_eki(**kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(mf_module, "run_mf_eki", fake_run_mf_eki)
    parameter_space = UniformParameterSpace(
        ["u", "v"],
        np.zeros(2),
        np.ones(2),
        sampler=MonteCarloSampler,
    )
    mf_module.mf_eki_with_auto_rom(
        model=LinearQoiModel(),
        parameter_space=parameter_space,
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        rom_substep_start_iteration=2,
        rom_substep_end_iteration=6,
        num_rom_substeps=4,
    )
    assert captured["rom_substep_start_iteration"] == 2
    assert captured["rom_substep_end_iteration"] == 6
    assert captured["num_rom_substeps"] == 4

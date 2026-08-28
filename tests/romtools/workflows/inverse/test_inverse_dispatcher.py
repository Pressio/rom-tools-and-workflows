"""
Dispatcher wiring for the inverse workflow drivers.

Drivers must keep working for callers that pass no dispatcher at all, forward a
supplied dispatcher to every model evaluation and restart write, and keep
in-process surrogate (ROM) evaluations on the local machine.
"""

import os

import numpy as np
import pytest

import romtools.workflows
from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher
from romtools.rom.qoi_surrogates import GaussianProcessQoiModel
from romtools.workflows.inverse import mf_eki_drivers, mf_vi_drivers
from romtools.workflows.parameter_spaces import GaussianParameterSpace, MonteCarloSampler


class RecordingDispatcher(BaseDispatcher):
    """
    Dispatcher that performs local IO while recording what it was asked to do.

    Deliberately not a LocalDispatcher subclass, so that resolve_local_dispatcher
    replaces it and the tests can tell FOM work (routed here) apart from ROM work
    (which must stay on a plain LocalDispatcher).
    """

    def __init__(self):
        super().__init__(argv=[])
        self.created_dirs = []
        self.saved_npz = []
        self.written_text = []

    def path_exists(self, path: str) -> bool:
        return os.path.exists(path)

    def create_empty_dir(self, dir_name: str):
        self.created_dirs.append(dir_name)
        os.makedirs(dir_name, exist_ok=True)

    def list_dir(self, path: str) -> list:
        return os.listdir(path) if os.path.isdir(path) else []

    def remove(self, path: str) -> None:
        if os.path.exists(path):
            os.remove(path)

    def write_text(self, path: str, content: str) -> None:
        self.written_text.append(path)
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as text_file:
            text_file.write(content)

    def np_savez(self, path: str, **arrays) -> None:
        self.saved_npz.append(path)
        np.savez(path, **arrays)


class LinearQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([2.0 * float(parameter_sample["theta"])])


class SingleParameterSpace(romtools.workflows.ParameterSpace):
    def get_names(self):
        return ["theta"]

    def get_dimensionality(self) -> int:
        return 1

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else 3)
        return rng.uniform(-1.0, 1.0, size=(number_of_samples, 1))


class TwoParameterLinearQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        theta0 = float(parameter_sample["theta0"])
        theta1 = float(parameter_sample["theta1"])
        return np.array([theta0 + 0.2 * theta1, -0.3 * theta0 + 0.5 * theta1])


class TwoParameterSpace(romtools.workflows.ParameterSpace):
    def get_names(self):
        return ["theta0", "theta1"]

    def get_dimensionality(self) -> int:
        return 2

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else 3)
        return rng.uniform(-1.0, 1.0, size=(number_of_samples, 2))


def _gaussian_parameter_space():
    return GaussianParameterSpace(
        parameter_names=["theta"],
        means=np.array([0.0]),
        stds=np.array([1.0]),
        sampler=MonteCarloSampler,
    )


def _run_eki(directory, dispatcher=None, evaluation_concurrency=1):
    return romtools.workflows.run_eki(
        model=LinearQoiModel(),
        parameter_space=SingleParameterSpace(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_eki_directory=directory,
        ensemble_size=4,
        max_iterations=2,
        evaluation_concurrency=evaluation_concurrency,
        dispatcher=dispatcher,
    )


def _run_vi(directory, dispatcher=None, evaluation_concurrency=1):
    return romtools.workflows.run_vi(
        model=LinearQoiModel(),
        prior_parameter_space=_gaussian_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_vi_directory=directory,
        sample_size=4,
        evaluation_concurrency=evaluation_concurrency,
        bounded_parameter_handling="clip",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(max_iterations=1),
        random_seed=5,
        dispatcher=dispatcher,
    )


def _run_mf_eki(directory, dispatcher=None):
    return mf_eki_drivers.mf_eki_with_auto_rom(
        model=TwoParameterLinearQoiModel(),
        parameter_space=TwoParameterSpace(),
        observations=np.zeros(2),
        observations_covariance=np.eye(2),
        absolute_eki_directory=directory,
        fom_ensemble_size=4,
        rom_extra_ensemble_size=4,
        max_iterations=2,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        dispatcher=dispatcher,
    )


# ----------------------------------------------------------------------
# Backwards compatibility: callers that never heard of a dispatcher
# ----------------------------------------------------------------------

@pytest.mark.mpi_skip
def test_run_eki_without_dispatcher_writes_local_run_directories(tmp_path):
    _run_eki(str(tmp_path))

    assert (tmp_path / "iteration_0" / "run_mean").is_dir()
    assert (tmp_path / "iteration_0" / "restart.npz").is_file()


@pytest.mark.mpi_skip
def test_run_eki_without_dispatcher_still_requires_an_absolute_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match="must provide an absolute path"):
        _run_eki("relative_work")


@pytest.mark.mpi_skip
def test_mf_eki_with_auto_rom_without_dispatcher_builds_a_gp_rom(tmp_path):
    parameter_samples, qois = _run_mf_eki(str(tmp_path))

    assert parameter_samples.shape == (4, 2)
    assert qois.shape == (2, 4)


# ----------------------------------------------------------------------
# Regression: the GP ROM builder must return a model, dispatcher or not
# ----------------------------------------------------------------------

@pytest.mark.mpi_skip
@pytest.mark.parametrize("dispatcher", [None, LocalDispatcher(), RecordingDispatcher()])
def test_mf_eki_with_auto_rom_builds_a_rom_for_every_dispatcher(tmp_path, dispatcher):
    """
    Regression test: the GP builder used to return None whenever a dispatcher was
    present, and mf_eki_with_auto_rom always supplied one, so every ROM model was
    None and the first ROM evaluation raised AttributeError.
    """
    parameter_samples, _ = _run_mf_eki(str(tmp_path), dispatcher=dispatcher)

    assert parameter_samples.shape == (4, 2)


@pytest.mark.mpi_skip
def test_gp_rom_builder_returns_a_gaussian_process_model():
    builder = mf_eki_drivers.GaussianProcessQoiModelBuilderWithTrainingData(
        parameter_names=["theta"],
    )

    rom_model = builder.build_from_training_dirs(
        "offline",
        ["run_0", "run_1", "run_2"],
        np.array([[0.0], [0.5], [1.0]]),
        np.array([[0.0], [1.0], [2.0]]),
    )

    assert isinstance(rom_model, GaussianProcessQoiModel)


# ----------------------------------------------------------------------
# A supplied dispatcher is forwarded to every evaluation and restart write
# ----------------------------------------------------------------------

@pytest.mark.mpi_skip
def test_run_eki_forwards_dispatcher_to_runs_and_restarts(tmp_path):
    dispatcher = RecordingDispatcher()

    _run_eki(str(tmp_path), dispatcher=dispatcher)

    assert str(tmp_path) in dispatcher.created_dirs
    assert any(name.endswith("iteration_0/run_mean") for name in dispatcher.created_dirs)
    assert any(name.endswith("iteration_0/restart.npz") for name in dispatcher.saved_npz)


@pytest.mark.mpi_skip
def test_run_eki_accepts_a_relative_directory_for_a_non_local_dispatcher(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    _run_eki("relative_work", dispatcher=RecordingDispatcher())

    assert (tmp_path / "relative_work" / "iteration_0" / "run_mean").is_dir()


@pytest.mark.mpi_skip
def test_mf_eki_dispatches_fom_runs_but_keeps_rom_runs_local(tmp_path):
    dispatcher = RecordingDispatcher()

    _run_mf_eki(str(tmp_path), dispatcher=dispatcher)

    assert any("run_fom_sample_set" in name for name in dispatcher.created_dirs)
    assert not any("run_rom_sample_set" in name for name in dispatcher.created_dirs)
    # ROM run directories are still created, just by a local dispatcher
    assert (tmp_path / "iteration_0" / "run_rom_sample_set_0_mean").is_dir()


@pytest.mark.mpi_skip
def test_run_vi_forwards_dispatcher_to_runs_restarts_and_stats(tmp_path):
    dispatcher = RecordingDispatcher()

    _run_vi(str(tmp_path), dispatcher=dispatcher)

    assert any("iteration_0/run_" in name for name in dispatcher.created_dirs)
    assert any(name.endswith("restart.npz") for name in dispatcher.saved_npz)
    assert any(name.endswith("history.npz") for name in dispatcher.saved_npz)
    assert any(name.endswith("stats.txt") for name in dispatcher.written_text)


@pytest.mark.mpi_skip
def test_run_eki_with_a_local_dispatcher_survives_concurrent_evaluation(tmp_path):
    """
    run_eki_iteration submits prepare_and_run to a ProcessPoolExecutor, so the
    dispatcher has to survive being pickled into the workers.
    """
    _run_eki(str(tmp_path), dispatcher=LocalDispatcher(), evaluation_concurrency=2)

    assert (tmp_path / "iteration_0" / "run_mean").is_dir()


@pytest.mark.mpi_skip
def test_run_vi_with_a_local_dispatcher_survives_concurrent_evaluation(tmp_path):
    """Same as above, but run_vi_iteration uses a 'spawn' context rather than 'fork'."""
    _run_vi(str(tmp_path), dispatcher=LocalDispatcher(), evaluation_concurrency=2)

    assert (tmp_path / "iteration_0" / "run_0").is_dir()


@pytest.mark.mpi_skip
def test_mf_vi_dispatches_fom_runs_but_keeps_rom_runs_local(tmp_path):
    dispatcher = RecordingDispatcher()

    mf_vi_drivers.mf_vi_with_auto_rom(
        model=LinearQoiModel(),
        prior_parameter_space=_gaussian_parameter_space(),
        observations=np.array([0.0]),
        observations_covariance=np.eye(1),
        absolute_vi_directory=str(tmp_path),
        fom_sample_size=4,
        rom_extra_sample_size=2,
        rom_tolerance=0.0,
        bounded_parameter_handling="clip",
        optimizer_config=romtools.workflows.VIGradientOptimizerConfig(max_iterations=1),
        random_seed=5,
        fom_evaluation_concurrency=1,
        rom_evaluation_concurrency=1,
        dispatcher=dispatcher,
    )

    assert any("run_fom_sample_set" in name for name in dispatcher.created_dirs)
    assert not any("run_rom_sample_set" in name for name in dispatcher.created_dirs)
    assert (tmp_path / "iteration_0" / "run_rom_sample_set_0_0").is_dir()

import inspect

import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import mf_eki_drivers


class TwoParameterSpace(romtools.workflows.ParameterSpace):
    def get_names(self):
        return ["theta0", "theta1"]

    def get_dimensionality(self) -> int:
        return 2

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed if seed is not None else 11)
        return rng.uniform(-1.0, 1.0, size=(number_of_samples, 2))


class LinearQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        theta0 = float(parameter_sample["theta0"])
        theta1 = float(parameter_sample["theta1"])
        return np.array([
            theta0 + 0.2 * theta1,
            -0.3 * theta0 + 0.5 * theta1,
        ])


class BiasedLinearQoiModel(LinearQoiModel):
    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return super().compute_qoi(run_directory, parameter_sample) + np.array([2.0, -1.0])


class RefreshingRomBuilder:
    def __init__(self):
        self.build_count = 0

    def build_from_training_dirs(
            self,
            offline_data_dir,
            training_data_dirs,
            training_parameters,
            training_qois):
        self.build_count += 1
        if self.build_count == 1:
            return BiasedLinearQoiModel()
        return LinearQoiModel()


class ExactRomBuilder:
    def build_from_training_dirs(
            self,
            offline_data_dir,
            training_data_dirs,
            training_parameters,
            training_qois):
        return LinearQoiModel()


def test_mf_eki_rejuvenation_defaults_to_five_percent_reference_std():
    signature = inspect.signature(mf_eki_drivers.run_mf_eki)
    beta = signature.parameters["rejuvenation_prior_weight"].default
    assert beta == pytest.approx(0.05**2)


@pytest.mark.mpi_skip
def test_mf_eki_rejuvenation_rechecks_rom_and_rebuilds_when_needed(tmp_path):
    builder = RefreshingRomBuilder()
    parameter_samples, qois = mf_eki_drivers.run_mf_eki(
        model=LinearQoiModel(),
        rom_model_builder=builder,
        parameter_space=TwoParameterSpace(),
        observations=np.array([0.7, -0.2]),
        observations_covariance=np.eye(2),
        absolute_eki_directory=str(tmp_path),
        fom_ensemble_size=4,
        rom_extra_ensemble_size=4,
        rom_tolerance=1e-8,
        max_iterations=2,
        random_seed=5,
        rejuvenation_strategy="adaptive",
        rejuvenation_inflation=1.2,
        rejuvenation_prior_weight=0.05**2,
        max_rejuvenations=1,
        delta_params_tolerance=1e12,
        error_norm_tolerance=1e-12,
    )

    assert parameter_samples.shape == (4, 2)
    assert qois.shape == (2, 4)
    assert builder.build_count >= 2

    restart = np.load(tmp_path / "iteration_0" / "restart.npz", allow_pickle=True)
    assert int(restart["rejuvenation_count"]) == 1
    assert "rejuvenation_reference_covariance" in restart


@pytest.mark.mpi_skip
def test_mf_eki_periodic_rejuvenation_records_restart_state(tmp_path):
    mf_eki_drivers.run_mf_eki(
        model=LinearQoiModel(),
        rom_model_builder=ExactRomBuilder(),
        parameter_space=TwoParameterSpace(),
        observations=np.array([0.7, -0.2]),
        observations_covariance=np.eye(2),
        absolute_eki_directory=str(tmp_path),
        fom_ensemble_size=4,
        rom_extra_ensemble_size=4,
        rom_tolerance=1e-8,
        max_iterations=3,
        random_seed=5,
        rejuvenation_strategy="periodic",
        rejuvenation_interval=1,
        max_rejuvenations=1,
        delta_params_tolerance=0.0,
    )

    restart = np.load(tmp_path / "iteration_1" / "restart.npz", allow_pickle=True)
    assert int(restart["rejuvenation_count"]) == 1
    assert restart["rejuvenation_reference_covariance"].shape == (2, 2)

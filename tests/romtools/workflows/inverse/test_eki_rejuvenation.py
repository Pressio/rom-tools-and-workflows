import inspect

import numpy as np
import pytest

import romtools.workflows
from romtools.workflows.inverse import eki_drivers


class ConstantQoiModel:
    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        return None

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        return np.array([0.0])


class DeterministicTwoParameterSpace(romtools.workflows.ParameterSpace):
    def get_names(self):
        return ["theta0", "theta1"]

    def get_dimensionality(self) -> int:
        return 2

    def generate_samples(self, number_of_samples: int, seed=None) -> np.ndarray:
        base_samples = np.array([
            [-1.0, -0.5],
            [-0.5, 0.5],
            [0.0, -1.0],
            [0.5, 1.0],
            [1.0, -0.25],
            [0.25, 0.75],
        ])
        repeats = int(np.ceil(number_of_samples / base_samples.shape[0]))
        return np.tile(base_samples, (repeats, 1))[:number_of_samples].copy()


def test_rejuvenation_is_opt_in_by_default():
    signature = inspect.signature(romtools.workflows.run_eki)
    assert signature.parameters["rejuvenation_strategy"].default == "none"


def test_rejuvenation_defaults_to_five_percent_reference_std():
    signature = inspect.signature(romtools.workflows.run_eki)
    beta = signature.parameters["rejuvenation_prior_weight"].default
    assert beta == pytest.approx(0.05**2)


def test_parameter_scaled_reference_covariance_uses_current_mean():
    samples = np.array([
        [1.0, -5.0],
        [3.0, -3.0],
    ])

    reference_covariance = (
        eki_drivers._compute_parameter_scaled_reference_covariance(samples)
    )

    np.testing.assert_allclose(
        reference_covariance,
        np.diag([2.0**2, 4.0**2]),
    )


def test_build_rejuvenation_covariance_combines_current_and_scaled_reference():
    samples = np.array([
        [1.0, -5.0],
        [3.0, -3.0],
        [2.0, -4.5],
        [2.0, -3.5],
    ])
    fallback_covariance = np.eye(2)
    current_covariance = eki_drivers._compute_ensemble_covariance(samples)
    scaled_reference = np.diag(np.abs(np.mean(samples, axis=0))**2)

    covariance = eki_drivers._build_rejuvenation_covariance(
        samples,
        fallback_covariance,
        rejuvenation_inflation=1.2,
        rejuvenation_prior_weight=0.05**2,
    )

    expected = 1.2 * current_covariance + 0.05**2 * scaled_reference
    np.testing.assert_allclose(covariance, expected)


def test_collapsed_covariance_gives_five_percent_parameter_std():
    samples = np.tile(np.array([[2.0, -4.0]]), (6, 1))

    covariance = eki_drivers._build_rejuvenation_covariance(
        samples,
        reference_covariance=np.eye(2),
        rejuvenation_inflation=1.1,
        rejuvenation_prior_weight=0.05**2,
    )

    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)),
        np.array([0.1, 0.2]),
    )


def test_zero_mean_parameter_uses_bound_range_as_reference_scale():
    samples = np.tile(np.array([[0.0, 2.0]]), (6, 1))

    covariance = eki_drivers._build_rejuvenation_covariance(
        samples,
        reference_covariance=np.eye(2),
        rejuvenation_inflation=1.1,
        rejuvenation_prior_weight=0.05**2,
        parameter_mins=np.array([-10.0, -100.0]),
        parameter_maxes=np.array([10.0, 100.0]),
    )

    # theta0 has zero mean, so its 20-unit admissible range sets the scale.
    # theta1 is nonzero, so its current mean magnitude remains the scale.
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)),
        np.array([1.0, 0.1]),
    )


def test_zero_mean_unbounded_parameter_uses_initial_covariance_fallback():
    samples = np.zeros((6, 2))
    fallback_covariance = np.diag([3.0**2, 4.0**2])

    covariance = eki_drivers._build_rejuvenation_covariance(
        samples,
        reference_covariance=fallback_covariance,
        rejuvenation_inflation=1.1,
        rejuvenation_prior_weight=0.05**2,
    )

    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)),
        np.array([0.15, 0.2]),
    )


def test_rejuvenation_preserves_mean_without_bounds_and_is_reproducible():
    samples = np.array([
        [-1.0, -0.5],
        [-0.5, 0.5],
        [0.5, -1.0],
        [1.0, 1.0],
        [0.25, 0.75],
    ])
    reference_covariance = eki_drivers._compute_ensemble_covariance(samples)

    first = eki_drivers._rejuvenate_parameter_samples(
        samples,
        reference_covariance,
        rejuvenation_inflation=1.1,
        rejuvenation_prior_weight=0.05**2,
        random_seed=19,
    )
    second = eki_drivers._rejuvenate_parameter_samples(
        samples,
        reference_covariance,
        rejuvenation_inflation=1.1,
        rejuvenation_prior_weight=0.05**2,
        random_seed=19,
    )

    np.testing.assert_allclose(np.mean(first, axis=0), np.mean(samples, axis=0))
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, samples)


def test_rejuvenation_respects_parameter_bounds():
    samples = np.array([
        [-1.0, -1.0],
        [1.0, 1.0],
        [-0.5, 0.5],
        [0.5, -0.5],
    ])
    reference_covariance = eki_drivers._compute_ensemble_covariance(samples)

    rejuvenated = eki_drivers._rejuvenate_parameter_samples(
        samples,
        reference_covariance,
        rejuvenation_inflation=5.0,
        rejuvenation_prior_weight=1.0,
        random_seed=7,
        parameter_mins=np.array([-0.25, -0.4]),
        parameter_maxes=np.array([0.3, 0.2]),
    )

    assert np.all(rejuvenated >= np.array([-0.25, -0.4]))
    assert np.all(rejuvenated <= np.array([0.3, 0.2]))


def test_adaptive_rejuvenation_distinguishes_stagnation_from_convergence():
    assert eki_drivers._should_rejuvenate(
        "adaptive",
        iteration=4,
        dp_norm=1e-8,
        error_norm=1e-2,
        delta_params_tolerance=1e-6,
        error_norm_tolerance=1e-5,
        rejuvenation_count=0,
        max_rejuvenations=3,
        rejuvenation_interval=5,
    )
    assert not eki_drivers._should_rejuvenate(
        "adaptive",
        iteration=4,
        dp_norm=1e-8,
        error_norm=1e-7,
        delta_params_tolerance=1e-6,
        error_norm_tolerance=1e-5,
        rejuvenation_count=0,
        max_rejuvenations=3,
        rejuvenation_interval=5,
    )


def test_periodic_rejuvenation_triggers_only_on_requested_interval():
    assert eki_drivers._should_rejuvenate(
        "periodic",
        iteration=10,
        dp_norm=1.0,
        error_norm=1.0,
        delta_params_tolerance=1e-6,
        error_norm_tolerance=1e-5,
        rejuvenation_count=0,
        max_rejuvenations=3,
        rejuvenation_interval=5,
    )
    assert not eki_drivers._should_rejuvenate(
        "periodic",
        iteration=11,
        dp_norm=1.0,
        error_norm=1.0,
        delta_params_tolerance=1e-6,
        error_norm_tolerance=1e-5,
        rejuvenation_count=0,
        max_rejuvenations=3,
        rejuvenation_interval=5,
    )


def test_periodic_rejuvenation_does_not_run_after_residual_convergence():
    assert not eki_drivers._should_rejuvenate(
        "periodic",
        iteration=10,
        dp_norm=1.0,
        error_norm=1e-7,
        delta_params_tolerance=1e-6,
        error_norm_tolerance=1e-5,
        rejuvenation_count=0,
        max_rejuvenations=3,
        rejuvenation_interval=5,
    )


@pytest.mark.mpi_skip
def test_adaptive_rejuvenation_is_restart_reproducible(tmp_path):
    common_args = dict(
        model=ConstantQoiModel(),
        parameter_space=DeterministicTwoParameterSpace(),
        observations=np.array([1.0]),
        observations_covariance=np.eye(1),
        ensemble_size=6,
        max_iterations=2,
        random_seed=13,
        rejuvenation_strategy="adaptive",
        rejuvenation_inflation=1.2,
        rejuvenation_prior_weight=0.05**2,
        delta_params_tolerance=1e-12,
        error_norm_tolerance=1e-8,
    )

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted_samples, _ = romtools.workflows.run_eki(
        absolute_eki_directory=str(uninterrupted_dir),
        max_rejuvenations=2,
        **common_args,
    )

    first_dir = tmp_path / "first"
    romtools.workflows.run_eki(
        absolute_eki_directory=str(first_dir),
        max_rejuvenations=1,
        **common_args,
    )
    restart_file = first_dir / "iteration_0" / "restart.npz"
    restart_data = np.load(restart_file)
    assert int(restart_data["rejuvenation_count"]) == 1

    restarted_dir = tmp_path / "restarted"
    restarted_samples, _ = romtools.workflows.run_eki(
        absolute_eki_directory=str(restarted_dir),
        max_rejuvenations=2,
        restart_file=str(restart_file),
        **common_args,
    )

    np.testing.assert_allclose(restarted_samples, uninterrupted_samples)
    restarted_state = np.load(restarted_dir / "iteration_0" / "restart.npz")
    assert int(restarted_state["rejuvenation_count"]) == 2

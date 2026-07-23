import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import romtools.workflows
from romtools.workflows.parameter_spaces import (
    GaussianParameterSpace,
    MonteCarloSampler,
    UniformParameterSpace,
)


PARAMETER_NAMES = ["bmag", "theta", "nu", "sigma"]
PARAMETER_MINS = np.array([1.0e-5, 0.5, 1.0e-10, 0.1], dtype=float)
PARAMETER_MAXES = np.array([3.0, 1.6, 10.0, 2.0], dtype=float)
PRIOR_MEAN = np.array([0.5, 0.8, 2.0e-2, 0.5], dtype=float)
PRIOR_STDS = 0.5 * PRIOR_MEAN
TRUTH_PARAMETERS = {
    "bmag": 0.65,
    "theta": 0.95,
    "nu": 3.0e-2,
    "sigma": 0.6,
}
OBSERVATION_NOISE_SEED = 7
WORKFLOW_RANDOM_SEED = 3
GRID_SHAPE = (10, 10)


def _build_advection_matrices(nx: int, ny: int, dx: float, dy: float):
    ax_builder = _SparseMatrixBuilder(nx * ny, nx * ny)
    ay_builder = _SparseMatrixBuilder(nx * ny, nx * ny)

    for j in range(1, ny):
        for i in range(1, nx):
            index = i + j * nx
            index_im1 = index - 1
            index_im2 = index - 2
            index_jm1 = index - nx
            index_jm2 = index - 2 * nx

            ax_builder.add_entry(index, index, 1.5 / dx)
            ax_builder.add_entry(index, index_im1, -2.0 / dx)
            if i > 1:
                ax_builder.add_entry(index, index_im2, 0.5 / dx)

            ay_builder.add_entry(index, index, 1.5 / dy)
            ay_builder.add_entry(index, index_jm1, -2.0 / dy)
            if j > 1:
                ay_builder.add_entry(index, index_jm2, 0.5 / dy)

    for i in range(1, nx):
        index = i
        index_im1 = index - 1
        index_im2 = index - 2

        ax_builder.add_entry(index, index, 1.5 / dx)
        ax_builder.add_entry(index, index_im1, -2.0 / dx)
        if i > 1:
            ax_builder.add_entry(index, index_im2, 0.5 / dx)

        ay_builder.add_entry(index, index, 1.0 / dy)

    for j in range(1, ny):
        index = j * nx
        index_jm1 = index - nx
        index_jm2 = index - 2 * nx

        ax_builder.add_entry(index, index, 1.0 / dx)
        ay_builder.add_entry(index, index, 1.5 / dy)
        ay_builder.add_entry(index, index_jm1, -2.0 / dy)
        if j > 1:
            ay_builder.add_entry(index, index_jm2, 0.5 / dy)

    ax_builder.add_entry(0, 0, 1.0 / dx)
    ay_builder.add_entry(0, 0, 1.0 / dy)
    return ax_builder.assemble(), ay_builder.assemble()


def _build_diffusion_matrix(nx: int, ny: int, dx: float, dy: float):
    builder = _SparseMatrixBuilder(nx * ny, nx * ny)
    for j in range(ny):
        for i in range(nx):
            index = i + j * nx
            builder.add_entry(index, index, -2.0 / dx**2 - 2.0 / dy**2)
            if i > 0:
                builder.add_entry(index, index - 1, 1.0 / dx**2)
            if i < nx - 1:
                builder.add_entry(index, index + 1, 1.0 / dx**2)
            if j > 0:
                builder.add_entry(index, index - nx, 1.0 / dy**2)
            if j < ny - 1:
                builder.add_entry(index, index + nx, 1.0 / dy**2)
    return builder.assemble()


class _SparseMatrixBuilder:
    def __init__(self, rows: int, cols: int):
        self._rows = rows
        self._cols = cols
        self._row_indices = []
        self._col_indices = []
        self._values = []

    def add_entry(self, row_index: int, col_index: int, value: float):
        self._row_indices.append(row_index)
        self._col_indices.append(col_index)
        self._values.append(value)

    def assemble(self):
        return scipy.sparse.csr_matrix(
            (self._values, (self._row_indices, self._col_indices)),
            shape=(self._rows, self._cols),
        )


class SteadyCdrSystem:
    def __init__(self, nx: int, ny: int):
        self.nx = nx
        self.ny = ny
        self.num_dofs = nx * ny
        self.dx = 1.0 / (nx + 1)
        self.dy = 1.0 / (ny + 1)
        self.source = np.ones(self.num_dofs)
        self.diffusion = _build_diffusion_matrix(nx, ny, self.dx, self.dy)
        self.advection_x, self.advection_y = _build_advection_matrices(nx, ny, self.dx, self.dy)
        self.identity = scipy.sparse.identity(self.num_dofs, format="csr")


def solve_steady_cdr(system: SteadyCdrSystem, parameter_sample: dict) -> np.ndarray:
    velocity = np.array(
        [
            parameter_sample["bmag"] * np.cos(parameter_sample["theta"]),
            parameter_sample["bmag"] * np.sin(parameter_sample["theta"]),
        ],
        dtype=float,
    )
    lhs = (
        parameter_sample["nu"] * system.diffusion
        - velocity[0] * system.advection_x
        - velocity[1] * system.advection_y
        - parameter_sample["sigma"] * system.identity
    )
    return scipy.sparse.linalg.spsolve(lhs, -system.source)


class CdrFullStateQoiModel:
    def __init__(self, nx: int, ny: int):
        self.system = SteadyCdrSystem(nx, ny)
        self._latest_state = None

    def populate_run_directory(self, run_directory: str, parameter_sample: dict) -> None:
        _ = run_directory
        _ = parameter_sample

    def run_model(self, run_directory: str, parameter_sample: dict) -> int:
        _ = run_directory
        self._latest_state = solve_steady_cdr(self.system, parameter_sample)
        return 0

    def compute_qoi(self, run_directory: str, parameter_sample: dict) -> np.ndarray:
        _ = run_directory
        _ = parameter_sample
        return np.asarray(self._latest_state, dtype=float)


def build_prior_parameter_space() -> GaussianParameterSpace:
    return GaussianParameterSpace(
        parameter_names=PARAMETER_NAMES,
        means=PRIOR_MEAN.copy(),
        stds=PRIOR_STDS.copy(),
        sampler=MonteCarloSampler,
    )


def build_model() -> CdrFullStateQoiModel:
    return CdrFullStateQoiModel(*GRID_SHAPE)


def build_eki_parameter_space() -> UniformParameterSpace:
    return UniformParameterSpace(
        parameter_names=PARAMETER_NAMES,
        lower_bounds=PARAMETER_MINS.copy(),
        upper_bounds=PARAMETER_MAXES.copy(),
        sampler=MonteCarloSampler,
    )


def build_observations():
    truth_model = build_model()
    truth_model.run_model("", TRUTH_PARAMETERS)
    truth_qoi = truth_model.compute_qoi("", TRUTH_PARAMETERS)
    noise_std = max(0.05 * abs(np.mean(truth_qoi)), 1.0e-6)
    rng = np.random.default_rng(OBSERVATION_NOISE_SEED)
    observations = truth_qoi + rng.normal(loc=0.0, scale=noise_std, size=truth_qoi.shape)
    observations_covariance = np.eye(truth_qoi.size) * noise_std**2
    return observations.astype(float), observations_covariance.astype(float)


def build_vi_kwargs(absolute_vi_directory: str):
    observations, observations_covariance = build_observations()
    return {
        "model": build_model(),
        "prior_parameter_space": build_prior_parameter_space(),
        "observations": observations,
        "observations_covariance": observations_covariance,
        "parameter_mins": PARAMETER_MINS.copy(),
        "parameter_maxes": PARAMETER_MAXES.copy(),
        "absolute_vi_directory": absolute_vi_directory,
        "sample_size": 8,
        "optimizer_method": "gradient",
        "optimizer_config": romtools.workflows.VIGradientOptimizerConfig(
            gradient_method="standard",
            max_iterations=10,
            gradient_norm_tolerance=0.0,
            min_variational_std=1.0e-6,
            max_variational_std=2.0,
        ),
        "line_search_method": "legacy",
        "line_search_config": romtools.workflows.VILegacyLineSearchConfig(
            initial_step_size=1.0e-2,
            max_step_size=1.0e-2,
            step_size_growth_factor=1.0,
            step_size_decay_factor=2.0,
            max_step_size_decrease_trys=20,
            relaxation_parameter=10.0,
            line_search_sample_growth_factor=1.0,
        ),
        "elbo_relative_tolerance": None,
        "evaluation_concurrency": 1,
        "baseline_method": "loo",
        "bounded_parameter_handling": "transform",
        "random_seed": WORKFLOW_RANDOM_SEED,
        "sampling_method": "mc",
    }


def build_eki_kwargs(absolute_eki_directory: str):
    observations, observations_covariance = build_observations()
    return {
        "model": build_model(),
        "parameter_space": build_eki_parameter_space(),
        "observations": observations,
        "observations_covariance": observations_covariance,
        "parameter_mins": PARAMETER_MINS.copy(),
        "parameter_maxes": PARAMETER_MAXES.copy(),
        "absolute_eki_directory": absolute_eki_directory,
        "ensemble_size": 8,
        "initial_step_size": 1.0e-2,
        "regularization_parameter": 1.0e-6,
        "step_size_growth_factor": 1.05,
        "step_size_decay_factor": 2.0,
        "max_step_size_decrease_trys": 20,
        "relaxation_parameter": 1.2,
        "error_norm_tolerance": 0.0,
        "delta_params_tolerance": 0.0,
        "max_iterations": 10,
        "random_seed": WORKFLOW_RANDOM_SEED,
        "evaluation_concurrency": 1,
    }


def build_mf_vi_kwargs(absolute_vi_directory: str):
    observations, observations_covariance = build_observations()
    return {
        "model": build_model(),
        "prior_parameter_space": build_prior_parameter_space(),
        "observations": observations,
        "observations_covariance": observations_covariance,
        "parameter_mins": PARAMETER_MINS.copy(),
        "parameter_maxes": PARAMETER_MAXES.copy(),
        "absolute_vi_directory": absolute_vi_directory,
        "fom_sample_size": 8,
        "rom_extra_sample_size": 32,
        "rom_tolerance": 0.0,
        "max_rom_training_history": 2,
        "optimizer_method": "newton",
        "optimizer_config": romtools.workflows.VINewtonOptimizerConfig(
            max_iterations=10,
            gradient_norm_tolerance=0.0,
            newton_metric="standard",
            newton_regularization=1.0e-8,
        ),
        "line_search_method": "stochastic_nonmonotone",
        "line_search_config": romtools.workflows.VIStochasticNonmonotoneLineSearchConfig(
            initial_step_size=1.0,
            step_size_growth_factor=1.05,
            max_step_size=1.0,
            max_step_size_decrease_trys=10,
            line_search_armijo_coefficient=1.0e-4,
            line_search_uncertainty_sigma=1.0,
        ),
        "elbo_relative_tolerance": None,
        "fom_evaluation_concurrency": 1,
        "rom_evaluation_concurrency": 1,
        "baseline_method": "loo",
        "use_mfmc_control_variate": True,
        "mfmc_control_variate_mode": "componentwise",
        "bounded_parameter_handling": "transform",
        "rom_type": "gp",
        "rom_args": {
            "normalize_parameters": True,
            "normalize_targets": True,
            "auto_noise_variance": False,
            "noise_variance_fraction": 1.0e-6,
        },
        "random_seed": WORKFLOW_RANDOM_SEED,
        "sampling_method": "mc",
    }


def build_mf_eki_kwargs(absolute_eki_directory: str):
    observations, observations_covariance = build_observations()
    return {
        "model": build_model(),
        "parameter_space": build_eki_parameter_space(),
        "observations": observations,
        "observations_covariance": observations_covariance,
        "parameter_mins": PARAMETER_MINS.copy(),
        "parameter_maxes": PARAMETER_MAXES.copy(),
        "absolute_eki_directory": absolute_eki_directory,
        "fom_ensemble_size": 8,
        "rom_extra_ensemble_size": 32,
        "rom_tolerance": 0.0,
        "use_updated_rom_in_update_on_rebuild": True,
        "initial_step_size": 1.0e-2,
        "regularization_parameter": 1.0e-6,
        "step_size_growth_factor": 1.05,
        "step_size_decay_factor": 2.0,
        "max_step_size_decrease_trys": 20,
        "relaxation_parameter": 1.2,
        "error_norm_tolerance": 0.0,
        "delta_params_tolerance": 0.0,
        "max_rom_training_history": 2,
        "max_iterations": 10,
        "random_seed": WORKFLOW_RANDOM_SEED,
        "fom_evaluation_concurrency": 1,
        "rom_evaluation_concurrency": 1,
        "rom_type": "gp",
        "rom_args": {
            "normalize_parameters": True,
            "normalize_targets": True,
            "auto_noise_variance": False,
            "noise_variance_fraction": 1.0e-6,
        },
    }

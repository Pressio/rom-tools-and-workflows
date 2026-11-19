"""CDR PDE integration test for multifidelity random adaptive GP training.

Reuses the established steady CDR model with two spatial mesh resolutions.
A grid-independent scalar QoI (domain integral of the state) is extracted from
the same solver output; the original vector-QoI wrapper is not modified.
"""
import sys
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[3] / "examples" / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from steady_cdr_model import SteadyCdrQoiModel

from romtools.rom.multifidelity import (
    AutoregressiveGPBuilder, AutoregressiveGPConfig, GPLevelConfig,
)
from romtools.rom.qoi_surrogates import GaussianProcessKernel
from romtools.workflows.multifidelity import (
    FidelityLevel, MultiFidelityKFoldValidation, run_adaptive_multifidelity,
)
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import MonteCarloSampler


class IntegralSteadyCdrQoiModel(SteadyCdrQoiModel):
    """Scalar physical-domain integral of the steady CDR solution."""

    def compute_qoi(self, run_directory, parameter_sample):
        with np.load(Path(run_directory) / "solution.npz") as saved:
            solution = saved["state"]
        return np.array([solution.sum() * self.system.dx * self.system.dy])


def test_adaptive_multifidelity_on_steady_cdr(tmp_path):
    names = ["bmag", "theta", "nu", "sigma"]
    parameter_space = UniformParameterSpace(
        names,
        [0.2, 0.3, 0.002, 0.4],
        [0.8, 1.2, 0.010, 1.5],
        MonteCarloSampler,
    )
    low = IntegralSteadyCdrQoiModel(nx=9, ny=9)
    high = IntegralSteadyCdrQoiModel(nx=21, ny=21)

    builder = AutoregressiveGPBuilder(
        AutoregressiveGPConfig(
            levels=[
                GPLevelConfig(GaussianProcessKernel(length_scale=0.5),
                              noise_variance=1e-8),
                GPLevelConfig(GaussianProcessKernel(length_scale=0.5),
                              rho=1.0, noise_variance=1e-8),
            ],
            tune_hyperparameters=False,
        ),
        parameter_names=names,
    )
    result = run_adaptive_multifidelity(
        fidelities=[
            FidelityLevel(low, "coarse_9x9", cost=1),
            FidelityLevel(high, "fine_21x21", cost=10),
        ],
        surrogate_builder=builder,
        parameter_space=parameter_space,
        absolute_work_dir=str(tmp_path / "cdr_multifidelity"),
        tolerance=0.0,  # force all rounds unless the holdout error is exactly zero
        initial_samples=[9, 4],
        batch_sizes=[3, 1],
        adaptive_sampling_strategy="random",
        pair_fidelity_samples=True,
        validation_strategy=MultiFidelityKFoldValidation(num_folds=2),
        max_iterations=2,
        random_seed=2026,
    )

    np.testing.assert_array_equal(result.sample_counts, [[9, 4], [12, 5]])
    assert result.total_cost == 62
    assert result.validation_errors.shape == (2,)
    assert np.all(np.isfinite(result.validation_errors))
    assert np.all(result.validation_errors >= 0)
    assert result.termination_reason == "max_iterations"

    lf_data, hf_data = result.training_data
    assert lf_data.qois.shape == (12, 1)
    assert hf_data.qois.shape == (5, 1)
    assert set(map(tuple, hf_data.parameters)).issubset(
        set(map(tuple, lf_data.parameters))
    )
    # Discretization levels are correlated but not identical.
    paired_lf = {
        tuple(row): float(qoi[0])
        for row, qoi in zip(lf_data.parameters, lf_data.qois)
    }
    paired_errors = np.array([
        float(qoi[0]) - paired_lf[tuple(row)]
        for row, qoi in zip(hf_data.parameters, hf_data.qois)
    ])
    assert np.all(np.isfinite(paired_errors))
    assert np.any(np.abs(paired_errors) > 1e-8)

    # Check the final hierarchical surrogate against an *unseen* fine-grid solve.
    sample = dict(zip(names, [0.47, 0.73, 0.006, 0.91]))
    validation_dir = tmp_path / "independent_fine_grid"
    validation_dir.mkdir()
    high.populate_run_directory(str(validation_dir), sample)
    assert high.run_model(str(validation_dir), sample) == 0
    truth = high.compute_qoi(str(validation_dir), sample)
    prediction = result.rom_model.compute_qoi("", sample)
    std = result.rom_model.compute_error_estimate("", sample)
    assert truth.shape == prediction.shape == (1,)
    assert np.all(np.isfinite(prediction))
    assert np.isfinite(std) and std >= 0
    assert np.isfinite(abs(float(prediction[0] - truth[0])))
    assert (tmp_path / "cdr_multifidelity" / "multifidelity_stats.npz").is_file()

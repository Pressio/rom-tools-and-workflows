"""Regression tests for dry-run Monte Carlo workflow preparation."""

import os

import numpy as np
import pytest

from romtools.workflows.uq.monte_carlo import (
    MonteCarloDryRunResult,
    run_monte_carlo,
    run_multifidelity_monte_carlo,
)


class DummySpace:
    def get_names(self):
        return ["x"]

    def generate_samples(self, n, seed=1):
        return np.random.default_rng(seed).normal(size=(n, 1))


class PreparationOnlyModel:
    def __init__(self):
        self.prepared = []

    def populate_run_directory(self, run_directory, parameters):
        self.prepared.append((run_directory, float(parameters["x"])))
        with open(os.path.join(run_directory, "input.txt"), "w") as output:
            output.write(str(parameters["x"]))

    def run_model(self, run_directory, parameters):
        raise AssertionError("dry run must not execute the model")

    def compute_qoi(self, run_directory, parameters):
        raise AssertionError("dry run must not compute QoI")


def test_monte_carlo_dry_run(tmp_path):
    model = PreparationOnlyModel()
    work_dir = str(tmp_path / "mc")
    kwargs = dict(model=model, parameter_space=DummySpace(),
                  absolute_work_dir=work_dir, number_of_samples=3,
                  random_seed=5, dry_run=True)
    result = run_monte_carlo(**kwargs)
    assert isinstance(result, MonteCarloDryRunResult)
    assert len(result.run_directories) == 3
    assert len(model.prepared) == 3
    for directory in result.run_directories:
        assert os.path.isfile(os.path.join(directory, "input.txt"))
        assert not os.path.exists(os.path.join(directory, "passed.txt"))
    assert not os.path.exists(os.path.join(work_dir, "uq_stats.npz"))
    repeat = run_monte_carlo(**kwargs)
    np.testing.assert_array_equal(result.parameter_samples, repeat.parameter_samples)


def test_dry_run_skips_successful_samples(tmp_path):
    model = PreparationOnlyModel()
    directory = str(tmp_path / "mc")
    os.makedirs(os.path.join(directory, "run_0"))
    with open(os.path.join(directory, "run_0", "passed.txt"), "w") as output:
        output.write("0")
    run_monte_carlo(model, DummySpace(), directory, 2, dry_run=True)
    assert len(model.prepared) == 1
    run_monte_carlo(model, DummySpace(), directory, 2, dry_run=True, overwrite=True)
    assert len(model.prepared) == 3


def test_multifidelity_fixed_dry_run(tmp_path):
    high = PreparationOnlyModel()
    low = PreparationOnlyModel()
    result = run_multifidelity_monte_carlo(
        high, low, DummySpace(), str(tmp_path / "mf"),
        number_of_high_fidelity_samples=2,
        number_of_low_fidelity_samples=4,
        dry_run=True,
    )
    assert isinstance(result, MonteCarloDryRunResult)
    assert not result.pilot_only
    assert len(result.high_fidelity_run_directories) == 2
    assert len(result.low_fidelity_run_directories) == 4
    assert len(high.prepared) == 2
    assert len(low.prepared) == 4


def test_multifidelity_pilot_dry_run(tmp_path):
    high = PreparationOnlyModel()
    low = PreparationOnlyModel()
    result = run_multifidelity_monte_carlo(
        high, low, DummySpace(), str(tmp_path / "pilot"),
        pilot_sample_count=3, high_fidelity_equivalent_budget=15,
        dry_run=True,
    )
    assert result.pilot_only
    assert len(result.high_fidelity_run_directories) == 3
    assert len(result.low_fidelity_run_directories) == 3
    assert len(high.prepared) == len(low.prepared) == 3


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_dry_run_rejects_invalid_concurrency(tmp_path, concurrency):
    model = PreparationOnlyModel()
    with pytest.raises(ValueError, match="evaluation_concurrency"):
        run_monte_carlo(model, DummySpace(), str(tmp_path / "mc"),
                        2, evaluation_concurrency=concurrency, dry_run=True)
    assert model.prepared == []


@pytest.mark.parametrize("which", ["high", "low"])
def test_multifidelity_dry_run_rejects_invalid_concurrency(tmp_path, which):
    kwargs = {f"{which}_fidelity_evaluation_concurrency": 0}
    with pytest.raises(ValueError, match="evaluation_concurrency"):
        run_multifidelity_monte_carlo(
            PreparationOnlyModel(), PreparationOnlyModel(), DummySpace(),
            str(tmp_path / "mf"), number_of_high_fidelity_samples=2,
            number_of_low_fidelity_samples=3, dry_run=True, **kwargs,
        )


def test_dry_run_overwrite_removes_success_marker(tmp_path):
    model = PreparationOnlyModel()
    work_dir = str(tmp_path / "mc")
    completed = os.path.join(work_dir, "run_0")
    os.makedirs(completed)
    with open(os.path.join(completed, "passed.txt"), "w") as output:
        output.write("0")
    run_monte_carlo(model, DummySpace(), work_dir, 2, overwrite=True, dry_run=True)
    assert not os.path.exists(os.path.join(completed, "passed.txt"))


@pytest.mark.parametrize("budget,ratio,coefficients", [
    (4, 1.0, None),
    (0, 1.0, None),
    (float("nan"), 1.0, None),
    (15, 1.0, np.array([1.0])),
])
def test_pilot_dry_run_rejects_invalid_configuration(
    tmp_path, budget, ratio, coefficients,
):
    high = PreparationOnlyModel()
    low = PreparationOnlyModel()
    with pytest.raises(ValueError):
        run_multifidelity_monte_carlo(
            high, low, DummySpace(), str(tmp_path / "pilot"),
            pilot_sample_count=3, high_fidelity_equivalent_budget=budget,
            low_to_high_fidelity_cost_ratio=ratio,
            control_variate_coefficients=coefficients,
            dry_run=True,
        )
    assert not high.prepared and not low.prepared

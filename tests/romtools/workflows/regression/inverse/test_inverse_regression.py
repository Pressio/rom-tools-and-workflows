from pathlib import Path

import numpy as np
import pytest

import romtools.workflows

from tests.romtools.workflows.regression.inverse.cdr_regression_fixture import (
    build_eki_kwargs,
    build_mf_eki_kwargs,
    build_mf_vi_kwargs,
    build_vi_kwargs,
)


GOLD_DIRECTORY = Path(__file__).resolve().parent / "gold"


def _assert_matches_gold(result, gold_name: str):
    first, second, parameter_samples, qois = result
    with np.load(GOLD_DIRECTORY / gold_name) as gold:
        assert first.shape == gold["first"].shape
        assert second.shape == gold["second"].shape
        assert parameter_samples.shape == gold["parameter_samples"].shape
        assert qois.shape == gold["qois"].shape
        assert np.all(np.isfinite(first))
        assert np.all(np.isfinite(second))
        assert np.all(np.isfinite(parameter_samples))
        assert np.all(np.isfinite(qois))
        assert np.allclose(first, gold["first"], atol=1.0e-10, rtol=1.0e-9)
        assert np.allclose(second, gold["second"], atol=1.0e-10, rtol=1.0e-9)
        assert np.allclose(parameter_samples, gold["parameter_samples"], atol=1.0e-10, rtol=1.0e-9)
        assert np.allclose(qois, gold["qois"], atol=1.0e-10, rtol=1.0e-9)


@pytest.mark.mpi_skip
@pytest.mark.regression
def test_vi_inverse_workflow_regression(tmp_path):
    result = romtools.workflows.run_vi(
        **build_vi_kwargs(str(tmp_path / "vi")),
    )
    _assert_matches_gold(result, "vi_cdr_inverse_regression_gold.npz")


@pytest.mark.mpi_skip
@pytest.mark.regression
def test_mf_vi_inverse_workflow_regression(tmp_path):
    result = romtools.workflows.mf_vi_with_auto_rom(
        **build_mf_vi_kwargs(str(tmp_path / "mf_vi")),
    )
    _assert_matches_gold(result, "mf_vi_cdr_inverse_regression_gold.npz")


@pytest.mark.mpi_skip
@pytest.mark.regression
def test_eki_inverse_workflow_regression(tmp_path):
    parameter_samples, qois = romtools.workflows.run_eki(
        **build_eki_kwargs(str(tmp_path / "eki")),
    )
    _assert_matches_gold(
        (
            np.mean(parameter_samples, axis=0),
            np.std(parameter_samples, axis=0),
            parameter_samples,
            qois,
        ),
        "eki_cdr_inverse_regression_gold.npz",
    )


@pytest.mark.mpi_skip
@pytest.mark.regression
def test_mf_eki_inverse_workflow_regression(tmp_path):
    parameter_samples, qois = romtools.workflows.mf_eki_with_auto_rom(
        **build_mf_eki_kwargs(str(tmp_path / "mf_eki")),
    )
    _assert_matches_gold(
        (
            np.mean(parameter_samples, axis=0),
            np.std(parameter_samples, axis=0),
            parameter_samples,
            qois,
        ),
        "mf_eki_cdr_inverse_regression_gold.npz",
    )

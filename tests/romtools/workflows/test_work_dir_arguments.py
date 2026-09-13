"""Tests for standardized workflow work-directory arguments."""

import inspect

import pytest

from romtools.workflows.greedy.run_greedy import run_greedy
from romtools.workflows.inverse.ego_drivers import run_batch_ego, run_ego
from romtools.workflows.inverse.eki_drivers import run_eki
from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom, run_mf_eki
from romtools.workflows.inverse.mf_vi_drivers import mf_vi_with_auto_rom, run_mf_vi
from romtools.workflows.inverse.vi_drivers import run_vi
from romtools.workflows.sampling.sampling import run_sampling
from romtools.workflows.sampling_with_holdout.sampling_with_holdout import run_sampling_with_holdout
from romtools.workflows.uq.monte_carlo import run_monte_carlo, run_multifidelity_monte_carlo


WORKFLOWS = (
    run_sampling,
    run_sampling_with_holdout,
    run_greedy,
    run_monte_carlo,
    run_multifidelity_monte_carlo,
    run_ego,
    run_batch_ego,
    run_eki,
    run_mf_eki,
    mf_eki_with_auto_rom,
    run_vi,
    run_mf_vi,
    mf_vi_with_auto_rom,
)


def test_workflows_use_absolute_work_dir():
    for workflow in WORKFLOWS:
        assert "absolute_work_dir" in inspect.signature(workflow).parameters


def test_deprecated_sampling_keyword_warns_and_is_rejected_with_new_keyword():
    with pytest.warns(DeprecationWarning, match="absolute_sampling_directory"):
        with pytest.raises(TypeError, match="absolute_work_dir"):
            run_sampling(
                None,
                None,
                absolute_work_dir="new",
                absolute_sampling_directory="old",
            )


def test_deprecated_eki_keyword_warns_and_is_rejected_with_new_keyword():
    with pytest.warns(DeprecationWarning, match="absolute_eki_directory"):
        with pytest.raises(TypeError, match="absolute_work_dir"):
            run_eki(
                None,
                None,
                None,
                None,
                absolute_work_dir="new",
                absolute_eki_directory="old",
            )


def test_deprecated_vi_keyword_warns_and_is_rejected_with_new_keyword():
    with pytest.warns(DeprecationWarning, match="absolute_vi_directory"):
        with pytest.raises(TypeError, match="absolute_work_dir"):
            run_vi(
                None,
                None,
                None,
                None,
                absolute_work_dir="new",
                absolute_vi_directory="old",
            )

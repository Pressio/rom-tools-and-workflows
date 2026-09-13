"""Tests for standardized workflow working-directory arguments."""

import inspect

import pytest

from romtools.workflows._work_dir_compat import standardize_work_dir_argument


def _legacy_workflow(value, absolute_old_directory="default"):
    return value, absolute_old_directory


def test_new_work_dir_keyword_is_forwarded():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )
    assert workflow(1, absolute_work_dir="new") == (1, "new")


def test_old_work_dir_keyword_warns():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )
    with pytest.warns(DeprecationWarning, match="absolute_work_dir"):
        assert workflow(1, absolute_old_directory="old") == (1, "old")


def test_duplicate_work_dir_is_rejected():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )
    with pytest.raises(TypeError, match="specified more than once"):
        workflow(
            1,
            absolute_work_dir="new",
            absolute_old_directory="old",
        )


def test_workflow_signatures_use_absolute_work_dir():
    from romtools.workflows.greedy.run_greedy import run_greedy
    from romtools.workflows.inverse.ego_drivers import run_ego
    from romtools.workflows.inverse.eki_drivers import run_eki
    from romtools.workflows.inverse.mf_eki_drivers import mf_eki_with_auto_rom, run_mf_eki
    from romtools.workflows.inverse.mf_vi_drivers import mf_vi_with_auto_rom, run_mf_vi
    from romtools.workflows.inverse.vi_drivers import run_vi
    from romtools.workflows.sampling.sampling import run_sampling
    from romtools.workflows.sampling_with_holdout.sampling_with_holdout import run_sampling_with_holdout
    from romtools.workflows.uq.monte_carlo import run_monte_carlo, run_multifidelity_monte_carlo

    workflows = (
        run_sampling,
        run_sampling_with_holdout,
        run_greedy,
        run_monte_carlo,
        run_multifidelity_monte_carlo,
        run_ego,
        run_eki,
        run_mf_eki,
        mf_eki_with_auto_rom,
        run_vi,
        run_mf_vi,
        mf_vi_with_auto_rom,
    )
    for workflow in workflows:
        assert "absolute_work_dir" in inspect.signature(workflow).parameters

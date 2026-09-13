"""Tests for the standardized workflow working-directory argument."""

import inspect

import pytest

from romtools.workflows._work_dir_compat import standardize_work_dir_argument


def _legacy_workflow(value, absolute_old_directory="default", other=None):
    return value, absolute_old_directory, other


def test_standardized_work_dir_keyword_is_forwarded():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    result = workflow(1, absolute_work_dir="new", other=2)

    assert result == (1, "new", 2)


def test_legacy_work_dir_keyword_warns_and_is_forwarded():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    with pytest.warns(DeprecationWarning, match="absolute_work_dir"):
        result = workflow(1, absolute_old_directory="legacy")

    assert result == (1, "legacy", None)


def test_both_work_dir_keywords_are_rejected():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    with pytest.raises(TypeError, match="Specify only 'absolute_work_dir'"):
        workflow(
            1,
            absolute_work_dir="new",
            absolute_old_directory="legacy",
        )


def test_positional_work_dir_remains_backwards_compatible():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    assert workflow(1, "positional", 2) == (1, "positional", 2)


def test_positional_and_new_keyword_work_dir_are_rejected():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    with pytest.raises(TypeError, match="provided both positionally"):
        workflow(1, "positional", absolute_work_dir="new")


def test_public_signature_uses_standardized_name():
    workflow = standardize_work_dir_argument(
        _legacy_workflow, "absolute_old_directory"
    )

    parameters = inspect.signature(workflow).parameters
    assert "absolute_work_dir" in parameters
    assert "absolute_old_directory" not in parameters


def test_repo_workflow_signatures_use_absolute_work_dir():
    from romtools.workflows.greedy.run_greedy import run_greedy
    from romtools.workflows.inverse.ego_drivers import run_ego
    from romtools.workflows.inverse.eki_drivers import run_eki
    from romtools.workflows.inverse.mf_eki_drivers import (
        mf_eki_with_auto_rom,
        run_mf_eki,
    )
    from romtools.workflows.inverse.mf_vi_drivers import (
        mf_vi_with_auto_rom,
        run_mf_vi,
    )
    from romtools.workflows.inverse.vi_drivers import run_vi
    from romtools.workflows.sampling.sampling import run_sampling
    from romtools.workflows.sampling_with_holdout.sampling_with_holdout import (
        run_sampling_with_holdout,
    )
    from romtools.workflows.uq.monte_carlo import (
        run_monte_carlo,
        run_multifidelity_monte_carlo,
    )

    workflows = [
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
    ]

    for workflow in workflows:
        parameters = inspect.signature(workflow).parameters
        assert "absolute_work_dir" in parameters, workflow.__qualname__

import inspect

import romtools.workflows.inverse as inverse
from romtools.workflows.inverse import full_covariance_router as router


def test_newton_defaults_are_natural_full_lagged_and_regularized():
    config = inverse.VINewtonOptimizerConfig()
    assert config.newton_metric == "natural"
    assert config.newton_hessian_type == "full"
    assert config.newton_regularization == 5.0e-4
    assert config.newton_curvature_strategy == "lagged"
    assert config.newton_hessian_averaging_factor == 0.25


def test_line_search_initial_step_defaults_to_point_one():
    assert inverse.VILegacyLineSearchConfig().initial_step_size == 0.1
    assert (
        inverse.VIStochasticNonmonotoneLineSearchConfig().initial_step_size
        == 0.1
    )


def test_public_mf_vi_signatures_expose_full_training_history_default():
    assert (
        inspect.signature(inverse.run_mf_vi)
        .parameters["max_rom_training_history"]
        .default
        is None
    )
    assert (
        inspect.signature(inverse.mf_vi_with_auto_rom)
        .parameters["max_rom_training_history"]
        .default
        is None
    )


def test_router_materializes_default_newton_config():
    _, kwargs = router._ensure_default_newton_config(
        router._legacy_run_vi,
        (),
        {"optimizer_method": "newton"},
    )
    config = kwargs["optimizer_config"]
    assert config.newton_metric == "natural"
    assert config.newton_hessian_type == "full"
    assert config.newton_regularization == 5.0e-4
    assert config.newton_curvature_strategy == "lagged"
    assert config.newton_hessian_averaging_factor == 0.25


def test_router_uses_effectively_full_mf_training_history_by_default():
    _, kwargs = router._ensure_full_rom_training_history(
        router._legacy_run_mf_vi,
        (),
        {"fom_sample_size": 16},
    )
    assert kwargs["max_rom_training_history"] > 1_000_000


def test_explicit_mf_training_history_is_preserved():
    _, kwargs = router._ensure_full_rom_training_history(
        router._legacy_run_mf_vi,
        (),
        {"fom_sample_size": 16, "max_rom_training_history": 4},
    )
    assert kwargs["max_rom_training_history"] == 4

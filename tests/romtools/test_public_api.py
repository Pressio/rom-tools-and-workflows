"""Regression tests for the supported romtools public import surface."""

from importlib import metadata

import romtools


def test_version_matches_installed_distribution():
    try:
        expected_version = metadata.version("romtools")
    except metadata.PackageNotFoundError:
        expected_version = "0+unknown"

    assert romtools.__version__ == expected_version


def test_top_level_public_api_is_domain_oriented():
    assert set(romtools.__all__) == {
        "__version__",
        "vector_space",
        "composite_vector_space",
        "hyper_reduction",
        "linalg",
        "rom",
        "workflows",
        "hpc",
    }


def test_top_level_star_import_does_not_export_implementation_details():
    namespace = {}
    exec("from romtools import *", namespace)  # pylint: disable=exec-used

    assert "np" not in namespace
    assert "Protocol" not in namespace
    assert "Callable" not in namespace
    assert "utils" not in namespace
    assert "run_eki" not in namespace
    assert "GaussianProcessQoiModel" not in namespace


def test_historical_flat_aliases_remain_available():
    assert romtools.VectorSpace is romtools.vector_space.VectorSpace
    assert romtools.VectorSpaceFromPOD is romtools.vector_space.VectorSpaceFromPOD
    assert romtools.DictionaryVectorSpace is romtools.vector_space.DictionaryVectorSpace
    assert romtools.run_eki is romtools.workflows.run_eki
    assert romtools.NeuralNetworkConfig is romtools.rom.NeuralNetworkConfig
    assert romtools.DEIM is romtools.hyper_reduction.DEIM


def test_domain_packages_define_public_exports():
    assert "run_eki" in romtools.workflows.__all__
    assert "run_sampling" in romtools.workflows.__all__
    assert "run_monte_carlo" in romtools.workflows.__all__

    assert "GaussianProcessQoiModel" in romtools.rom.__all__
    assert "NeuralNetworkQoiModelBuilderWithTrainingData" in romtools.rom.__all__

    assert "DEIM" in romtools.hyper_reduction.__all__
    assert "QDEIM" in romtools.hyper_reduction.__all__
    assert "ecsw" in romtools.hyper_reduction.__all__


def test_inverse_exports_preserve_wrapper_layers():
    import inspect
    from romtools.workflows import inverse
    from romtools.workflows.inverse import eki_bound_transforms, full_covariance_router

    assert inverse.run_eki is eki_bound_transforms.run_eki
    assert inverse.run_mf_eki is eki_bound_transforms.run_mf_eki
    assert inverse.run_vi is full_covariance_router.run_vi
    assert inverse.run_mf_vi is full_covariance_router.run_mf_vi
    assert inverse.mf_vi_with_auto_rom is full_covariance_router.mf_vi_with_auto_rom
    assert inverse.vi_drivers.run_vi is inverse.run_vi
    assert inverse.mf_vi_drivers.run_mf_vi is inverse.run_mf_vi
    assert inverse.eki_drivers.run_eki is inverse.run_eki
    assert "VISampleReuseConfig" in inverse.__all__
    for driver in (inverse.run_vi, inverse.run_mf_vi, inverse.mf_vi_with_auto_rom):
        parameters = inspect.signature(driver).parameters
        assert "initial_variational_parameter_space" in parameters
        assert "sample_reuse_config" in parameters
        assert "create_run_directories" in parameters


def test_workflow_configuration_aliases_remain_available():
    from romtools.workflows.parameter_spaces import ParameterSpace
    from romtools.workflows.inverse import vi_optimization_methods

    assert romtools.workflows.ParameterSpace is ParameterSpace
    for name in (
        "VIAdamOptimizerConfig", "VIGradientOptimizerConfig", "VINewtonOptimizerConfig",
        "VILegacyLineSearchConfig", "VIStochasticNonmonotoneLineSearchConfig",
    ):
        expected = getattr(vi_optimization_methods, name)
        assert getattr(romtools.workflows, name) is expected
        assert getattr(romtools.workflows.inverse, name) is expected


def test_vector_space_exports_exclude_imported_implementation_helpers():
    from romtools import vector_space
    from romtools.vector_space import utils

    for package in (vector_space, utils):
        namespace = {}
        exec("from " + package.__name__ + " import *", namespace)
        assert set(namespace) - {"__builtins__"} == set(package.__all__)
        for helper in ("np", "Protocol", "Callable", "Tuple", "la"):
            assert helper not in namespace
    assert "VectorSpaceFromStreamingPOD" in vector_space.__all__
    assert "ScalarScaler" in utils.__all__
    assert "create_streaming_average_shifter" in utils.__all__


def test_historical_hyper_reduction_procedural_imports_remain_available():
    from romtools import hyper_reduction

    exports = {
        "deim": (
            "qdeim_get_indices", "deim_get_indices", "multi_state_deim_get_indices",
            "deim_get_approximation_matrix", "multi_state_deim_get_test_basis",
            "deim_get_test_basis",
        ),
        "ecsw": ("ecsw_fixed_test_basis", "ecsw_varying_test_basis", "ecsw_lspg_zero_residual"),
    }
    for module_name, names in exports.items():
        module = getattr(hyper_reduction, module_name)
        for name in names:
            expected = getattr(module, name)
            assert getattr(hyper_reduction, name) is expected
            assert getattr(romtools, name) is expected
            assert name in hyper_reduction.__all__

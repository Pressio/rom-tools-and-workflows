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

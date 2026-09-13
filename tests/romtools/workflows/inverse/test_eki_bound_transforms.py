import inspect
from importlib import import_module

import numpy as np
import pytest

import romtools.workflows

transforms = import_module("romtools.workflows.inverse.eki_bound_transforms")


class DeterministicParameterSpace(romtools.workflows.ParameterSpace):
    def get_names(self):
        return ["theta"]

    def get_dimensionality(self):
        return 1

    def generate_samples(self, number_of_samples: int, seed=None):
        return np.linspace(-1.5, 1.5, number_of_samples)[:, None]


class LinearQoiModel:
    def __init__(self):
        self.seen_parameters = []

    def populate_run_directory(self, run_directory, parameter_sample):
        self.seen_parameters.append(float(parameter_sample["theta"]))

    def run_model(self, run_directory, parameter_sample):
        self.seen_parameters.append(float(parameter_sample["theta"]))
        return 0

    def compute_qoi(self, run_directory, parameter_sample):
        theta = float(parameter_sample["theta"])
        self.seen_parameters.append(theta)
        return np.array([theta])


class RecordingBuilder:
    def __init__(self):
        self.training_parameters = None
        self.model = LinearQoiModel()

    def build_from_training_dirs(self,
                                 offline_data_dir,
                                 training_data_dirs,
                                 training_parameters,
                                 training_qois):
        self.training_parameters = np.asarray(training_parameters).copy()
        return self.model


def test_public_eki_modules_expose_bound_handling_options():
    eki_module = import_module("romtools.workflows.inverse.eki_drivers")
    mf_eki_module = import_module("romtools.workflows.inverse.mf_eki_drivers")
    assert "bounded_parameter_handling" in inspect.signature(eki_module.run_eki).parameters
    assert "bounded_parameter_handling" in inspect.signature(mf_eki_module.run_mf_eki).parameters
    assert "bounded_parameter_handling" in inspect.signature(
        mf_eki_module.mf_eki_with_auto_rom
    ).parameters


@pytest.mark.parametrize("transform_map", ["sigmoid", "arctan"])
def test_optimizer_parameter_space_preserves_physical_initial_draw(transform_map):
    parameter_space = DeterministicParameterSpace()
    lower = np.array([-1.0])
    upper = np.array([1.0])
    adapter = transforms._OptimizerParameterSpace(
        parameter_space,
        lower,
        upper,
        1e-6,
        transform_map,
    )

    optimizer_samples = adapter.generate_samples(5)
    physical_samples = transforms._map_optimizer_to_physical(
        optimizer_samples,
        lower,
        upper,
        1e-6,
        transform_map,
    )
    expected = np.clip(parameter_space.generate_samples(5), lower, upper)
    expected = np.clip(expected, lower + 2e-6, upper - 2e-6)
    np.testing.assert_allclose(physical_samples, expected, rtol=0.0, atol=3e-6)
    assert np.all(physical_samples > lower)
    assert np.all(physical_samples < upper)


def test_model_adapter_only_exposes_physical_parameters():
    model = LinearQoiModel()
    lower = np.array([-2.0])
    upper = np.array([4.0])
    adapter = transforms._PhysicalParameterQoiModel(
        model,
        ["theta"],
        lower,
        upper,
        0.0,
        "sigmoid",
    )

    adapter.populate_run_directory("unused", {"theta": 0.0})
    adapter.run_model("unused", {"theta": 0.0})
    qoi = adapter.compute_qoi("unused", {"theta": 0.0})

    np.testing.assert_allclose(qoi, np.array([1.0]))
    np.testing.assert_allclose(model.seen_parameters, np.ones(3))


def test_mf_builder_receives_physical_training_parameters():
    builder = RecordingBuilder()
    lower = np.array([-1.0])
    upper = np.array([3.0])
    adapter = transforms._PhysicalParameterQoiModelBuilder(
        builder,
        ["theta"],
        lower,
        upper,
        0.0,
        "sigmoid",
    )
    optimizer_training = np.array([[0.0], [np.log(3.0)]])
    wrapped_model = adapter.build_from_training_dirs(
        "unused",
        ["a", "b"],
        optimizer_training,
        np.zeros((2, 1)),
    )

    np.testing.assert_allclose(
        builder.training_parameters[:, 0],
        np.array([1.0, 2.0]),
    )
    qoi = wrapped_model.compute_qoi("unused", {"theta": 0.0})
    np.testing.assert_allclose(qoi, np.array([1.0]))


def test_legacy_restart_is_converted_to_optimizer_coordinates(tmp_path):
    lower = np.array([-1.0])
    upper = np.array([1.0])
    physical_samples = np.array([[-0.5], [0.0], [0.5]])
    restart_file = tmp_path / "restart.npz"
    np.savez(
        restart_file,
        parameter_samples=physical_samples,
        iteration=1,
        step_size=0.1,
        qois=np.zeros((1, 3)),
        mean_qoi=np.zeros(1),
        errors=np.zeros((1, 3)),
        rejuvenation_reference_covariance=np.eye(1),
    )

    with transforms._optimizer_restart_file(
        str(restart_file),
        lower,
        upper,
        0.0,
        "sigmoid",
    ) as optimizer_restart:
        with np.load(optimizer_restart) as data:
            optimizer_samples = data["parameter_samples"]
            assert str(data["parameter_sample_coordinates"]) == "optimizer"
            mapped_back = transforms._map_optimizer_to_physical(
                optimizer_samples,
                lower,
                upper,
                0.0,
                "sigmoid",
            )
            np.testing.assert_allclose(mapped_back, physical_samples)
            expected_covariance = np.cov(optimizer_samples, rowvar=False, ddof=1)
            np.testing.assert_allclose(
                np.atleast_2d(data["rejuvenation_reference_covariance"]),
                np.atleast_2d(expected_covariance),
            )


def test_transform_requires_complete_valid_bounds(tmp_path):
    parameter_space = DeterministicParameterSpace()
    with pytest.raises(ValueError, match="required"):
        romtools.workflows.run_eki(
            model=LinearQoiModel(),
            parameter_space=parameter_space,
            observations=np.zeros(1),
            observations_covariance=np.eye(1),
            parameter_mins=np.array([-1.0]),
            parameter_maxes=None,
            absolute_work_dir=str(tmp_path),
            bounded_parameter_handling="transform",
            max_iterations=1,
        )


def test_transformed_eki_returns_and_restarts_in_physical_coordinates(tmp_path):
    model = LinearQoiModel()
    lower = np.array([-1.0])
    upper = np.array([1.0])
    work_dir = tmp_path / "transformed"

    samples, _ = romtools.workflows.run_eki(
        model=model,
        parameter_space=DeterministicParameterSpace(),
        observations=np.array([0.9]),
        observations_covariance=np.array([[1e-2]]),
        parameter_mins=lower,
        parameter_maxes=upper,
        absolute_work_dir=str(work_dir),
        ensemble_size=5,
        initial_step_size=0.5,
        relaxation_parameter=2.0,
        delta_params_tolerance=0.0,
        max_iterations=2,
        bounded_parameter_handling="transform",
        transform_interior_margin=1e-6,
        transform_map="sigmoid",
    )

    assert np.all(samples > lower)
    assert np.all(samples < upper)
    assert np.all(np.asarray(model.seen_parameters) > lower[0])
    assert np.all(np.asarray(model.seen_parameters) < upper[0])

    restart_files = sorted(work_dir.glob("iteration_*/restart.npz"))
    assert restart_files
    with np.load(restart_files[-1]) as data:
        assert str(data["bounded_parameter_handling"]) == "transform"
        assert str(data["parameter_sample_coordinates"]) == "physical"
        assert str(data["transform_map"]) == "sigmoid"
        persisted_samples = data["parameter_samples"]
        assert np.all(persisted_samples > lower)
        assert np.all(persisted_samples < upper)


def test_transformed_mf_eki_adapts_model_builder_and_returned_samples(monkeypatch, tmp_path):
    lower = np.array([-1.0])
    upper = np.array([1.0])
    builder = RecordingBuilder()
    physical_model = LinearQoiModel()

    def fake_legacy_run_mf_eki(**kwargs):
        assert kwargs["parameter_mins"] is None
        assert kwargs["parameter_maxes"] is None
        optimizer_samples = kwargs["parameter_space"].generate_samples(4)
        wrapped_rom = kwargs["rom_model_builder"].build_from_training_dirs(
            "unused",
            ["a", "b"],
            optimizer_samples[:2],
            np.zeros((2, 1)),
        )
        wrapped_rom.compute_qoi("unused", {"theta": optimizer_samples[0, 0]})
        return optimizer_samples[:2], np.zeros((1, 2))

    monkeypatch.setattr(transforms, "_legacy_run_mf_eki", fake_legacy_run_mf_eki)
    samples, _ = transforms.run_mf_eki(
        model=physical_model,
        rom_model_builder=builder,
        parameter_space=DeterministicParameterSpace(),
        observations=np.zeros(1),
        observations_covariance=np.eye(1),
        parameter_mins=lower,
        parameter_maxes=upper,
        absolute_work_dir=str(tmp_path),
        bounded_parameter_handling="transform",
    )

    assert np.all(samples > lower)
    assert np.all(samples < upper)
    assert builder.training_parameters is not None
    assert np.all(builder.training_parameters > lower)
    assert np.all(builder.training_parameters < upper)
    assert builder.model.seen_parameters
    assert all(lower[0] < value < upper[0] for value in builder.model.seen_parameters)

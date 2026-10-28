"""Batch boundaries, validation data flow, process isolation, and failures."""

import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pytest

from romtools.workflows.inverse.mf_eki_drivers import GaussianProcessQoiModelBuilderWithTrainingData
from romtools.workflows.space_filling import (
    KFoldValidation, relative_l2_error, run_space_filling,
)
from .test_sampling import FixedSpace


class FileModel:
    """Importable spawn-compatible model, including deliberately legacy chdir."""

    def __init__(self, failure=None, chdir=False):
        self.failure = failure
        self.chdir = chdir

    def populate_run_directory(self, directory, parameters):
        if self.chdir:
            os.chdir(directory)
        if self.failure == "prepare" and parameters["p0"] == 0:
            raise ValueError("preparation failed")
        np.savez(os.path.join(directory, "input.npz"), x=parameters["p0"])

    def run_model(self, directory, parameters):
        if self.chdir:
            os.chdir(directory)
        x = float(parameters["p0"])
        np.testing.assert_array_equal(np.load(os.path.join(directory, "input.npz"))["x"], x)
        if self.failure == "code" and x == 0:
            return 2
        if self.failure == "run" and x == 0:
            raise ValueError("solver failed")
        qoi = np.array([1 + x, 2 * x - 1])
        if self.failure == "nonfinite" and x == 0:
            qoi[0] = np.nan
        if self.failure == "dimension" and x == 0:
            qoi = np.ones(3)
        if self.failure == "empty" and x == 0:
            qoi = np.array([])
        np.savez(os.path.join(directory, "solution.npz"), qoi=qoi)
        return 0

    def compute_qoi(self, directory, parameters):
        if self.failure == "qoi" and parameters["p0"] == 0:
            raise ValueError("QoI extraction failed")
        with np.load(os.path.join(directory, "solution.npz")) as data:
            return data["qoi"]


class TrackingModel(FileModel):
    def __init__(self, active, peak, lock, event):
        super().__init__(chdir=True)
        self.active, self.peak, self.lock, self.event = active, peak, lock, event

    def run_model(self, directory, parameters):
        with self.lock:
            self.active.value += 1
            self.peak.value = max(self.peak.value, self.active.value)
            if self.active.value == 2:
                self.event.set()
        try:
            if not self.event.wait(15):
                raise RuntimeError("second worker did not start")
            # Arrange completion order differently from selection order.
            time.sleep(0.02 * (5 - int(parameters["p0"])))
            return super().run_model(directory, parameters)
        finally:
            with self.lock:
                self.active.value -= 1


class LinearRom:
    def __init__(self, coefficients, trained, reject_training_eval=False):
        self.coefficients = coefficients
        self.trained = trained
        self.reject_training_eval = reject_training_eval

    def populate_run_directory(self, directory, parameters):
        pass

    def run_model(self, directory, parameters):
        return 0

    def compute_qoi(self, directory, parameters):
        x = float(parameters["p0"])
        if self.reject_training_eval and x in self.trained:
            raise AssertionError("held-out sample leaked into this ROM's training set")
        return np.array([1.0, x]) @ self.coefficients


class RecordingBuilder:
    def __init__(self, reject_training_eval=False):
        self.calls = []
        self.reject_training_eval = reject_training_eval

    def build_from_training_dirs(self, offline, directories, parameters, qois):
        assert Path(offline).is_dir()
        assert len(directories) == len(parameters) == len(qois)
        for directory, parameter, qoi in zip(directories, parameters, qois):
            assert Path(directory, "passed.txt").exists()
            np.testing.assert_array_equal(np.load(Path(directory, "input.npz"))["x"], parameter[0])
            np.testing.assert_array_equal(np.load(Path(directory, "solution.npz"))["qoi"], qoi)
        self.calls.append((offline, tuple(directories), parameters.copy(), qois.copy()))
        coefficients = np.linalg.lstsq(np.c_[np.ones(len(parameters)), parameters[:, 0]], qois, rcond=None)[0]
        return LinearRom(coefficients, set(parameters[:, 0]), self.reject_training_eval)


def request(tmp_path, **kwargs):
    options = dict(
        fom_model=FileModel(), rom_model_builder=RecordingBuilder(),
        parameter_space=FixedSpace(np.arange(8)[:, None]),
        absolute_work_dir=str(tmp_path), tolerance=0,
        initial_sample_size=3, batch_size=2, max_concurrent_fom_runs=1,
        max_iterations=3, validation_strategy=lambda context: 1.0,
    )
    options.update(kwargs)
    return options


def test_accumulation_budget_and_partial_final_batch(tmp_path):
    options = request(tmp_path, max_fom_samples=6)
    result = run_space_filling(**options)
    np.testing.assert_array_equal(result.sample_counts, [3, 5, 6])
    assert not result.converged and result.termination_reason == "max_fom_samples"
    assert [len(call[1]) for call in options["rom_model_builder"].calls] == [3, 5, 6]
    assert len(set(result.training_parameters[:, 0])) == 6
    assert result.rom_model is not None
    with np.load(tmp_path / "space_filling_stats.npz") as stats:
        np.testing.assert_array_equal(stats["training_qois"], result.training_qois)
        assert str(stats["termination_reason"]) == result.termination_reason
        assert stats["fom_run_times"].shape == (6,)


def test_callback_and_convergence(tmp_path):
    contexts = []
    def validate(context):
        contexts.append(context)
        assert not context.training_parameters.flags.writeable
        assert not context.training_qois.flags.writeable
        assert context.error_functor is relative_l2_error
        assert Path(context.validation_directory).is_dir()
        return 1.0 / len(contexts)
    result = run_space_filling(**request(tmp_path, tolerance=0.5, validation_strategy=validate))
    assert result.converged and result.termination_reason == "tolerance"
    np.testing.assert_array_equal(result.validation_errors, [1, 0.5])
    assert [c.iteration for c in contexts] == [0, 1]
    assert contexts[-1].rom_model is result.rom_model


def test_iteration_limit_and_pool_exhaustion(tmp_path):
    limited = run_space_filling(**request(tmp_path / "limited", max_iterations=1))
    assert limited.termination_reason == "max_iterations"
    exhausted = run_space_filling(**request(
        tmp_path / "exhausted", parameter_space=FixedSpace([[0], [1]]),
    ))
    assert exhausted.termination_reason == "candidate_pool_exhausted"
    np.testing.assert_array_equal(exhausted.sample_counts, [2])
    with np.load(tmp_path / "exhausted/space_filling_stats.npz") as stats:
        assert str(stats["termination_reason"]) == "candidate_pool_exhausted"


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("failure", ["prepare", "code", "run", "qoi", "nonfinite", "empty", "dimension"])
def test_failed_batches_never_build_and_finish_all_tasks(tmp_path, concurrency, failure):
    options = request(
        tmp_path, fom_model=FileModel(failure=failure, chdir=True),
        parameter_space=FixedSpace([[0], [1], [2]]),
        max_concurrent_fom_runs=concurrency,
    )
    original_cwd = os.getcwd()
    with pytest.raises(RuntimeError, match="sample.*run_"):
        run_space_filling(**options)
    assert os.getcwd() == original_cwd
    assert not options["rom_model_builder"].calls
    # The rest of a failing batch still completed, even at concurrency one.
    assert len(list(tmp_path.glob("fom/run_*/solution.npz"))) >= 2
    assert not (tmp_path / "space_filling_stats.npz").exists()
    assert "Failed:" in (tmp_path / "space_filling_status.log").read_text()


def test_concurrent_limit_is_independent_of_batch_size_and_results_are_ordered(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        active, peak = manager.Value("i", 0), manager.Value("i", 0)
        model = TrackingModel(active, peak, manager.Lock(), manager.Event())
        result = run_space_filling(**request(
            tmp_path, fom_model=model, initial_sample_size=5,
            parameter_space=FixedSpace(np.arange(5)[:, None]),
            max_concurrent_fom_runs=2, tolerance=1,
        ))
        assert peak.value == 2 and active.value == 0
    np.testing.assert_allclose(result.training_qois[:, 0], 1 + result.training_parameters[:, 0])
    assert [Path(d).name for d in result.training_dirs] == [f"run_{i}" for i in range(5)]
    assert all(Path(d, "input.npz").exists() for d in result.training_dirs)


def test_kfold_is_disjoint_and_production_rom_uses_all_data(tmp_path):
    builder = RecordingBuilder(reject_training_eval=True)
    result = run_space_filling(**request(
        tmp_path, initial_sample_size=6, rom_model_builder=builder,
        validation_strategy=KFoldValidation(3), tolerance=1e-10,
    ))
    assert result.converged
    assert [len(call[1]) for call in builder.calls] == [6, 4, 4, 4]
    assert len(result.rom_model.trained) == 6
    assert len(list(tmp_path.glob("iteration_0/validation/fold_*/run_*/passed.txt"))) == 6
    counts = {d: sum(d not in call[1] for call in builder.calls[1:]) for d in result.training_dirs}
    assert set(counts.values()) == {1}


def test_custom_error_functor_is_not_normalized_and_needs_no_pickling(tmp_path):
    seen = []
    def functor(prediction, reference):
        assert prediction.shape == reference.shape == (2,)
        seen.append(reference[0])
        return reference[0]  # Deliberately different from relative L2.
    result = run_space_filling(**request(
        tmp_path, initial_sample_size=5, validation_strategy=KFoldValidation(2),
        error_functor=functor, tolerance=100, max_concurrent_fom_runs=2,
    ))
    assert len(seen) == 5
    assert result.validation_errors[0] == max(result.training_qois[:, 0])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1, [1], True, 1j])
def test_invalid_error_functor_results_have_fold_sample_context(tmp_path, invalid):
    with pytest.raises(ValueError, match="validation fold.*sample.*finite nonnegative scalar"):
        run_space_filling(**request(
            tmp_path, validation_strategy=KFoldValidation(2), error_functor=lambda a, b: invalid,
        ))


def test_relative_l2_and_zero_reference_floor():
    assert relative_l2_error(np.array([3, 4]), np.zeros(2)) == pytest.approx(5e12)
    assert relative_l2_error(np.zeros(2), np.zeros(2)) == 0
    assert relative_l2_error(np.array([6, 8]), np.array([3, 4])) == 1


def test_gp_builder_integration_and_default_validation(tmp_path):
    builder = GaussianProcessQoiModelBuilderWithTrainingData(parameter_names=["p0"])
    result = run_space_filling(**request(
        tmp_path, rom_model_builder=builder, validation_strategy=None,
        initial_sample_size=5, tolerance=100,
    ))
    assert result.converged
    assert len(list(tmp_path.glob("iteration_0/validation/fold_*/offline_data"))) == 5
    assert np.isfinite(result.validation_errors).all()


def test_selection_is_reproducible_independent_of_validation_randomness(tmp_path):
    def random_validation(context):
        np.random.uniform(size=100)
        return 1.0
    first = run_space_filling(**request(tmp_path / "first"))
    second = run_space_filling(**request(tmp_path / "second", validation_strategy=random_validation))
    np.testing.assert_array_equal(first.training_parameters, second.training_parameters)
    np.testing.assert_array_equal(first.training_qois, second.training_qois)


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"max_concurrent_fom_runs": True}, {"max_iterations": 0},
    {"max_fom_samples": 0}, {"tolerance": -1}, {"tolerance": np.nan},
    {"error_functor": 2}, {"validation_strategy": 3},
    {"validation_strategy": None, "initial_sample_size": 3},
    {"validation_strategy": None, "initial_sample_size": 10, "max_fom_samples": 4},
    {"absolute_work_dir": "relative"},
])
def test_bad_configuration_fails_before_creating_campaign(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        run_space_filling(**request(tmp_path / "campaign", **options))
    assert not (tmp_path / "campaign").exists()


def test_nonempty_output_directory_is_preserved(tmp_path):
    (tmp_path / "important.txt").write_text("keep me")
    with pytest.raises(ValueError, match="absent or empty"):
        run_space_filling(**request(tmp_path))
    assert (tmp_path / "important.txt").read_text() == "keep me"


def test_insufficient_distinct_initial_samples_do_not_launch_fom(tmp_path):
    with pytest.raises(ValueError, match="distinct samples"):
        run_space_filling(**request(
            tmp_path, initial_sample_size=5, parameter_space=FixedSpace([[0], [0]]),
            validation_strategy=None,
        ))
    assert not list(tmp_path.glob("fom/run_*"))


def test_builder_failure_keeps_previous_round_history(tmp_path):
    class FailingBuilder(RecordingBuilder):
        def build_from_training_dirs(self, *args):
            if len(self.calls) == 1:
                raise ValueError("build failed")
            return super().build_from_training_dirs(*args)
    with pytest.raises(ValueError, match="build failed"):
        run_space_filling(**request(tmp_path, rom_model_builder=FailingBuilder()))
    with np.load(tmp_path / "space_filling_stats.npz") as stats:
        np.testing.assert_array_equal(stats["sample_counts"], [3])
    assert len(list(tmp_path.glob("fom/run_*/passed.txt"))) == 5


class NumpyRandomModel(FileModel):
    def run_model(self, directory, parameters):
        np.savez(os.path.join(directory, "solution.npz"), qoi=np.random.normal(size=2))
        return 0


def test_stochastic_fom_reproducible_across_concurrency_limits(tmp_path):
    serial = run_space_filling(**request(tmp_path / "serial", fom_model=NumpyRandomModel()))
    parallel = run_space_filling(**request(
        tmp_path / "parallel", fom_model=NumpyRandomModel(), max_concurrent_fom_runs=2,
    ))
    np.testing.assert_array_equal(serial.training_parameters, parallel.training_parameters)
    np.testing.assert_array_equal(serial.training_qois, parallel.training_qois)


def test_second_batch_failure_preserves_previous_round(tmp_path):
    class BatchSpace(FixedSpace):
        calls = 0
        def generate_samples(self, count, seed=None):
            self.calls += 1
            return np.array([[1], [2]]) if self.calls == 1 else np.array([[0], [3]])
    options = request(
        tmp_path, initial_sample_size=2, parameter_space=BatchSpace([[0]]),
        fom_model=FileModel(failure="code"), max_concurrent_fom_runs=2,
    )
    with pytest.raises(RuntimeError, match="nonzero model return code"):
        run_space_filling(**options)
    assert len(options["rom_model_builder"].calls) == 1
    with np.load(tmp_path / "space_filling_stats.npz") as stats:
        np.testing.assert_array_equal(stats["sample_counts"], [2])
        assert len(stats["training_parameters"]) == 2
        assert not bool(stats["converged"])
    assert len(list(tmp_path.glob("fom/run_*/passed.txt"))) == 3


def test_converged_result_takes_precedence_over_limits(tmp_path):
    result = run_space_filling(**request(
        tmp_path, initial_sample_size=10, max_fom_samples=2, max_iterations=1, tolerance=1,
    ))
    assert result.termination_reason == "tolerance"
    np.testing.assert_array_equal(result.sample_counts, [2])


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, [1], True])
def test_invalid_validation_callback_result(tmp_path, value):
    with pytest.raises(ValueError, match="validation strategy result"):
        run_space_filling(**request(tmp_path, validation_strategy=lambda c: value))
    assert not (tmp_path / "space_filling_stats.npz").exists()


def test_error_functor_exception_has_context(tmp_path):
    def broken(a, b):
        raise LookupError("custom error failed")
    with pytest.raises(ValueError, match="validation fold.*sample.*custom error failed") as error:
        run_space_filling(**request(
            tmp_path, validation_strategy=KFoldValidation(2), error_functor=broken,
        ))
    assert isinstance(error.value.__cause__, LookupError)


def test_custom_validation_receives_error_functor(tmp_path):
    functor = lambda a, b: 7.0
    def validate(context):
        assert context.error_functor is functor
        return context.error_functor(np.ones(2), np.ones(2))
    result = run_space_filling(**request(
        tmp_path, validation_strategy=validate, error_functor=functor, tolerance=7,
    ))
    assert result.converged and result.validation_errors[0] == 7


def test_validation_builder_failure_is_not_reported_as_convergence(tmp_path):
    class FoldFailureBuilder(RecordingBuilder):
        def build_from_training_dirs(self, *args):
            if self.calls:
                raise ValueError("fold build failed")
            return super().build_from_training_dirs(*args)
    with pytest.raises(RuntimeError, match="validation fold 0 ROM construction failed"):
        run_space_filling(**request(
            tmp_path, rom_model_builder=FoldFailureBuilder(), validation_strategy=KFoldValidation(2),
        ))
    assert not (tmp_path / "space_filling_stats.npz").exists()


def test_parent_working_directory_and_numpy_state_are_restored(tmp_path):
    class ChdirBuilder(RecordingBuilder):
        def build_from_training_dirs(self, *args):
            os.chdir(args[0])
            np.random.uniform(size=10)
            return super().build_from_training_dirs(*args)
    before_dir, before_rng = os.getcwd(), np.random.get_state()
    run_space_filling(**request(tmp_path, rom_model_builder=ChdirBuilder(), fom_model=FileModel(chdir=True)))
    assert os.getcwd() == before_dir
    after_rng = np.random.get_state()
    np.testing.assert_array_equal(before_rng[1], after_rng[1])
    assert before_rng[0] == after_rng[0] and before_rng[2:] == after_rng[2:]


def test_nonserializable_fom_reports_failure_without_build(tmp_path):
    model = FileModel()
    model.unpicklable = lambda: None
    options = request(tmp_path, fom_model=model, max_concurrent_fom_runs=2)
    with pytest.raises(RuntimeError, match="FOM batch failed.*",):
        run_space_filling(**options)
    assert not options["rom_model_builder"].calls


@pytest.mark.parametrize("folds", [0, 1, True, 2.5])
def test_bad_fold_counts(folds):
    with pytest.raises(ValueError, match="num_folds"):
        KFoldValidation(folds)

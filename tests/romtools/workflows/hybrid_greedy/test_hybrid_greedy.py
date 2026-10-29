"""Hybrid batch selection, accumulated data, validation, and failures."""

import multiprocessing
import os
from pathlib import Path

import numpy as np
import pytest

from romtools.workflows import HybridGreedyResult, run_hybrid_greedy, select_maximin_samples
from romtools.workflows.parameter_spaces import UniformParameterSpace
from romtools.workflows.sampling_methods import LatinHypercubeSampler, MonteCarloSampler
from romtools.workflows.space_filling import KFoldValidation, relative_l2_error
from ..space_filling.test_workflow import FileModel, LinearRom, RecordingBuilder, TrackingModel


class SequenceSpace:
    def __init__(self, initial=((0,), (1,)), candidates=((0,), (.125,), (.5,), (.75,), (1,))):
        self.initial = np.asarray(initial)
        self.candidates = np.asarray(candidates)
        self.calls = []

    def get_names(self):
        return [f"p{i}" for i in range(self.initial.shape[1])]

    def get_dimensionality(self):
        return self.initial.shape[1]

    def generate_samples(self, count, seed=None):
        self.calls.append((count, seed))
        return (self.initial if len(self.calls) == 1 else self.candidates).copy()


class IndicatorRom(LinearRom):
    def __init__(self, rom, evaluations, indicator, failure, filesystem):
        self.__dict__.update(rom.__dict__)
        self.evaluations = evaluations
        self.indicator = indicator
        self.failure = failure
        self.filesystem = filesystem

    def populate_run_directory(self, directory, parameters):
        if self.filesystem:
            Path(directory, "input.txt").write_text(str(parameters))
        if self.failure == "prepare":
            raise ValueError("bad preparation")

    def run_model(self, directory, parameters):
        if self.failure == "run":
            raise ValueError("bad solver")
        if self.failure == "code":
            return 3
        if self.filesystem:
            assert Path(directory, "input.txt").exists()
            Path(directory, "solution.txt").write_text("done")
        return 0

    def compute_error_estimate(self, directory, parameters):
        if self.filesystem:
            assert Path(directory, "solution.txt").exists()
        self.evaluations.append((parameters.copy(), directory, Path(directory).exists()))
        if self.failure == "indicator":
            raise LookupError("bad estimator")
        return self.indicator(parameters)


class IndicatorBuilder(RecordingBuilder):
    def __init__(self, indicator=None, failure=None, filesystem=False, reject_training_eval=False):
        super().__init__(reject_training_eval)
        self.indicator = indicator if indicator is not None else lambda p: 1 - abs(p["p0"] - .5)
        self.failure = failure
        self.filesystem = filesystem
        self.evaluations = []

    def build_from_training_dirs(self, *args):
        rom = super().build_from_training_dirs(*args)
        return IndicatorRom(rom, self.evaluations, self.indicator, self.failure, self.filesystem)


def request(tmp_path, **kwargs):
    options = dict(
        fom_model=FileModel(), rom_model_builder=IndicatorBuilder(),
        parameter_space=SequenceSpace(), absolute_work_dir=str(tmp_path),
        tolerance=0, initial_sample_size=2, batch_size=2, candidate_pool_size=7,
        max_concurrent_fom_runs=1, max_iterations=2,
        validation_strategy=lambda context: 1.0,
    )
    options.update(kwargs)
    return options


def test_initial_sampling_direct_and_initial_validation_can_stop(tmp_path):
    space = SequenceSpace(initial=[[.3], [.2], [.8], [.9], [.5]])
    builder = IndicatorBuilder()
    contexts = []
    def validate(context):
        contexts.append(context)
        assert not builder.evaluations
        assert context.iteration == 0
        assert not context.training_parameters.flags.writeable
        assert context.error_functor is relative_l2_error
        return 0
    result = run_hybrid_greedy(**request(
        tmp_path, parameter_space=space, rom_model_builder=builder,
        initial_sample_size=5, validation_strategy=validate,
    ))
    assert isinstance(result, HybridGreedyResult)
    assert [c[0] for c in space.calls] == [5]
    np.testing.assert_array_equal(result.training_parameters, space.initial)
    assert result.converged and result.termination_reason == "tolerance"
    assert result.greedy_parameters.shape == (0, 1)
    assert len(contexts) == 1


def test_one_greedy_point_and_maximin_accounts_for_it(tmp_path):
    options = request(tmp_path, batch_size=3)
    result = run_hybrid_greedy(**options)
    # .5 has the largest indicator; .75 is then farther than .125 from {0,1,.5}.
    np.testing.assert_array_equal(result.training_parameters[:, 0], [0, 1, .5, .75, .125])
    np.testing.assert_array_equal(result.greedy_parameters, [[.5]])
    np.testing.assert_array_equal(result.greedy_error_indicators, [1])
    assert [c[0] for c in options["parameter_space"].calls] == [2, 7]
    np.testing.assert_array_equal(
        [p[0]["p0"] for p in options["rom_model_builder"].evaluations], [.125, .5, .75],
    )
    assert not list(tmp_path.glob("candidates_*"))
    assert result.termination_reason == "max_iterations"


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7])
def test_multiple_rounds_complete_dataset_and_no_fom_reevaluation(tmp_path, batch_size):
    options = request(
        tmp_path, batch_size=batch_size, max_iterations=4,
        parameter_space=SequenceSpace(candidates=np.linspace(0, 1, 40)[:, None]),
    )
    result = run_hybrid_greedy(**options)
    counts = [2 + i * batch_size for i in range(4)]
    np.testing.assert_array_equal(result.sample_counts, counts)
    assert len(result.greedy_parameters) == 3
    assert [len(call[1]) for call in options["rom_model_builder"].calls] == counts
    for call in options["rom_model_builder"].calls:
        np.testing.assert_array_equal(call[2], result.training_parameters[:len(call[2])])
        np.testing.assert_array_equal(call[3], result.training_qois[:len(call[3])])
    assert len(np.unique(result.training_parameters)) == counts[-1]
    assert len(list(tmp_path.glob("fom/run_*/passed.txt"))) == counts[-1]
    if batch_size == 1:
        np.testing.assert_array_equal(result.training_parameters[2:], result.greedy_parameters)
    with np.load(tmp_path / "hybrid_greedy_stats.npz") as stats:
        np.testing.assert_array_equal(stats["training_parameters"], result.training_parameters)
        np.testing.assert_array_equal(stats["greedy_parameters"], result.greedy_parameters)
        np.testing.assert_array_equal(stats["greedy_error_indicators"], result.greedy_error_indicators)
        assert len(stats["fom_run_times"]) == counts[-1]


def test_ties_and_duplicates_short_batch_then_exhaustion(tmp_path):
    result = run_hybrid_greedy(**request(
        tmp_path, rom_model_builder=IndicatorBuilder(indicator=lambda p: 0),
        parameter_space=SequenceSpace(candidates=[[0], [.75], [.75], [.25], [1]]),
        batch_size=10, max_iterations=4,
    ))
    np.testing.assert_array_equal(result.training_parameters[:, 0], [0, 1, .75, .25])
    np.testing.assert_array_equal(result.greedy_parameters, [[.75]])
    assert result.termination_reason == "candidate_pool_exhausted"
    with np.load(tmp_path / "hybrid_greedy_stats.npz") as stats:
        assert str(stats["termination_reason"]) == "candidate_pool_exhausted"


def test_default_five_fold_validation_and_relative_l2(tmp_path):
    builder = IndicatorBuilder(reject_training_eval=True)
    result = run_hybrid_greedy(**request(
        tmp_path, initial_sample_size=5, rom_model_builder=builder,
        parameter_space=SequenceSpace(initial=np.arange(5)[:, None]),
        validation_strategy=None, tolerance=1e-10,
    ))
    assert result.converged
    assert [len(c[1]) for c in builder.calls] == [5, 4, 4, 4, 4, 4]
    assert len(list(tmp_path.glob("iteration_0/validation/fold_*/run_*/passed.txt"))) == 5
    assert not builder.evaluations


def test_kfold_custom_functor_maximum_and_stopping_each_round(tmp_path):
    seen = []
    def error(prediction, reference):
        seen.append(reference.copy())
        return 1 if len(seen) <= 5 else .25
    result = run_hybrid_greedy(**request(
        tmp_path, initial_sample_size=5, tolerance=.25,
        parameter_space=SequenceSpace(initial=np.linspace(0, .4, 5)[:, None],
                                       candidates=[[.5], [.7], [.9]]),
        validation_strategy=None, error_functor=error,
    ))
    np.testing.assert_array_equal(result.validation_errors, [1, .25])
    assert len(seen) == 5 + 7
    assert result.converged


@pytest.mark.parametrize("limit,reason,counts", [
    ({"max_iterations": 1}, "max_iterations", [2]),
    ({"max_fom_samples": 3}, "max_fom_samples", [2, 3]),
    ({"max_fom_samples": 1}, "max_fom_samples", [1]),
    ({"max_fom_samples": 2, "max_iterations": 1}, "max_fom_samples", [2]),
    ({"max_fom_samples": 2, "max_iterations": 1, "tolerance": 1}, "tolerance", [2]),
])
def test_limits_and_precedence(tmp_path, limit, reason, counts):
    result = run_hybrid_greedy(**request(tmp_path, **limit))
    assert result.termination_reason == reason
    np.testing.assert_array_equal(result.sample_counts, counts)


@pytest.mark.parametrize("create_directories", [True, False])
def test_candidate_directories_and_parameter_dictionaries(tmp_path, create_directories):
    builder = IndicatorBuilder(filesystem=create_directories)
    run_hybrid_greedy(**request(
        tmp_path, rom_model_builder=builder, create_rom_run_directories=create_directories,
    ))
    for parameters, directory, existed in builder.evaluations:
        assert set(parameters) == {"p0"}
        assert os.path.isabs(directory)
        assert existed == create_directories
        assert not Path(directory).exists()
    assert not list(tmp_path.glob("candidates_*"))


@pytest.mark.parametrize("indicator", [np.nan, np.inf, -1, [1], True, 1j, None])
def test_invalid_indicators_have_candidate_diagnostics(tmp_path, indicator):
    options = request(tmp_path, rom_model_builder=IndicatorBuilder(indicator=lambda p: indicator))
    with pytest.raises(RuntimeError, match="candidate 0, parameters=.*failed during compute_error_estimate.*finite nonnegative scalar"):
        run_hybrid_greedy(**options)
    assert len(options["rom_model_builder"].calls) == 1
    assert len(list(tmp_path.glob("fom/run_*"))) == 2
    assert not list(tmp_path.glob("candidates_*"))
    with np.load(tmp_path / "hybrid_greedy_stats.npz") as stats:
        np.testing.assert_array_equal(stats["sample_counts"], [2])


@pytest.mark.parametrize("failure,stage", [
    ("prepare", "populate_run_directory"), ("run", "run_model"),
    ("code", "run_model"), ("indicator", "compute_error_estimate"),
])
def test_failed_candidate_evaluation_does_not_launch_fom(tmp_path, failure, stage):
    with pytest.raises(RuntimeError, match=f"candidate 0, parameters=.*failed during {stage}"):
        run_hybrid_greedy(**request(tmp_path, rom_model_builder=IndicatorBuilder(failure=failure)))
    assert len(list(tmp_path.glob("fom/run_*"))) == 2


def test_missing_error_estimator_is_diagnosed_before_validation(tmp_path):
    with pytest.raises(TypeError, match="QoiModelWithErrorEstimate.*compute_error_estimate"):
        run_hybrid_greedy(**request(tmp_path, rom_model_builder=RecordingBuilder(), tolerance=1))
    assert not list(tmp_path.glob("iteration_0/validation"))


@pytest.mark.parametrize("concurrency", [1, 2])
def test_adaptive_fom_failure_preserves_prior_stats_and_waits_for_batch(tmp_path, concurrency):
    options = request(
        tmp_path, fom_model=FileModel(failure="code"), max_concurrent_fom_runs=concurrency,
        parameter_space=SequenceSpace(initial=[[1], [2]], candidates=[[0], [3]]),
        rom_model_builder=IndicatorBuilder(indicator=lambda p: 3 - p["p0"]),
    )
    with pytest.raises(RuntimeError, match="FOM batch failed.*"):
        run_hybrid_greedy(**options)
    assert len(options["rom_model_builder"].calls) == 1
    assert len(list(tmp_path.glob("fom/run_*/passed.txt"))) == 3
    with np.load(tmp_path / "hybrid_greedy_stats.npz") as stats:
        np.testing.assert_array_equal(stats["sample_counts"], [2])
        assert len(stats["greedy_parameters"]) == 0


def test_fom_concurrency_in_initial_and_adaptation_batches(tmp_path):
    with multiprocessing.get_context("spawn").Manager() as manager:
        active, peak = manager.Value("i", 0), manager.Value("i", 0)
        fom = TrackingModel(active, peak, manager.Lock(), manager.Event())
        result = run_hybrid_greedy(**request(
            tmp_path, fom_model=fom, max_concurrent_fom_runs=2,
            rom_model_builder=IndicatorBuilder(indicator=lambda p: float(p["p0"])),
            parameter_space=SequenceSpace(candidates=[[2], [3], [4]]),
        ))
        assert peak.value == 2 and active.value == 0
    np.testing.assert_array_equal(result.sample_counts, [2, 4])
    np.testing.assert_array_equal(result.training_qois[:, 0], 1 + result.training_parameters[:, 0])


def test_reproducibility_and_global_state_preservation(tmp_path):
    def run(path, seed):
        return run_hybrid_greedy(**request(
            path, parameter_space=UniformParameterSpace(["p0"], [0], [1], MonteCarloSampler),
            random_seed=seed, max_iterations=3,
        ))
    directory, state = os.getcwd(), np.random.get_state()
    first, second, third = run(tmp_path / "a", 17), run(tmp_path / "b", 17), run(tmp_path / "c", 18)
    np.testing.assert_array_equal(first.training_parameters, second.training_parameters)
    np.testing.assert_array_equal(first.greedy_error_indicators, second.greedy_error_indicators)
    assert not np.array_equal(first.training_parameters, third.training_parameters)
    assert os.getcwd() == directory
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[0] == after[0] and state[2:] == after[2:]


def test_initial_lhs_is_controlled_by_parameter_space(tmp_path):
    result = run_hybrid_greedy(**request(
        tmp_path, initial_sample_size=10, max_iterations=1,
        parameter_space=UniformParameterSpace(["p0"], [0], [1], LatinHypercubeSampler),
    ))
    np.testing.assert_array_equal(np.sort((result.training_parameters[:, 0] * 10).astype(int)), np.arange(10))


class ConstantFom:
    def populate_run_directory(self, directory, parameters):
        pass

    def run_model(self, directory, parameters):
        return 0

    def compute_qoi(self, directory, parameters):
        return np.array([1.0])


class GenericRom(ConstantFom):
    def compute_error_estimate(self, directory, parameters):
        return 1.0


class GenericBuilder:
    def build_from_training_dirs(self, offline, directories, parameters, qois):
        return GenericRom()


def test_string_parameters_and_metric_equivalent_points(tmp_path):
    space = SequenceSpace(initial=[["A"]], candidates=[["a"], ["B"], ["b"], ["C"]])
    result = run_hybrid_greedy(**request(
        tmp_path, initial_sample_size=1, batch_size=5, parameter_space=space,
        fom_model=ConstantFom(), rom_model_builder=GenericBuilder(),
        distance_metric=lambda a, b: float(a[0].lower() != b[0].lower()),
    ))
    np.testing.assert_array_equal(result.training_parameters[:, 0], ["A", "B", "C"])
    np.testing.assert_array_equal(result.greedy_parameters, [["B"]])


def test_irregular_domain_custom_metric(tmp_path):
    class CircleSpace:
        def get_names(self): return ["x", "y"]
        def get_dimensionality(self): return 2
        def generate_samples(self, count, seed=None):
            a = np.random.default_rng(seed).uniform(0, 2 * np.pi, count)
            return np.c_[np.cos(a), np.sin(a)]
    metric = lambda a, b: np.linalg.norm((a - b) * [3, 1])
    result = run_hybrid_greedy(**request(
        tmp_path, parameter_space=CircleSpace(), fom_model=ConstantFom(),
        rom_model_builder=GenericBuilder(), distance_metric=metric,
    ))
    np.testing.assert_allclose(np.linalg.norm(result.training_parameters, axis=1), 1)


def test_initial_duplicates_removed_without_reordering(tmp_path):
    result = run_hybrid_greedy(**request(
        tmp_path, initial_sample_size=4, max_iterations=1,
        parameter_space=SequenceSpace(initial=[[1], [0], [1], [0]]),
    ))
    np.testing.assert_array_equal(result.training_parameters, [[1], [0]])


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"max_iterations": 0}, {"initial_sample_size": True},
    {"max_concurrent_fom_runs": 0}, {"candidate_pool_size": -1}, {"max_fom_samples": 0},
    {"tolerance": np.nan}, {"distance_metric": 3}, {"error_functor": 2},
    {"validation_strategy": 5}, {"create_rom_run_directories": 1},
    {"absolute_work_dir": "relative"}, {"validation_strategy": None},
])
def test_bad_configuration_fails_before_creating_campaign(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        run_hybrid_greedy(**request(tmp_path / "campaign", **options))
    assert not (tmp_path / "campaign").exists()


def test_nonempty_work_directory_preserved(tmp_path):
    (tmp_path / "keep").touch()
    with pytest.raises(ValueError, match="absent or empty"):
        run_hybrid_greedy(**request(tmp_path))
    assert (tmp_path / "keep").exists()


def test_default_kfold_rejects_short_initial_design_before_fom(tmp_path):
    with pytest.raises(ValueError, match="fewer distinct"):
        run_hybrid_greedy(**request(tmp_path, initial_sample_size=5, validation_strategy=None))
    assert not list(tmp_path.glob("fom/run_*"))


def test_explicit_pool_maximin_uses_input_order_and_preserves_arrays():
    candidates = np.array([[0], [.25], [.5], [.75], [1]])
    existing = np.array([[0]])
    copy = candidates.copy()
    np.testing.assert_array_equal(select_maximin_samples(candidates, existing, 3), [[1], [.5], [.25]])
    np.testing.assert_array_equal(select_maximin_samples(candidates, None, 1), [[0]])
    np.testing.assert_array_equal(candidates, copy)
    np.testing.assert_array_equal(existing, [[0]])


def test_custom_metric_changes_hybrid_space_filling_selection(tmp_path):
    def run(path, metric):
        return run_hybrid_greedy(**request(
            path, fom_model=ConstantFom(), rom_model_builder=GenericBuilder(),
            parameter_space=SequenceSpace(initial=[[0, 0]], candidates=[[1, 1], [4, 0], [0, 3]]),
            initial_sample_size=1, distance_metric=metric,
        ))
    euclidean = run(tmp_path / "euclidean", None)
    scaled = run(tmp_path / "scaled", lambda a, b: np.linalg.norm((a - b) * [1, 10]))
    np.testing.assert_array_equal(euclidean.training_parameters, [[0, 0], [1, 1], [4, 0]])
    np.testing.assert_array_equal(scaled.training_parameters, [[0, 0], [1, 1], [0, 3]])


def test_zero_dimensional_space_exhausts_and_saves_empty_greedy_history(tmp_path):
    result = run_hybrid_greedy(**request(
        tmp_path, parameter_space=SequenceSpace(initial=np.empty((2, 0)), candidates=np.empty((5, 0))),
        fom_model=ConstantFom(), rom_model_builder=GenericBuilder(),
    ))
    assert result.training_parameters.shape == (1, 0)
    assert result.greedy_parameters.shape == (0, 0)
    assert result.termination_reason == "candidate_pool_exhausted"


def test_existing_gp_builder_integration(tmp_path):
    from examples.hybrid_greedy_demo import ScalarFom, UncertaintyBuilder
    builder = UncertaintyBuilder()
    result = run_hybrid_greedy(**request(
        tmp_path, fom_model=ScalarFom(), rom_model_builder=builder,
        parameter_space=UniformParameterSpace(["x"], [0], [1], LatinHypercubeSampler),
        initial_sample_size=5, tolerance=0, validation_strategy=None,
        create_rom_run_directories=False,
    ))
    assert len(result.greedy_parameters) == 1
    assert np.isfinite(result.validation_errors).all()
    assert result.training_qois.shape == (7, 1)

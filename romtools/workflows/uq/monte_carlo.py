"""Monte Carlo and two-level multifidelity Monte Carlo workflows."""

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, LocalDispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.uq._evaluation import evaluate_qoi_model
from romtools.workflows.uq.sample_allocation import allocate_samples_from_pilot
from romtools.workflows.uq.statistics import (
    compute_monte_carlo_statistics,
    compute_multifidelity_statistics,
    compute_paired_statistics,
)


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo mean estimator."""

    parameter_samples: np.ndarray
    qoi_values: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    standard_deviation: np.ndarray
    standard_error: np.ndarray
    run_times: np.ndarray


@dataclass
class MultifidelityMonteCarloResult:
    """Results from fixed or pilot-allocated two-level MFMC."""

    parameter_samples: np.ndarray
    high_fidelity_qoi_values: np.ndarray
    low_fidelity_qoi_values: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    standard_error: np.ndarray
    control_variate_coefficients: np.ndarray
    paired_correlations: np.ndarray
    high_fidelity_sample_count: int
    low_fidelity_sample_count: int
    pilot_sample_count: Optional[int]
    low_to_high_fidelity_cost_ratio: float
    high_fidelity_equivalent_cost: float
    used_multifidelity_estimator: bool
    high_fidelity_run_times: np.ndarray
    low_fidelity_run_times: np.ndarray


def _require_absolute_directory(directory: str) -> None:
    if not os.path.isabs(directory):
        raise ValueError("absolute_uq_directory must be an absolute path")


def _save_monte_carlo_result(
    dispatcher: BaseDispatcher,
    directory: str,
    result: MonteCarloResult,
    random_seed: int,
) -> None:
    dispatcher.np_savez(
        os.path.join(directory, "uq_stats"),
        parameter_samples=result.parameter_samples,
        qoi_values=result.qoi_values,
        mean=result.mean,
        variance=result.variance,
        standard_deviation=result.standard_deviation,
        standard_error=result.standard_error,
        run_times=result.run_times,
        random_seed=np.array([random_seed]),
        number_of_samples=np.array([result.qoi_values.shape[0]], dtype=int),
    )


def _save_multifidelity_result(
    dispatcher: BaseDispatcher,
    directory: str,
    result: MultifidelityMonteCarloResult,
    random_seed: int,
    requested_budget: float,
    pilot_correlations: np.ndarray,
) -> None:
    dispatcher.np_savez(
        os.path.join(directory, "uq_stats"),
        parameter_samples=result.parameter_samples,
        high_fidelity_qoi_values=result.high_fidelity_qoi_values,
        low_fidelity_qoi_values=result.low_fidelity_qoi_values,
        mean=result.mean,
        variance=result.variance,
        standard_error=result.standard_error,
        control_variate_coefficients=result.control_variate_coefficients,
        paired_correlations=result.paired_correlations,
        pilot_correlations=pilot_correlations,
        high_fidelity_sample_count=np.array(
            [result.high_fidelity_sample_count], dtype=int
        ),
        low_fidelity_sample_count=np.array(
            [result.low_fidelity_sample_count], dtype=int
        ),
        pilot_sample_count=np.array(
            [-1 if result.pilot_sample_count is None else result.pilot_sample_count],
            dtype=int,
        ),
        low_to_high_fidelity_cost_ratio=np.array(
            [result.low_to_high_fidelity_cost_ratio]
        ),
        high_fidelity_equivalent_cost=np.array(
            [result.high_fidelity_equivalent_cost]
        ),
        requested_high_fidelity_equivalent_budget=np.array([requested_budget]),
        used_multifidelity_estimator=np.array(
            [result.used_multifidelity_estimator], dtype=bool
        ),
        high_fidelity_run_times=result.high_fidelity_run_times,
        low_fidelity_run_times=result.low_fidelity_run_times,
        random_seed=np.array([random_seed]),
    )


def run_monte_carlo(
    model: QoiModel,
    parameter_space: ParameterSpace,
    absolute_uq_directory: str,
    number_of_samples: int,
    random_seed: int = 1,
    evaluation_concurrency: int = 1,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
) -> MonteCarloResult:
    """Estimate a QoI mean and its standard error by Monte Carlo."""
    _require_absolute_directory(absolute_uq_directory)
    if number_of_samples < 2:
        raise ValueError("number_of_samples must be at least two")
    dispatcher = dispatcher if dispatcher is not None else LocalDispatcher()
    dispatcher.create_empty_dir(absolute_uq_directory)
    samples = parameter_space.generate_samples(number_of_samples, seed=random_seed)
    qois, run_times = evaluate_qoi_model(
        model,
        parameter_space.get_names(),
        samples,
        os.path.join(absolute_uq_directory, "run_"),
        evaluation_concurrency=evaluation_concurrency,
        overwrite=overwrite,
        dispatcher=dispatcher,
    )
    statistics = compute_monte_carlo_statistics(qois)
    result = MonteCarloResult(
        parameter_samples=np.asarray(samples),
        qoi_values=qois,
        mean=statistics.mean,
        variance=statistics.variance,
        standard_deviation=statistics.standard_deviation,
        standard_error=statistics.standard_error,
        run_times=run_times,
    )
    _save_monte_carlo_result(dispatcher, absolute_uq_directory, result, random_seed)
    return result


def _validate_multifidelity_modes(
    number_high: Optional[int],
    number_low: Optional[int],
    pilot_count: Optional[int],
    budget: Optional[float],
) -> str:
    fixed_requested = number_high is not None or number_low is not None
    pilot_requested = pilot_count is not None or budget is not None
    if fixed_requested and pilot_requested:
        raise ValueError("fixed and pilot allocation modes are mutually exclusive")
    if fixed_requested:
        if number_high is None or number_low is None:
            raise ValueError("fixed allocation requires both sample counts")
        if number_high < 2 or number_low < number_high:
            raise ValueError(
                "fixed allocation requires low count >= high count >= 2"
            )
        return "fixed"
    if pilot_count is None or budget is None:
        raise ValueError("pilot allocation requires pilot_sample_count and budget")
    if pilot_count < 2:
        raise ValueError("pilot_sample_count must be at least two")
    return "pilot"


def _finite_mean(values: np.ndarray, name: str) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError(
            f"cannot estimate {name} from reused samples; supply the cost ratio"
        )
    value = float(np.mean(finite_values))
    if value <= 0.0:
        raise ValueError(f"estimated {name} must be positive")
    return value


def _build_multifidelity_result(
    parameter_samples: np.ndarray,
    high_qois: np.ndarray,
    low_qois: np.ndarray,
    high_run_times: np.ndarray,
    low_run_times: np.ndarray,
    pilot_count: Optional[int],
    cost_ratio: float,
    equivalent_cost: float,
    use_multifidelity: bool,
    coefficients: Optional[np.ndarray],
) -> MultifidelityMonteCarloResult:
    if use_multifidelity:
        statistics = compute_multifidelity_statistics(
            high_qois, low_qois, coefficients
        )
        mean = statistics.mean
        variance = statistics.variance
        standard_error = statistics.standard_error
        used_coefficients = statistics.control_variate_coefficients
        correlations = statistics.paired_correlations
    else:
        statistics = compute_monte_carlo_statistics(high_qois)
        _, _, _, _, correlations = compute_paired_statistics(
            high_qois[: low_qois.shape[0]], low_qois
        )
        mean = statistics.mean
        variance = statistics.variance / high_qois.shape[0]
        standard_error = statistics.standard_error
        used_coefficients = np.zeros(high_qois.shape[1])

    return MultifidelityMonteCarloResult(
        parameter_samples=parameter_samples,
        high_fidelity_qoi_values=high_qois,
        low_fidelity_qoi_values=low_qois,
        mean=mean,
        variance=variance,
        standard_error=standard_error,
        control_variate_coefficients=used_coefficients,
        paired_correlations=correlations,
        high_fidelity_sample_count=high_qois.shape[0],
        low_fidelity_sample_count=low_qois.shape[0],
        pilot_sample_count=pilot_count,
        low_to_high_fidelity_cost_ratio=cost_ratio,
        high_fidelity_equivalent_cost=equivalent_cost,
        used_multifidelity_estimator=use_multifidelity,
        high_fidelity_run_times=high_run_times,
        low_fidelity_run_times=low_run_times,
    )


def run_multifidelity_monte_carlo(
    high_fidelity_model: QoiModel,
    low_fidelity_model: QoiModel,
    parameter_space: ParameterSpace,
    absolute_uq_directory: str,
    number_of_high_fidelity_samples: Optional[int] = None,
    number_of_low_fidelity_samples: Optional[int] = None,
    pilot_sample_count: Optional[int] = None,
    high_fidelity_equivalent_budget: Optional[float] = None,
    low_to_high_fidelity_cost_ratio: Optional[float] = None,
    allocation_qoi_index: int = 0,
    random_seed: int = 1,
    high_fidelity_evaluation_concurrency: int = 1,
    low_fidelity_evaluation_concurrency: int = 1,
    control_variate_coefficients: Optional[np.ndarray] = None,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
) -> MultifidelityMonteCarloResult:
    """Estimate a QoI mean with fixed or pilot-allocated two-level MFMC."""
    _require_absolute_directory(absolute_uq_directory)
    mode = _validate_multifidelity_modes(
        number_of_high_fidelity_samples,
        number_of_low_fidelity_samples,
        pilot_sample_count,
        high_fidelity_equivalent_budget,
    )
    if low_to_high_fidelity_cost_ratio is not None and (
        not np.isfinite(low_to_high_fidelity_cost_ratio)
        or low_to_high_fidelity_cost_ratio <= 0.0
    ):
        raise ValueError("low_to_high_fidelity_cost_ratio must be positive")

    dispatcher = dispatcher if dispatcher is not None else LocalDispatcher()
    dispatcher.create_empty_dir(absolute_uq_directory)
    high_directory_base = os.path.join(
        absolute_uq_directory, "high_fidelity", "run_"
    )
    low_directory_base = os.path.join(
        absolute_uq_directory, "low_fidelity", "run_"
    )
    parameter_names = parameter_space.get_names()

    if mode == "fixed":
        samples = np.asarray(
            parameter_space.generate_samples(
                number_of_low_fidelity_samples, seed=random_seed
            )
        )
        high_qois, high_times = evaluate_qoi_model(
            high_fidelity_model,
            parameter_names,
            samples[:number_of_high_fidelity_samples],
            high_directory_base,
            high_fidelity_evaluation_concurrency,
            overwrite,
            dispatcher,
        )
        low_qois, low_times = evaluate_qoi_model(
            low_fidelity_model,
            parameter_names,
            samples,
            low_directory_base,
            low_fidelity_evaluation_concurrency,
            overwrite,
            dispatcher,
        )
        if low_to_high_fidelity_cost_ratio is None:
            try:
                cost_ratio = _finite_mean(low_times, "low-fidelity cost") / _finite_mean(
                    high_times, "high-fidelity cost"
                )
            except ValueError:
                cost_ratio = np.nan
        else:
            cost_ratio = float(low_to_high_fidelity_cost_ratio)
        equivalent_cost = (
            number_of_high_fidelity_samples + cost_ratio * number_of_low_fidelity_samples
            if np.isfinite(cost_ratio)
            else np.nan
        )
        result = _build_multifidelity_result(
            samples,
            high_qois,
            low_qois,
            high_times,
            low_times,
            None,
            cost_ratio,
            equivalent_cost,
            True,
            control_variate_coefficients,
        )
        _save_multifidelity_result(
            dispatcher,
            absolute_uq_directory,
            result,
            random_seed,
            np.nan,
            result.paired_correlations,
        )
        return result

    # Pilot mode.
    pilot_count = int(pilot_sample_count)
    if control_variate_coefficients is not None:
        raise ValueError(
            "pilot allocation estimates and freezes control-variate coefficients"
        )
    pilot_samples = np.asarray(
        parameter_space.generate_samples(pilot_count, seed=random_seed)
    )
    pilot_high_qois, pilot_high_times = evaluate_qoi_model(
        high_fidelity_model,
        parameter_names,
        pilot_samples,
        high_directory_base,
        high_fidelity_evaluation_concurrency,
        overwrite,
        dispatcher,
    )
    pilot_low_qois, pilot_low_times = evaluate_qoi_model(
        low_fidelity_model,
        parameter_names,
        pilot_samples,
        low_directory_base,
        low_fidelity_evaluation_concurrency,
        overwrite,
        dispatcher,
    )
    if low_to_high_fidelity_cost_ratio is None:
        cost_ratio = _finite_mean(
            pilot_low_times, "low-fidelity cost"
        ) / _finite_mean(pilot_high_times, "high-fidelity cost")
    else:
        cost_ratio = float(low_to_high_fidelity_cost_ratio)

    allocation = allocate_samples_from_pilot(
        pilot_high_qois,
        pilot_low_qois,
        float(high_fidelity_equivalent_budget),
        cost_ratio,
        allocation_qoi_index,
    )
    number_high = allocation.number_high_fidelity_samples
    number_low = allocation.number_low_fidelity_samples
    additional_count = max(number_high, number_low) - pilot_count
    if additional_count:
        additional_samples = np.asarray(
            parameter_space.generate_samples(additional_count, seed=random_seed + 1)
        )
    else:
        additional_samples = np.empty((0, pilot_samples.shape[1]))
    samples = np.vstack([pilot_samples, additional_samples])

    high_extra_count = number_high - pilot_count
    low_extra_count = number_low - pilot_count
    if high_extra_count:
        high_extra_qois, high_extra_times = evaluate_qoi_model(
            high_fidelity_model,
            parameter_names,
            additional_samples[:high_extra_count],
            high_directory_base,
            high_fidelity_evaluation_concurrency,
            overwrite,
            dispatcher,
            starting_sample_index=pilot_count,
        )
        high_qois = np.vstack([pilot_high_qois, high_extra_qois])
        high_times = np.concatenate([pilot_high_times, high_extra_times])
    else:
        high_qois, high_times = pilot_high_qois, pilot_high_times

    if low_extra_count:
        low_extra_qois, low_extra_times = evaluate_qoi_model(
            low_fidelity_model,
            parameter_names,
            additional_samples[:low_extra_count],
            low_directory_base,
            low_fidelity_evaluation_concurrency,
            overwrite,
            dispatcher,
            starting_sample_index=pilot_count,
        )
        low_qois = np.vstack([pilot_low_qois, low_extra_qois])
        low_times = np.concatenate([pilot_low_times, low_extra_times])
    else:
        low_qois, low_times = pilot_low_qois, pilot_low_times

    result = _build_multifidelity_result(
        samples,
        high_qois,
        low_qois,
        high_times,
        low_times,
        pilot_count,
        cost_ratio,
        allocation.high_fidelity_equivalent_cost,
        allocation.use_multifidelity_estimator,
        allocation.control_variate_coefficients,
    )
    _save_multifidelity_result(
        dispatcher,
        absolute_uq_directory,
        result,
        random_seed,
        float(high_fidelity_equivalent_budget),
        allocation.pilot_correlations,
    )
    return result

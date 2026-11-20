"""Two-level multifidelity Sobol sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Tuple

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.uq._evaluation import (
    evaluate_qoi_model,
    prepare_qoi_model,
    validate_evaluation_concurrency,
)
from romtools.workflows.uq.sobol import (
    SobolIndicesDryRunResult,
    _estimate_indices,
    _pick_freeze_samples,
    _sobol_run_prefix,
    _validate_base_count,
    _validate_sobol_inputs,
)
from romtools.workflows.uq.statistics import compute_paired_statistics


@dataclass
class MultifidelitySobolIndicesDryRunResult:
    """Prepared pilot and production inputs at both fidelities."""

    high_fidelity_pilot: SobolIndicesDryRunResult
    low_fidelity_pilot: SobolIndicesDryRunResult
    high_fidelity: SobolIndicesDryRunResult
    low_fidelity: SobolIndicesDryRunResult


@dataclass
class MultifidelitySobolIndicesResult:
    """Indices and diagnostics for the fixed-count multifidelity estimate.

    Indices have shape (number of parameters, number of QoIs). Coefficients
    and pilot correlations have one row for the variance, followed by the
    first-order and total-effect numerator rows in parameter order. Coefficients
    apply to terms computed from QoIs rescaled by the corresponding fidelity's
    QoI scale. A component flagged in ``used_high_fidelity_only`` has the
    single-fidelity indices from the high-fidelity production design.

    Samples, QoIs, and timings are ordered by pick-freeze block (A, B, and
    AB_i in parameter order). Within each low-fidelity production block, the
    first high-fidelity sample count rows match the high-fidelity block.
    """

    parameter_names: Tuple[str, ...]
    first_order: np.ndarray
    total_effect: np.ndarray
    control_variate_coefficients: np.ndarray
    pilot_correlations: np.ndarray
    used_high_fidelity_only: np.ndarray
    high_fidelity_parameter_samples: np.ndarray
    low_fidelity_parameter_samples: np.ndarray
    pilot_parameter_samples: np.ndarray
    high_fidelity_qoi_values: np.ndarray
    low_fidelity_qoi_values: np.ndarray
    high_fidelity_pilot_qoi_values: np.ndarray
    low_fidelity_pilot_qoi_values: np.ndarray
    high_fidelity_qoi_scale: np.ndarray
    low_fidelity_qoi_scale: np.ndarray
    high_fidelity_run_times: np.ndarray
    low_fidelity_run_times: np.ndarray
    high_fidelity_pilot_run_times: np.ndarray
    low_fidelity_pilot_run_times: np.ndarray


def _sobol_terms(pilot_qois, production_qois, pilot_count, production_count):
    dimension = pilot_qois.shape[1]
    scale = np.max(np.abs(pilot_qois), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    pilot = (pilot_qois / scale).reshape(-1, pilot_count, dimension)
    center = np.mean(pilot[:2], axis=(0, 1))

    def contributions(blocks):
        a, b = blocks[:2]
        differences = blocks[2:] - a
        # A and B are independent under the target distribution, so this
        # squared difference has expectation Var[f(X)].
        variance = 0.5 * (a - b) ** 2
        first_order = (b - center) * differences
        total_effect = 0.5 * differences ** 2
        return np.concatenate((variance[None, :, :], first_order, total_effect))

    with np.errstate(over="ignore", invalid="ignore"):
        production = (production_qois / scale).reshape(-1, production_count, dimension)
        production_terms = contributions(production)
    return contributions(pilot), production_terms, scale


def _estimate_multifidelity_indices(high, low, high_pilot, low_pilot, high_count, low_count,
                                   pilot_count):
    if len({values.shape[1] for values in (high, low, high_pilot, low_pilot)}) != 1:
        raise ValueError("pilot and production QoIs must have matching dimensions at both fidelities")

    high_pilot_terms, high_terms, high_scale = _sobol_terms(high_pilot, high, pilot_count, high_count)
    low_pilot_terms, low_terms, low_scale = _sobol_terms(low_pilot, low, pilot_count, low_count)
    term_count, _, qoi_count = high_terms.shape
    _, _, _, coefficients, correlations = compute_paired_statistics(
        high_pilot_terms.transpose(1, 0, 2).reshape(pilot_count, -1),
        low_pilot_terms.transpose(1, 0, 2).reshape(pilot_count, -1),
    )
    coefficients = coefficients.reshape(term_count, qoi_count)
    correlations = correlations.reshape(term_count, qoi_count)
    if high_count == low_count:
        coefficients.fill(0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        estimates = high_terms.mean(axis=1) + coefficients * (
            low_terms.mean(axis=1) - low_terms[:, :high_count].mean(axis=1)
        )
        variance = estimates[0]
        denominator = np.where(variance > 0.0, variance, np.nan)
        indices = estimates[1:] / denominator
    fallback = (
        ~np.all(np.isfinite(estimates), axis=0)
        | ~np.all(np.isfinite(indices), axis=0)
        | ~np.any(coefficients != 0.0, axis=0)
    )
    first_order, total_effect = np.split(indices, 2, axis=0)
    if np.any(fallback):
        high_first, high_total = _estimate_indices(high[:, fallback], high_count)
        first_order[:, fallback] = high_first
        total_effect[:, fallback] = high_total
    return first_order, total_effect, coefficients, correlations, fallback, high_scale, low_scale


def run_multifidelity_sobol_indices(
    high_fidelity_model: QoiModel,
    low_fidelity_model: QoiModel,
    parameter_space: HeterogeneousParameterSpace,
    absolute_work_dir: str,
    number_of_high_fidelity_base_samples: int,
    number_of_low_fidelity_base_samples: int,
    pilot_sample_count: int = 16,
    random_seed: int = 1,
    high_fidelity_evaluation_concurrency: int = 1,
    low_fidelity_evaluation_concurrency: int = 1,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
    *,
    dry_run: bool = False,
) -> MultifidelitySobolIndicesResult | MultifidelitySobolIndicesDryRunResult:
    """Estimate high-fidelity Sobol indices using paired low-fidelity runs.

    All counts refer to base rows and must be powers of two of at least two.
    The low-fidelity production design must be at least as large as the
    high-fidelity design. An independently scrambled paired pilot fits a
    control coefficient for each term and QoI component. The pilot is
    additional to the specified production sample counts. Both models must
    return the same deterministic QoI components at independent scalar inputs.

    For each term, the correction adds the pilot coefficient times the
    difference between the full and paired low-fidelity production means to
    the high-fidelity production mean. Ratios are formed after correction.
    A QoI with a nonpositive variance, a nonfinite estimate, or no active
    correction uses its high-fidelity-only estimate.

    With d inputs, each fidelity runs (production count + pilot count) *
    (d + 2) evaluations. Successful evaluations can be reused by rerunning
    with the same design. Set overwrite=True when a model or QoI changes.
    Dry runs prepare both designs at both fidelities. Completed runs write
    multifidelity_sobol_stats.npz under absolute_work_dir.
    """
    names = _validate_sobol_inputs(parameter_space, absolute_work_dir, random_seed)
    high_count = _validate_base_count(
        "number_of_high_fidelity_base_samples", number_of_high_fidelity_base_samples
    )
    low_count = _validate_base_count(
        "number_of_low_fidelity_base_samples", number_of_low_fidelity_base_samples
    )
    pilot_count = _validate_base_count("pilot_sample_count", pilot_sample_count)
    if high_count > low_count:
        raise ValueError("the low-fidelity base sample count must be at least the high-fidelity count")
    validate_evaluation_concurrency(high_fidelity_evaluation_concurrency)
    validate_evaluation_concurrency(low_fidelity_evaluation_concurrency)

    pilot_seed, production_seed = np.random.SeedSequence(int(random_seed)).spawn(2)
    pilot_samples = _pick_freeze_samples(parameter_space, pilot_count, np.random.default_rng(pilot_seed))
    low_samples = _pick_freeze_samples(parameter_space, low_count, np.random.default_rng(production_seed))
    low_blocks = low_samples.reshape(len(names) + 2, low_count, len(names))
    high_samples = np.ascontiguousarray(low_blocks[:, :high_count]).reshape(-1, len(names))

    dispatcher = resolve_dispatcher(dispatcher)
    batches = (
        ("pilot/high_fidelity", high_fidelity_model, pilot_samples, high_fidelity_evaluation_concurrency),
        ("pilot/low_fidelity", low_fidelity_model, pilot_samples, low_fidelity_evaluation_concurrency),
        ("production/high_fidelity", high_fidelity_model, high_samples, high_fidelity_evaluation_concurrency),
        ("production/low_fidelity", low_fidelity_model, low_samples, low_fidelity_evaluation_concurrency),
    )
    runs = []
    for phase_directory, model, samples, concurrency in batches:
        directory = os.path.join(absolute_work_dir, phase_directory)
        prefix = _sobol_run_prefix(dispatcher, directory, names, samples)
        if dry_run:
            directories = prepare_qoi_model(
                model, names, samples, prefix, overwrite=overwrite, dispatcher=dispatcher
            )
            runs.append(SobolIndicesDryRunResult(samples, directories))
        else:
            runs.append(evaluate_qoi_model(
                model, names, samples, prefix, evaluation_concurrency=concurrency,
                overwrite=overwrite, dispatcher=dispatcher,
            ))

    if dry_run:
        return MultifidelitySobolIndicesDryRunResult(*runs)

    (high_pilot, high_pilot_times), (low_pilot, low_pilot_times), (high, high_times), (low, low_times) = runs
    first, total, coefficients, correlations, fallback, high_scale, low_scale = (
        _estimate_multifidelity_indices(high, low, high_pilot, low_pilot, high_count, low_count, pilot_count)
    )
    result = MultifidelitySobolIndicesResult(
        parameter_names=names,
        first_order=first,
        total_effect=total,
        control_variate_coefficients=coefficients,
        pilot_correlations=correlations,
        used_high_fidelity_only=fallback,
        high_fidelity_parameter_samples=high_samples,
        low_fidelity_parameter_samples=low_samples,
        pilot_parameter_samples=pilot_samples,
        high_fidelity_qoi_values=high,
        low_fidelity_qoi_values=low,
        high_fidelity_pilot_qoi_values=high_pilot,
        low_fidelity_pilot_qoi_values=low_pilot,
        high_fidelity_qoi_scale=high_scale,
        low_fidelity_qoi_scale=low_scale,
        high_fidelity_run_times=high_times,
        low_fidelity_run_times=low_times,
        high_fidelity_pilot_run_times=high_pilot_times,
        low_fidelity_pilot_run_times=low_pilot_times,
    )
    dispatcher.np_savez(
        os.path.join(absolute_work_dir, "multifidelity_sobol_stats.npz"),
        **vars(result),
        number_of_high_fidelity_base_samples=np.asarray(high_count),
        number_of_low_fidelity_base_samples=np.asarray(low_count),
        pilot_sample_count=np.asarray(pilot_count),
        random_seed=np.asarray(str(random_seed)),
    )
    return result

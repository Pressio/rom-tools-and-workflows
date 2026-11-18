"""Variance-based global sensitivity analysis for independent inputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import os
from typing import Optional, Tuple

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import HeterogeneousParameterSpace
from romtools.workflows.sampling_methods import RandomizedQuasiMonteCarloSampler
from romtools.workflows.uq._evaluation import (
    evaluate_qoi_model,
    prepare_qoi_model,
    validate_evaluation_concurrency,
)


@dataclass
class SobolIndicesDryRunResult:
    """Ordered sample rows and their prepared model run directories."""

    parameter_samples: np.ndarray
    run_directories: Tuple[str, ...]


@dataclass
class SobolIndicesResult:
    """Sensitivity indices with shape (number of parameters, number of QoIs).

    The sample, QoI, and timing rows are ordered in blocks A, B, AB_0, ...,
    where AB_i is A with parameter i replaced by the corresponding column of B.
    Timings are NaN for reused evaluations.
    """

    parameter_names: Tuple[str, ...]
    parameter_samples: np.ndarray
    qoi_values: np.ndarray
    first_order: np.ndarray
    total_effect: np.ndarray
    run_times: np.ndarray


def _pick_freeze_samples(parameter_space, number_of_base_samples, random_seed):
    dimension = parameter_space.get_dimensionality()
    unit_samples = RandomizedQuasiMonteCarloSampler(
        number_of_base_samples, 2 * dimension, seed=random_seed
    )
    # Inverse CDFs for unbounded marginals are infinite at the endpoints.
    unit_samples = np.clip(
        unit_samples, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
    )
    unit_a, unit_b = np.split(unit_samples, 2, axis=1)
    a = parameter_space.transform_unit_samples(unit_a)
    b = parameter_space.transform_unit_samples(unit_b)
    blocks = np.repeat(a[np.newaxis, :, :], dimension + 2, axis=0)
    blocks[1] = b
    for i in range(dimension):
        blocks[2 + i, :, i] = b[:, i]
    return blocks.reshape(-1, dimension)


def _estimate_indices(qoi_values, number_of_base_samples):
    # Scaling and centering preserve variance ratios and reduce cancellation.
    scale = np.max(np.abs(qoi_values), axis=0)
    blocks = (qoi_values / np.where(scale > 0, scale, 1.0)).reshape(
        -1, number_of_base_samples, qoi_values.shape[1]
    )
    population = np.concatenate((blocks[0], blocks[1]), axis=0)
    variance = np.var(population, axis=0)
    denominator = np.where(variance > 0, variance, np.nan)
    differences = blocks[2:] - blocks[0]
    centered_b = blocks[1] - population.mean(axis=0)
    first_order = np.mean(centered_b * differences, axis=1)
    total_effect = 0.5 * np.mean(differences ** 2, axis=1)
    return first_order / denominator, total_effect / denominator


def run_sobol_indices(
    model: QoiModel,
    parameter_space: HeterogeneousParameterSpace,
    absolute_work_dir: str,
    number_of_base_samples: int,
    random_seed: int = 1,
    evaluation_concurrency: int = 1,
    overwrite: bool = False,
    dispatcher: Optional[BaseDispatcher] = None,
    *,
    dry_run: bool = False,
) -> SobolIndicesResult | SobolIndicesDryRunResult:
    """Estimate first-order and total-effect Sobol indices for each flattened QoI.

    A scrambled Sobol design supplies two base matrices. A pick-freeze design
    combines them for the Saltelli first-order and squared-difference total-effect
    estimators. The number of base samples must be a power of two (at least two).
    For d independent scalar parameters, the design has N * (d + 2) model runs.
    The workflow uses the parameter space's marginal transforms and its own
    sampling design. The model should produce a deterministic QoI at each input.

    Successful evaluations can be reused for the same sample design. Use
    overwrite=True when the model or QoI definition changes. Set dry_run=True
    to prepare inputs without evaluating the model. Otherwise, the run saves
    results and sampling metadata to sobol_stats.npz under absolute_work_dir.

    Constant QoI components return NaN indices. Finite-sample estimates are
    returned without clipping.
    """
    if not os.path.isabs(absolute_work_dir):
        raise ValueError("absolute_work_dir must be an absolute path")
    if isinstance(number_of_base_samples, bool) or not isinstance(number_of_base_samples, Integral):
        raise ValueError("number_of_base_samples must be a power of two of at least 2")
    number_of_base_samples = int(number_of_base_samples)
    if number_of_base_samples < 2 or number_of_base_samples & (number_of_base_samples - 1):
        raise ValueError("number_of_base_samples must be a power of two of at least 2")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral) or random_seed < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not isinstance(parameter_space, HeterogeneousParameterSpace):
        raise ValueError("Sobol indices require a HeterogeneousParameterSpace with independent inputs")
    names = tuple(parameter_space.get_names())
    if not names or len(names) != parameter_space.get_dimensionality() or len(set(names)) != len(names):
        raise ValueError("Sobol indices require distinct names for independent scalar parameters")
    validate_evaluation_concurrency(evaluation_concurrency)

    samples = _pick_freeze_samples(parameter_space, number_of_base_samples, int(random_seed))
    if samples.dtype.kind not in "biuf" or not np.all(np.isfinite(samples)):
        raise ValueError("Sobol indices require finite numeric parameter samples")

    # The design determines which inputs correspond to cached run directories.
    digest = hashlib.sha256(repr((names, samples.shape, samples.dtype.str)).encode())
    digest.update(memoryview(samples).cast("B"))
    dispatcher = resolve_dispatcher(dispatcher)
    design_directory = os.path.join(absolute_work_dir, f"design_{digest.hexdigest()}")
    dispatcher.create_empty_dir(design_directory)
    run_prefix = os.path.join(design_directory, "run_")
    if dry_run:
        directories = prepare_qoi_model(
            model, names, samples, run_prefix, overwrite=overwrite, dispatcher=dispatcher
        )
        return SobolIndicesDryRunResult(samples, directories)

    qois, run_times = evaluate_qoi_model(
        model, names, samples, run_prefix,
        evaluation_concurrency=evaluation_concurrency, overwrite=overwrite, dispatcher=dispatcher,
    )
    first_order, total_effect = _estimate_indices(qois, number_of_base_samples)
    result = SobolIndicesResult(names, samples, qois, first_order, total_effect, run_times)
    dispatcher.np_savez(
        os.path.join(absolute_work_dir, "sobol_stats.npz"),
        parameter_names=np.asarray(result.parameter_names),
        parameter_samples=result.parameter_samples,
        qoi_values=result.qoi_values,
        first_order=result.first_order,
        total_effect=result.total_effect,
        run_times=result.run_times,
        number_of_base_samples=np.asarray(number_of_base_samples),
        random_seed=np.asarray(str(random_seed)),
        design_directory=np.asarray(os.path.basename(design_directory)),
    )
    return result

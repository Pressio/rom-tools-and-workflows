"""Leakage-free K-fold QoI validation and user-defined error measures."""

from dataclasses import dataclass
from typing import Callable, Tuple
import os

import numpy as np

from romtools.workflows.model_builders import QoiModelBuilderWithTrainingData
from romtools.workflows.models import QoiModel
from .execution import _evaluate_sample, _process_state
from .sampling import _positive_integer


def _error_value(value, description):
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
        raise ValueError(f"{description} must be a finite nonnegative scalar")
    result = float(scalar)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{description} must be a finite nonnegative scalar")
    return result


def relative_l2_error(qoi_rom: np.ndarray, qoi_fom: np.ndarray) -> float:
    """Return ||q_rom-q_fom||_2 / max(||q_fom||_2, 1e-12)."""
    return float(np.linalg.norm(qoi_rom - qoi_fom) / max(np.linalg.norm(qoi_fom), 1e-12))


@dataclass(frozen=True)
class SpaceFillingValidationContext:
    """Inputs to a validation callable returning one nonnegative scalar error.

    Parameter/QoI rows and training directories have matching order. Arrays are
    read-only copies. Builders must fit independent models using only their
    supplied subsets; a fold must not reuse the production ROM's fitted basis.
    A custom strategy owns aggregation and can use ``error_functor`` to compare
    its predictions with independently prepared reference QoIs.
    """

    rom_model: QoiModel
    rom_model_builder: QoiModelBuilderWithTrainingData
    parameter_names: Tuple[str, ...]
    training_parameters: np.ndarray
    training_qois: np.ndarray
    training_dirs: Tuple[str, ...]
    iteration: int
    validation_directory: str
    random_seed: int
    error_functor: Callable


def _build_rom(builder, directory, training_dirs, parameters, qois, seed):
    os.makedirs(directory, exist_ok=False)
    with _process_state(seed):
        return builder.build_from_training_dirs(
            directory, list(training_dirs), parameters.copy(), qois.copy()
        )


@dataclass(frozen=True)
class KFoldValidation:
    """Maximum held-out sample error from independent subset-trained ROMs.

    No additional FOM evaluations are performed. Each validation uses
    ``num_folds`` builds and one held-out ROM evaluation per training sample.
    The error functor comes from the context; its result is not renormalized.
    """

    num_folds: int = 5

    def __post_init__(self):
        _positive_integer(self.num_folds, "num_folds")
        if self.num_folds < 2:
            raise ValueError("num_folds must be at least 2")

    def __call__(self, context: SpaceFillingValidationContext) -> float:
        count = len(context.training_parameters)
        if count < self.num_folds:
            raise ValueError("K-fold validation requires at least num_folds training samples")
        rng = np.random.default_rng(context.random_seed)
        folds = np.array_split(rng.permutation(count), self.num_folds)
        maximum_error = 0.0
        for fold_index, held_out in enumerate(folds):
            training = np.ones(count, dtype=bool)
            training[held_out] = False
            indices = np.flatnonzero(training)
            directory = os.path.join(context.validation_directory, f"fold_{fold_index}")
            try:
                rom = _build_rom(
                    context.rom_model_builder, os.path.join(directory, "offline_data"),
                    [context.training_dirs[i] for i in indices],
                    context.training_parameters[indices], context.training_qois[indices],
                    int(rng.integers(0, 2**32)),
                )
            except Exception as error:
                raise RuntimeError(f"validation fold {fold_index} ROM construction failed: {error}") from error
            for index in held_out:
                index = int(index)
                prediction, _ = _evaluate_sample(
                    rom, context.parameter_names, context.training_parameters[index],
                    os.path.join(directory, f"run_{index}"), index,
                    int(rng.integers(0, 2**32)),
                )
                reference = context.training_qois[index]
                if prediction.shape != reference.shape:
                    raise ValueError(f"validation fold {fold_index}, sample {index}: QoI dimension mismatch")
                try:
                    error = _error_value(
                        context.error_functor(prediction.copy(), reference.copy()),
                        "error_functor result",
                    )
                except Exception as exception:
                    raise ValueError(
                        f"validation fold {fold_index}, sample {index}: {exception}"
                    ) from exception
                maximum_error = max(maximum_error, error)
        return maximum_error


def _readonly_copy(array):
    result = array.copy()
    result.setflags(write=False)
    return result

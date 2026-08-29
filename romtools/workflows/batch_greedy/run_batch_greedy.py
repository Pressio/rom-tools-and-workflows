"""Batch greedy workflow for QoI models with error estimates."""

import concurrent.futures
import multiprocessing
from numbers import Integral
import time

import numpy as np

from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.model_builders import QoiModelWithErrorEstimateBuilder
from romtools.workflows.batch_greedy.selection import select_batch


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def _run_fom_sample(fom_model, sample_index, run_directory, parameter_dict):
    """Run one FOM sample and return its index and QoI.

    This helper is defined at module scope so it can be used by a spawned
    ``ProcessPoolExecutor`` worker.
    """
    flag = fom_model.run_model(run_directory, parameter_dict)
    if flag != 0:
        raise RuntimeError(
            f"FOM evaluation failed for sample {sample_index} with flag {flag}"
        )
    qoi = fom_model.compute_qoi(run_directory, parameter_dict)
    return int(sample_index), qoi


def _run_fom_samples(
    fom_model,
    parameter_names,
    parameter_samples,
    sample_indices,
    greedy_directory,
    run_directory_prefix,
    fom_evaluation_concurrency,
):
    """Run FOM samples concurrently while preserving sample association."""
    jobs = []
    for sample_index in sample_indices:
        parameter_dict = _create_parameter_dict(
            parameter_names, parameter_samples[sample_index]
        )
        run_directory = (
            f"{greedy_directory}/fom/{run_directory_prefix}{sample_index}"
        )
        jobs.append((int(sample_index), run_directory, parameter_dict))

    if fom_evaluation_concurrency == 1:
        results = [
            _run_fom_sample(fom_model, sample_index, run_directory, parameter_dict)
            for sample_index, run_directory, parameter_dict in jobs
        ]
    else:
        mp_context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=fom_evaluation_concurrency,
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(
                    _run_fom_sample,
                    fom_model,
                    sample_index,
                    run_directory,
                    parameter_dict,
                )
                for sample_index, run_directory, parameter_dict in jobs
            ]
            results = [future.result() for future in futures]

    return {sample_index: qoi for sample_index, qoi in results}


class QoIvsErrorIndicatorRegressor:
    """Simple multiplicative calibration between error indicator and QoI error."""

    def __init__(self, calibrated_error: bool = True):
        self.__c = 1.0
        self.__calibrated_error = calibrated_error

    def fit(self, x, y):
        """Fit the multiplicative calibration factor."""
        if self.__calibrated_error:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            mean_x = np.mean(x)
            self.__c = 1.0 if mean_x == 0.0 else np.mean(y) / mean_x

    def predict(self, x):
        """Predict QoI error from an error indicator."""
        return self.__c * x


def run_batch_greedy(
    fom_model: QoiModel,
    rom_model_builder: QoiModelWithErrorEstimateBuilder,
    parameter_space: ParameterSpace,
    absolute_greedy_work_directory: str,
    tolerance: float,
    batch_size: int,
    testing_sample_size: int = 10,
    random_seed: int = 1,
    calibrated_error: bool = True,
    distance_exponent: float = 1.0,
    fom_evaluation_concurrency: int = 1,
):
    """Construct a ROM using batches of error-greedy FOM samples.

    This workflow mirrors :func:`romtools.workflows.greedy.run_greedy`, but
    selects multiple FOM training points from one set of ROM error estimates
    before rebuilding the ROM. The first point in each batch maximizes the
    model's error estimate; subsequent points use within-batch distance
    penalization through :func:`select_batch`.

    FOM evaluations in each selected batch are executed concurrently using
    spawned worker processes. The QoI model is responsible for defining
    ``compute_error_estimate``; the workflow is agnostic to whether the
    indicator comes from a residual, learned error model, predictive variance,
    ensemble, or another source.

    Parameters
    ----------
    fom_model : QoiModel
        Full-order model used to generate training data and reference QoIs.
    rom_model_builder : QoiModelWithErrorEstimateBuilder
        Builder returning a QoI model that also supplies an error estimate.
    parameter_space : ParameterSpace
        Parameter space used to generate the candidate set.
    absolute_greedy_work_directory : str
        Work directory for FOM, ROM, and statistics data.
    tolerance : float
        Maximum calibrated QoI error estimate used for convergence.
    batch_size : int
        Number of FOM points selected per outer iteration.
    testing_sample_size : int, default=10
        Initial size of the candidate/training pool. The first two samples are
        used to initialize the ROM, matching the sequential greedy workflow.
    random_seed : int, default=1
        NumPy random seed used before candidate generation.
    calibrated_error : bool, default=True
        Whether to calibrate the ROM error indicator against observed QoI error.
    distance_exponent : float, default=1.0
        Strength of within-batch diversity. Zero gives top-k error selection.
    fom_evaluation_concurrency : int, default=1
        Maximum number of FOM evaluations run concurrently. A value of one
        preserves serial execution.
    """
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(fom_evaluation_concurrency, bool)
        or not isinstance(fom_evaluation_concurrency, Integral)
        or fom_evaluation_concurrency < 1
    ):
        raise ValueError("fom_evaluation_concurrency must be a positive integer")
    if testing_sample_size < 3:
        raise ValueError("testing_sample_size must be at least 3")
    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")

    greedy_directory = absolute_greedy_work_directory
    create_empty_dir(greedy_directory)
    offline_directory_prefix = "offline_data"
    run_directory_prefix = "run_"

    status_path = f"{greedy_directory}/batch_greedy_status.log"
    greedy_file = open(status_path, "w", encoding="utf-8")  # pylint: disable=consider-using-with
    greedy_file.write("Batch greedy reduced basis status\n")

    fom_time = 0.0
    rom_time = 0.0
    basis_time = 0.0

    np.random.seed(random_seed)
    parameter_samples = parameter_space.generate_samples(testing_sample_size)
    parameter_names = parameter_space.get_names()

    for sample_index, sample in enumerate(parameter_samples):
        fom_run_directory = f"{greedy_directory}/fom/{run_directory_prefix}{sample_index}"
        create_empty_dir(fom_run_directory)
        parameter_dict = _create_parameter_dict(parameter_names, sample)
        fom_model.populate_run_directory(fom_run_directory, parameter_dict)

    training_samples = np.array([0, 1], dtype=int)
    samples_left = np.arange(2, testing_sample_size, dtype=int)

    # Initial FOM training cases use the same concurrency mechanism as later
    # batches. With the default concurrency of one this matches run_greedy.
    t0 = time.time()
    for sample_index in training_samples:
        greedy_file.write(f"Running initial FOM sample {sample_index}\n")
    fom_qois_by_sample = _run_fom_samples(
        fom_model,
        parameter_names,
        parameter_samples,
        training_samples,
        greedy_directory,
        run_directory_prefix,
        fom_evaluation_concurrency,
    )
    training_dirs = [
        f"{greedy_directory}/fom/{run_directory_prefix}{sample_index}"
        for sample_index in training_samples
    ]
    fom_time += time.time() - t0

    t0 = time.time()
    outer_loop_counter = 0
    updated_offline_data_dir = (
        f"{greedy_directory}/rom_iteration_{outer_loop_counter}/"
        f"{offline_directory_prefix}/"
    )
    create_empty_dir(updated_offline_data_dir)
    rom_model = rom_model_builder.build_from_training_dirs(
        updated_offline_data_dir, training_dirs
    )
    basis_time += time.time() - t0

    reg = QoIvsErrorIndicatorRegressor(calibrated_error)
    selected_error_indicators = []
    qoi_errors = []
    predicted_max_qoi_errors = []
    batch_history = []
    greedy_file.flush()

    while samples_left.size > 0:
        greedy_file.write(f"Batch greedy iteration # {outer_loop_counter}\n")
        error_indicators = np.zeros(samples_left.size)

        t0 = time.time()
        for counter, sample_index in enumerate(samples_left):
            parameter_dict = _create_parameter_dict(
                parameter_names, parameter_samples[sample_index]
            )
            rom_run_directory = (
                f"{greedy_directory}/rom_iteration_{outer_loop_counter}/"
                f"{run_directory_prefix}{sample_index}"
            )
            create_empty_dir(rom_run_directory)
            rom_model.populate_run_directory(rom_run_directory, parameter_dict)
            rom_model.run_model(rom_run_directory, parameter_dict)
            error_indicators[counter] = rom_model.compute_error_estimate(
                rom_run_directory, parameter_dict
            )
        rom_time += time.time() - t0

        predicted_errors = reg.predict(error_indicators)
        predicted_max_qoi_error = float(np.max(predicted_errors))
        predicted_max_qoi_errors.append(predicted_max_qoi_error)
        greedy_file.write(
            "Maximum predicted QoI error before batch selection: "
            f"{predicted_max_qoi_error}\n"
        )

        have_calibration_history = len(qoi_errors) > 0 or not calibrated_error
        if have_calibration_history and predicted_max_qoi_error < tolerance:
            greedy_file.write("Converged before launching a new batch.\n")
            greedy_file.flush()
            break

        local_selected = select_batch(
            parameter_samples[samples_left],
            error_indicators,
            batch_size=batch_size,
            distance_exponent=distance_exponent,
        )
        batch_sample_indices = samples_left[local_selected]
        batch_history.append(batch_sample_indices.copy())

        greedy_file.write(
            f"Selected batch: {batch_sample_indices.tolist()} with error indicators "
            f"{error_indicators[local_selected].tolist()}\n"
        )
        greedy_file.flush()

        # Run the selected FOM cases concurrently, then rebuild the ROM once
        # after every member of the batch has completed successfully.
        t0 = time.time()
        batch_fom_qois = _run_fom_samples(
            fom_model,
            parameter_names,
            parameter_samples,
            batch_sample_indices,
            greedy_directory,
            run_directory_prefix,
            fom_evaluation_concurrency,
        )
        fom_time += time.time() - t0
        fom_qois_by_sample.update(batch_fom_qois)

        for sample_index in batch_sample_indices:
            parameter_dict = _create_parameter_dict(
                parameter_names, parameter_samples[sample_index]
            )
            fom_qoi = batch_fom_qois[int(sample_index)]
            rom_run_directory = (
                f"{greedy_directory}/rom_iteration_{outer_loop_counter}/"
                f"{run_directory_prefix}{sample_index}"
            )
            rom_qoi = rom_model.compute_qoi(rom_run_directory, parameter_dict)
            denominator = np.linalg.norm(fom_qoi)
            qoi_error = np.linalg.norm(rom_qoi - fom_qoi)
            if denominator > 0.0:
                qoi_error /= denominator

            local_position = int(np.flatnonzero(samples_left == sample_index)[0])
            selected_error_indicators.append(error_indicators[local_position])
            qoi_errors.append(float(qoi_error))
            greedy_file.write(
                f"Sample {sample_index} had relative QoI error {qoi_error}\n"
            )

        reg.fit(selected_error_indicators, qoi_errors)

        training_samples = np.append(training_samples, batch_sample_indices)
        selected_set = set(int(i) for i in batch_sample_indices)
        samples_left = np.asarray(
            [i for i in samples_left if int(i) not in selected_set], dtype=int
        )

        outer_loop_counter += 1

        t0 = time.time()
        training_dirs = [
            f"{greedy_directory}/fom/{run_directory_prefix}{i}"
            for i in training_samples
        ]
        updated_offline_data_dir = (
            f"{greedy_directory}/rom_iteration_{outer_loop_counter}/"
            f"{offline_directory_prefix}/"
        )
        create_empty_dir(updated_offline_data_dir)
        rom_model = rom_model_builder.build_from_training_dirs(
            updated_offline_data_dir, training_dirs
        )
        basis_time += time.time() - t0

        number_of_new_samples = len(batch_sample_indices)
        new_parameter_samples = parameter_space.generate_samples(number_of_new_samples)
        first_new_sample = parameter_samples.shape[0]
        parameter_samples = np.append(parameter_samples, new_parameter_samples, axis=0)

        for offset, new_parameter_sample in enumerate(new_parameter_samples):
            new_sample_number = first_new_sample + offset
            fom_run_directory = (
                f"{greedy_directory}/fom/{run_directory_prefix}{new_sample_number}"
            )
            create_empty_dir(fom_run_directory)
            parameter_dict = _create_parameter_dict(
                parameter_names, new_parameter_sample
            )
            fom_model.populate_run_directory(fom_run_directory, parameter_dict)
            samples_left = np.append(samples_left, new_sample_number)

        max_batch_width = max((len(batch) for batch in batch_history), default=0)
        padded_batches = np.full(
            (len(batch_history), max_batch_width), -1, dtype=int
        )
        for batch_index, batch in enumerate(batch_history):
            padded_batches[batch_index, : len(batch)] = batch

        np.savez(
            f"{greedy_directory}/batch_greedy_stats",
            selected_error_indicators=np.asarray(selected_error_indicators),
            qoi_errors=np.asarray(qoi_errors),
            predicted_max_qoi_errors=np.asarray(predicted_max_qoi_errors),
            training_samples=training_samples,
            batch_indices=padded_batches,
            batch_sizes=np.asarray([len(batch) for batch in batch_history], dtype=int),
            fom_evaluation_concurrency=np.asarray([fom_evaluation_concurrency], dtype=int),
            fom_time=fom_time,
            rom_time=rom_time,
            basis_time=basis_time,
        )
        greedy_file.flush()

    greedy_file.close()

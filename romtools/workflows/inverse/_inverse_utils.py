from functools import partial
from typing import Optional

import numpy as np

from romtools.hpc.dispatchers import BaseDispatcher, resolve_dispatcher
from romtools.workflows.model_execution import (
    ModelEvaluation, execute_models, require_supported_concurrency,
)


def require_relative_or_absolute_path(dispatcher: BaseDispatcher, directory: str):
    """Check that a working directory is usable by both the FOM and ROM dispatcher.

    Multifidelity workflows always create their ROM run directories locally, so the
    directory must be absolute for a LocalDispatcher and relative for a RemoteDispatcher.
    """
    dispatcher.require_absolute_path(directory)
    dispatcher.require_relative_path(directory)


def _prepare_evaluation(model, evaluation, dispatcher, create_run_directory):
    if create_run_directory:
        dispatcher.create_empty_dir(evaluation.run_directory)
    model.populate_run_directory(evaluation.run_directory, evaluation.parameter_sample)


def _finish_evaluation(model, evaluation, observations):
    qoi = model.compute_qoi(evaluation.run_directory, evaluation.parameter_sample)
    assert isinstance(qoi, np.ndarray), "Error, compute_qoi must return an np.ndarray"
    return qoi.copy(), observations - qoi, evaluation.run_time


def prepare_and_run_batch(model, observations, run_directories, parameter_names,
                          parameter_samples, evaluation_concurrency,
                          dispatcher=None, create_run_directories=True, start_method="spawn"):
    dispatcher = resolve_dispatcher(dispatcher)
    require_supported_concurrency(model, dispatcher, evaluation_concurrency)
    evaluations = (
        ModelEvaluation(directory, dict(zip(parameter_names, sample)))
        for directory, sample in zip(run_directories, parameter_samples)
    )
    completed = execute_models(
        model, evaluations, evaluation_concurrency, start_method=start_method,
        prepare=partial(_prepare_evaluation, dispatcher=dispatcher,
                        create_run_directory=create_run_directories),
        postprocess=partial(_finish_evaluation, observations=observations),
        ignore_blocking_status=True,
    )
    return [evaluation.output for evaluation in completed]


def prepare_and_run(model, observations, run_directory, parameter_names, parameter_sample,
                    dispatcher: Optional[BaseDispatcher] = None, create_run_directory: bool = True):
    """Prepare the model run and compute the QoI and error."""
    return prepare_and_run_batch(
        model, observations, [run_directory], parameter_names, [parameter_sample], 1,
        dispatcher, create_run_directory,
    )[0]


def bound_samples(samples,samples_min,samples_max):
    if samples_max is not None:
      samples[:,:] = np.fmin(samples[:,:],samples_max)
    if samples_min is not None:
      samples[:,:] = np.fmax(samples[:,:],samples_min)
    return samples


def run_eki_iteration(model, observations, run_directory_base, parameter_names, parameter_samples,
                      evaluation_concurrency, dispatcher: Optional[BaseDispatcher] = None):
    """Run the EKI iteration for the specified parameters."""
    ensemble_size = len(parameter_samples)
    parameter_mean = np.mean(parameter_samples, axis=0)
    directories = [f"{run_directory_base}mean"]
    directories.extend(f"{run_directory_base}{i}" for i in range(ensemble_size))
    mean, *members = prepare_and_run_batch(
        model, observations, directories, parameter_names, [parameter_mean, *parameter_samples],
        evaluation_concurrency, dispatcher, start_method="fork",
    )
    mean_qoi = mean[0]
    qois = np.zeros((mean_qoi.size, ensemble_size))
    errors = np.zeros_like(qois)
    for i, (qoi, error, _) in enumerate(members):
        qois[:, i] = qoi
        errors[:, i] = error
    return {"qois": qois, "mean-qoi": mean_qoi, "errors": errors}


def run_vi_iteration(model, observations, run_directory_base, parameter_names, parameter_samples,
                     evaluation_concurrency, dispatcher: Optional[BaseDispatcher] = None,
                     create_run_directories: bool = True):
    """Run a VI iteration for sampled parameters only (no explicit mean run)."""
    ensemble_size = len(parameter_samples)
    assert ensemble_size > 0, "parameter_samples must contain at least one sample"
    members = prepare_and_run_batch(
        model, observations, (f"{run_directory_base}{i}" for i in range(ensemble_size)),
        parameter_names, parameter_samples, evaluation_concurrency, dispatcher,
        create_run_directories,
    )
    qois = np.zeros((members[0][0].size, ensemble_size))
    errors = np.zeros_like(qois)
    for i, (qoi, error, _) in enumerate(members):
        qois[:, i] = qoi
        errors[:, i] = error
    return {"qois": qois, "mean-qoi": np.mean(qois, axis=1), "errors": errors}

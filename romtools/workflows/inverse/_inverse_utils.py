import concurrent.futures
import multiprocessing
import numpy as np
import os
import time

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))

def prepare_and_run(model, observations, run_directory, parameter_names, parameter_sample):
    """Prepare the model run and compute the QoI and error."""
    os.makedirs(run_directory, exist_ok=True)
    parameter_dict = _create_parameter_dict(parameter_names, parameter_sample)
    model.populate_run_directory(run_directory, parameter_dict)
    ts = time.time()
    flag = model.run_model(run_directory, parameter_dict)
    qoi = model.compute_qoi(run_directory, parameter_dict)
    assert isinstance(qoi,np.ndarray), "Error, compute_qoi must return an np.ndarray"
    error = observations - qoi
    run_time = time.time() - ts
    summary = None
    if hasattr(model, "get_last_run_summary"):
        summary = model.get_last_run_summary(run_directory, parameter_dict)

    return qoi, error, run_time, summary

def bound_samples(samples,samples_min,samples_max):
    if samples_max is not None: 
      samples[:,:] = np.fmin(samples[:,:],samples_max)
    if samples_min is not None: 
      samples[:,:] = np.fmax(samples[:,:],samples_min)
    return samples 


def run_eki_iteration(model, observations, run_directory_base, parameter_names, parameter_samples, evaluation_concurrency):
    """Run the EKI iteration for the specified parameters."""
    mp_cntxt = multiprocessing.get_context("fork")
    ensemble_size = np.shape(parameter_samples)[0]

    if evaluation_concurrency == 1:
        # Run at parameter mean
        run_directory = f'{run_directory_base}mean'
        parameter_means = np.mean(parameter_samples, axis=0)
        mean_qoi, mean_error, run_time, mean_summary = prepare_and_run(model, observations, run_directory, parameter_names, parameter_means)
        qois = np.zeros((mean_qoi.size, ensemble_size))
        errors = np.zeros((mean_qoi.size, ensemble_size))
        sample_summaries = []
        for ensemble_member in range(ensemble_size):
            run_directory = f'{run_directory_base}{ensemble_member}'
            qois[:, ensemble_member], errors[:, ensemble_member], run_time, summary = prepare_and_run(model, observations, run_directory, parameter_names, parameter_samples[ensemble_member])
            sample_summaries.append(summary)
    else:
        samples_to_run = list(range(ensemble_size))
        with concurrent.futures.ProcessPoolExecutor(max_workers=evaluation_concurrency, mp_context=mp_cntxt) as executor:
            #these_futures = [executor.submit(my_print,ensemble_member) for ensemble_member in samples_to_run]
            parameter_means = np.mean(parameter_samples, axis=0)
            run_directory = f'{run_directory_base}mean'
            these_futures = [executor.submit(prepare_and_run, model, observations, run_directory, parameter_names, parameter_means)]
            these_futures.extend([executor.submit(prepare_and_run, model, observations, f'{run_directory_base}{ensemble_member}', parameter_names, parameter_samples[ensemble_member]) for ensemble_member in samples_to_run])
            concurrent.futures.wait(these_futures)
        
        for i, future in enumerate(these_futures):
            if i == 0:
                mean_qoi, mean_error, run_time, mean_summary = future.result() 
                qois = np.zeros((mean_qoi.size, ensemble_size))
                errors = np.zeros((mean_qoi.size, ensemble_size))
                sample_summaries = []
            else:
                qois[:, i-1], errors[:, i-1], run_time, summary = future.result()
                sample_summaries.append(summary)
    
    results = {}
    results['qois'] = qois
    results['mean-qoi'] = mean_qoi
    results['errors'] = errors
    results['sample_summaries'] = sample_summaries
    results['mean_summary'] = mean_summary
    if hasattr(model, "print_iteration_summary"):
        model.print_iteration_summary(
            run_directory_base,
            parameter_names,
            parameter_samples,
            results,
        )
    return results 


def run_vi_iteration(model, observations, run_directory_base, parameter_names, parameter_samples, evaluation_concurrency):
    """Run a VI iteration for sampled parameters only (no explicit mean run)."""
    mp_cntxt = multiprocessing.get_context("spawn")
    ensemble_size = np.shape(parameter_samples)[0]
    assert ensemble_size > 0, "parameter_samples must contain at least one sample"

    if evaluation_concurrency == 1:
        first_run_directory = f'{run_directory_base}0'
        first_qoi, first_error, run_time, first_summary = prepare_and_run(
            model,
            observations,
            first_run_directory,
            parameter_names,
            parameter_samples[0],
        )
        qois = np.zeros((first_qoi.size, ensemble_size))
        errors = np.zeros((first_qoi.size, ensemble_size))
        qois[:, 0] = first_qoi
        errors[:, 0] = first_error
        sample_summaries = [first_summary]
        for ensemble_member in range(1, ensemble_size):
            run_directory = f'{run_directory_base}{ensemble_member}'
            qois[:, ensemble_member], errors[:, ensemble_member], run_time, summary = prepare_and_run(
                model,
                observations,
                run_directory,
                parameter_names,
                parameter_samples[ensemble_member],
            )
            sample_summaries.append(summary)
    else:
        samples_to_run = list(range(ensemble_size))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=evaluation_concurrency,
            mp_context=mp_cntxt,
        ) as executor:
            these_futures = [
                executor.submit(
                    prepare_and_run,
                    model,
                    observations,
                    f'{run_directory_base}{ensemble_member}',
                    parameter_names,
                    parameter_samples[ensemble_member],
                )
                for ensemble_member in samples_to_run
            ]
            concurrent.futures.wait(these_futures)

        first_qoi, first_error, run_time, first_summary = these_futures[0].result()
        qois = np.zeros((first_qoi.size, ensemble_size))
        errors = np.zeros((first_qoi.size, ensemble_size))
        qois[:, 0] = first_qoi
        errors[:, 0] = first_error
        sample_summaries = [first_summary]
        for i, future in enumerate(these_futures[1:], start=1):
            qois[:, i], errors[:, i], run_time, summary = future.result()
            sample_summaries.append(summary)

    results = {}
    results['qois'] = qois
    results['mean-qoi'] = np.mean(qois, axis=1)
    results['errors'] = errors
    results['sample_summaries'] = sample_summaries
    if hasattr(model, "print_iteration_summary"):
        model.print_iteration_summary(
            run_directory_base,
            parameter_names,
            parameter_samples,
            results,
        )
    return results

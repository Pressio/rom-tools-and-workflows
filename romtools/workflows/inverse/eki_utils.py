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
    error = observations - qoi
    run_time = time.time() - ts
    
    return qoi, error, run_time 


def run_eki_iteration(model, observations, run_directory_base, parameter_names, parameter_samples, evaluation_concurrency):
    """Run the EKI iteration for the specified parameters."""
    mp_cntxt = multiprocessing.get_context("spawn")
    ensemble_size = np.shape(parameter_samples)[0]
    run_directory = f'{run_directory_base}mean'

    # Run at parameter mean
    parameter_means = np.mean(parameter_samples, axis=0)
    mean_qoi, mean_error, run_time = prepare_and_run(model, observations, run_directory, parameter_names, parameter_means)
    qois = np.zeros((mean_qoi.size, ensemble_size))
    errors = np.zeros((mean_qoi.size, ensemble_size))
    if evaluation_concurrency == 1:
        for ensemble_member in range(ensemble_size):
            run_directory = f'{run_directory_base}{ensemble_member}'
            qois[:, ensemble_member], errors[:, ensemble_member], run_time = prepare_and_run(model, observations, run_directory, parameter_names, parameter_samples[ensemble_member])
    else:
        samples_to_run = list(range(ensemble_size))
        with concurrent.futures.ProcessPoolExecutor(max_workers=evaluation_concurrency, mp_context=mp_cntxt) as executor:
            these_futures = [executor.submit(prepare_and_run, model, observations, f'{run_directory_base}{ensemble_member}', parameter_names, parameter_samples[ensemble_member]) for ensemble_member in samples_to_run]
            concurrent.futures.wait(these_futures)
        
        for i, future in enumerate(these_futures):
            qois[:, i], errors[:, i], run_time = future.result()
    
    results = {}
    results['qois'] = qois
    results['mean-qoi'] = mean_qoi
    results['errors'] = errors
    return results 

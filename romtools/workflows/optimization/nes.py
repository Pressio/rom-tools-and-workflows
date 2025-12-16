import os
import time
import numpy as np
import sys
import concurrent.futures
import multiprocessing
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.model_builders import QoiModelBuilder
import scipy

def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def prepare_and_run(model, run_directory, parameter_names, parameter_sample):
    """Prepare the model run and compute the QoI and runtime."""
    os.makedirs(run_directory, exist_ok=True)
    parameter_dict = _create_parameter_dict(parameter_names, parameter_sample)
    model.populate_run_directory(run_directory, parameter_dict)
    ts = time.time()
    flag = model.run_model(run_directory, parameter_dict)
    qoi = model.compute_qoi(run_directory, parameter_dict)
    assert isinstance(qoi, float), "Error, compute_qoi must return a float"
    run_time = time.time() - ts
    return qoi, run_time


def _evaluate_samples(model, parameter_names, samples, work_dir, iteration, bound, label_prefix="run_", evaluation_concurrency=1, start_index=0):
    """Evaluate QoI for a batch of samples, optionally in parallel."""
    count = samples.shape[0]
    qois = np.zeros(count)
    if evaluation_concurrency <= 1:
        for i in range(count):
            run_directory = f"{work_dir}/iteration_{iteration}/{label_prefix}{start_index + i}"
            sample = samples[i]
            if callable(bound):
                sample = bound(sample.reshape(1, -1))[0]
            qois[i], _ = prepare_and_run(model, run_directory, parameter_names, sample)
    else:
        mp_ctx = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=evaluation_concurrency, mp_context=mp_ctx) as executor:
            futures = []
            for i in range(count):
                run_directory = f"{work_dir}/iteration_{iteration}/{label_prefix}{start_index + i}"
                sample = samples[i]
                if callable(bound):
                    sample = bound(sample.reshape(1, -1))[0]
                futures.append(executor.submit(prepare_and_run, model, run_directory, parameter_names, sample))
            concurrent.futures.wait(futures)
            for i, fut in enumerate(futures):
                qois[i], _ = fut.result()
    return qois


def compute_nes_gradients(qois,parameters,mean,covariance,utilities_type):   
    order = np.argsort(qois)
    qois = qois[order]
    parameters = parameters[order]
    centered = parameters - mean[None]
    ## Compute gradients
    sample_size =  parameters.shape[0]
    
    #utilities = np.ones(qois.shape)/mu
    if utilities_type == 'none':
      mu = sample_size
      utilities = -(qois*1.0 / mu)
      utilities = utilities #/ np.sum(np.abs(utilities))
   
    if utilities_type == 'linear':
      utilities = np.zeros(sample_size)
      mu = max(1,int(sample_size/10))
      utilities[0:mu] = 1./np.linspace(1,mu,mu)
      utilities = utilities / np.sum(utilities)

    if utilities_type == 'log': 
      mu = sample_size
      ranks = np.linspace(1,sample_size,sample_size)
      utilities = np.maximum(0, np.log(sample_size/2 + 1) - np.log(ranks))
      utilities /= np.sum(utilities)
      utilities -= 1.0 / sample_size
   
    #ranks = np.linspace(1,sample_size,sample_size)
    #utilities = np.maximum(0, np.log(sample_size/2 + 1) - np.log(ranks))
    #utilities /= np.sum(utilities)
    #utilities -= 1.0 / sample_size

    dJdmu = 0.0
    for i in range(mu):
      dJdmu += utilities[i] * centered[i] 

    # Now compute for covariance
    dJdcov = 0.0
    for i in range(mu):
      term = (centered[i])[:,None] @ (centered[i])[:,None].transpose() - covariance
      dJdcov += utilities[i]*term

    true_utils = np.zeros(sample_size)
    for i in range(0,sample_size):
        true_utils[i] = utilities[order[i]]

    return dJdmu,dJdcov,qois


def nes(model: QoiModel,
        parameter_space: ParameterSpace,
        sample_size: int,
        iterations: int = 100,
        mean_learning_rate: float = 0.8,
        covariance_learning_rate: float = 0.8,
        work_dir="work/",
        utilities_type="linear",
        random_seed: int = 0,
        restart_file: str = None,
        evaluation_concurrency: int = 1):
    """
    Natural Evolution Strategies / EMNA-style optimizer with Gaussian search distribution.

    The search distribution is a multivariate Gaussian parameterized by mean and covariance.
    Parameters are updated with natural gradient ascent on the expected fitness (negative QoI).

    Args:
        model: QoiModel used to evaluate QoI for sampled parameters.
        parameter_space: ParameterSpace that supplies names, dimensionality, and optional bounds.
        sample_size: Number of samples per NES iteration (population size).
        iterations: Number of optimization iterations.
        mean_learning_rate: Step size for the mean update.
        covariance_learning_rate: Step size for the covariance update.
        work_dir: Base directory for model runs.
        utilities_type: Utility weighting scheme ("linear" or "log").
        random_seed: Seed for reproducible sampling.
        restart_file: Optional path to restart.npz to resume/append history.
        evaluation_concurrency: Number of concurrent workers for model evaluations (1 = serial).

    Returns:
        Tuple containing best_parameters (np.ndarray) and best_qoi (float).
    """
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    restart_file_path = os.path.join(work_dir, "restart.npz")
    parameter_names = parameter_space.get_names()
    dim = parameter_space.get_dimensionality()
    bound = getattr(parameter_space, "bound_samples", None)

    start_iteration = 0
    objective_history = np.zeros(0)
    wall_time_history = np.zeros(0)
    samples_history = []
    qois_history = []
    start_time = time.time()
    if restart_file  and os.path.isfile(restart_file):
        data = np.load(restart_file, allow_pickle=True)
        mean = data["mean"]
        covariance = data["covariance"]
        best_qoi = float(data["best_qoi"])
        best_parameters = data["best_parameters"]
        start_iteration = int(data["iteration"])
        samples_history = data.get("samples_history", []).tolist() if hasattr(data, "get") else []
        qois_history = data.get("qois_history", []).tolist() if hasattr(data, "get") else []
        if "rng_state" in data:
            rng_state = data["rng_state"].item() if getattr(data["rng_state"], "dtype", None) == object else data["rng_state"]
            rng.bit_generator.state = rng_state
    else:
        initial_samples = parameter_space.generate_samples(max(sample_size, dim + 1), seed=random_seed)
        mean = np.mean(initial_samples, axis=0)
        covariance = np.diag(np.diag(np.cov(initial_samples.transpose())))
        best_qoi = np.inf
        best_parameters = mean.copy()

    for iteration in range(start_iteration, iterations):
        samples = rng.multivariate_normal(mean=mean, cov=covariance, size=sample_size)

        qois = _evaluate_samples(model, parameter_names, samples, work_dir, iteration, bound, evaluation_concurrency=evaluation_concurrency)
        samples_history.append(samples)
        qois_history.append(qois)

        dJdmu,dJdcov,_ = compute_nes_gradients(qois,samples,mean,covariance,utilities_type)

        mean = mean +  mean_learning_rate * dJdmu
        covariance = covariance + covariance_learning_rate * dJdcov
        covariance = np.abs(np.diag(covariance))
        covariance = np.diag(np.fmax(covariance,1e-10)) 

        if callable(bound):
            mean = bound(mean.reshape(1, -1))[0]

        iteration_best_idx = np.argmin(qois)
        iteration_best_qoi = qois[iteration_best_idx]
        if iteration_best_qoi < best_qoi:
            best_qoi = iteration_best_qoi
            best_parameters = samples[iteration_best_idx].copy()

        objective_history = np.append(objective_history,np.mean(qois))
        wall_time = time.time() - start_time
        wall_time_history = np.append(wall_time_history,wall_time)
        np.savez(restart_file_path,
                 mean=mean,
                 covariance=covariance,
                 best_qoi=best_qoi,
                 objective_history=objective_history,
                 wall_time=wall_time_history,
                 samples_history=np.array(samples_history, dtype=object),
                 qois_history=np.array(qois_history, dtype=object),
                 best_parameters=best_parameters,
                 iteration=iteration + 1,
                 rng_state=rng.bit_generator.state)
        print("==================================")
        print(f"Iteration={iteration}, Wall time={wall_time:.3f}s Best Error={best_qoi:.6f}, Mean Error={np.mean(qois):.6f}")
        print(f"Mean Parameters={mean}")
        print(f"Best Parameters={best_parameters}")
        print("==================================")

    return best_parameters, float(best_qoi)



def mf_nes(model: QoiModel,
        rom_model_builder: QoiModelBuilder,
        parameter_space: ParameterSpace,
        sample_size: int,
        aux_sample_size: int,
        iterations: int = 100,
        mean_learning_rate: float = 0.8,
        covariance_learning_rate: float = 0.8,
        work_dir="work/",
        utilities_type="linear",
        method="MFMC",
        random_seed: int = 0,
        restart_file: str = None,
        evaluation_concurrency: int = 1):
    """
    Natural Evolution Strategies / EMNA-style optimizer with Gaussian search distribution.

    The search distribution is a multivariate Gaussian parameterized by mean and covariance.
    Parameters are updated with natural gradient ascent on the expected fitness (negative QoI).

    Args:
        model: QoiModel used to evaluate QoI for sampled parameters.
        parameter_space: ParameterSpace that supplies names, dimensionality, and optional bounds.
        sample_size: Number of full-order model (FOM) samples per iteration.
        aux_sample_size: Number of auxiliary/ROM samples per iteration.
        iterations: Number of optimization iterations.
        mean_learning_rate: Step size for the mean update.
        covariance_learning_rate: Step size for the covariance update.
        work_dir: Base directory for model runs.
        utilities_type: Utility weighting scheme ("linear" or "log").
        method: Multi-fidelity strategy ("MFMC" or "vanilla").
        random_seed: Seed for reproducible sampling.
        restart_file: Optional path to restart.npz to resume/append history.
        evaluation_concurrency: Number of concurrent workers for model evaluations (1 = serial).

    Returns:
        Tuple containing best_parameters (np.ndarray) and best_qoi (float).
    """
    np.random.seed(random_seed)
    rom_aux_sample_size = aux_sample_size 
    rng = np.random.default_rng(random_seed)
    restart_file_path = os.path.join(work_dir, "restart.npz")
    parameter_names = parameter_space.get_names()
    dim = parameter_space.get_dimensionality()
    bound = getattr(parameter_space, "bound_samples", None)

    start_iteration = 0
    training_dirs = []
    samples_history_fom = []
    qois_history_fom = []
    start_time = time.time()
    if restart_file and os.path.isfile(restart_file):
        data = np.load(restart_file, allow_pickle=True)
        mean = data["mean"]
        covariance = data["covariance"]
        best_qoi = float(data["best_qoi"])
        best_parameters = data["best_parameters"]
        start_iteration = int(data["iteration"])
        training_dirs = data["training_dirs"].tolist()
        if "samples_history_fom" in data:
            samples_history_fom = data["samples_history_fom"].tolist()
        if "qois_history_fom" in data:
            qois_history_fom = data["qois_history_fom"].tolist()
        rng_state = data["rng_state"].item() if getattr(data["rng_state"], "dtype", None) == object else data["rng_state"]
        rng.bit_generator.state = rng_state
    else:
        initial_samples = parameter_space.generate_samples(max(sample_size, dim + 1), seed=random_seed)
        mean = np.mean(initial_samples, axis=0)
        covariance = np.diag(np.diag(np.cov(initial_samples.transpose())))
        best_qoi = np.inf
        best_parameters = mean.copy()
        training_dirs = []
        samples_history_fom = []
        qois_history_fom = []

    objective_history = np.zeros(0)
    wall_time_history = np.zeros(0)
    for iteration in range(start_iteration, iterations):
        samples = rng.multivariate_normal(mean=mean, cov=covariance, size=sample_size + rom_aux_sample_size)

        if method == 'vanilla':
            rho = 0
            qois = _evaluate_samples(model, parameter_names, samples[:sample_size], work_dir, iteration, bound, evaluation_concurrency=evaluation_concurrency)
            samples_history_fom.append(samples[:sample_size])
            qois_history_fom.append(qois)
            for i in range(sample_size):
                training_dirs.append(f"{work_dir}/iteration_{iteration}/run_{i}")
    
            offline_dir = f'{work_dir}/iteration_{iteration}/'
            t0 = time.time()
            rom_model = rom_model_builder.build_from_training_dirs(
                offline_dir,
                training_dirs[-sample_size:],
                parameters=np.vstack(samples_history_fom) if samples_history_fom else None,
                qois=np.vstack(qois_history_fom) if qois_history_fom else None)

            rom_qois = _evaluate_samples(rom_model, parameter_names, samples[sample_size:sample_size + rom_aux_sample_size], work_dir, iteration, bound, label_prefix="run_rom_", evaluation_concurrency=evaluation_concurrency, start_index=sample_size)
            qois = np.concatenate([qois, rom_qois])
    
            ## Compute gradients
            dJdmu_FOM,dJdCov_FOM,_ = compute_nes_gradients(qois,samples,mean,covariance,utilities_type)
    
            dJdmu = dJdmu_FOM
            dJdCov = dJdCov_FOM

        if method == 'MFMC':
            qois = _evaluate_samples(model, parameter_names, samples[:sample_size], work_dir, iteration, bound, evaluation_concurrency=evaluation_concurrency)
            samples_history_fom.append(samples[:sample_size])
            qois_history_fom.append(qois)
            for i in range(sample_size):
                training_dirs.append(f"{work_dir}/iteration_{iteration}/run_{i}")


            offline_dir = f'{work_dir}/iteration_{iteration}/'
            rom_model = rom_model_builder.build_from_training_dirs(
                offline_dir,
                training_dirs[-sample_size:],
                parameters=np.vstack(samples_history_fom) if samples_history_fom else None,
                qois=np.vstack(qois_history_fom) if qois_history_fom else None)

            rom_qois = _evaluate_samples(rom_model, parameter_names, samples[:sample_size + rom_aux_sample_size], work_dir, iteration, bound, label_prefix="run_rom_", evaluation_concurrency=evaluation_concurrency)
 

            holdout = False
            if holdout:
                val_qois = np.zeros(holdout_size)
                for i in range(sample_size,sample_size + holdout_size):
                    run_directory = work_dir + "/iteration_" + str(iteration) + "/tmp_" + str(i)
                    samples_to_pass = samples[i]
                    if callable(bound):
                      samples_to_pass = bound(samples_to_pass.reshape(1, -1))[0]
                    val_qois[i-sample_size], run_time = prepare_and_run(model, run_directory, parameter_names, samples_to_pass)
    
                for i in range(sample_size):
                    run_directory = f"{work_dir}/iteration_{iteration}/run_{i}"
                    samples_to_pass = samples[i]
                    if callable(bound):
                      samples_to_pass = bound(samples_to_pass.reshape(1, -1))[0]
                    qois[i], _ = prepare_and_run(model, run_directory, parameter_names, samples_to_pass)
                    training_dirs.append(run_directory)
       
                _,_,_,utilities_FOM_val = compute_nes_gradients(val_qois,samples[sample_size:sample_size + holdout_size],mean,covariance,utilities_type)
                _,_,_,utilities_ROM_val = compute_nes_gradients(rom_qois[sample_size:sample_size + holdout_size],samples[sample_size:sample_size + holdout_size],mean,covariance,utilities_type)
                rho = np.corrcoef(utilities_FOM_val,utilities_ROM_val)[0,1]
            else:
                rho = 1.0

            ## Compute gradients
            dJdmu_FOM,dJdCov_FOM,utilities_FOM = compute_nes_gradients(qois,samples,mean,covariance,utilities_type)
            dJdmu_ROMa,dJdCov_ROMa,utilities_ROMa = compute_nes_gradients(rom_qois[0:sample_size],samples[0:sample_size],mean,covariance,utilities_type)
            dJdmu_ROMb,dJdCov_ROMb,utilities_ROMb = compute_nes_gradients(rom_qois,samples,mean,covariance,utilities_type)
            dJdmu = dJdmu_FOM + 1*(dJdmu_ROMb - dJdmu_ROMa)
            dJdCov = dJdCov_FOM + 1*(dJdCov_ROMb - dJdCov_ROMa)

        mean = mean +  mean_learning_rate * dJdmu
        covariance = covariance + covariance_learning_rate * dJdCov
        covariance = np.abs(np.diag(covariance))
        covariance = np.diag(np.fmax(covariance,1e-10)) 
        if callable(bound):
            mean = bound(mean.reshape(1, -1))[0]

        iteration_best_idx = np.argmin(qois)
        iteration_best_qoi = qois[iteration_best_idx]
        if iteration_best_qoi < best_qoi:
            best_qoi = iteration_best_qoi
            best_parameters = samples[iteration_best_idx].copy()

        objective_history = np.append(objective_history,np.mean(qois))
        wall_time = time.time() - start_time
        wall_time_history = np.append(wall_time_history,wall_time)
        np.savez(restart_file_path,
                 mean=mean,
                 covariance=covariance,
                 best_qoi=best_qoi,
                 objective_history=objective_history,
                 wall_time=wall_time_history,
                 best_parameters=best_parameters,
                 training_dirs=np.array(training_dirs, dtype=object),
                 samples_history_fom=np.array(samples_history_fom, dtype=object),
                 qois_history_fom=np.array(qois_history_fom, dtype=object),
                 iteration=iteration + 1,
                 rng_state=rng.bit_generator.state)
        print("==================================")
        print(f"Iteration={iteration}, Wall time={wall_time:.3f}s Best Error={best_qoi:.6f}, Mean Error={np.mean(qois):.6f}")
        print(f"Mean Parameters={mean}")
        print(f"Best Parameters={best_parameters}")
        print("==================================")

    return best_parameters, float(best_qoi)

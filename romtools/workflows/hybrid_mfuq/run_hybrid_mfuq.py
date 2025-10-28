#
# ************************************************************************
#
#                         ROM Tools and Workflows
# Copyright 2019 National Technology & Engineering Solutions of Sandia,LLC
#                              (NTESS)
#
# Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.
#
# ROM Tools and Workflows is licensed under BSD-3-Clause terms of use:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Questions? Contact Eric Parish (ejparis@sandia.gov)
#
# ************************************************************************
#

import time
import os
import numpy as np

import romtools.workflows.hybrid_mfuq.mfuq_methods as mfmc
from romtools.workflows.models import QoiModel
from romtools.workflows.parameter_spaces import ParameterSpace
from romtools.workflows.workflow_utils import create_empty_dir
from romtools.workflows.model_builders import QoiModelBuilder


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


def _prepare_directory_and_run(model, run_directory, parameter_dict, overwrite=False):
    create_empty_dir(run_directory)
    model.populate_run_directory(run_directory, parameter_dict)

    passed_file = os.path.join(run_directory, "passed.txt")

    if os.path.isfile(passed_file) and not overwrite:
        print("Skipping (Sample has already run successfully) \n")
        return

    print("Running...\n")
    model.run_model(run_directory, parameter_dict)
    np.savetxt(passed_file, [0], fmt="%i")


# WARNING: the computed time will be wrong if sample is run ahead of time somewhere else
#          (so passed.txt exists) and QoI/time files are never saved
def _compute_QoI_and_time(model, run_directory, parameter_dict, overwrite=False):
    t0 = time.time()
    _prepare_directory_and_run(model, run_directory, parameter_dict, overwrite)
    
    qoi_path = os.path.join(run_directory, "qoi.txt")
    time_path = os.path.join(run_directory, "time.txt")
    qoi_exists = os.path.isfile(qoi_path) and os.path.isfile(time_path)
    
    if not overwrite and qoi_exists:
        print("Reading in QoI value and runtime \n")
        model_qoi = np.loadtxt(qoi_path)
        model_time = np.loadtxt(time_path)
    else:
        print("Computing QoI value and runtime \n")
        model_qoi = model.compute_qoi(run_directory, parameter_dict)
        model_time = time.time() - t0
        np.savetxt(qoi_path, [model_qoi])
        np.savetxt(time_path, [model_time])

    return np.array(model_qoi), np.array(model_time)


# Helper function: reshape data consistent with rom runs
def _fancy_reshape(data_master, labels_list):
    return [
        np.array([
            [data_master[sample_index] for sample_index in test_group]
            for test_group in test_labels
        ])
        for test_labels in labels_list
    ]


# Helper function: run ACV many times and get best result
def _solve_best_mfuq(obj, n_trials=50, log=True):
    """Solve MFMC object multiple times and return best result."""
    best_fval = float('inf')
    best_x = None
    for _ in range(n_trials):
        obj.solve()
        if obj.result.success:
            if log: 
                fval = np.exp(obj.result.fun)
            else:
                fval = obj.result.fun
            # fval = obj.result.fun
            if 0 <= fval <= best_fval:
                best_fval = fval
                best_x = obj.result.x
    # print(f'aaaa {best_fval}')
    return best_fval, best_x


# Function which handles the pilot sampling and saves data
def do_pilot_sampling(fom_model: QoiModel,
               aux_model: QoiModel,
               rom_model_builder: QoiModelBuilder,
               parameter_space: ParameterSpace,
               hybrid_MFMC_directory: str,
               pilot_manager,
               max_combinations: int = 25,
               random_seed: int = 2025,
               overwrite: bool = False
               ):

    hybrid_file = open(f"{hybrid_MFMC_directory}/pilot_status.log", "w", encoding="utf-8") 

    # Extract from manager class (def'd in mfuq_methods.py)
    pilot_list = pilot_manager.s_list
    pilot_sample_size = pilot_manager.num_pilot

    # Split train and test
    out = "Creating train and test labels \n"
    hybrid_file.write(out), print(out)

    pilot_manager.set_train_and_test_labels(max_groups=max_combinations)
    train_labels_list = pilot_manager.train_labels
    test_labels_list = pilot_manager.test_labels

    # create parameter samples for pilot
    parameter_samples = parameter_space.generate_samples(pilot_sample_size)
    parameter_names = parameter_space.get_names()

    # Make fixed model directories and get fixed model QoIs
    out = "Creating fixed model samples and computing QoIs \n"
    hybrid_file.write(out), print(out)

    training_dirs = []
    fom_qois, fom_times = [], []
    aux_qois, aux_times = [], []

    for sample_index, sample in enumerate(parameter_samples):
        print(f"===========  Sample {sample_index} ============ \n")

        fom_run_directory = f'{hybrid_MFMC_directory}/pilot/fom/run_{sample_index}'
        aux_run_directory = f'{hybrid_MFMC_directory}/pilot/aux/run_{sample_index}'
        parameter_dict = _create_parameter_dict(parameter_names, sample)

        fom_qoi, fom_time = _compute_QoI_and_time(fom_model, fom_run_directory, 
                                                 parameter_dict, overwrite=overwrite)
        aux_qoi, aux_time = _compute_QoI_and_time(aux_model, aux_run_directory,
                                                parameter_dict, overwrite=overwrite)    
         
        fom_qois.append(fom_qoi), fom_times.append(fom_time)
        aux_qois.append(aux_qoi), aux_times.append(aux_time)
        
        training_dirs.append(fom_run_directory)

    fom_qois_master, fom_times_master = np.array(fom_qois), np.array(fom_times)
    aux_qois_master, aux_times_master = np.array(aux_qois), np.array(aux_times)

    # Make variable ROM directories and train all ROMs
    out = "Creating ROM bases \n"
    hybrid_file.write(out), print(out)
    rom_models = []

    for pilot_index, basis_size in enumerate(pilot_list):
        out = f"Basis size {basis_size} \n"
        hybrid_file.write(out), print(out)

        base_rom_directory = f'{hybrid_MFMC_directory}/pilot/rom/basis_size_{basis_size}'
        train_label_set = train_labels_list[pilot_index]
        rom_models_at_pilot_index = []

        for train_label in train_label_set:
            out = f"Training ROM from samples {train_label} \n"
            hybrid_file.write(out), print(out)

            combination_id = '-'.join(str(index) for index in train_label)
            rom_offline_directory = os.path.join(base_rom_directory, f'combination_{combination_id}')
            # rom_offline_directory = os.path.join(base_rom_directory, f'combination_{train_label}')
            create_empty_dir(rom_offline_directory)
            training_dir = [training_dirs[sample_index] for sample_index in train_label]

            rom_model = rom_model_builder.build_from_training_dirs(rom_offline_directory, training_dir)
            rom_models_at_pilot_index.append(rom_model)

        rom_models.append(rom_models_at_pilot_index)

    # Sample ROMs and compute QoIs
    out = "Sampling ROMs on test parameters \n"
    hybrid_file.write(out), print(out)
    rom_qois, rom_times = [], []

    for i, basis_size in enumerate(pilot_list):
        out = f"Basis size {basis_size} \n"
        hybrid_file.write(out), print(out)

        base_rom_directory_i = f'{hybrid_MFMC_directory}/pilot/rom/basis_size_{basis_size}'
        train_labels_i = train_labels_list[i]
        test_labels_i = test_labels_list[i]

        rom_qois_i, rom_times_i = [], []

        for j, test_label in enumerate(test_labels_i):
            out = f"Testing ROM built from samples {train_labels_i[j]} \n"
            hybrid_file.write(out), print(out)

            combination_id = '-'.join(str(index) for index in train_labels_i[j])
            base_rom_directory_ij = os.path.join(base_rom_directory_i, f'combination_{combination_id}')
            # base_rom_directory_ij = os.path.join(base_rom_directory_i, f'combination_{train_labels_i[j]}')
            test_samples = [parameter_samples[sample_index] for sample_index in test_label]

            rom_model = rom_models[i][j]
            rom_qois_ij, rom_times_ij = [], []

            for k, sample in enumerate(test_samples):
                out = f"Testing on sample {test_label[k]} \n"
                hybrid_file.write(out), print(out)

                rom_run_directory = os.path.join(base_rom_directory_ij, f'run_test_sample_{test_label[k]}')
                parameter_dict = _create_parameter_dict(parameter_names, sample)

                rom_qoi, rom_time = _compute_QoI_and_time(rom_model, rom_run_directory,
                                                          parameter_dict, overwrite=overwrite)
                
                rom_qois_ij.append(rom_qoi), rom_times_ij.append(rom_time)
            rom_qois_i.append(rom_qois_ij), rom_times_i.append(rom_times_ij)
        rom_qois.append(np.array(rom_qois_i)), rom_times.append(np.array(rom_times_i))

    # Reshape master lists to be compared with list of ROMs
    fom_qois = _fancy_reshape(fom_qois_master, test_labels_list)
    aux_qois = _fancy_reshape(aux_qois_master, test_labels_list)
    fom_times = _fancy_reshape(fom_times_master, test_labels_list)
    # aux_times = fancy_reshape(aux_times_master, test_labels_list)

    # Compute pilot data for surrogate training
    out = "Computing average cost and correlation data \n"
    hybrid_file.write(out), print(out)

    fom_aux_corr = pilot_manager.estimate_FOM_correlations(
        [fom_qois_master[None,:]], [aux_qois_master[None,:]])[0]
    fom_rom_corrs = pilot_manager.estimate_FOM_correlations(fom_qois, rom_qois)
    aux_rom_corrs = pilot_manager.estimate_FOM_correlations(aux_qois, rom_qois)
    
    normalized_aux_time = np.mean([aux_time / fom_time for (aux_time, fom_time) 
                                   in zip(aux_times_master, fom_times_master)])
    # normalized_aux_times = [np.mean(aux_group / fom_group) for (aux_group, fom_group)
    #                          in zip(aux_times, fom_times)]
    # normalized_aux_time = np.mean(normalized_aux_times)
    normalized_rom_times = [np.mean(rom_group / fom_group) for (rom_group, fom_group) 
                            in zip(rom_times, fom_times)]

    data_dict = {'fom_qois_master': fom_qois_master, 'aux_qois_master': aux_qois_master,
                 'fom_aux_corr': fom_aux_corr, 'fom_rom_corrs': fom_rom_corrs,
                 'aux_rom_corrs': aux_rom_corrs, 
                 'fom_times_master': fom_times_master,
                 'normalized_aux_time': normalized_aux_time, 
                 'normalized_rom_times': normalized_rom_times,
                 'parameter_samples': parameter_samples,
                 'training_dirs': np.array(training_dirs)}

    # print(normalized_rom_times)
    # rom_shapes = [rom_time.shape for rom_time in rom_times]
    # fom_shapes = [fom_time.shape for fom_time in fom_times]
    # print(f'aaaaaa {rom_shapes, fom_shapes}')

    np.savez(f"{hybrid_MFMC_directory}/pilot_results.npz", **data_dict)

    hybrid_file.close()


# Function which trains surrogates based on data collected during pilot
def build_surrogates(data_npz, pilot_list):

    with np.load(data_npz) as data:
        fom_aux_corr = data['fom_aux_corr']
        fom_rom_corrs = data['fom_rom_corrs']
        aux_rom_corrs = data['aux_rom_corrs']
        normalized_aux_time = data['normalized_aux_time']
        normalized_rom_times = data['normalized_rom_times']

    pilot_sizes = np.tile(np.array(pilot_list), (2,1))

    # Correlations fit with sigmoids
    # HACKED to prevent tensor product when one s doesn't vary...
    def rho12(s): return fom_aux_corr
    rho13_half = mfmc.fit_sigmoid(pilot_sizes[0][None,:], fom_rom_corrs)
    rho23_half = mfmc.fit_sigmoid(pilot_sizes[0][None,:], aux_rom_corrs)  
    def rho13(s): return rho13_half(s[0])
    def rho23(s): return rho23_half(s[0])

    # UN-HACKED VERSION (kind of weird)
    # rho13 = mfmc.fit_polynomial(pilot_sizes, fom_rom_corrs, order=1)
    # rho23 = mfmc.fit_polynomial(pilot_sizes, aux_rom_corrs, order=1)
    # rho13 = mfmc.fit_sigmoid(pilot_sizes, fom_rom_corrs)
    # rho23 = mfmc.fit_sigmoid(pilot_sizes, aux_rom_corrs)

    # Costs fit with polynomials
    def cost2(s): return normalized_aux_time
    # def cost2(s): return normalized_rom_times[0]
    cost3_half = mfmc.fit_polynomial(pilot_sizes[0][None,:], normalized_rom_times, order=1)
    cost3 = lambda s: cost3_half(s[0])
    # cost3 = mfmc.fit_polynomial(pilot_sizes, normalized_rom_times, order=1)

    # Lists of correlations and costs for ACV procedure
    hifi_corr_list = [rho12, rho13]
    lofi_corr_list = [rho23]
    cost_list = [cost2, cost3]

    return (hifi_corr_list, lofi_corr_list, cost_list)


# Function which trains the optimal ROM and computes its pilot statistics
def train_opimized_rom_and_compute_stats(hybrid_file, hybrid_MFMC_directory, pilot_manager,
                                         rom_basis_num, parameter_space, parameter_samples,
                                         training_dirs, fom_model, rom_model_builder, data_npz,
                                         overwrite=False):

    out = f"Training ROM from first {rom_basis_num} samples \n"
    hybrid_file.write(out), print(out)
    parameter_names = parameter_space.get_names()

    rom_offline_directory = f'{hybrid_MFMC_directory}/pilot/rom_optimized/basis_size_{rom_basis_num}'
    create_empty_dir(rom_offline_directory)
    num_existing = len(training_dirs)
    
    if num_existing < rom_basis_num: 
        out = "Sampling extra FOMs for training \n"
        hybrid_file.write(out), print(out)    
        
        num_extra = rom_basis_num - num_existing
        extra_parameter_samples = parameter_space.generate_samples(num_extra)

        for sample_index, sample in enumerate(extra_parameter_samples):
            print(f"===========  Sample {sample_index} ============ \n")
            
            run_index = num_existing + sample_index
            fom_run_directory = f'{hybrid_MFMC_directory}/pilot/fom/run_{run_index}'
            
            parameter_dict = _create_parameter_dict(parameter_names, sample)
            _prepare_directory_and_run(fom_model, fom_run_directory, parameter_dict, overwrite=overwrite)

        training_dirs.append(fom_run_directory)
    
    rom_model = rom_model_builder.build_from_training_dirs(
        rom_offline_directory, training_dirs[:rom_basis_num])

    # Get pilot QoIs for correlation and cost computation
    out = "Creating trained ROM samples and computing pilot QoIs \n"
    hybrid_file.write(out), print(out)
    trained_rom_qois, trained_rom_times = [], []

    for sample_index, sample in enumerate(parameter_samples):
        print(f"===========  Sample {sample_index} ============ \n")
        
        rom_run_directory = os.path.join(rom_offline_directory, f'run_{sample_index}')
        parameter_dict = _create_parameter_dict(parameter_names, sample)
        
        rom_qoi, rom_time = _compute_QoI_and_time(rom_model, rom_run_directory,
                                                  parameter_dict, overwrite=overwrite)
        
        trained_rom_qois.append(rom_qoi), trained_rom_times.append(rom_time)
    trained_rom_qois_master, trained_rom_times_master = np.array(trained_rom_qois), np.array(trained_rom_times)

    # Compute correlation and cost data
    with np.load(data_npz) as data:
        fom_qois_master = data['fom_qois_master']
        aux_qois_master = data['aux_qois_master']
        fom_times_master = data['fom_times_master']

    fom_rom_corr = pilot_manager.estimate_FOM_correlations(
        [fom_qois_master[None,:]], [trained_rom_qois_master[None,:]])[0]
    aux_rom_corr = pilot_manager.estimate_FOM_correlations(
        [aux_qois_master[None,:]], [trained_rom_qois_master[None,:]])[0]
    normalized_rom_time = np.mean([rom_time / fom_time for (rom_time, fom_time) 
                                   in zip(trained_rom_times_master, fom_times_master)])

    # Save data to file
    data_dict = {'fom_rom_corr': fom_rom_corr, 
                 'aux_rom_corr': aux_rom_corr,
                 'normalized_rom_time': normalized_rom_time}
    np.savez(f"{hybrid_MFMC_directory}/trained_{rom_basis_num}_sample_rom_results.npz",
              **data_dict)


def run_hybrid_mfuq(fom_model: QoiModel,
               aux_model: QoiModel,
               rom_model_builder: QoiModelBuilder,
               parameter_space: ParameterSpace,
               absolute_hybrid_MFMC_work_directory: str,
               pilot_sample_size: int = 20, 
               pilot_list: list = [1,3,5,7,9],
               max_combinations: int = 25,
               tunable_range: list = [1,20],
               budget: int = 40,
               allocate_based_on: str = 'min',
               log_of_objective: bool = True,
               overwrite: bool = True,
               random_seed: int = 2025
               ):
    '''
    Main implementation of the hybrid MFUQ algorithm.
    Right now, it is assumed that we have a fom_model, another
        fixed aux_model (e.g., a trained ROM), and a variable ROM. 

    The first step is a pilot sampling which uses pilot_sample_size number
        of pilot samples to compute correlation data at the basis sizes in
        pilot_list.  This is done through a combinatorial average with number
        of groups controlled by max_combinations.  
    The second step computes surrogates from these data and uses them to
        solve the ACV optimization problem, returning the optimal basis
        size and sampling strategy.
    The third step computes the optimal rom, and recomputes the FOM/ROM
        correlation information using the pilot sampling.  This is used to
        get an accurate projection of the optimal ACV estimator variance.    
    '''
    # Initialize stuff
    np.random.seed(random_seed)
    if allocate_based_on not in ['min', 'max']:
        raise ValueError("Allocation type not implemented!")
    hybrid_MFMC_directory = absolute_hybrid_MFMC_work_directory
    create_empty_dir(hybrid_MFMC_directory)
    hybrid_file = open(f"{hybrid_MFMC_directory}/hybrid_status.log", "w", encoding="utf-8") 
    hybrid_file.write("Hybrid MFUQ status \n")

    # Do pilot sampling (or load previous results)
    pilot_manager = mfmc.Pilot(pilot_list, pilot_sample_size, random_seed=random_seed)
    if not os.path.exists(f"{hybrid_MFMC_directory}/pilot_results.npz"):
        msg = "Doing pilot sampling \n"
        hybrid_file.write(msg), print(msg)
        do_pilot_sampling(fom_model, aux_model, rom_model_builder,
               parameter_space, hybrid_MFMC_directory,
               pilot_manager, max_combinations,
               random_seed, overwrite)
    else:
        msg = "Pilot sampling done previously \n"
        hybrid_file.write(msg), print(msg)
    data_npz = f"{hybrid_MFMC_directory}/pilot_results.npz"

    # Loading stats from pilot
    with np.load(data_npz) as data:
        fom_rom_corrs = data['fom_rom_corrs']
        aux_rom_corrs = data['aux_rom_corrs']
        normalized_rom_times = data['normalized_rom_times']
        parameter_samples = data['parameter_samples']
        training_dirs = data['training_dirs'].tolist()

    # Train surrogates for cost and correlation
    out = "Training surrogates for cost and correlation \n"
    hybrid_file.write(out), print(out)
    hifi_corr_list, lofi_corr_list, cost_list = build_surrogates(
        data_npz, pilot_list)

    # Solve MFUQ problem based on surrogates to get predictions
    # Budget and bounds on optimization variables N, r, s
    # Elements of budget_list are HARD CODED based on given budget
    print("Solving the hybrid MFUQ optimization problem \n")
    budget_list = [budget * (i+1) for i in range(6)]
    bounds = [(1, None), (1.001, None), (1.001, None), 
              (0, 0), tuple(tunable_range)]
    
    funcMFs, xMFs = [], []
    funcISs, xISs = [], []
    for budget in budget_list:
        for model_type, func_list, x_list in [('MF', funcMFs, xMFs), ('IS', funcISs, xISs)]:
            obj = mfmc.MFMC(budget, model_type, hybrid=False)
            obj.set_corrs_and_costs(hifi_corr_list, lofi_corr_list, cost_list)
            obj.set_objective_and_constraint(log=log_of_objective,bounds=bounds)

            fval, x = _solve_best_mfuq(obj, n_trials=50, log=log_of_objective)
            out = f'Predicted variance ratio for {model_type} at budget {budget} is {fval} and occurs at {x} \n'
            hybrid_file.write(out), print(out)

            func_list.append(fval)
            x_list.append(x)

    # Allocate training samples based on IS at lowest or highest budget
    if allocate_based_on == 'min':
        s_star = xISs[0][3:]
    elif allocate_based_on == 'max':
        s_star = xISs[-1][3:]

    # Train the optimized ROM or load from previous
    rom_basis_num = int(round(s_star[1]))
    if not os.path.exists(f"{hybrid_MFMC_directory}/trained_{rom_basis_num}_sample_rom_results.npz"):
        out = "Doing ROM training and computing pilot stats \n"
        hybrid_file.write(out), print(out)
        train_opimized_rom_and_compute_stats(hybrid_file, hybrid_MFMC_directory, pilot_manager,
                                         rom_basis_num, parameter_space, parameter_samples,
                                         training_dirs, fom_model, rom_model_builder, data_npz,
                                         overwrite=overwrite)
    else:
        out = "ROM training done previously \n"
        hybrid_file.write(out), print(out)

    # Load statistics of trained ROM and build surrogates
    with np.load(f"{hybrid_MFMC_directory}/trained_{rom_basis_num}_sample_rom_results.npz") as data:
        fom_trained_rom_corr = data['fom_rom_corr']
        aux_trained_rom_corr = data['aux_rom_corr']
        normalized_trained_rom_time = data['normalized_rom_time'] 

    def exact_rho13(s): return fom_trained_rom_corr
    def exact_rho23(s): return aux_trained_rom_corr
    def exact_cost3(s): return normalized_trained_rom_time

    hifi_corr_list_exact = [hifi_corr_list[0], exact_rho13]
    lofi_corr_list_exact = [exact_rho23]
    cost_list_exact = [cost_list[0], exact_cost3]

    # Solve fixed MFUQ problem with optimized data to validate prediction
    out = "Solving the MFUQ optimization problem with fixed models \n"
    hybrid_file.write(out), print(out)
    bounds = [(1, None), (1.001, None), (1.001, None), 
              (0, 0), (rom_basis_num, rom_basis_num)]

    funcMFs_exact, xMFs_exact = [], []
    funcISs_exact, xISs_exact = [], []
    for budget in budget_list:
        for model_type, func_list, x_list in [
            ('MF', funcMFs_exact, xMFs_exact),
            ('IS', funcISs_exact, xISs_exact) ]:

            obj = mfmc.MFMC(budget, model_type, hybrid=True)
            obj.set_corrs_and_costs(hifi_corr_list_exact, lofi_corr_list_exact, cost_list_exact)
            obj.set_objective_and_constraint(log=log_of_objective,bounds=bounds)

            fval, x = _solve_best_mfuq(obj, n_trials=50, log=log_of_objective)
            out = f'Variance ratio for {model_type} at budget {budget} is {fval} and occurs at {x} \n'
            hybrid_file.write(out), print(out)

            func_list.append(fval)
            x_list.append(x)

    # Save data for visualization
    xx = np.array(budget_list)
    pilot_sizes = np.tile(pilot_manager.s_list, (2,1))
    rho12, rho13 = hifi_corr_list
    cost2, cost3 = cost_list
    rho23 = lofi_corr_list[0]
    s_plot = np.tile(np.arange(1,int(tunable_range[-1])), (2,1))
    rho12s = np.array([rho12(i) for i in s_plot[0]])
    rho13s = rho13(s_plot)
    rho23s = rho23(s_plot)
    cost2s = np.array([cost2(i) for i in s_plot[0]])
    cost3s = cost3(s_plot)
    data_dict = {'rho12s':rho12s, 'rho13s':rho13s, 'rho23s':rho23s,
                 'cost2s':cost2s, 'cost3s':cost3s,
                 'fom_rom_corrs':fom_rom_corrs, 'aux_rom_corrs':aux_rom_corrs,
                 'normalized_rom_times':normalized_rom_times, 
                 'ss':s_plot, 'pp':pilot_sizes, 's_star':s_star,
                 'xx':xx, 'fMFs':np.array(funcMFs), 'fMFs_ex':np.array(funcMFs_exact),
                 'fISs':np.array(funcISs), 'fISs_ex':np.array(funcISs_exact)}
    np.savez('visualization_data.npz', **data_dict)

    hybrid_file.close()

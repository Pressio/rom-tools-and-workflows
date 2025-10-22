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
               overwrite: bool = True,
               show_plots: bool = False,
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
    np.random.seed(random_seed)
    if allocate_based_on not in ['min', 'max']:
        raise ValueError("Allocation type not implemented!")

    hybrid_MFMC_directory = absolute_hybrid_MFMC_work_directory
    create_empty_dir(hybrid_MFMC_directory)

    hybrid_file = open(f"{hybrid_MFMC_directory}/hybrid_status.log", "w", encoding="utf-8") 
    # pylint: disable=consider-using-with
    hybrid_file.write("Hybrid MFMC status \n")

    # Set up manager for pilot sampling and label splitting
    pilot_manager = mfmc.Pilot(pilot_list, pilot_sample_size, random_seed=random_seed)

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
    def fancy_reshape(data_master, labels_list):
        return [
            np.array([
                [data_master[sample_index] for sample_index in test_group]
                for test_group in test_labels
            ])
            for test_labels in labels_list
        ]
    
    fom_qois = fancy_reshape(fom_qois_master, test_labels_list)
    aux_qois = fancy_reshape(aux_qois_master, test_labels_list)
    fom_times = fancy_reshape(fom_times_master, test_labels_list)
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

    # print(normalized_rom_times)
    # rom_shapes = [rom_time.shape for rom_time in rom_times]
    # fom_shapes = [fom_time.shape for fom_time in fom_times]
    # print(f'aaaaaa {rom_shapes, fom_shapes}')

    # Train surrogates for cost and correlation
    out = "Training surrogates for cost and correlation \n"
    hybrid_file.write(out), print(out)
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

    # Budget and bounds on optimization variables N, r, s
    # Elements of budget_list are HARD CODED based on given budget
    budget_list = [budget * (i+1) for i in range(6)]
    bounds = [(1, None), (1.001, None), (1.001, None), 
              (0, 0), tuple(tunable_range)]

    # Helper function: run ACV many times and get best result
    def solve_best_mfmc(obj, n_trials=50):
        """Solve MFMC object multiple times and return best result."""
        best_fval = float('inf')
        best_x = None
        for _ in range(n_trials):
            obj.solve()
            if obj.result.success:
                fval = np.exp(obj.result.fun)
                # fval = obj.result.fun
                if 0 <= fval <= best_fval:
                    best_fval = fval
                    best_x = obj.result.x
        # print(f'aaaa {best_fval}')
        return best_fval, best_x

    # Initialize output and results
    print("Solving the hybrid MFMC optimization problem \n")
    funcMFs, xMFs = [], []
    funcISs, xISs = [], []

    for budget in budget_list:
        for model_type, func_list, x_list in [('MF', funcMFs, xMFs), ('IS', funcISs, xISs)]:
            obj = mfmc.MFMC(budget, model_type, hybrid=False)
            obj.set_corrs_and_costs(hifi_corr_list, lofi_corr_list, cost_list)
            obj.set_objective_and_constraint(bounds)

            fval, x = solve_best_mfmc(obj, n_trials=50) #HARD CODED
            out = f'Predicted variance ratio for {model_type} at budget {budget} is {fval} and occurs at {x} \n'
            hybrid_file.write(out)
            print(out)

            func_list.append(fval)
            x_list.append(x)

    # Allocate based on IS at lowest or highest budget (IS THIS OK?)
    if allocate_based_on == 'min':
        s_star = xISs[0][3:]
    elif allocate_based_on == 'max':
        s_star = xISs[-1][3:]
    # r_star = xMFs[0][1:3]
    # N_star = xMFs[0][0]

    # # save corrs and costs for plotting
    # s_plot = np.tile(np.arange(1,int(tunable_range[-1])), (2,1))
    # rho12s = np.array([rho12(i) for i in s_plot[0]])
    # rho13s = rho13(s_plot)
    # rho23s = rho23(s_plot)
    # cost2s = np.array([cost2(i) for i in s_plot[0]])
    # cost3s = cost3(s_plot)
    # data_dict = {'rho12s':rho12s, 'rho13s':rho13s, 'rho23s':rho23s,
    #              'cost2s':cost2s, 'cost3s':cost3s,
    #              'fom_rom_corrs':fom_rom_corrs, 'aux_rom_corrs':aux_rom_corrs,
    #              'normalized_rom_times':normalized_rom_times, 
    #              'ss':s_plot, 'pp':pilot_sizes, 's_star':s_star}
    # np.savez('corrcost.npz', **data_dict)

    # # Plot correlations and costs
    # fig, ax = plt.subplots()
    # ax.plot(s_plot[0], [rho12(i) for i in s_plot[0]], color='blue', label='rho12')
    # ax.plot(s_plot[0], rho13(s_plot), color='orange', label='rho13')
    # ax.scatter(pilot_sizes[0], fom_rom_corrs, color='orange', label='FOM-ROM Corrs')
    # ax.plot(s_plot[0], rho23(s_plot), color='green', label='rho23')
    # ax.scatter(pilot_sizes[0], aux_rom_corrs, color='green', label='AUX-ROM Corrs')
    # ax.scatter(s_star[1], rho13(s_star), marker='*', s=200, color='grey')
    # ax.set_xlabel('Basis size')
    # ax.set_ylabel('Correlation')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('correlation_plot.pdf', transparent=True)
    # if show_plots: plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(s_plot[0], [cost2(i) for i in s_plot[0]], color='blue', label='cost2')
    # ax.plot(s_plot[0], cost3(s_plot), color='orange', label='cost3')
    # ax.scatter(pilot_sizes[0], normalized_rom_times, color='orange', label='Normalized ROM Time')
    # ax.scatter(s_star[1], cost3_half(s_star[1]), marker='*', s=200, color='grey')
    # ax.set_xlabel('Basis size')
    # ax.set_ylabel('Model Costs')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('cost_plot.pdf', transparent=True)
    # if show_plots: plt.show()

    # Train the optimized ROM
    rom_basis_num = int(round(s_star[1]))
    out = f"Training ROM from {rom_basis_num} samples \n"
    hybrid_file.write(out), print(out)
    
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
    fom_rom_corr = pilot_manager.estimate_FOM_correlations(
        [fom_qois_master[None,:]], [trained_rom_qois_master[None,:]])[0]
    aux_rom_corr = pilot_manager.estimate_FOM_correlations(
        [aux_qois_master[None,:]], [trained_rom_qois_master[None,:]])[0]
    normalized_rom_time = np.mean([rom_time / fom_time for (rom_time, fom_time) 
                                   in zip(trained_rom_times_master, fom_times_master)])

    def exact_rho13(s): return fom_rom_corr
    def exact_rho23(s): return aux_rom_corr
    def exact_cost3(s): return normalized_rom_time

    hifi_corr_list_exact = [rho12, exact_rho13]
    lofi_corr_list_exact = [exact_rho23]
    cost_list_exact = [cost2, exact_cost3]


    # Budget and bounds on optimization variables N, r (for fixed s)
    bounds = [(1, None), (1.001, None), (1.001, None), 
              (0, 0), (rom_basis_num, rom_basis_num)]

    # MFMC solve
    out = "Solving the MFMC optimization problem with fixed models \n"
    hybrid_file.write(out), print(out)

    # Output lists
    funcMFs_exact, xMFs_exact = [], []
    funcISs_exact, xISs_exact = [], []

    # Loop over budgets and model types
    for budget in budget_list:
        for model_type, func_list, x_list in [
            ('MF', funcMFs_exact, xMFs_exact),
            ('IS', funcISs_exact, xISs_exact) ]:

            obj = mfmc.MFMC(budget, model_type, hybrid=True)
            obj.set_corrs_and_costs(hifi_corr_list_exact, lofi_corr_list_exact, cost_list_exact)
            obj.set_objective_and_constraint(bounds)

            fval, x = solve_best_mfmc(obj, n_trials=50) #HARD CODED

            out = f'Variance ratio for {model_type} at budget {budget} is {fval} and occurs at {x} \n'
            hybrid_file.write(out)
            print(out)

            func_list.append(fval)
            x_list.append(x)

    ### Save data for plotting
    xx = np.array(budget_list)  # Convert budget_list to a NumPy array
    # xMFs = np.array(xMFs)
    # xMFs_exact = np.array(xMFs_exact)
    # xISs = np.array(xISs)
    # xISs_exact = np.array(xISs_exact)
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
    np.savez('corrcostvar.npz', **data_dict)

    # fig, ax = plt.subplots()
    # ax.loglog(xx, 1/xx, color='black', label='MC')
    # # plt.loglog(xx, 1/xx * np.array(funcMFs), linestyle=':', color='blue', label='ACV-MF Prediction')
    # # plt.loglog(xx, 1/xx * np.array(funcMFs_exact), color='blue', label='ACV-MF Actual')
    # # plt.loglog(xx, 1/xx * np.array(funcISs), linestyle=':', color='orange', label='ACV-IS Predicted')
    # # plt.loglog(xx, 1/xx * np.array(funcISs_exact), color='orange', label='ACV-IS Actual')
    # # plt.loglog(xx, np.exp(np.array(funcMFs)), linestyle=':', color='blue', label='ACV-MF Prediction')
    # # plt.loglog(xx, np.exp(np.array(funcMFs_exact)), color='blue', label='ACV-MF Actual')
    # # plt.loglog(xx, np.exp(np.array(funcISs)), linestyle=':', color='orange', label='ACV-IS Predicted')
    # # plt.loglog(xx, np.exp(np.array(funcISs_exact)), color='orange', label='ACV-IS Actual')
    # ax.loglog(xx, np.array(funcMFs), linestyle=':', color='blue', label='ACV-MF Prediction')
    # ax.loglog(xx, np.array(funcMFs_exact), color='blue', label='ACV-MF Actual')
    # ax.loglog(xx, np.array(funcISs), linestyle=':', color='orange', label='ACV-IS Predicted')
    # ax.loglog(xx, np.array(funcISs_exact), color='orange', label='ACV-IS Actual')
    # # plt.loglog(xx, 1/xMFs[:,0] * np.array(funcMFs), linestyle=':', color='blue', label='ACV-MF Prediction')
    # # plt.loglog(xx, 1/xMFs_exact[:,0] * np.array(funcMFs_exact), color='blue', label='ACV-MF Actual')
    # # plt.loglog(xx, 1/xISs[:,0] * np.array(funcISs), linestyle=':', color='orange', label='ACV-IS Predicted')
    # # plt.loglog(xx, 1/xISs_exact[:,0] * np.array(funcISs_exact), color='orange', label='ACV-IS Actual')
    # # plt.xlabel('Budget')
    # # plt.ylabel('Estimator Variance')
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig('variance_plot.pdf', transparent=True)
    # if show_plots: plt.show()

    hybrid_file.close()


    # ### Validate MFMC solution
    # out = "Validating the MFMC sampling strategy \n"
    # hybrid_file.write(out), print(out)

    # # Define quantities needed for validation sampling
    # lofi_sample_nums = np.ceil(r_star*N_star).astype(int)
    # num_fom_samples = np.ceil(N_star).astype(int)
    # num_model_samples_list = [num_fom_samples]
    # for sample_num in lofi_sample_nums:
    #     num_model_samples_list.append(sample_num)
    # if obj.type == "ACV-MF":
    #     total_lofi_samples = max(num_model_samples_list) - num_fom_samples
    # elif obj.type == "ACV-IS":
    #     total_lofi_samples = sum(num_model_samples_list) - \
    #         len(num_model_samples_list) * num_fom_samples
    # parameter_samples = parameter_space.generate_samples(num_fom_samples + total_lofi_samples)

    # # Make FOM directories and get its QoIs
    # out = "Sampling FOM model and computing N* QoIs \n"
    # hybrid_file.write(out), print(out)
    # parameter_samples_fom = parameter_samples[:num_model_samples_list[0]]
    # for sample_index, sample in enumerate(parameter_samples_fom):
    #     print("===========  Sample " + str(sample_index) + " ============")
    #     fom_run_directory = f'{hybrid_MFMC_directory}/sampling/fom/run_{sample_index}'
    #     parameter_dict = _create_parameter_dict(parameter_names, sample)
    #     _prepare_directory_and_run(fom_model, fom_run_directory, parameter_dict)
    #     fom_qoi = fom_model.compute_qoi(fom_run_directory, parameter_dict)
    #     if sample_index == 0:
    #         sampled_fom_qois = fom_qoi[None]
    #     else:
    #         sampled_fom_qois = np.append(sampled_fom_qois, fom_qoi[None], axis=0)

    # # Define parameter samples for aux and rom
    # # Aux strategy is identical for MF and IS
    # parameter_samples_aux = parameter_samples[:num_model_samples_list[1]]
    # if obj.type == "ACV-MF":
    #     parameter_samples_rom = parameter_samples[:num_model_samples_list[2]]
    # elif obj.type == "ACV-IS":
    #     independent_lofi_samples = parameter_samples[num_model_samples_list[1]:]
    #     parameter_samples_rom = parameter_samples_fom.append(independent_lofi_samples)

    # # Make Aux directories and get its QoIs
    # out = "Sampling auxilliary model and computing its QoIs \n"
    # hybrid_file.write(out), print(out)
    # for sample_index, sample in enumerate(parameter_samples_aux):
    #     print("===========  Sample " + str(sample_index) + " ============")
    #     aux_run_directory = f'{hybrid_MFMC_directory}/sampling/aux/run_{sample_index}'
    #     parameter_dict = _create_parameter_dict(parameter_names, sample)
    #     _prepare_directory_and_run(aux_model, aux_run_directory, parameter_dict)        
    #     aux_qoi = aux_model.compute_qoi(aux_run_directory, parameter_dict)
    #     if sample_index == 0:
    #         sampled_aux_qois = aux_qoi[None]
    #     else:
    #         sampled_aux_qois = np.append(sampled_aux_qois, aux_qoi[None], axis=0)

    # # Make ROM directories and get its QoIs
    # out = "Sampling ROM model and computing its QoIs \n"
    # hybrid_file.write(out), print(out)
    # for sample_index, sample in enumerate(parameter_samples_rom):
    #     print("===========  Sample " + str(sample_index) + " ============")
    #     rom_run_directory = rom_offline_directory + f'/run_{sample_index}'
    #     parameter_dict = _create_parameter_dict(parameter_names, sample)
    #     _prepare_directory_and_run(rom_model, rom_run_directory, parameter_dict)        
    #     rom_qoi = rom_model.compute_qoi(rom_run_directory, parameter_dict)
    #     if sample_index == 0:
    #         sampled_rom_qois = rom_qoi[None]
    #     else:
    #         sampled_rom_qois = np.append(sampled_rom_qois, rom_qoi[None], axis=0)            



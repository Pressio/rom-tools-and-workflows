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

'''
The greedy procedure iteratively constructs a reduced basis ROM until it reaches a desired tolerance.
The algorithm is as follows:
 1. We generate a parameter training set, $\\mathcal{D}_{\\mathrm{train}}, |\\mathcal{D}_{\\mathrm{train}}
    | = N_{\\mathrm{train}}$
 2. We select an initial sample, $\\mu_1 \\in \\mathcal{D}_{\\mathrm{train}} \\text{ and set }
    \\mathcal{D}_{\\mathrm{train}}=\\mathcal{D}_{\\mathrm{train}} - \\{\\mu_1\\}$
 3. We then solve the FOM to obtain the solution, $\\mathbf{u}(\\mu_1)$.
 4. We select a second sample, $\\mu_2 \\in \\mathcal{D}_{\\mathrm{train}} \\text{ and set }
    \\mathcal{D}_{\\mathrm{train}}=\\mathcal{D}_{\\mathrm{train}} - \\{\\mu_2\\}$
 4. We then solve the FOM to obtain the solution, $\\mathbf{u}(\\mu_2)$.
 5. We employ the first two solutions to compute the trial space, e.g.,
    $$\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\mathbf{u}(\\mu_2)) -  \\mathbf{u}_{\\mathrm{shift}},
    \\; \\mathbf{u}_{\\mathrm{shift}} = \\mathbf{u}(\\mu_1)$$
 6. We then solve the resulting ROM for the remaining parameter samples $\\mathcal{D}_{\\mathrm{train}}$ to
    generate approximate solutions $\\mathbf{u}(\\mu), \\mu \\in \\mathcal{D}_{\\mathrm{train}}$
 7. For each ROM solution $\\mathbf{u}(\\mu), \\mu \\in \\mathcal{D}_{\\mathrm{train}}$ we compuate an error
    estimate, $ e \\left(\\mu \\right) $
 8. If the maximum error estimate is less than some tolerance, we exit the algorithm. If not, we:
   - Set $ \\mu^* = \\underset{ \\mu \\in \\mathcal{D}_{\\mathrm{train}} }{ \\mathrm{arg\\; max} } \\; e
        \\left(\\mu \\right) $
   - Remove $\\mu^{\\*}$ from the training set, $\\mathcal{D}_{\\mathrm{train}}=
        \\mathcal{D}_{\\mathrm{train}} - \\{\\mu^{\\*}\\}$
   - Solve the FOM for $\\mathbf{u}(\\mu^{\\*})$
   - Set the reduced basis to be $\\boldsymbol \\Phi = \\mathrm{orthogonalize}(\\boldsymbol \\Phi,
      \\mathbf{u}(\\mu^{\\*})) -  \\mathbf{u}_{\\mathrm{shift}}$
   - Go back to step 6 and continue until convergence.

This function implements the basic greedy workflow. In addition, we enable an adaptive
error estimate based on a QoI. This is based on the fact that, throughout the greedy algorithm, we have a
history of error indicators as well as a set of FOM and ROM runs for the same parameter instance. We leverage this data
to improve the error estimate. Specifically, for the max error estimates over the first $j$ iterations,
$e_1,\\ldots,e_j$, we additionally can access QoI errors $e^q_1,\\ldots,e^q_j$ from our FOM-ROM runs, and we define a
scaling based on $$C = \\frac{\\sum_{i=1}^j | e^q_j| }{ \\sum_{i=1}^j | e_j | }.$$
The error at the $j+1$ iteration can be approximated by $C e(\\mu)$.
This will adaptively scale the error estimate to match the QoI error as
closely as possible, which can be helpful for defining exit criterion.
'''

import os
import time
import numpy as np

from romtools.workflows.greedy.\
    greedy_coupler_base import GreedyCouplerBase
from romtools.workflows.parameter_spaces import monte_carlo_sample
from romtools.workflows.models import *
from romtools.workflows.vector_space_builder import *
from romtools.workflows.parameter_spaces import *

def run_greedy(fom_model: QoiModel,
               rom_model: QoiModelWithErrorEstimate,
               vector_space_builder: VectorSpaceBuilder, 
               parameter_space: ParameterSpace,
               tolerance: float,
               testing_sample_size: int = 10):
    '''
    Main implementation of the greedy algorithm.
    '''
    base_directory = os.path.realpath(os.getcwd())
    run_directory_prefix = "run_"
    greedy_file = open("greedy_status.log", "w", encoding="utf-8")
    greedy_file.write("Greedy reduced basis status \n")
    fom_time = 0.
    rom_time = 0.
    basis_time = 0.

    np.random.seed(1)

    
    # create parameter samples 
    parameter_samples = monte_carlo_sample(parameter_space, testing_sample_size)
    parameter_names = parameter_space.get_names()

    # Make FOM/ROM directories
    end_sample_index = parameter_samples.shape[0]
    for sample_index in range(0, end_sample_index):
        rom_run_directory = base_directory + f'/rom_{run_directory_prefix}{sample_index}'
        fom_run_directory = base_directory + f'/fom_{run_directory_prefix}{sample_index}'
        create_empty_dir(rom_run_directory)
        create_empty_dir(fom_run_directory)
        parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])

        rom_model.populate_run_directory(rom_run_directory,parameter_dict)
        os.chdir(base_directory)
        fom_model.populate_run_directory(fom_run_directory,parameter_dict)
        os.chdir(base_directory)


    training_samples = np.array([0, 1], dtype='int')
    samples_left = np.arange(2, testing_sample_size)

    # Run FOM training cases
    t0 = time.time()
    training_dirs = []
    for i in training_samples:
        greedy_file.write(f"Running FOM sample {i} \n")
        sample_index = i
        parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])
        fom_run_directory = base_directory + f'/fom_{run_directory_prefix}{sample_index}'
        fom_model.run_model(fom_run_directory,parameter_dict)
        fom_qoi = fom_model.compute_qoi(fom_run_directory)
        if i == 0:
          fom_qois = fom_qoi[None]
        else:
          fom_qois = np.append(fom_qois,fom_qoi[None],axis=0) 
        os.chdir(base_directory)
        training_dirs.append(fom_run_directory)
    fom_time += time.time() - t0

    # Create ROM bases
    t0 = time.time()
    greedy_file.write("Creating ROM bases \n")
    vector_space_builder.build_and_save_vector_space_from_list_of_directories(training_dirs)
    basis_time += time.time() - t0

    # Evaluate ROM at training samples
    #     Do we actually need to do this?
    initial_errors = np.zeros(2)
    for i in training_samples:
        greedy_file.write(f"Running ROM sample {i}\n")
        t0 = time.time()
        sample_index = i
        rom_run_directory = base_directory + f'/rom_{run_directory_prefix}{sample_index}'
        os.chdir(rom_run_directory)
        parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])
        rom_model.run_model(parameter_dict)
        rom_qoi = rom_model.compute_qoi()
        error_indicator = rom_model.compute_error_indicator()
        if i == 0:
          rom_qois = rom_qoi[None]
          error_indicators = error_indicator[None]
        else:
          rom_qois = np.append(rom_qois,rom_qoi[None],axis=0) 
          error_indicators = np.append(error_indicators,error_indicator[None],axis=0)
        os.chdir(base_directory)
        rom_time += time.time() - t0
        greedy_file.write(f"Computing ROM/FOM error for sample {i} \n")
        initial_errors[i] = np.linalg.norm(rom_qois[i] - fom_qois[i]) / np.linalg.norm(fom_qois[i])

    converged = False
    max_error_indicators = np.zeros(0)
    reg = QoIvsErrorIndicatorRegressor()
    qoi_errors = np.zeros(0)
    outer_loop_counter = 0
    predicted_qoi_errors = np.zeros(0)
    greedy_file.flush()

    while converged is False:
        print(f'Greedy iteration # {outer_loop_counter}')
        error_indicators = np.zeros(samples_left.size)

        t0 = time.time()
        greedy_file.write(f"Greedy iteration # {outer_loop_counter}\n" +
                          f"Parameter samples: \n {parameter_samples}\n")
        greedy_file.flush()
        for counter, sample_index in enumerate(samples_left):
            greedy_file.write(f"    Running ROM sample {sample_index}\n")
            greedy_file.flush()
            parameter_dict = _create_parameter_dict(parameter_names,parameter_samples[sample_index])
            rom_run_directory = base_directory + f'/rom_{run_directory_prefix}{sample_index}'
            rom_model.run_model(rom_run_directory,parameter_dict)
            error_indicators[counter] = rom_model.compute_error_indicator(rom_run_directory,parameter_dict)
            os.chdir(base_directory)

        rom_time += time.time() - t0

        sample_with_highest_error_indicator = samples_left[np.argmax(error_indicators)]
        max_error_indicators = np.append(max_error_indicators,
                                         np.amax(error_indicators))
        greedy_file.write(f"Sample {sample_with_highest_error_indicator}"
                          " had the highest error indicator of"
                          f" {max_error_indicators[-1]}")

        outer_loop_counter += 1
        if outer_loop_counter > 1:
            predicted_max_qoi_error = reg.predict(error_indicators[np.argmax(error_indicators)])
            greedy_file.write("Our MLEM error estimate is "
                              f"{predicted_max_qoi_error}\n")
            greedy_file.flush()
            predicted_qoi_errors = np.append(predicted_qoi_errors,
                                             predicted_max_qoi_error)
            if np.amax(predicted_max_qoi_error < tolerance):
                converged = True
                print('Run converged, max approximate qoi error'
                      f' = {predicted_max_qoi_error}')
                break

        t0 = time.time()
        greedy_file.write("Running FOM sample"
                          f" {sample_with_highest_error_indicator}\n")
        greedy_file.flush()


        ## Identify the sample with the highest error and run FOM
        fom_run_directory = base_directory + f'/fom_{run_directory_prefix}{sample_with_highest_error_indicator}'
        parameter_dict = _create_parameter_dict(parameter_names,samples_left[sample_with_highest_error_indicator])
        fom_model.run_model(fom_run_directory,parameter_dict)
        fom_qoi = fom_model.compute_qoi(fom_run_directory,parameter_dict)
        os.chdir(base_directory)

        fom_time += time.time() - t0

        # Get ROM QoI to calibrate our error estimator
        rom_run_directory = base_directory + f'/rom_{run_directory_prefix}{sample_with_highest_error_indicator}'
        rom_qoi = rom_model.compute_qoi(rom_run_directory,parameter_dict)
        qoi_error = np.linalg.norm(rom_qoi - fom_qoi) / np.linalg.norm(fom_qoi) 
        greedy_file.write(f"Sample {sample_with_highest_error_indicator} had "
                          f"an error of {qoi_error}\n")
        qoi_errors = np.append(qoi_errors, qoi_error)
        reg.fit(max_error_indicators, qoi_errors)

        os.chdir(base_directory)

        ## Update our samples
        training_samples = np.append(training_samples,
                                     sample_with_highest_error_indicator)
        samples_left = np.delete(samples_left, np.argmax(error_indicators))

        # Update ROM basis
        t0 = time.time()
        greedy_file.write("Computing ROM bases \n")
        greedy_file.flush()
        training_dirs = [f'/rom_{run_directory_prefix}{i}' for i in training_samples]
        vector_space_builder.build_and_save_vector_space_from_list_of_directories(training_dirs)
        basis_time += time.time() - t0

        # Add a new sample
        new_parameter_sample = monte_carlo_sample(parameter_space, 1)
        parameter_samples = np.append(parameter_samples,
                                      new_parameter_sample, axis=0)
        new_sample_number = testing_sample_size + outer_loop_counter - 1
        greedy_coupler.create_fom_and_rom_cases(new_sample_number,
                                                new_parameter_sample)

        rom_run_directory = base_directory + f'/rom_{run_directory_prefix}{new_sample_number}'
        fom_run_directory = base_directory + f'/fom_{run_directory_prefix}{new_sample_number}'
        create_empty_dir(rom_run_directory)
        create_empty_dir(fom_run_directory)
        parameter_dict = _create_parameter_dict(parameter_names,new_parameter_sample)

        rom_model.populate_run_directory(rom_run_directory,parameter_dict)
        os.chdir(base_directory)
        fom_model.populate_run_directory(fom_run_directory,parameter_dict)
        os.chdir(base_directory)


        samples_left = np.append(samples_left, new_sample_number)
        greedy_file.flush()
        np.savez('greedy_stats',
                 max_error_indicators=max_error_indicators,
                 qoi_errors=qoi_errors,
                 predicted_qoi_errors=predicted_qoi_errors,
                 training_samples=training_samples,
                 fom_time=fom_time,
                 rom_time=rom_time,
                 basis_time=basis_time)

    greedy_file.close()


class QoIvsErrorIndicatorRegressor:
    '''
    Regressor for modeling the relationship between QoI error and error
    indicator.

    This class provides a simple linear regression model for estimating the
    scaling factor (c) between QoI error and the error indicator.
    '''
    def __init__(self):
        '''
        Initializes the regressor with a default scaling factor of 1.0.
        '''
        self.__c = 1.

    def fit(self, x, y):
        '''
        Fits the regression model to the provided data points (x, y) to
        estimate the scaling factor.

        Args:
            x: Error indicator values.
            y: Corresponding QoI error values.
        '''
        self.__c = np.mean(y) / np.mean(x)

    def predict(self, x):
        '''
        Predicts QoI error based on the error indicator using the estimated
        scaling factor.

        Args:
            x: Error indicator value(s) for prediction.
        '''
        return self.__c*x

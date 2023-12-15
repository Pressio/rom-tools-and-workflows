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


def runGreedy(greedyCoupler, tolerance, testing_sample_size=10):
    '''
    Main implementation of the greedy algorithm.
    '''
    greedy_file = open("greedy_status.log", "w", encoding="utf-8")
    greedy_file.write("Greedy reduced basis status \n")
    fom_time = 0.
    rom_time = 0.
    basis_time = 0.

    np.random.seed(1)
    # create parameter domain
    parameterSpace = greedyCoupler.getParameterSpace()

    # lower_bounds,upper_bounds = greedyCoupler.get_parameter_bounds()
    parameter_samples = parameterSpace.generate_samples(testing_sample_size)
    # Make FOM/ROM directories
    starting_sample_index = 0
    greedyCoupler.createFomAndRomCases(starting_sample_index, parameter_samples)
    # Run first FOM case
    fom_directory = (
        greedyCoupler.getWorkDirectoryBaseName() + '/' +
        greedyCoupler.getFomDirectoryBaseName() + '_' +
        str(starting_sample_index) + '/'
    )
    t0 = time.time()
    greedy_file.write("Running FOM sample 0 \n")
    os.chdir(fom_directory)
    greedyCoupler.runFom(greedyCoupler.getFomInputFileName(), parameter_samples[0])
    os.chdir(greedyCoupler.getBaseDirectory())

    fom_time += time.time() - t0

    # Run second FOM case
    fom_directory = (
        greedyCoupler.getWorkDirectoryBaseName() + '/' +
        greedyCoupler.getFomDirectoryBaseName() + '_' +
        str(starting_sample_index + 1) + '/'
    )
    t0 = time.time()
    greedy_file.write("Running FOM sample 1 \n")
    os.chdir(fom_directory)
    greedyCoupler.runFom(greedyCoupler.getFomInputFileName(),parameter_samples[1])
    os.chdir(greedyCoupler.getBaseDirectory())
    fom_time += time.time() - t0

    # Create ROM cases
    training_samples = np.array([0,1],dtype='int')
    # greedyCoupler.create_rom_cases(starting_sample_index,samples)
    t0 = time.time()
    greedy_file.write("Creating ROM bases \n")
    # Create list of directories for training
    training_dirs = []
    for i in training_samples:
        path_to_dir = (
            greedyCoupler.getBaseDirectory() + '/' +
            greedyCoupler.getWorkDirectoryBaseName() + '/' +
            greedyCoupler.getFomDirectoryBaseName() + '_' + str(i)
        )
        training_dirs.append(path_to_dir)

    greedyCoupler.createTrialSpace(training_dirs)
    basis_time += time.time() - t0

    initial_errors = np.zeros(2)
    initial_error_indicators = np.zeros(2)
    greedy_file.write("Running ROM sample 0 \n")
    rom_directory = greedyCoupler.getWorkDirectoryBaseName() + '/' + greedyCoupler.getRomDirectoryBaseName() + '_0/'
    t0 = time.time()
    os.chdir(rom_directory)
    greedyCoupler.runRom(greedyCoupler.getRomInputFileName(),parameter_samples[0])
    initial_error_indicators[0] = greedyCoupler.computeErrorIndicator()
    os.chdir(greedyCoupler.getBaseDirectory())
    rom_time += time.time() - t0
    greedy_file.write("Computing ROM/FOM error for sample 0 \n")
    fom_directory =  greedyCoupler.getWorkDirectoryBaseName() + '/' + greedyCoupler.getFomDirectoryBaseName() + '_0/'

    initial_errors[0] = greedyCoupler.computeError(rom_directory,fom_directory)

    greedy_file.write("Running ROM sample 1 \n")
    rom_directory = greedyCoupler.getWorkDirectoryBaseName() + '/' + greedyCoupler.getRomDirectoryBaseName() + '_1/'
    t0 = time.time()
    os.chdir(rom_directory)
    greedyCoupler.runRom(greedyCoupler.getRomInputFileName(),parameter_samples[1])
    initial_error_indicators[1] = greedyCoupler.computeErrorIndicator()
    os.chdir(greedyCoupler.getBaseDirectory())
    rom_time += time.time() - t0

    greedy_file.write("Computing ROM/FOM error for sample 1 \n")
    fom_directory = greedyCoupler.getWorkDirectoryBaseName() + '/' + greedyCoupler.getFomDirectoryBaseName() + '_1/'
    initial_errors[1] = greedyCoupler.computeError(rom_directory,fom_directory)


    samples_left = np.arange(0,testing_sample_size)
    samples_left = np.delete(samples_left,0)
    samples_left = np.delete(samples_left,0)

    converged = False
    max_error_indicators = np.zeros(0)
    qoi_errors = np.zeros(0)
    outer_loop_counter = 0

    def reg(val):
        return val

    predicted_qoi_errors = np.zeros(0)
    greedy_file.flush()
    while converged is False:
        print('Greedy iteration # ' + str(outer_loop_counter))
        error_indicators = np.zeros(samples_left.size)
        counter = 0
        t0 = time.time()
        greedy_file.write("Greedy iteration # " + str(outer_loop_counter) + " \n" +
                          "Parameter samples: \n " + str(parameter_samples) + " \n")
        greedy_file.flush()
        for i in samples_left:
            rom_directory = (
                greedyCoupler.getWorkDirectoryBaseName() + '/' +
                greedyCoupler.getRomDirectoryBaseName() + '_' + str(i) + '/'
            )
            greedy_file.write("    Running ROM sample " + str(i) + " \n")
            greedy_file.flush()
            os.chdir(rom_directory)
            greedyCoupler.runRom(greedyCoupler.getRomInputFileName(), parameter_samples[i])
            error_indicators[counter] = greedyCoupler.computeErrorIndicator()
            os.chdir(greedyCoupler.getBaseDirectory())
            counter += 1
        rom_time += time.time() - t0

        sample_with_highest_error_indicator = samples_left[np.argmax(error_indicators)]
        max_error_indicators = np.append(max_error_indicators, np.amax(error_indicators))
        greedy_file.write("Sample " + str(sample_with_highest_error_indicator) +
                  " had the highest error indicator of " +
                  str(max_error_indicators[-1]) + " \n")

        outer_loop_counter += 1
        if outer_loop_counter > 1:
            predicted_max_qoi_error = reg.predict(error_indicators[np.argmax(error_indicators)])
            greedy_file.write("Our MLEM error estimate is " + str(predicted_max_qoi_error) + " \n")
            greedy_file.flush()
            predicted_qoi_errors = np.append(predicted_qoi_errors, predicted_max_qoi_error)
            if np.amax(predicted_max_qoi_error < tolerance):
                converged = True
                print('Run converged, max approximate qoi error = ' + str(predicted_max_qoi_error) )
                break

        fom_directory = (
            greedyCoupler.getWorkDirectoryBaseName() + '/' +
            greedyCoupler.getFomDirectoryBaseName() + '_' +
            str(sample_with_highest_error_indicator)
        )
        t0 = time.time()
        greedy_file.write("Running FOM sample " + str(sample_with_highest_error_indicator) + " \n")
        greedy_file.flush()
        os.chdir(fom_directory)
        greedyCoupler.runFom(greedyCoupler.getFomInputFileName(), parameter_samples[sample_with_highest_error_indicator])
        os.chdir(greedyCoupler.getBaseDirectory())
        fom_time += time.time() - t0
        training_samples = np.append(training_samples, sample_with_highest_error_indicator)

        samples_left = np.delete(samples_left, np.argmax(error_indicators))

        rom_directory = (
            greedyCoupler.getWorkDirectoryBaseName() + '/' +
            greedyCoupler.getRomDirectoryBaseName() + '_' +
            str(sample_with_highest_error_indicator) + '/'
        )
        fom_directory = (
            greedyCoupler.getWorkDirectoryBaseName() + '/' +
            greedyCoupler.getFomDirectoryBaseName() + '_' +
            str(sample_with_highest_error_indicator) + '/'
        )
        qoi_error = greedyCoupler.computeError(rom_directory, fom_directory)
        greedy_file.write("Sample " + str(sample_with_highest_error_indicator) + " had an error of " + str(qoi_error) +  " \n")
        qoi_errors = np.append(qoi_errors, qoi_error)
        reg = qoi_vs_error_indicator_regressor()
        reg.fit(max_error_indicators, qoi_errors)
        t0 = time.time()
        greedy_file.write("Computing ROM bases \n")
        greedy_file.flush()

        training_dirs = []
        for i in training_samples:
            path_to_dir = (
                greedyCoupler.getBaseDirectory() + '/' +
                greedyCoupler.getWorkDirectoryBaseName() + '/' +
                greedyCoupler.getFomDirectoryBaseName() + '_' + str(i)
            )
            training_dirs.append(path_to_dir)

        greedyCoupler.createTrialSpace(training_dirs)
        basis_time += time.time() - t0
        # Add a new sample
        new_parameter_sample = parameterSpace.generate_samples(1)
        parameter_samples = np.append(parameter_samples, new_parameter_sample, axis=0)
        new_sample_number = testing_sample_size + outer_loop_counter - 1
        greedyCoupler.createFomAndRomCases(new_sample_number, new_parameter_sample)
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



class qoi_vs_error_indicator_regressor:
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

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

"""
This module implements the class required to couple a model to Dakota.
To couple to Dakota, a user should
1. Complete a Model class for their application of interest
2. Complete a driver script that instantiates the model and calls this coupler for use as the Dakota analysis driver
"""

import sys
import numpy as np
import os

from romtools.workflows.models import QoiModel


def _create_parameter_dict(parameter_names, parameter_values):
    return dict(zip(parameter_names, parameter_values))


class DakotaSamplingCoupler:

    def __init__(self, model: QoiModel):
        """
        Initializes a DakotaSamplingCouplerBase object.

        Args:
            model (QoiModel): A FOM or ROM that returns a QoI
        """
        self.model = model

    def run_model(self, run_directory: str, parameter_sample: dict):
        """
        Executes the model.

        Args:
            run_directory (str): Where the model is run
            parameter_sample (dict): A dictionary defining a single parameter sample
        """
        self.model.run_model(run_directory, parameter_sample)
        pass

    def compute_qoi_and_save_to_file(
        self, run_directory, parameter_sample, results_file
    ):
        """
        Computes the QoI and saves it to a file. The output file
        should match what is specified in the Dakota input script
        """
        qoi = self.model.compute_qoi(run_directory, parameter_sample)
        np.savetxt(results_file, qoi)
        pass

    def __call__(self):
        """
        This class should be used in a driver script that will be called with
            `python <driver.py> params.in results.out`

        by Dakota. The parameter space is built by parsing params.in, 
        while the output QoI will be saved to results.out
        """
        # Read parameters from param.in file (1st command line argument)
        dtype = [("floats", float), ("strings", "U100")]
        data = np.genfromtxt(sys.argv[1], dtype=dtype, encoding=None, delimiter=" ")

        data_floats = data["floats"]
        data_strings = data["strings"]

        num_vars = int(data_floats[0])
        parameter_values = np.array(data_floats[1 : 1 + num_vars])
        parameter_names = data_strings[1 : 1 + num_vars]

        # Run directory is current dir, results dir is from Dakota
        run_directory = os.getcwd()
        results_file = sys.argv[2]

        # Initialize and run ROM
        parameter_sample = _create_parameter_dict(parameter_names, parameter_values)
        self.model.populate_run_directory(run_directory, parameter_sample)
        self.run_model(run_directory, parameter_sample)

        # Save ROM QoI
        self.compute_qoi_and_save_to_file(run_directory, parameter_sample, results_file)
